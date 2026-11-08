# Results

Measured on a base M3 Mac (10-core GPU, 16 GB, ~100 GB/s memory bandwidth, ~3.5 TFLOPS), MLX 0.32.4,
mlx-audio 0.6.4, Qwen3-TTS-12Hz-1.7B-Base in bf16. The test script is a documentary-style narration script
(~1,400 words, 66 chunks, about 9 minutes; not included in the repo). The reference voices were made with Gemini TTS (voice
Alnilam) using `scripts/make_ref_gemini.py`: a brisk read (180 wpm, 20 s) and a paced read (148 wpm, 35 s,
`scripts/reference_text.txt`). "Compute per narration second" is seconds of compute per second of finished
narration, excluding model loading. The raw per-run numbers are in `perf/runs.jsonl`
(`uv run python scripts/perf_table.py`), and the benchmark outputs are in `perf/bench/`.

Speed varies between runs on this machine: 0.98 and 1.16 for two runs of one configuration, and 0.85 and 0.96
for the 50-frame decoder with and without a cache cap (which changes no computation). Single-run differences
of 15–20% or less are noise.

## Every chunk is cloned from the reference

Three ways of conditioning chunk *i*, first 15 chunks, one take each:

| conditioning | WER | speaker similarity, mean (worst) | first→last change |
|---|---|---|---|
| clean reference only | 0.9% | **0.891** (0.861) | −0.002 |
| previous chunk as the prompt, speaker vector pinned to the reference | 1.1% | 0.756 (0.628) | −0.131 |
| reference + previous chunk as the prompt | 0.7% | 0.619 (0.223) | −0.427 |

Continuing from its own audio, the voice drifts within a minute or two, and it gets darker (−4 dB above
4 kHz). The pipeline never feeds generated audio back. Continuity comes from text instead: each chunk is read
with a lead-in and a guard that are cut away afterwards. In a blind listen, chunks read this way joined more
naturally than chunks read on their own.

Over the full script, speaker similarity (ECAPA, 15 s windows) averages 0.89, with the lowest window at 0.83.
It changes by less than 0.01 between the first and last minute. Loudness varies by 0.7 LU (standard deviation
over windows).

## Speed

| step | compute per narration second |
|---|---|
| one chunk at a time (up to 3 takes, best kept, a Whisper pass per take; a Kokoro reference voice) | 8.70 |
| batched (32 chunks at a time), cached decode, packed Whisper, first passing take kept | 1.78 |
| the clipped-ending check only adds cost, instead of forcing retakes | 1.04 (brisk reference) |
| paced reference, decoder in 50-frame steps | 0.85–1.16 |

In the final runs, generation is ~0.40 and Whisper ~0.15 s of compute per second of *generated* audio. Each
second of finished narration takes ~1.74 s of generated audio: lead-ins, guards and retakes. Checking takes
on the CPU runs alongside the next batch, so it costs nothing.

### Batch size

`scripts/bench_batch.py`, 16–64 texts, mlx-audio's own decode (before the cached decoder):

| batch | compute per audio second | generation loop only | decode |
|---|---|---|---|
| 1 | 1.44 | 1.09 | 0.35 |
| 4 | 0.79 | 0.44 | 0.35 |
| 16 | 0.65 | 0.31 | 0.34 |
| 32 | 0.59 | 0.24 | 0.35 |
| 64 | 0.58 | 0.22 | 0.35 |

At batch 1 the generation step is bound by memory bandwidth (reading the weights). Batching shares that read
across streams. Gate pass rates did not trend with batch size.

### One generation step

`scripts/profile_step.py`, ms per step, stages synced:

| batch | talker | code predictor (15 sub-steps) | sampling | per stream-frame | prefill |
|---|---|---|---|---|---|
| 1 | 36 | 41 | 0.7 | 79 | 0.4 s |
| 16 | 98 | 140 | 3.2 | 15 | 10.5 s |
| 64 | 179 | 293 | 11 | 7.6 | 41.5 s |

Sampling and Python bookkeeping are about 2% of a step. At batch 64 the talker reaches about 1 TFLOPS and the
code predictor about 0.5 TFLOPS. Compiling the code predictor into one graph gave identical codes but only
1–4% speed, so its remaining gap is kernel efficiency. Prefill costs about 0.65 s per stream. Qwen's cloning
prompt puts the target text before the reference, so it can't be shared across chunks.

### Audio decode

`scripts/check_prefix_decode.py`. Every chunk is decoded with the whole reference before it as acoustic
context. The ground truth is one unchunked decoder pass over reference + chunk, compared sample by sample:

| decode | compute per audio second | match with the truth |
|---|---|---|
| mlx-audio default (re-decodes the reference per chunk, 300-frame pieces with 25 frames of overlap, proportional trim) | 0.361 | 9.4 dB SNR |
| batched decode, 2 / 4 / 8 streams | 0.33 / 0.34 / 0.53 | same as default; 15–27 GB peak memory |
| **reference-prefix cache** (`qwen_fast.py`) | **0.135** | **118 dB SNR (float rounding)** |
| only the last 25 reference frames as context | 0.149 | 17 dB (lossy, not used) |

The decoder is causal (causal convolutions and attention), so its state after the reference is the same for
every chunk. The prefix cache decodes the reference once, snapshots that state (convolution buffers and the
attention KV cache) and resumes from it per chunk. It is more exact than mlx-audio's default as well as 2.7×
faster. The decoder is compute-bound, so batching it only adds memory.

### Whisper

`scripts/bench_whisper.py`, 66 real chunks (486 s). One clip at a time costs 0.41 s per audio second. The
encoder always processes a padded 30 s window, so a 7 s clip costs as much as a 30 s one. Packing several
clips into one window, separated by silence, divides that cost:

| variant | speedup | gate decisions changed | word-start shift (mean / p95) |
|---|---|---|---|
| packed, 28 s windows, 1.0 s gaps | 2.05× | 1 of 66 | 15 / 50 ms |
| **packed, 28 s windows, 0.6 s gaps** | **2.70×** | **0 of 66** | 15 / 50 ms |

The remaining WER differences are Whisper's wording choices (one word or two for a compound name), not words assigned to the
wrong clip. Each take is transcribed once, and that transcript serves both the cut and the gate. Taking only
the words inside the cut gave the same WER as re-transcribing the cut clip on 12 of 12 chunks.

## The take gate

- **Best-of-N bought nothing.** When the first take already passed, a later take was picked about half the time
  and speaker similarity changed by −0.002 to −0.007 on average. The pipeline keeps the first passing take.
- **A clipped ending is a cost, not a failure.** The gate used to reject takes whose last 80 ms were within
  12 dB of the speech level, as possibly cut off mid-word. In a blind listen, takes failing only that check
  sounded as good as passing ones. What did sound wrong were short gaps, inside takes and at joins. Making it
  a cost alone took the full script from 1.78 to 1.04 s of compute per narration second, and from 1.77 to
  1.26 takes per chunk.
- **Cut robustness.**
  - Whisper often stamps the pause before a word as part of that word, so the pause search runs up to 0.3 s
    into the next word.
  - A chunk whose cut fails on every take is read once more without lead-in or guard, so it can't drop out
    of the narration.

## Reference pacing

Qwen3-TTS copies the reference's pacing along with its timbre. Full script, same pipeline:

| | brisk reference (180 wpm) | paced reference (148 wpm) |
|---|---|---|
| reference's own pauses after full stops | 0.15–0.32 s | 0.41–0.73 s |
| sentence breaks run together inside takes (< 150 ms) | 32% | 25% |
| median sentence pause inside takes | 0.21 s | 0.26 s |
| narration pace | 168 wpm | 155 wpm |
| speaker similarity to its reference | 0.884 | 0.890 |
| WER | 1.9% | 2.0% |
| takes per chunk | 1.26 | 1.15 |

In a blind listen, the paced reference's narration was preferred over the brisk one and over a second paced
take. The other two had flaws in pronunciation and tone. The paced reference's text also covers every phoneme
of British English. The longer reference costs about 20% more generation per take, because the prompt is
longer.

## Memory

`mx.get_peak_memory()` counts only live arrays. MLX also keeps freed buffers cached for reuse, by default up
to the whole memory limit, and PyTorch (the ECAPA model) is invisible to MLX's counters. Each run logs MLX's
live peak, live + cached (sampled every 0.25 s), and the process's peak physical footprint as macOS counts it
(`proc_pid_rusage`), which is what decides whether the machine swaps.

| | memory |
|---|---|
| Qwen weights | 4.6 GB, held |
| Whisper large-v3-turbo | 1.6 GB held, +2 GB while transcribing |
| encoding the reference | +2.1 GB briefly |
| audio decoder, per call | ~0.8 GB at 10 frames, 1.6 at 25, 2.5 at 50, 3.7 at 100, ~4 at 300 |
| talker KV cache | ~114 KB per token per stream: ~4 GB at batch 32 with a 35 s reference |

The decoder's memory grows with the frames it takes per call. Each step, and the decoder state it hands on, is
evaluated before the next step is built. Otherwise MLX runs all the steps as one lazy graph and the step size
bounds nothing. The output is the same for any step size.

Full script, paced reference, batch 32 (same seeds):

| | decode step 300, no cache cap | decode step 50, cache cap 2 GB |
|---|---|---|
| peak live | 12.61 GB | 11.18 GB |
| peak live + cached | 12.61 GB | 12.08 GB |
| peak footprint | 13.39 GB | 13.31 GB |
| compute per narration second | 0.98 | 0.96 |

With the decoder bounded, the peak is the first generation batch, mostly the KV cache. On 34 chunks, batch 16
instead of 32 cut the live peak from 10.6 to 9.4 GB and the footprint from 12.85 to 12.28 GB. The two runs
above differ in one chunk's Whisper transcript, from a ~1e-5 difference in the audio, which crossed the WER
threshold. Every later round then batched different chunks with different seeds. Both are equally valid
samples.
