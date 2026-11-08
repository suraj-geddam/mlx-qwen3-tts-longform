# mlx-qwen3-tts-longform

Long-form narration (chapters, video scripts, 10+ minutes) from
[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) on Apple Silicon, with
[MLX](https://github.com/ml-explore/mlx). Give it a reference voice and a script. It clones that voice for the
whole script with no drift, makes natural joins between chunks, checks every take with Whisper, and runs at
about realtime on a base M3 Mac.

> **Status: experimental.** Everything here, the code, the experiments and these docs, was implemented with
> Claude Opus 5.5 under human direction, with listening tests by a human. It has been run end to
> end on one machine (M3, 16 GB) and it probably works, but expect rough edges.

## Why this exists

I wanted long-form narration from a local model whose output I'm allowed to use commercially, for example in
monetised YouTube videos. Long-form TTS has two failure modes: the voice drifts when a model keeps continuing
from its own audio, and chunk joins sound stitched together. Qwen3-TTS 1.7B is Apache-2.0 and clones voices
well. This repo makes it hold one voice for as long as you like, with natural joins, at about realtime.

**If you don't need commercial use:** for long-form voice cloning, I've found
[Echo-TTS](https://huggingface.co/jordand/echo-tts-base) much better, run through
[longecho](https://github.com/suraj-geddam/longecho). Its weights are non-commercial (CC-BY-NC-SA 4.0). Why I
think Echo is better at long-form, and what this repo does to make up for it, is in [NOTES.md](NOTES.md).

## Requirements

- A Mac with Apple Silicon. Memory: a run's footprint peaks at about 13–14 GB with the defaults, so 16 GB works if
  other memory-hungry apps are closed; see [Memory](#memory) for lower-memory options.
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs Python 3.12+
  and the dependencies.
- About 6 GB of disk for models, downloaded from Hugging Face on first use: Qwen3-TTS 1.7B bf16 (4.2 GB),
  Whisper large-v3-turbo (1.5 GB), ECAPA speaker encoder (85 MB).
- A reference voice: see below.

## Quickstart

```bash
git clone https://github.com/suraj-geddam/mlx-qwen3-tts-longform
cd mlx-qwen3-tts-longform
uv sync
uv run qwen3-longform --ref my_voice.wav --script examples/long.txt --out out/test --limit 8
```

`--limit 8` narrates only the first 8 chunks (about a minute of audio). Drop it to narrate the whole of
`examples/long.txt`, O. Henry's *The Gift of the Magi* (public domain, about 13 minutes). The test output will
be at `out/test/narration.wav`. `examples/short.txt` is a 45-second example with two chapters.

## Your reference voice

The repo ships no audio. Bring a clean recording of a voice you have the rights to use:

- **10–40 s** of one speaker, no music or background noise, ideally 24 kHz or higher.
- **An exact transcript** in a `.txt` file with the same name beside the wav (`my_voice.wav` and
  `my_voice.txt`), or pass it with `--ref-text`. Without one, Whisper transcribes the reference and saves it
  there for you to correct. Cloning is better with an exact transcript.
- **Pacing matters.** Qwen3-TTS copies the reference's pacing along with its timbre. A brisk read makes the
  clone run sentences together; an unhurried read with clear pauses after full stops makes it pause too.
- Text that covers many different sounds helps the clone pronounce words the reference doesn't contain.
  `scripts/reference_text.txt` covers all 41 phonemes of British English.

**Optional:** `scripts/make_ref_gemini.py` is the script I used to make my reference voice with Gemini TTS.
It reads `scripts/reference_text.txt` with an unhurried direction and writes `refs/<voice>_<n>.wav` plus the
transcript. With `GEMINI_API_KEY` set in your environment, run:

```bash
uv run python scripts/make_ref_gemini.py Alnilam --takes 2
```

## Your script

- **Plain text** (`.txt` or `.md`): paragraphs separated by blank lines. A line starting with `#` starts a new
  chapter. The heading isn't read aloud; chapters get a longer pause between them.
- **JSON:** `{"lines": [{"ch": "intro", "text": "...", "say": "..."}]}`. `say` is the spoken form, read and
  checked instead of `text` when they differ (for example `"Dr"` read as `"Doctor"`, or a respelled name).

Before reading, the text is normalized for speech (from [longecho](https://github.com/suraj-geddam/longecho);
`--no-normalize` turns it off):
- currency: "$1.87" becomes "1 dollar and 87 cents", "an $8 flat" becomes "an 8-dollar flat", "$5M" becomes
  "5 million dollars";
- common abbreviations: "Dr." becomes "Doctor", "Mrs." becomes "Missus", "e.g." becomes "for example";
- symbols: "50%" becomes "50 percent", "&" becomes "and";
- parentheses become commas.

Plain numbers are left as digits: Qwen reads them in context ("in 1999" vs "1999 people"). The take
checker normalizes both the script and Whisper's transcript, so "$1.87" and "one dollar and eighty-seven
cents" count as the same words. Anything else that should be read a particular way, such as an unusual name,
is best respelled in the script.

## Output

`out/<name>/`:
- `narration.wav`: the finished narration (24 kHz mono).
- `chunks/NNN.wav`: each kept chunk.
- `run.json`: every take of every chunk with its scores, the pause inserted after each chunk, speed and memory.
- `metrics.json` (with `--analyse`): drift over the narration in 15 s windows (speaker similarity, loudness,
  pitch, brightness, pace) and the WER of the whole narration.

Each run also appends its speed and memory to `out/perf_runs.jsonl`; `uv run python scripts/perf_table.py
out/perf_runs.jsonl` prints them as a table.

## How it works

1. **Chunks.** The script is split into runs of whole sentences of about 6–10 s of speech, never across a
   chapter.
2. **Every chunk is cloned from the clean reference.** Feeding a chunk's own output back as the next prompt
   makes the voice drift: in testing, speaker similarity fell from 0.87 to 0.65 over 15 chunks. Cloning each
   chunk directly from the reference keeps it steady for any length (0.89 on average over 9 minutes, no trend).
3. **Lead-in and guard.** Each chunk is read with the end of the previous chunk before it and the start of the
   next one after it. Both are cut off at real pauses, found with Whisper word timestamps, so every kept chunk
   starts and ends with the prosody it would have mid-paragraph. The pause inserted after a chunk is the
   narrator's own pause before the guard, never shorter than the reference's typical sentence pause.
4. **Take gate.** Each take is checked: Whisper word error rate against the script (≤ 8%), speaker similarity
   to the reference (ECAPA, ≥ 0.55), loudness within 4 LU of the reference, and a sane pace. The first take
   that passes is kept. A failing chunk is regenerated with a new seed, up to `--takes` times, and the best
   take is kept if none passes.
5. **Fast.**
   - **Batching:** 32 chunks are generated together, so one read of the weights per step serves 32 streams.
   - **Exact cached decode:** the audio decoder is causal, so its state after the reference is the same for
     every chunk. It is computed once and reused, which is exact and 2.7× faster than re-decoding the
     reference per chunk.
   - **Packed Whisper:** several takes share one 30 s Whisper window, 2.7× faster with the same gate decisions.
   - **Overlap:** CPU-side checks run while the GPU generates the next batch.

## Options

| flag | default | |
|---|---|---|
| `--ref` | required | reference voice (wav) |
| `--ref-text` | `<ref>.txt` | its exact transcript |
| `--script` | required | text to narrate (.txt, .md or .json) |
| `--out` | required | output folder |
| `--no-normalize` | off | read the text as written, without normalization |
| `--takes` | 3 | most takes per chunk when takes fail the gate |
| `--limit N` | all | narrate only the first N chunks |
| `--batch` | 32 | chunks generated together |
| `--decode-step` | 50 | frames per audio-decoder call |
| `--cache-limit GB` | none | cap on MLX's buffer cache |
| `--perf-log` | `out/perf_runs.jsonl` | where to append speed and memory |
| `--analyse` | off | measure drift over the finished narration |

### Memory

With the defaults, a run's peak footprint is about 13–14 GB. It is roughly the live peak plus what MLX keeps
cached, plus about 1 GB outside MLX. The live peak is Qwen's weights (4.6 GB), Whisper (1.6 GB, plus ~2 GB
while transcribing), the talker's KV cache (grows with `--batch`) and the decoder's working memory (grows
with `--decode-step`). None of these options changes the output:

- `--batch 16`: about 1.2 GB less; generation up to ~10% slower.
- `--decode-step 25`: about 1 GB less than the default 50; decoding a little slower.
- `--cache-limit 2`: MLX keeps at most 2 GB of freed buffers for reuse.

Runs record macOS swap activity. If a run swapped, it prints a warning and its timings aren't representative.

## Speed and quality

On a base M3 Mac (10-core GPU, 16 GB), a 9-minute narration script, defaults:

| | |
|---|---|
| compute per second of narration | 0.85–1.16 s (about realtime; model loading ~12 s extra) |
| one chunk at a time, same pipeline | 8.7 s |
| word error rate (Whisper, whole narration) | ~2% |
| speaker similarity to the reference | 0.89 average, 0.83 lowest 15 s window, no trend over the narration |
| takes per chunk | 1.15–1.2 |

The included 13-minute example (`examples/long.txt`, one run): 1.36 s of compute per second of narration,
84 of 97 chunks passed on the first take, 1.2 takes per chunk, WER 1.1%, speaker similarity 0.88.

Details, ablations and benchmarks are in [RESULTS.md](RESULTS.md).

`mlx-audio` is pinned to 0.6.4 because `qwen3_longform/qwen_fast.py` patches its Qwen3-TTS decode internals.
`uv run pytest` runs the tests (text handling; no models needed).

## Limitations and future work

- **Dialogue pace.** The take check rejects reads faster than 4.4 or slower than 1.6 words per second. A
  character blurting out a line can rightly go faster, so such takes may be rejected or ranked lower.
  Narration whose pace follows the text, fast where the line is urgent, would need tuning against human
  listening. It isn't attempted here.
- **No acoustic continuity across joins.** Every chunk is cloned from the reference, so energy and room tone
  don't carry over from one chunk to the next; a join is a cut plus silence. Echo-TTS continues from the
  previous chunk's audio instead (see [NOTES.md](NOTES.md)).
- **Pauses follow the reference.** The clone copies the reference's pacing. Even with a deliberately paced
  reference, about half the commas are read without a pause, and the model occasionally pauses where you
  wouldn't expect. Pause cues written into the text are an option if this matters more than it has so far.
- **Rescued chunks.** If no take of a chunk can be cut cleanly, the chunk is read on its own, without a
  lead-in or guard, so its joins sound less natural. This is rare: it hasn't happened in recent full runs.
- **Speed.** What's left:
  - Whisper's token-by-token text decoding is the largest cost after generation. Batching the decoding of
    the packed windows would cut it.
  - Custom Metal kernels for the code predictor and the audio decoder. The code predictor is ~60% of a
    generation step at large batch, and it runs well below the GPU's peak.
  - Retakes only start once a whole round has finished. Extra takes for likely failures, in spare batch
    slots, would cut that wait.

## Licences

Code: MIT ([LICENSE](LICENSE)). Models, downloaded at run time:
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base): Apache-2.0. The MLX
  conversion used is
  [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16).
- [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo): MIT.
- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb): Apache-2.0.

You are responsible for having the rights to the reference voice you clone and for how you use the output.
