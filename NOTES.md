# Notes: why Echo-TTS is better at long-form, and what this repo does instead

*As of November 2026. Measurements are from experiments on an M3 Mac with the same test script and metrics as
this repo. ECAPA speaker similarity is measured against the clean reference (about 0.85–0.89 is "same voice"
for these clones).*

For long-form voice cloning where commercial use doesn't matter, I've found
[Echo-TTS](https://huggingface.co/jordand/echo-tts-base) (run through
[longecho](https://github.com/suraj-geddam/longecho)) much better than Qwen3-TTS and Ming-3.0-tts. This is my
best understanding of why, and of how this repo makes up for it.

## Two ways to build long-form narration

1. **Continuation.** Each chunk continues from the end of the previous chunk's audio, so energy, room tone and
   where the sentence was heading carry over naturally. This is how a human narrator sounds, and it's what
   longecho does.
2. **Independent clones.** Each chunk is cloned fresh from the reference and the chunks are joined. The voice
   can't drift, but nothing carries over acoustically from one chunk to the next.

## Why continuation works for Echo

Echo gets *who is speaking* and *what came before* from two separate inputs, and keeps them apart:

- **Speaker.** The clean reference is encoded once into a speaker latent and fed through its own conditioning
  path, into every chunk. Generated audio never replaces it. In training, the reference came from *other*
  segments of the same speaker, so the model learned identity from separate audio, not from the clip it was
  generating.
- **Continuation.** The end of the previous chunk, re-encoded, is the fixed start of the sequence and the model
  fills in what comes next. It carries continuity, not identity.
- **Separate guidance.** Speaker and text were each dropped during training, so each gets its own
  classifier-free guidance at inference. longecho pushes the speaker condition 8× (`cfg_scale_speaker=8`). If the
  previous chunk drifted slightly, the continuation nudges the new chunk that way, but the guidance pulls it
  back toward the clean reference much harder. Errors are corrected every chunk instead of compounding.

Measured: 12 chunks, each continuing the previous one, gave speaker similarities of 0.87, 0.79, 0.85, 0.89,
0.86, 0.79, 0.91, 0.86, 0.90, 0.87, 0.87, 0.85. There's no trend: the last minute differed from the first by
−0.007.

## Why it doesn't work for Qwen3-TTS or Ming

In both, one input does both jobs. The cloning prompt (reference audio plus its transcript) is the identity
*and* the thing being continued. Give either model the previous chunk as its prompt, and the previous chunk
*becomes* the voice to clone: a copy of a copy. Each generation's small deviations become the next one's
target, and they compound.

- **Ming-3.0-tts-0.5B:** previous chunk as the prompt, its speaker embedding pinned to the clean reference.
  Similarity went 0.82, 0.75, 0.76, 0.62, 0.45, 0.34, 0.18, 0.17 over 8 chunks. Level, noise floor, pitch and
  brightness drifted too, and the text eventually turned to garbage. A probe showed why: given one speaker's
  audio prompt and another's speaker embedding, Ming follows the prompt (0.79–0.82 similarity to the prompt's
  voice, 0.22 to the embedding's).
- **Qwen3-TTS 1.7B:** previous chunk as the prompt, its speaker vector (x-vector) pinned to the clean reference.
  Similarity sank from 0.87 to about 0.65–0.75 over 15 chunks and the voice got darker. The x-vector slows the
  drift but doesn't stop it: it's a single token next to hundreds of audio codes, and nothing amplifies it the
  way Echo's speaker guidance does.

## What this repo does instead

It never lets generated audio into the prompt. Every chunk is cloned directly from the clean reference (one hop),
so there's nothing to drift from. Continuity has to come from somewhere else:

- **Text instead of audio.** Each chunk is read with the end of the previous chunk before it (lead-in) and the
  start of the next one after it (guard). Both are cut off at real pauses, so each kept chunk starts and ends
  with mid-paragraph prosody instead of the "first sentence" and "last sentence" reading of an isolated chunk.
- **The narrator's own pauses.** The silence after a chunk is the pause the model actually took before the guard,
  never shorter than the reference's typical sentence pause.
- **A gate.** Whisper and a speaker-similarity check catch the takes that go wrong, and the chunk is regenerated.

What it can't recover is the acoustic carry-over that continuation gives for free. Each join is a cut plus
inserted silence, not a continuous performance. It also generates about 1.7 seconds of audio per second kept.
Batching, an exact cached decoder and packed Whisper buy that back: about realtime on a base M3, where Echo
runs at roughly 9× slower than realtime on the same machine.

Hopefully someday soon there'll be a TTS model as good as Echo at long-form that allows commercial use.
