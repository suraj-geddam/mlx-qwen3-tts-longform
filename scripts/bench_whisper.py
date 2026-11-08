"""Packed vs one-clip-at-a-time Whisper on the chunks of a finished run.

    uv run python scripts/bench_whisper.py out/<run> [--packed 28:0.6,28:1.0]

Whisper's encoder always processes a 30 s window, so a 7 s clip costs as much as a 30 s one. Packing several
clips into one window (separated by `gap` s of silence) divides that cost. For each window:gap setting this
times transcribe_packed() against transcribe() per clip and checks agreement where the take gate depends on it:
WER per clip, how many clips would flip pass/fail at the 0.08 WER threshold, and word start times.
"""
import argparse, difflib, json, os, sys, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qwen3_longform import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--packed", default="28:0.6,28:1.0", help="window:gap pairs, comma separated")
a = ap.parse_args()

run = json.load(open(f"{a.run}/run.json"))
texts, ys = [], []
for c in run["chunks"]:
    y, sr = sf.read(f"{a.run}/chunks/{c['i']:03d}.wav", dtype="float32")
    ys.append(M.to16(y, sr)); texts.append(c["text"])
audio_s = sum(len(y) for y in ys) / M.SR16
print(f"{len(ys)} clips, {audio_s:.0f} s of audio")
M.transcribe(ys[0])  # load + warm up


def starts(res, text):
    st = [t for w in text.split() for t in M.toks(w)]
    ht, hs = [], []
    for w, s0, _ in M.words_of(res):
        for t in M.toks(w):
            ht.append(t); hs.append(s0)
    d = {}
    for b in difflib.SequenceMatcher(a=st, b=ht, autojunk=False).get_matching_blocks():
        for k in range(b.size):
            d[b.a + k] = hs[b.b + k]
    return d


t0 = time.time(); seq = [M.transcribe(y) for y in ys]; t_seq = time.time() - t0
wer_seq = [M.wer_ops(t, r["text"])[0] for t, r in zip(texts, seq)]
st_seq = [starts(r, t) for r, t in zip(seq, texts)]
print(f"{'one at a time':14s} {t_seq:6.1f}s  RTF {t_seq/audio_s:.4f}  mean WER {np.mean(wer_seq):.4f}")
for p in a.packed.split(","):
    t0 = time.time(); bat = M.transcribe_packed(ys, *map(float, p.split(":"))); tb = time.time() - t0
    wer_b = [M.wer_ops(t, r["text"])[0] for t, r in zip(texts, bat)]
    diffs = []
    for r, t, sref in zip(bat, texts, st_seq):
        sb = starts(r, t)
        diffs += [abs(sb[k] - sref[k]) for k in sref.keys() & sb.keys()]
    same = sum(abs(x - y) < 1e-9 for x, y in zip(wer_seq, wer_b))
    flips = sum((x <= 0.08) != (y <= 0.08) for x, y in zip(wer_seq, wer_b))
    print(f"{'packed ' + p:14s} {tb:6.1f}s  x{t_seq/tb:.2f}  mean WER {np.mean(wer_b):.4f}  same WER {same}/{len(ys)}  "
          f"gate flips {flips}  word-start |d| mean {1000*np.mean(diffs):.0f} ms p95 {1000*np.percentile(diffs, 95):.0f} ms",
          flush=True)
