"""Punctuation adherence: how long the narrator actually pauses at each punctuation mark.

    uv run python -m qwen3_longform.pauses out/<run> [out/<run> ...]

Per chunk (text known, so no long-form ASR drift): Whisper word timestamps are aligned to the script words;
the pause after word i is the longest run of 10 ms frames quieter than (speech median - 25 dB) between the
two words. Joins between chunks use the pause the runner inserted. Pauses are grouped by the punctuation that
ends word i: sentence (. ! ?), colon/semicolon/dash (: ; —), comma, none. A sentence break under 150 ms or a
comma/colon under 60 ms counts as "missed" (run together).
"""
import difflib, json, os, re, sys
import numpy as np
from . import metrics as M

KINDS = (("sentence", r'[.!?]["”’]?$'), ("colon", r'[:;—]$'), ("comma", r',$'), ("none", r''))
MISS = {"sentence": 0.15, "colon": 0.06, "comma": 0.06}


def kind_of(word):
    for k, pat in KINDS:
        if pat and re.search(pat, word):
            return k
    return "none"


def chunk_pauses(y, sr, text):
    words = M.words_of(M.transcribe(M.to16(y, sr), words=True))
    orig = text.split()
    stoks, owner = [], []
    for i, w in enumerate(orig):
        for t in M.toks(w):
            stoks.append(t); owner.append(i)
    htoks, hs, he = [], [], []
    for w, a, b in words:
        for t in M.toks(w):
            htoks.append(t); hs.append(a); he.append(b)
    first, last = {}, {}
    for blk in difflib.SequenceMatcher(a=stoks, b=htoks, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            o = owner[blk.a + k]
            first.setdefault(o, hs[blk.b + k]); last[o] = he[blk.b + k]
    e = M.frame_db(y, sr, 0.01)
    thr = np.median(e[e > e.max() - 30]) - 25
    out = []
    for i in range(len(orig) - 1):
        if i not in last or (i + 1) not in first:
            continue
        a, b = int((last[i] - 0.05) / 0.01), int((first[i + 1] + 0.05) / 0.01)
        seg = e[max(0, a):max(a + 1, b)] < thr
        run = best = 0
        for q in seg:
            run = run + 1 if q else 0; best = max(best, run)
        out.append((orig[i], kind_of(orig[i]), best * 0.01))
    return out


def sentence_pause(wav, text, default=0.42, lo=0.38, hi=0.8):
    """Median pause the speaker takes after a full stop / ? / ! in `wav` (e.g. the clone reference), clamped to
    [lo, hi]; `default` if the text has no internal sentence break."""
    y, sr = M.load(wav)
    v = [p for _, k, p in chunk_pauses(y, sr, text) if k == "sentence"]
    return float(np.clip(np.median(v), lo, hi)) if v else default


def analyse_run(d):
    import soundfile as sf
    run = json.load(open(f"{d}/run.json"))
    rows = []
    for c in run["chunks"]:
        y, sr = sf.read(f"{d}/chunks/{c['i']:03d}.wav", dtype="float32")
        rows += [(w, k, p, "inside") for w, k, p in chunk_pauses(y, sr, c["text"])]
        if c is not run["chunks"][-1]:
            p = c.get("pause_after") or (0.95 if c.get("end_of_ch") else 0.42)
            w = c["text"].split()[-1]
            rows.append((w, kind_of(w), p, "join"))
    return rows


def summarise(name, rows):
    print(f"== {name}")
    for k in ("sentence", "colon", "comma"):
        for where in ("inside", "join", None):
            v = [p for w, kk, p, wh in rows if kk == k and (where is None or wh == where)]
            if not v:
                continue
            miss = sum(p < MISS[k] for p in v)
            print(f"   {k:8s} {where or 'all':6s} n={len(v):3d} median {np.median(v):.2f}s  missed {miss:3d} ({miss/len(v):4.0%})")
    missed = [(w, round(p, 2)) for w, kk, p, wh in rows if kk in MISS and p < MISS[kk] and kk != "comma"]
    print("   run-together sentence/colon breaks:", missed[:12])


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rows = analyse_run(d)
        summarise(os.path.basename(d.rstrip('/')), rows)
        json.dump(rows, open(f"{d}/pauses.json", "w"), indent=1)
