"""Metrics for long-form narration: take scoring (Whisper WER, ECAPA speaker similarity, loudness) and
whole-file drift analysis (--analyse), measured on the finished narration in fixed windows.

Per window (default 15 s):
  spk     cosine similarity to the reference, ECAPA-TDNN (speechbrain, VoxCeleb). Independent of Qwen's own
          speaker encoder, so a take can't "pass" by matching the embedding it was conditioned on.
  lufs    integrated loudness of the window (pauses are gated out by BS.1770).
  floor   noise floor: 10th percentile of 20 ms frame levels (dBFS). Rising floor = hiss/noise accumulating.
  hf      energy above 4 kHz relative to the whole band on speech frames (dB). Tracks brightness/fizz drift.
  f0      median pitch (Hz) of voiced frames.
  wpm     words per minute (Whisper word timestamps).
File level: WER vs the script, and for each metric the slope per minute and last-minute minus first-minute.
"""
import json, os, re, difflib, functools
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

from .normalize import TextNormalizer, spell_numbers

WHISPER = "mlx-community/whisper-large-v3-turbo"
SR16 = 16000


def load(path, sr=None):
    y, s = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr and s != sr:
        from math import gcd
        g = gcd(s, sr); y = resample_poly(y, sr // g, s // g).astype(np.float32); s = sr
    return y, s


def to16(y, sr):
    from math import gcd
    if sr == SR16:
        return y
    g = gcd(sr, SR16)
    return resample_poly(y, SR16 // g, sr // g).astype(np.float32)


@functools.lru_cache(1)
def _ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(4)
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          savedir=os.path.expanduser("~/.cache/qwen3_longform/ecapa"), run_opts={"device": "cpu"})


def spk_embed(y16):
    import torch
    with torch.no_grad():
        e = _ecapa().encode_batch(torch.from_numpy(y16)[None]).squeeze().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def frame_db(y, sr, hop=0.02):
    h = int(hop * sr); n = len(y) // h
    if n == 0:
        return np.array([-120.0])
    return 20 * np.log10(np.sqrt(np.mean(y[:n * h].reshape(n, h) ** 2, axis=1)) + 1e-9)


def lufs(y, sr):
    import pyloudnorm as pyln
    if len(y) < 0.5 * sr:
        return float("nan")
    v = pyln.Meter(sr).integrated_loudness(y.astype(np.float64))
    return float(v) if np.isfinite(v) else float("nan")


def hf_ratio(y, sr):
    """dB of >4 kHz energy relative to full-band energy, over speech frames only."""
    if sr < 9000:
        return float("nan")
    hi = sosfilt(butter(4, 4000, "hp", fs=sr, output="sos"), y)
    db = frame_db(y, sr); dbh = frame_db(hi, sr)
    sp = db > db.max() - 30
    if sp.sum() < 5:
        return float("nan")
    return float(10 * np.log10(np.mean(10 ** (dbh[sp] / 10)) / np.mean(10 ** (db[sp] / 10))))


def f0_median(y16):
    import librosa
    f = librosa.yin(y16, fmin=60, fmax=400, sr=SR16, frame_length=1024, hop_length=320)
    db = frame_db(y16, SR16)[: len(f)]
    f = f[: len(db)]
    v = f[(db > db.max() - 25)]
    return float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------- text / ASR
# spellings that sound the same, so a script and Whisper's transcript don't disagree over them
EQUIV = {"gonna": "going", "ok": "okay", "em": "them"}
for _uk, _us in [("grey", "gray"), ("colour", "color"), ("honour", "honor"), ("favour", "favor"),
                 ("neighbour", "neighbor"), ("labour", "labor"), ("humour", "humor"), ("harbour", "harbor"),
                 ("centre", "center"), ("theatre", "theater"), ("metre", "meter"), ("litre", "liter"),
                 ("travelled", "traveled"), ("travelling", "traveling"), ("cancelled", "canceled"),
                 ("jewellery", "jewelry"), ("mould", "mold"), ("plough", "plow"), ("cheque", "check"),
                 ("practise", "practice"), ("defence", "defense"), ("licence", "license"),
                 ("organise", "organize"), ("realise", "realize"), ("recognise", "recognize")]:
    for _suffix in ("", "s", "d", "ed", "ing"):
        EQUIV.setdefault(_uk + _suffix, _us + _suffix)


_NORM = None


def _base_toks(s):
    """Words for comparing a script with a transcript. Both sides go through the same normalization as the script
    ("$1.87" and "one dollar and eighty-seven cents" give the same words)."""
    global _NORM
    if _NORM is None:
        _NORM = TextNormalizer()
    s = re.sub(r"(\d) \.(\d)", r"\1.\2", s)  # Whisper's word timestamps split "$1.87" into "$1" + ".87"
    s = spell_numbers(_NORM.normalize(s)).replace(" and ", " ").replace(",", " ")
    s = s.lower().replace("—", " ").replace("-", " ").replace("’", "'").replace("%", " percent")
    return [EQUIV.get(w, w) for w in (w.strip(".,:;!?'\"“”()") for w in s.split()) if w]


def _apply_equiv(tokens, asr_equiv, spans=None):
    """One longest-first, whole-token substitution pass; never recursively rewrite a target."""
    rules = _equiv_rules(tuple(sorted((asr_equiv or {}).items())))
    out, times, i = [], [], 0
    while i < len(tokens):
        for source, target in rules.get(tokens[i], ()):
            if tuple(tokens[i:i + len(source)]) == source:
                out.extend(target)
                if spans is not None:
                    times.extend([(spans[i][0], spans[i + len(source) - 1][1])] * len(target))
                i += len(source)
                break
        else:
            out.append(tokens[i])
            if spans is not None:
                times.append(spans[i])
            i += 1
    return out, times


@functools.lru_cache(maxsize=64)
def _equiv_rules(items):
    rules = {}
    for source, target in items:
        source, target = tuple(_base_toks(source)), tuple(_base_toks(target))
        if not source or not target:
            raise ValueError("asr_equiv spellings must contain comparison tokens")
        rules.setdefault(source[0], []).append((source, target))
    for candidates in rules.values():
        candidates.sort(key=lambda r: -len(r[0]))
    return rules


def toks(s, asr_equiv=None):
    """Comparison words, with optional per-script spelling equivalences on both sides."""
    tokens = _base_toks(s)
    return _apply_equiv(tokens, asr_equiv)[0] if asr_equiv else tokens


def transcript_tokens(words, asr_equiv=None):
    """Comparison tokens with timestamps, including phrases spanning Whisper word entries."""
    tokens, spans = [], []
    for word, start, end in words:
        ts = _base_toks(word)
        tokens.extend(ts); spans.extend([(start, end)] * len(ts))
    if asr_equiv:
        tokens, spans = _apply_equiv(tokens, asr_equiv, spans)
    return [(token, start, end) for token, (start, end) in zip(tokens, spans)]


def wer_ops(ref, hyp, asr_equiv=None):
    r, h = toks(ref, asr_equiv), toks(hyp, asr_equiv)
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    errs = [(op, " ".join(r[i1:i2]), " ".join(h[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    n = sum(max(len(a.split()), len(b.split())) for _, a, b in errs)
    return n / max(1, len(r)), errs


def transcribe(path_or_y, prompt=None, words=True):
    import mlx_whisper
    return mlx_whisper.transcribe(path_or_y, path_or_hf_repo=WHISPER, language="en", word_timestamps=words,
                                  initial_prompt=prompt, condition_on_previous_text=False)


def transcribe_packed(ys16, window=28.0, gap=1.0):
    """Transcribe many short clips by packing them, separated by `gap` s of silence, into <= `window` s
    windows: Whisper's encoder costs the same for 7 s as for 30 s, so packing 3-4 clips per window divides
    that cost. Each window is a normal transcribe() call; words are given back to the clip whose span holds
    the word's midpoint, with times relative to that clip. Same output shape as transcribe()."""
    out = [None] * len(ys16)
    g = np.zeros(int(gap * SR16), np.float32)
    packs, cur, cur_len = [], [], 0
    for i, y in enumerate(ys16):
        if len(y) <= 1600:
            out[i] = {"text": "", "segments": []}; continue
        n = len(y) + (len(g) if cur else 0)
        if cur and cur_len + n > window * SR16:
            packs.append(cur); cur, cur_len, n = [], 0, len(y)
        cur.append(i); cur_len += n
    if cur:
        packs.append(cur)
    for p in packs:
        pieces, spans, t = [], [], 0
        for k, i in enumerate(p):
            if k:
                pieces.append(g); t += len(g)
            pieces.append(ys16[i]); spans.append((t / SR16, (t + len(ys16[i])) / SR16)); t += len(ys16[i])
        res = transcribe(np.concatenate(pieces))
        words = [[] for _ in p]
        for s in res["segments"]:
            for w in s.get("words", []):
                mid = (w["start"] + w["end"]) / 2
                k = min(range(len(p)), key=lambda k: 0 if spans[k][0] <= mid <= spans[k][1] else
                        min(abs(mid - spans[k][0]), abs(mid - spans[k][1])))
                a0 = spans[k][0]
                words[k].append(dict(w, start=max(0.0, w["start"] - a0), end=max(0.0, w["end"] - a0)))
        for k, i in enumerate(p):
            out[i] = {"text": "".join(w["word"] for w in words[k]), "segments": [{"words": words[k]}]}
    return out


def words_of(res):
    return [(w["word"].strip(), w["start"], w["end"]) for s in res["segments"] for w in s.get("words", [])]


# ---------------------------------------------------------------- analysis
def reference_profile(ref_path):
    y, sr = load(ref_path)
    y16 = to16(y, sr)
    return {"emb": spk_embed(y16), "lufs": lufs(y, sr), "hf": hf_ratio(y, sr), "f0": f0_median(y16),
            "floor": float(np.percentile(frame_db(y, sr), 10))}


def slope_per_min(t, v):
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(t[ok] / 60, v[ok], 1)[0])


def analyse(path, ref, script_text=None, win=15.0, asr=True, prompt=None, asr_equiv=None):
    """ref: reference_profile(). Returns dict with per-window rows and a summary."""
    y, sr = load(path)
    y16 = to16(y, sr)
    dur = len(y) / sr
    ws = []
    res = transcribe(y16, prompt=prompt) if asr else None
    wl = words_of(res) if res else []
    t = 0.0
    while t + 4.0 <= dur:
        a, b = int(t * sr), int(min(dur, t + win) * sr)
        seg, seg16 = y[a:b], y16[int(t * SR16): int(min(dur, t + win) * SR16)]
        db = frame_db(seg, sr)
        nw = sum(1 for w in wl if t <= w[1] < t + win)
        ws.append({"t": round(t, 1), "spk": float(spk_embed(seg16) @ ref["emb"]), "lufs": lufs(seg, sr),
                   "floor": float(np.percentile(db, 10)), "hf": hf_ratio(seg, sr), "f0": f0_median(seg16),
                   "wpm": nw / ((b - a) / sr) * 60 if wl else float("nan")})
        t += win
    summ = {"dur": round(dur, 1)}
    if res and script_text:
        w, errs = wer_ops(script_text, res["text"], asr_equiv)
        summ["wer"] = round(w, 4); summ["n_err"] = len(errs); summ["errs"] = errs[:40]
    if wl:
        summ["wpm_all"] = round(len(wl) / dur * 60, 1)
    tt = [r["t"] for r in ws]
    for k in ("spk", "lufs", "floor", "hf", "f0", "wpm"):
        v = np.array([r[k] for r in ws], float)
        fin = v[np.isfinite(v)]
        if not len(fin):
            continue
        n1 = max(1, int(round(60 / win)))
        first, last = np.nanmean(v[:n1]), np.nanmean(v[-n1:])
        summ[k] = {"mean": round(float(np.nanmean(v)), 3), "std": round(float(np.nanstd(v)), 3),
                   "min": round(float(np.nanmin(v)), 3), "slope_per_min": round(slope_per_min(tt, v), 4),
                   "last_minus_first": round(float(last - first), 3)}
    return {"file": path, "windows": ws, "summary": summ}


def fmt_summary(name, s):
    g = lambda k, f: s.get(k, {}).get(f, float("nan"))
    return (f"{name:28s} dur {s['dur']:6.1f}s wer {s.get('wer', float('nan')):.3f} wpm {s.get("wpm_all", float("nan")):5.1f} | "
            f"spk {g('spk','mean'):.3f} (min {g('spk','min'):.3f}, Δ {g('spk','last_minus_first'):+.3f}) | "
            f"lufs {g('lufs','mean'):6.1f} sd {g('lufs','std'):.2f} Δ {g('lufs','last_minus_first'):+.2f} | "
            f"floor {g('floor','mean'):6.1f} Δ {g('floor','last_minus_first'):+.1f} | hf Δ {g('hf','last_minus_first'):+.2f} | "
            f"f0 {g('f0','mean'):5.1f} Δ {g('f0','last_minus_first'):+.1f}")
