"""Turning a raw take into a kept chunk: cut at real pauses, tidy, score, and choose the pause after it.

A take reads `lead text guard`. Whisper word timestamps locate the boundaries; the cut goes at the quietest
10 ms frame of the pause after the lead-in and of the pause before the guard. A take with no pause at either
boundary (the narrator ran the sentences together) can't be cut cleanly and is rejected.

The gate (check) passes a take when Whisper's WER against the script is <= 0.08, the pace is 1.6-4.4 words/s,
the ECAPA speaker similarity to the reference is >= 0.55 and its loudness is within 4 LU of the reference.
`cost` ranks takes when none passes (WER dominates, then identity, loudness, pace, a clipped ending).
"""
import difflib
import numpy as np

from . import metrics as M


def env_db(y, sr, hop=0.01):
    h = int(hop * sr); n = max(1, len(y) // h)
    return 20 * np.log10(np.sqrt(np.mean(y[:n * h].reshape(n, h) ** 2, axis=1)) + 1e-9)


def tidy(y, sr, pre=0.05, post=0.12):
    """Trim leading/trailing silence, keeping the decay (down to peak-55 dB) plus `post` s."""
    if len(y) < sr * 0.2:
        return y
    e = env_db(y, sr); pk = e.max()
    on = np.where(e > pk - 38)[0]; off = np.where(e > pk - 55)[0]
    if not len(on):
        return y
    s0 = max(0, int((on[0] * 0.01 - pre) * sr)); s1 = min(len(y), int(((off[-1] + 1) * 0.01 + post) * sr))
    y = y[s0:s1].copy()
    k = int(0.005 * sr); y[:k] *= np.linspace(0, 1, k)
    k = min(len(y) // 4, int(0.05 * sr)); y[-k:] *= np.linspace(1, 0, k) ** 2
    return y


def cap_pauses(y, sr, max_pause=1.0):
    """Shorten any internal silence (< peak-45 dB) longer than max_pause to max_pause."""
    if len(y) < sr * 0.2:
        return y
    e = env_db(y, sr); quiet = e < e.max() - 45
    i, n = 0, len(e)
    keep = np.ones(len(y), bool)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.01 > max_pause:
                a = int((i * 0.01 + max_pause / 2) * sr); b = int((j * 0.01 - max_pause / 2) * sr)
                keep[a:b] = False
            i = j
        else:
            i += 1
    return y[keep]


def ends_loud(y, sr):
    """Level of the last 80 ms relative to the speech level (dB). Above -12 suggests a clipped ending."""
    e = env_db(y, sr); med = np.median(e[e > e.max() - 30])
    return float(e[-8:].max() - med)


def check(y, sr, text, ref, res=None, asr_equiv=None):
    """Score a take. ref: metrics.reference_profile(). res: a transcript already made for exactly this audio
    (else Whisper runs here)."""
    y16 = M.to16(y, sr)
    if res is None:
        res = M.transcribe(y16, words=True) if len(y16) > 1600 else {"text": "", "segments": []}
    hyp = res["text"].strip()
    w, errs = M.wer_ops(text, hyp, asr_equiv)
    dur = len(y) / sr
    wps = len(text.split()) / max(0.3, dur)
    spk = float(M.spk_embed(y16) @ ref["emb"]) if dur > 1 else 0.0
    L = M.lufs(y, sr)
    dl = (L - ref["lufs"]) if np.isfinite(L) else 99.0
    el = ends_loud(y, sr) if dur > 0.5 else 99.0
    # A clipped ending only adds cost: in a blind listen, takes failing it sounded as good as passing ones.
    ok = w <= 0.08 and 1.6 <= wps <= 4.4 and spk >= 0.55 and abs(dl) <= 4.0
    cost = 10 * w + 4 * max(0, 0.8 - spk) + 0.15 * abs(dl) + (0 if 1.9 <= wps <= 3.6 else 1) + (0 if el < -12 else 1)
    return {"wer": round(w, 3), "hyp": hyp, "errs": errs[:6], "dur": round(dur, 2), "wps": round(wps, 2),
            "spk": round(spk, 3), "dlufs": round(dl, 2), "end": round(el, 1), "ok": bool(ok), "cost": round(cost, 3)}


def cut_span(y, sr, lead, text, guard, words, asr_equiv=None):
    """The take read `lead text guard`; words: its Whisper [(word, start, end)]. Returns (audio, natural pause
    before the guard in s, (start, end) in samples), or (None, None, None) if a boundary can't be found or has
    no pause."""
    fail = (None, None, None)
    lt, tt, gt = [M.toks(part, asr_equiv) for part in (lead, text, guard)]
    stoks = lt + tt + gt
    n0, n1 = len(lt), len(lt) + len(tt)
    htoks, hs, he = [], [], []
    for token, a, b in M.transcript_tokens(words, asr_equiv):
        htoks.append(token); hs.append(a); he.append(b)
    first_s = first_e = first_s_end = guard_s = guard_s_end = lead_e = None
    for blk in difflib.SequenceMatcher(a=stoks, b=htoks, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            if i < n0:
                lead_e = he[j]
            elif i < n1:
                if first_s is None:
                    first_s, first_s_end = hs[j], he[j]
                first_e = he[j]
            elif guard_s is None:
                guard_s, guard_s_end = hs[j], he[j]
    if first_s is None or (lead and lead_e is None) or (guard and guard_s is None):
        return fail
    e = env_db(y, sr); pk = e.max(); fr = int(0.01 * sr)

    def quiet(t0, t1, t1_end):
        # Whisper often stamps the pause before a word as part of that word, so search up to 0.3 s into it
        t1 = max(t1, min(t1_end, t1 + 0.3))
        lo, hi = int(t0 / 0.01), max(int(t0 / 0.01) + 1, int(t1 / 0.01))
        hi = min(hi, len(e) - 1)
        if hi <= lo:
            return None, 0.0
        seg = e[lo:hi + 1]
        k = lo + int(np.argmin(seg))
        return (k if e[k] < pk - 35 else None), float(np.sum(seg < pk - 40) * 0.01)

    a = 0
    if lead:
        k, _ = quiet(lead_e, first_s, first_s_end)
        if k is None:
            return fail
        a = k * fr
    b, nat = len(y), None
    if guard:
        k, nat = quiet(first_e, guard_s, guard_s_end)
        if k is None:
            return fail
        b = k * fr
    return y[a:b], nat, (a, b)


def score_take(job, y, sr, words, ref, asr_equiv=None):
    """Cut, tidy and score one raw take, using the words of its transcript that fall inside the cut."""
    yc, nat, ab = cut_span(y, sr, job["lead"], job["text"], job["guard"], words, asr_equiv)
    if yc is None or len(yc) < 0.5 * sr:
        s = check(np.zeros(0, np.float32), sr, job["text"], ref, res={"text": "", "segments": []}, asr_equiv=asr_equiv)
        s.update(cut=False, nat_pause=None)
        return s, np.zeros(0, np.float32)
    t0, t1 = ab[0] / sr, ab[1] / sr
    inside = [(w, a - t0, b - t0) for w, a, b in words if t0 <= (a + b) / 2 <= t1]
    res = {"text": " ".join(w for w, _, _ in inside),
           "segments": [{"words": [dict(word=" " + w, start=a, end=b) for w, a, b in inside]}]}
    yc = cap_pauses(tidy(yc, sr), sr)
    s = check(yc, sr, job["text"], ref, res=res, asr_equiv=asr_equiv)
    s.update(cut=True, nat_pause=nat)
    return s, yc


def join_pause(nat, end_of_ch, floor, ch_gap=0.95, cap=0.9):
    """Silence after a chunk: the chapter gap at a chapter end, else the narrator's own pause before the guard
    (nat) but never shorter than `floor` (the reference's median sentence pause) nor longer than `cap`."""
    if end_of_ch:
        return ch_gap
    return min(cap, max(floor, nat or 0.0))
