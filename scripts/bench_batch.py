"""Batch-size sweep: speed, memory and take quality of batched generation per batch size.

    uv run python scripts/bench_batch.py --ref voice.wav [--sizes 1,2,4,8,16,32,64] [--script examples/long.txt]

Builds the real lead-in + chunk + guard texts for the script, then for each batch size B generates max(16, B)
of them (sorted by length so a batch's streams finish together) and reports compute seconds per second of
audio, split into the batched generation loop and the per-stream waveform decode, plus peak MLX memory. Each
output is then cut and gated, so a speedup that costs quality shows up. Writes out/bench_batch/results.json.
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qwen3_longform import metrics as M
from qwen3_longform.engine import Qwen, max_tokens_for
from qwen3_longform.narrate import reference_text
from qwen3_longform.script import load_script, make_chunks, overlap_jobs
from qwen3_longform.takes import score_take

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--script", default=f"{ROOT}/examples/long.txt")
ap.add_argument("--sizes", default="1,2,4,8,16,32,64")
ap.add_argument("--out", default=f"{ROOT}/out/bench_batch")
a = ap.parse_args()

jobs = overlap_jobs(make_chunks(load_script(a.script)))
eng = Qwen(a.ref, reference_text(a.ref))
ref = M.reference_profile(a.ref)
m, sr = eng.m, eng.sr
os.makedirs(a.out, exist_ok=True)
eng.generate([jobs[0]["full"]], 1, 60)  # warm-up: compile kernels, encode the reference

results = []
for B in map(int, a.sizes.split(",")):
    n = min(len(jobs), max(16, B))
    sel = sorted(jobs[:n], key=lambda j: len(j["full"]))
    outs = {}
    mx.clear_cache(); mx.reset_peak_memory()
    t_gen = t_dec = 0.0
    t0 = time.time()
    for s in range(0, n, B):
        grp = sel[s:s + B]
        mx.random.seed(7 + s)
        t1 = time.time(); first = None
        for r in m.batch_generate([j["full"] for j in grp], ref_audio=mx.array(eng.ref), ref_text=eng.ref_text,
                                  lang_code="English", temperature=0.7, top_p=0.9,
                                  max_tokens=max(max_tokens_for(j["full"]) for j in grp)):
            if first is None:
                first = r.processing_time_seconds  # batched loop done; the rest is per-stream decode
            outs[grp[r.sequence_idx]["i"]] = np.asarray(r.audio, np.float32).reshape(-1)
        el = time.time() - t1
        t_gen += first or el; t_dec += el - (first or el)
    wall = time.time() - t0
    peak = mx.get_peak_memory() / 1e9
    audio_s = sum(len(y) for y in outs.values()) / sr
    ys = [outs.get(j["i"], np.zeros(0, np.float32)) for j in sel]  # quality, outside the timer
    trs = M.transcribe_packed([M.to16(y, sr) for y in ys], 28.0, 0.6)
    rows = []
    for j, y, tr in zip(sel, ys, trs):
        s, _ = score_take(j, y, sr, M.words_of(tr), ref)
        rows.append(dict(i=j["i"], cut=s["cut"], wer=s["wer"], spk=s["spk"], ok=s["ok"], wps=s["wps"]))
    good = [r for r in rows if r["cut"]]
    res = dict(size=B, n=n, wall_s=round(wall, 1), gen_s=round(t_gen, 1), decode_s=round(t_dec, 1),
               audio_s=round(audio_s, 1), rtf=round(wall / audio_s, 3), rtf_gen=round(t_gen / audio_s, 3),
               peak_gb=round(peak, 2), cut_ok=len(good), take_ok=sum(r["ok"] for r in rows),
               wer_mean=round(float(np.mean([r["wer"] for r in good])), 3) if good else None,
               spk_mean=round(float(np.mean([r["spk"] for r in good])), 3) if good else None, rows=rows)
    results.append(res)
    print(f"B={B:>3} n={n:2d} wall {wall:6.1f}s (loop {t_gen:6.1f} dec {t_dec:5.1f}) audio {audio_s:6.1f}s "
          f"RTF {res['rtf']:.3f} (loop {res['rtf_gen']:.3f}) peak {peak:5.2f} GB | cut {len(good)}/{n} "
          f"ok {res['take_ok']}/{n} wer {res['wer_mean']} spk {res['spk_mean']}", flush=True)
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
