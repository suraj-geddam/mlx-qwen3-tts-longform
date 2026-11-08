"""Exactness and speed of the prefix-cached decoder (qwen3_longform/qwen_fast.py).

    uv run python scripts/check_prefix_decode.py --ref voice.wav [--n 16] [--steps 300,50]

Generates codes for n chunks once (cached in out/check_prefix_decode/codes.npz, so reruns compare the same
codes), then decodes them three ways and compares each, sample-aligned, with the ground truth: one unchunked
decoder pass over [whole reference + chunk], with the reference's samples cut off exactly.
  * mlx-audio's default ICL decode (300-frame pieces with 25 frames of overlap, proportional trim);
  * the prefix-cached decoder at each --steps size (should match the truth up to float rounding, ~120 dB SNR);
  * a short-context decode (only the last 25 reference frames), to show what a lossy shortcut costs.
Reports time, real-time factor, peak MLX memory, worst SNR over the chunk and over its first 0.5 s.
"""
import argparse, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mlx_audio.tts.utils import load_model
from qwen3_longform.engine import REPO
from qwen3_longform.metrics import load
from qwen3_longform.narrate import reference_text
from qwen3_longform.qwen_fast import RefPrefixDecoder
from qwen3_longform.script import load_script, make_chunks, overlap_jobs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--script", default=f"{ROOT}/examples/long.txt")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--steps", default="300,50")
a = ap.parse_args()
OUT = f"{ROOT}/out/check_prefix_decode"; os.makedirs(OUT, exist_ok=True)

m = load_model(REPO)  # unpatched: mlx-audio's own decode is one of the variants
st = m.speech_tokenizer; dec = st.decoder; up = st.decode_upsample_rate
cpath = f"{OUT}/codes.npz"
if os.path.exists(cpath):
    z = np.load(cpath)
    ref_codes = mx.array(z["ref"]); gens = [mx.array(z[k]) for k in sorted(z.files) if k.startswith("g")]
else:
    texts = [j["full"] for j in overlap_jobs(make_chunks(load_script(a.script)))[:a.n]]
    cap = {}

    def grab(generated_codes, rc):  # keep the codes, skip decoding
        cap.setdefault("ref", rc); cap.setdefault("gens", []).append(mx.stack(generated_codes, axis=1)[0])
        return mx.zeros((up,), dtype=mx.float32)
    m._decode_icl_generated_codes = grab
    mx.random.seed(7)
    for _ in m.batch_generate(texts, ref_audio=mx.array(load(a.ref, 24000)[0]), ref_text=reference_text(a.ref),
                              lang_code="English", temperature=0.7, top_p=0.9, max_tokens=600):
        pass
    del m._decode_icl_generated_codes  # back to the class method
    ref_codes, gens = cap["ref"], cap["gens"]
    np.savez(cpath, ref=np.array(ref_codes), **{f"g{i:02d}": np.array(g) for i, g in enumerate(gens)})
R = ref_codes.shape[2]; ref_t = mx.transpose(ref_codes, (0, 2, 1))[0]
audio_s = sum(g.shape[0] for g in gens) / 12.5
print(f"{len(gens)} chunks, {audio_s:.0f} s of audio, reference {R} frames")


def snr(x, y):
    n = min(len(x), len(y)); d = x[:n] - y[:n]; e = float(np.sum(d * d))
    return float("inf") if e == 0 else 10 * np.log10(float(np.sum(x[:n] ** 2)) / max(e, 1e-30))


def timed(fn):
    mx.clear_cache(); mx.reset_peak_memory()
    t0 = time.time(); out = [np.array(fn(g), np.float32) for g in gens]; dt = time.time() - t0
    return out, dt, mx.get_peak_memory() / 1e9


truth, dt, pk = timed(lambda g: dec(mx.transpose(mx.concatenate([ref_t, g])[None], (0, 2, 1)))[0, 0, R * up:])
print(f"{'truth (one unchunked pass)':32s} {dt:6.2f}s  RTF {dt/audio_s:.4f}  peak {pk:5.2f} GB")
variants = {"mlx-audio default": lambda g: m._decode_icl_generated_codes([r[None] for r in g], ref_codes)}
for step in map(int, a.steps.split(",")):
    variants[f"prefix-cached, step {step}"] = RefPrefixDecoder(st, ref_codes, step=step).decode
variants["short context 25 (lossy)"] = lambda g: st.decoder.chunked_decode(
    mx.transpose(mx.concatenate([ref_t[-25:], g])[None], (0, 2, 1)))[0, 0, 25 * up:]
for name, fn in variants.items():
    timed(fn)  # warm-up
    out, dt, pk = timed(fn)
    s_all = min(snr(t, o) for t, o in zip(truth, out))
    s_head = min(snr(t[:12000], o[:12000]) for t, o in zip(truth, out))
    print(f"{name:32s} {dt:6.2f}s  RTF {dt/audio_s:.4f}  peak {pk:5.2f} GB  worst SNR {s_all:6.1f} dB  "
          f"first 0.5 s {s_head:6.1f} dB", flush=True)
