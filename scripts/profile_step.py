"""Where one batched Qwen3-TTS generation step spends its time, per batch size.

    uv run python scripts/profile_step.py --ref voice.wav [--sizes 1,8,16,64] [--steps 40]

Prefills B lead-in + chunk + guard texts against the reference, then times `--steps` decode steps with a sync
after each stage: talker forward, first-codebook sampling (incl. repetition penalty), the 15-step code
predictor (forward + sampling), and next-input embedding. Synced timings slightly overstate the total (no
overlap between stages) but show the split.
"""
import argparse, os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qwen3_longform.engine import Qwen
from qwen3_longform.narrate import reference_text
from qwen3_longform.script import load_script, make_chunks, overlap_jobs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--script", default=f"{ROOT}/examples/long.txt")
ap.add_argument("--sizes", default="1,8,16,64")
ap.add_argument("--steps", type=int, default=40)
a = ap.parse_args()

texts = [j["full"] for j in overlap_jobs(make_chunks(load_script(a.script)))]
eng = Qwen(a.ref, reference_text(a.ref))
m = eng.m
sup = m._suppress_codec_tokens(m.config.talker_config.codec_eos_token_id)
ref = mx.array(eng.ref)

for B in map(int, a.sizes.split(",")):
    bi = m._prepare_batch_inputs(texts[:B], language="English", ref_audio=ref, ref_text=eng.ref_text,
                                 return_metadata=True)
    cache = m.talker.make_cache(); ccache = m.talker.code_predictor.make_cache()
    emb, mask = bi.input_embeds, (bi.attention_mask if B > 1 else None)
    trail = mx.zeros((B, 1), dtype=mx.int32)
    gen = [[] for _ in range(B)]
    t = {"talker": 0.0, "sample": 0.0, "codepred": 0.0, "embed": 0.0}
    for step in range(a.steps + 3):
        s0 = time.time()
        logits, hidden = m.talker(emb, cache=cache, attention_mask=mask); mx.eval(logits, hidden)
        s1 = time.time()
        tok = m._sample_token_batch(logits, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.5,
                                    generated_tokens_per_seq=gen, suppress_tokens=sup); mx.eval(tok)
        s2 = time.time()
        codes, allc = m._predict_code_tokens(tok, hidden, temperature=0.7, top_k=50, top_p=0.9, code_cache=ccache)
        mx.eval(allc)
        s3 = time.time()
        emb = m._next_batch_input_embeds(bi.trailing_text_hidden, bi.tts_pad_embed, trail, codes,
                                         pad_when_index_clamped=True)
        trail = trail + 1
        if mask is not None:
            mask = mx.concatenate([mask, mx.ones((B, 1))], axis=1)
        mx.eval(emb)
        s4 = time.time()
        for b, x in enumerate(tok[:, 0].tolist()):
            gen[b].append(x)
        if step == 0:
            pre = s1 - s0
        if step >= 3:  # skip prefill + warm-up
            for k, d in zip(t, (s1 - s0, s2 - s1, s3 - s2, s4 - s3)):
                t[k] += d
    tot = sum(t.values()) / a.steps
    print(f"B={B:3d} prefill {pre*1000:6.0f} ms | per step {tot*1000:6.1f} ms = " +
          "  ".join(f"{k} {v/a.steps*1000:5.1f}" for k, v in t.items()) +
          f" | per stream-frame {tot/B*1000:5.1f} ms  (RTF {tot/B*12.5:.3f})", flush=True)
    del cache, ccache; mx.clear_cache()
