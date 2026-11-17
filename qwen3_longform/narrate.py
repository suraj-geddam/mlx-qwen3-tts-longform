"""Long-form narration with Qwen3-TTS: the whole pipeline.

    uv run qwen3-longform --ref voice.wav --script chapter.txt --out out/chapter

Every chunk is cloned from the same clean reference, so the voice cannot drift however long the narration is.
For each chunk:
  * it is read with a lead-in (end of the previous chunk) and a guard (start of the next) and cut at real pauses
    (takes.cut_span), so it starts and ends with natural prosody;
  * the take is gated (Whisper WER, ECAPA speaker similarity, loudness, pace; takes.check). The first take that
    passes is kept; a failing chunk is regenerated with new seeds, up to --takes times (plus up to 3 more for a
    broken take), and the lowest-cost take wins if none passes;
  * a chunk whose cut failed on every take is read once more on its own (no lead-in or guard) so it can't drop
    out;
  * the pause after it is the narrator's own pause before the guard, never shorter than the reference's median
    sentence pause.
Speed:
  * chunks are generated --batch at a time (one weight read per step serves every stream), in batches of similar
    length so a batch's streams finish together;
  * the codec decoder resumes from a cached state after the reference instead of re-decoding it per chunk
    (qwen_fast: identical to one pass over reference + chunk);
  * Whisper runs on several takes packed into one 30 s window (its encoder costs the same for 7 s as for 30 s),
    and that one transcript serves both the cut and the gate;
  * the CPU side of the gate runs in a worker thread while the GPU generates the next batch.
"""
import argparse, json, os, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf
import mlx.core as mx

from . import metrics as M
from .engine import Qwen, max_tokens_for
from .pauses import sentence_pause
from .perf import MemWatch, machine, memory, record
from .script import load_script_document, make_chunks, overlap_jobs
from .takes import join_pause, score_take


def reference_text(ref_wav, ref_text=None):
    """The reference's transcript: given, else <ref>.txt beside the wav, else Whisper's (saved as <ref>.txt)."""
    if ref_text:
        return ref_text.strip()
    path = os.path.splitext(ref_wav)[0] + ".txt"
    if os.path.exists(path):
        return open(path).read().strip()
    y, sr = M.load(ref_wav)
    text = M.transcribe(M.to16(y, sr), words=False)["text"].strip()
    open(path, "w").write(text)
    print(f"no transcript for the reference; Whisper heard:\n  {text}\nsaved to {path}. Fix any mistakes there: "
          "cloning works best with an exact transcript.", flush=True)
    return text


def run(ref_wav, script, out, ref_text=None, B=32, takes=3, gap=0.42, ch_gap=0.95, seed0=7, limit=None,
        pack_gap=0.6, cache_gb=None, decode_step=50, perf_log=None, normalize=True):
    """cache_gb: cap on the memory MLX keeps cached for reuse after arrays are freed (None: MLX's default, up to
    the whole memory limit). decode_step: frames the codec decoder takes per call; any value gives the same
    audio, smaller uses less memory. Returns the path of the narration wav."""
    if cache_gb is not None:
        mx.set_cache_limit(int(cache_gb * 1e9))
    os.makedirs(f"{out}/chunks", exist_ok=True)
    document = load_script_document(script, normalize)
    asr_equiv = document["asr_equiv"]
    M.toks("", asr_equiv)  # validate comparison rules before loading models
    chunks = make_chunks(document["lines"])
    chunks = chunks[:limit] if limit else chunks
    jobs = overlap_jobs(chunks)
    ref_text = reference_text(ref_wav, ref_text)
    t_start = time.time()
    mem0 = memory()
    watch = MemWatch()
    eng = Qwen(ref_wav, ref_text, decode_step=decode_step)
    sr = eng.sr
    ref = M.reference_profile(ref_wav)
    floor = sentence_pause(ref_wav, ref_text, default=gap)
    print(f"{len(jobs)} chunks; join pause floor {floor:.2f}s (the reference's median sentence pause)", flush=True)
    t_load = time.time() - t_start

    cands = {j["i"]: [] for j in jobs}            # i -> [(score, audio)]
    done = {}                                     # i -> accepted (score, audio)
    tm = dict(gen=0.0, whisper=0.0, cpu_wait=0.0, gen_audio_s=0.0, takes=0, batches=0)
    pool = ThreadPoolExecutor(1)

    def generate_and_transcribe(grp, seed):
        t0 = time.time()
        ys = eng.generate([j["full"] for j in grp], seed, max(max_tokens_for(j["full"]) for j in grp))
        t1 = time.time()
        trs = M.transcribe_packed([M.to16(y, sr) for y in ys], 28.0, pack_gap)
        t2 = time.time()
        tm["gen"] += t1 - t0; tm["whisper"] += t2 - t1; tm["batches"] += 1
        tm["gen_audio_s"] += sum(len(y) for y in ys) / sr; tm["takes"] += len(grp)
        return ys, trs, t1 - t0, t2 - t1

    rnd = 0
    t_run = time.time()
    while True:
        todo = [j for j in jobs if j["i"] not in done and
                (len(cands[j["i"]]) < takes or
                 (len(cands[j["i"]]) < takes + 3 and cands[j["i"]][-1][0]["wer"] > 0.5))]
        if not todo:
            break
        todo.sort(key=lambda j: len(j["full"]))
        futures = []
        for b0 in range(0, len(todo), B):
            grp = todo[b0:b0 + B]
            seed = seed0 + 1000 * rnd + zlib.crc32("".join(j["full"] for j in grp).encode()) % 997
            ys, trs, tg, tw = generate_and_transcribe(grp, seed)
            print(f"round {rnd} batch {b0 // B}: {len(grp)} takes, {sum(len(y) for y in ys)/sr:6.1f}s audio, "
                  f"gen {tg:5.1f}s whisper {tw:5.1f}s", flush=True)
            futures.append(pool.submit(lambda grp=grp, ys=ys, trs=trs, seed=seed:
                                       [(j, seed) + score_take(j, y, sr, M.words_of(tr), ref, asr_equiv)
                                        for j, y, tr in zip(grp, ys, trs)]))
        tw = time.time()
        for f in futures:
            for j, seed, s, yc in f.result():
                s.update(seed=seed, round=rnd)
                cands[j["i"]].append((s, yc))
                if s["ok"]:
                    done[j["i"]] = (s, yc)
        tm["cpu_wait"] += time.time() - tw
        print(f"round {rnd}: {sum(1 for j in todo if j['i'] in done)}/{len(todo)} passed", flush=True)
        rnd += 1
    # A chunk whose cut failed on every take would otherwise vanish: read it on its own (no lead-in or guard,
    # nothing to cut) and gate that instead.
    rescue = [j for j in jobs if j["i"] not in done and not any(s.get("cut") and s["wer"] <= 0.5 for s, _ in cands[j["i"]])]
    if rescue:
        grp = [dict(j, lead="", guard="", full=j["text"]) for j in rescue for _ in range(2)]
        ys, trs, _, _ = generate_and_transcribe(grp, seed0 + 99991)
        for j, y, tr in zip(grp, ys, trs):
            s, yc = score_take(j, y, sr, M.words_of(tr), ref, asr_equiv)
            s.update(round="rescue", nat_pause=None)
            cands[j["i"]].append((s, yc))
            if s["ok"] and j["i"] not in done:
                done[j["i"]] = (s, yc)
        print(f"rescue: {len(rescue)} chunks read without lead-in/guard, "
              f"{sum(1 for j in rescue if j['i'] in done)} passed", flush=True)
    pool.shutdown()
    t_gate = time.time() - t_run

    log = {"ref": ref_wav, "script": script, "normalize": normalize, "asr_equiv": asr_equiv, "batch": B, "takes": takes, "chunks": []}
    pieces = []
    for j, c in zip(jobs, chunks):
        i = j["i"]
        cs = cands[i]
        best_s, best_y = done.get(i) or min(cs, key=lambda sy: (not sy[0]["ok"], sy[0]["cost"]))
        sf.write(f"{out}/chunks/{i:03d}.wav", best_y, sr)
        p = join_pause(best_s.get("nat_pause"), c["end_of_ch"], floor, ch_gap)
        log["chunks"].append({"i": i, "ch": c["ch"], "text": j["text"], "pick": best_s,
                              "cands": [s for s, _ in cs], "pause_after": round(p, 2)})
        pieces += [best_y, np.zeros(int(p * sr), np.float32)]
    full = np.concatenate(pieces[:-1])
    sf.write(f"{out}/narration.wav", full, sr)
    audio_s = len(full) / sr
    log.update(join_floor=round(floor, 2), load_s=round(t_load, 1), gate_s=round(t_gate, 1), audio_s=round(audio_s, 1),
               timing={k: round(v, 1) if isinstance(v, float) else v for k, v in tm.items()},
               rtf=round(t_gate / audio_s, 3), takes_per_chunk=round(tm["takes"] / len(jobs), 2),
               failed=[i for i in cands if i not in done], cache_limit_gb=cache_gb, decode_step=decode_step,
               **watch.stop())
    mem1 = memory()
    log["env"] = machine()
    log["memory"] = {"start": mem0, "end": mem1,
                     "swapped": bool(mem0["pageouts"] is not None and mem1["pageouts"] - mem0["pageouts"] > 1000)}
    if log["memory"]["swapped"]:
        print(f"WARNING: {mem1['pageouts'] - mem0['pageouts']} pageouts during the run (swap "
              f"{mem0['swap_used_gb']} -> {mem1['swap_used_gb']} GB), so the timings are not representative. Close "
              f"other apps, or lower memory with --batch / --cache-limit / --decode-step "
              f"(footprint {log['footprint_gb']} GB)", flush=True)
    json.dump(log, open(f"{out}/run.json", "w"), indent=1)
    record({"run": os.path.basename(out.rstrip("/")), "ref": os.path.basename(ref_wav), "chunks": len(jobs),
            **{k: v for k, v in log.items() if k not in ("chunks", "ref")},
            "first_pass": sum(1 for c in log["chunks"] if c["cands"][0]["ok"])},
           perf_log or os.path.join(os.path.dirname(os.path.abspath(out)), "perf_runs.jsonl"))
    print(f"done: {audio_s:.0f}s narration in {t_gate:.0f}s (+{t_load:.0f}s load) = {log['rtf']} s of compute per "
          f"second of narration; {log['takes_per_chunk']} takes/chunk; gen {tm['gen']:.0f}s whisper "
          f"{tm['whisper']:.0f}s; below the gate: {log['failed']}; memory peak {log['peak_active_gb']} GB live, "
          f"footprint {log['footprint_gb']} GB", flush=True)
    return f"{out}/narration.wav"


def main():
    ap = argparse.ArgumentParser(description="Long-form narration with Qwen3-TTS on Apple Silicon (MLX).")
    ap.add_argument("--ref", required=True, help="reference voice to clone: a clean 10-40 s wav")
    ap.add_argument("--ref-text", help="the reference's exact transcript (default: <ref>.txt beside the wav, "
                                       "else Whisper's, saved there)")
    ap.add_argument("--script", required=True, help="text to narrate: .txt/.md (blank lines between paragraphs, "
                                                    "'#' lines start chapters) or .json (see qwen3_longform/script.py)")
    ap.add_argument("--out", required=True, help="output folder: narration.wav, chunks/, run.json")
    ap.add_argument("--no-normalize", action="store_true",
                    help="read the text as written (default: spell out currency, abbreviations, %%, symbols and "
                         "parentheses first)")
    ap.add_argument("--batch", type=int, default=32,
                    help="chunks generated together (default 32); memory grows with it (~1.2 GB less at 16)")
    ap.add_argument("--takes", type=int, default=3, help="most takes per chunk when takes fail the gate (default 3)")
    ap.add_argument("--limit", type=int, help="only the first N chunks")
    ap.add_argument("--cache-limit", type=float, metavar="GB",
                    help="cap the memory MLX keeps cached after freeing arrays (default: MLX's own)")
    ap.add_argument("--decode-step", type=int, default=50, metavar="FRAMES",
                    help="frames per codec-decoder call (default 50): same audio for any value; larger needs "
                         "more memory (~2.5 GB at 50, ~4 GB at 300)")
    ap.add_argument("--perf-log", help="JSONL file to append this run's speed and memory to "
                                       "(default: perf_runs.jsonl beside the output folder)")
    ap.add_argument("--analyse", action="store_true", help="also measure drift over the finished narration "
                                                           "(metrics.json)")
    a = ap.parse_args()
    wav = run(a.ref, a.script, a.out, ref_text=a.ref_text, B=a.batch, takes=a.takes, limit=a.limit,
              cache_gb=a.cache_limit, decode_step=a.decode_step, perf_log=a.perf_log, normalize=not a.no_normalize)
    if a.analyse:
        document = load_script_document(a.script, not a.no_normalize)
        chunks = make_chunks(document["lines"])
        chunks = chunks[:a.limit] if a.limit else chunks
        r = M.analyse(wav, M.reference_profile(a.ref), " ".join(c["text"] for c in chunks), asr_equiv=document["asr_equiv"])
        json.dump(r, open(f"{a.out}/metrics.json", "w"), indent=1)
        print(M.fmt_summary(os.path.basename(a.out.rstrip("/")), r["summary"]))


if __name__ == "__main__":
    main()
