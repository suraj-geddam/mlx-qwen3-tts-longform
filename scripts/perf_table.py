"""Speed and memory of narration runs, as a markdown table.

    uv run python scripts/perf_table.py [log.jsonl ...]     # default: perf/runs.jsonl (the runs in RESULTS.md)

Every run appends a line to a JSONL log (by default out/perf_runs.jsonl for runs written to out/<name>).
"Compute per narration s" is seconds of compute per second of finished narration (lower is better; model
loading excluded). "gen" and "Whisper" are per second of generated audio: every take, including lead-ins,
guards and retakes. "peak GB" is MLX's live-array peak; "footprint" is the whole process's peak as macOS
counts it. Runs marked "swapped" ran short of memory (macOS paged to disk), so their timings are not
representative.
"""
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logs = sys.argv[1:] or [f"{ROOT}/perf/runs.jsonl"]
rows = [json.loads(l) for p in logs for l in open(p) if l.strip()]
f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "–"
print("| run | ref | chunks | batch | narration s | compute s | **per narration s** | takes/chunk | first pass | "
      "gen per gen. s | Whisper per gen. s | peak GB | footprint GB | cache cap / decode step | machine |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    t = r.get("timing", {}); ga = t.get("gen_audio_s") or 0
    env = r.get("env", {})
    print(f"| {r['run']} | {r.get('ref', '').replace('.wav', '')} | {r.get('chunks')} | {r.get('batch', '–')} | "
          f"{r.get('audio_s')} | {r.get('gate_s')} | **{f(r.get('rtf'))}** | {r.get('takes_per_chunk')} | "
          f"{r.get('first_pass')}/{r.get('chunks')} | {f(t['gen'] / ga) if ga and t.get('gen') else '–'} | "
          f"{f(t['whisper'] / ga) if ga and t.get('whisper') else '–'} | "
          f"{r.get('peak_active_gb', r.get('peak_gb', '–'))} | {r.get('footprint_gb') or '–'} | "
          f"{(str(r.get('cache_limit_gb') or '–') + ' / ' + str(r['decode_step'])) if 'decode_step' in r else '–'} | "
          f"{(env['chip'] + ' ' + str(env['mem_gb']) + ' GB') if env.get('chip') else '–'}"
          f"{' **swapped**' if r.get('memory', {}).get('swapped') else ''} |")
