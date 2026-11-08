"""Performance records: each run writes its timings and memory to its run.json and appends one JSON line to a
log beside the run's folder (out/<name> -> out/perf_runs.jsonl). perf/runs.jsonl holds the runs behind
RESULTS.md.

Each record holds the run's timings, swap use and pageouts at its start and end (growth means the machine ran
short of memory and the timings are suspect), and the environment they were measured in (chip, memory, macOS, MLX and
mlx-audio versions, git commit), so speed numbers stay comparable as the code and the machine change.
`scripts/perf_table.py` prints such a log as a table.
"""
import json, os, platform, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def machine():
    from importlib.metadata import version
    mem = _sh("sysctl", "-n", "hw.memsize")
    return {"chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "mem_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
            "macos": platform.mac_ver()[0], "mlx": version("mlx"), "mlx_audio": version("mlx-audio"),
            "git": _sh("git", "rev-parse", "--short", "HEAD") + ("+dirty" if _sh("git", "status", "--porcelain", "qwen3_longform") else "")}


def memory():
    """Swap in use and cumulative pageouts. If either grows during a run, the machine ran short of memory and
    the run's timings are not comparable."""
    sw = _sh("sysctl", "-n", "vm.swapusage")  # "total = 6144.00M  used = 5202.31M  free = ..."
    used = float(sw.split("used = ")[1].split("M")[0]) / 1024 if "used = " in sw else None
    vm = _sh("vm_stat")
    po = next((int(l.split(":")[1].strip(" .")) for l in vm.splitlines() if l.startswith("Pageouts")), None)
    return {"swap_used_gb": round(used, 2) if used is not None else None, "pageouts": po}


class MemWatch:
    """Samples MLX's memory every `every` seconds in a daemon thread. MLX's own peak (mx.get_peak_memory) counts
    only live arrays; freed buffers it keeps in its cache for reuse still hold RAM, and they are what can push a
    machine into swap. `peak_total_gb` is the largest live + cached total seen. `footprint_gb` is the process's
    peak memory footprint as macOS accounts it (everything: MLX, PyTorch, Python), which is what decides
    whether the machine swaps; None on other systems."""

    def __init__(self, every=0.25):
        import threading
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, args=(every,), daemon=True)
        self._t.start()

    def _loop(self, every):
        import mlx.core as mx
        while not self._stop.wait(every):
            self.peak = max(self.peak, mx.get_active_memory() + mx.get_cache_memory())

    def stop(self):
        import mlx.core as mx
        self._stop.set(); self._t.join()
        fp = peak_footprint()
        return {"peak_active_gb": round(mx.get_peak_memory() / 1e9, 2), "peak_total_gb": round(self.peak / 1e9, 2),
                "footprint_gb": round(fp / 1e9, 2) if fp else None}


def peak_footprint():
    """Lifetime peak physical footprint of this process in bytes (macOS proc_pid_rusage, rusage_info_v4), else
    None. Unlike ru_maxrss it includes GPU buffers."""
    import ctypes, sys
    if sys.platform != "darwin":
        return None
    try:
        buf = (ctypes.c_uint64 * 64)()
        if ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
            return None
        return buf[2 + 28]  # after the 16-byte uuid: ri_lifetime_max_phys_footprint
    except OSError:
        return None


def record(entry, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(dict(entry, logged=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
