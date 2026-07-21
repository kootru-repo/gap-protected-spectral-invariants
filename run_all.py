"""
Top-level verification runner. Executes every check script in the repository
in parallel and shows a live progress bar with an accurate estimated time to
completion. Exits nonzero if any script fails, so it can gate CI.

Parallelism: each script runs in its own OS process; a thread pool launches
them and waits concurrently (subprocess releases the GIL), so the wall-clock
is roughly the parallel makespan across the available cores, not the serial
sum.

The ETA is wall-clock, not a naive sum: it simulates the remaining schedule on
W workers from per-step durations, and recalibrates to this machine's speed
online (after the first step finishes) so it is accurate across fast and slow
hosts. Per-step durations are remembered between runs in `.verify_timings.json`
(an exponential moving average), seeded from a measured baseline.

Live display is shown only on an interactive terminal. Under CI / pipes /
redirects it prints one plain line per step as it finishes (with its duration),
plus a summary, so logs stay clean. Force the rich view anywhere with
VERIFY_RICH=1; disable it with VERIFY_RICH=0.

Run: python run_all.py
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TIMINGS_FILE = os.path.join(HERE, ".verify_timings.json")

# (label, working directory, script)
SCRIPTS = [
    ("Mode count, gap protection, decomposition", HERE, "verify.py"),
    ("Spectral radius and operator-norm interval", HERE, "verify_spectral_bounds.py"),
    ("Krawtchouk eigenvalues and exact identities", HERE, "verify_krein.py"),
    ("Crystallographic classification", HERE, "classify_crystallographic.py"),
    ("Computational cross-checks", HERE, "verify_computational.py"),
    ("Prime-family classification", HERE, "verify_prime_family.py"),
    ("Theta determination at determinant 16", HERE,
     "verify_theta_determination.py"),
    ("Substrate uniqueness over W(B_4)", os.path.join(HERE, "substrate-sweep"),
     "substrate_sweep.py"),
    ("Substrate uniqueness over 227 CARAT classes",
     os.path.join(HERE, "substrate-sweep"), "carat_sweep.py"),
    ("Adversarial / falsification suite",
     os.path.join(HERE, "adversarial"), "run_all.py"),
]

# Measured baseline (seconds, one reference machine). Used until the host's own
# .verify_timings.json exists; replaced/refined by the EMA after each run.
# Measured under the real parallel run (includes the contention the heavy
# steps see from the adversarial sub-runner), so the cold-start ETA already
# reflects wall-clock, not idealised standalone times.
SHIPPED_BASELINE = {
    "verify.py": 1.0,
    "verify_spectral_bounds.py": 0.5,
    "verify_krein.py": 1.0,
    "classify_crystallographic.py": 2.0,
    "verify_computational.py": 15.0,
    "verify_prime_family.py": 0.4,
    "verify_theta_determination.py": 290.0,
    "substrate_sweep.py": 5.0,
    "carat_sweep.py": 85.0,
    "run_all.py": 85.0,
}

# A completed step only recalibrates the speed factor if its expected time is at
# least this many seconds; shorter steps are dominated by fixed subprocess /
# interpreter startup and would otherwise skew the estimate the moment they
# finish (the cold-start over-estimate spike).
MIN_CALIB_SECONDS = 3.0

# Shown as the denominator ("N/TOTAL checks") before any local run has recorded
# the real total (the per-machine cache is gitignored, so a fresh clone -- e.g.
# Colab -- always starts cold). Refined automatically once a run completes.
SHIPPED_TOTAL_CHECKS = 708

COUNT_RE = re.compile(
    r"(\d+)\s*/\s*(\d+)\s+checks passed"
    r"|RESULTS?:\s*(\d+)\s+passed"
    r"|(\d+)\s+passed,\s*\d+\s+failed"
    r"|SUMMARY:\s*(\d+)\s+pass,"
    r"|TOTAL:\s*(\d+)/(\d+)"
)


def extract_count(text):
    p = None
    for m in COUNT_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                p = int(g)
    return p


def load_expected():
    base = dict(SHIPPED_BASELINE)
    try:
        with open(TIMINGS_FILE) as fh:
            saved = json.load(fh)
        for k, v in saved.items():
            if isinstance(v, (int, float)) and v > 0:
                base[k] = float(v)
    except (OSError, ValueError):
        pass
    return base


def save_expected(expected, actuals, total_checks=0):
    """EMA update: blend the just-measured times into the stored profile."""
    merged = load_expected()
    for script, dt in actuals.items():
        prev = merged.get(script)
        merged[script] = round(dt if prev is None else 0.5 * prev + 0.5 * dt, 3)
    if total_checks:
        merged["__checks_total__"] = int(total_checks)
    try:
        with open(TIMINGS_FILE, "w") as fh:
            json.dump(merged, fh, indent=2)
    except OSError:
        pass


def fmt(sec):
    if sec is None or sec != sec or sec < 0:
        return "--:--"
    sec = int(round(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


# --------------------------------------------------------------------------
# ETA: simulate the remaining schedule on W workers.
# --------------------------------------------------------------------------
def predict_eta(states, expected, sf, workers, now):
    import heapq
    running, queued = [], []
    for st in states:
        e = max(expected.get(st["script"], 4.0) * sf, 0.05)
        if st["state"] == "running":
            running.append(max(e - (now - st["t_start"]), 0.05))
        elif st["state"] == "queued":
            queued.append((st["order"], e))
    if not running and not queued:
        return 0.0
    heap = []
    for r in sorted(running)[:workers]:
        heapq.heappush(heap, r)
    while len(heap) < workers:
        heapq.heappush(heap, 0.0)
    for _, e in sorted(queued):           # submission order
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + e)
    return max(heap)


def speed_factor(states, expected, now):
    """Duration-weighted actual/expected over completed steps that are long
    enough to be informative; 1.0 until such a step finishes."""
    a = e = 0.0
    for st in states:
        exp = max(expected.get(st["script"], 4.0), 0.05)
        if st["state"] == "done" and exp >= MIN_CALIB_SECONDS:
            a += st["t_end"] - st["t_start"]
            e += exp
    if e <= 0 or a <= 0:
        return 1.0
    return min(max(a / e, 0.2), 5.0)


def progress_snapshot(states, expected, workers, t0, total_checks):
    """A JSON-serialisable tick: the same bar/ETA numbers the terminal draws,
    for consumers like the reproduce.ipynb notebook to render live."""
    now = time.perf_counter()
    elapsed = now - t0
    sf = speed_factor(states, expected, now)
    eta = predict_eta(states, expected, sf, workers, now)
    total = elapsed + eta
    frac = 0.0 if total <= 0 else min(elapsed / total, 0.999)
    steps = []
    for st in states:
        e = max(expected.get(st["script"], 4.0) * sf, 0.05)
        if st["state"] == "done":
            steps.append({"label": st["label"], "state": "done",
                          "ok": st["rc"] == 0, "count": st["count"],
                          "dur": round(st["t_end"] - st["t_start"], 1)})
        elif st["state"] == "running":
            el = now - st["t_start"]
            steps.append({"label": st["label"], "state": "running",
                          "elapsed": round(el, 1),
                          "remaining": round(max(e - el, 0), 1)})
        else:
            steps.append({"label": st["label"], "state": "queued"})
    return {"t": "tick", "elapsed": round(elapsed, 1), "eta": round(eta, 1),
            "frac": round(frac, 4),
            "steps_done": sum(1 for s in states if s["state"] == "done"),
            "steps_total": len(states),
            "checks": sum(s["count"] or 0 for s in states if s["state"] == "done"),
            "checks_total": total_checks, "steps": steps}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Ansi:
    def __init__(self, on):
        self.on = on
    def __getattr__(self, _):
        return ""

def _enable_vt():
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint()
        k.GetConsoleMode(h, ctypes.byref(mode))
        k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return True
    except Exception:
        return False


SPIN = "|/-\\"
BAR_FULL, BAR_EMPTY = "#", "."


def render_frame(states, expected, workers, t0, total_checks, color):
    now = time.perf_counter()
    elapsed = now - t0
    sf = speed_factor(states, expected, now)
    eta = predict_eta(states, expected, sf, workers, now)
    total = elapsed + eta
    frac = 0.0 if total <= 0 else min(elapsed / total, 0.999)
    done = sum(1 for s in states if s["state"] == "done")
    passed = sum(s["count"] or 0 for s in states if s["state"] == "done")

    width = 38
    fill = int(round(frac * width))
    bar = BAR_FULL * fill + BAR_EMPTY * (width - fill)
    C = (lambda s, c: "\x1b[%sm%s\x1b[0m" % (c, s)) if color else (lambda s, c: s)

    lines = []
    head = "Verification suite   %d workers   %d/%d steps" % (workers, done, len(states))
    if total_checks:
        head += "   %d/%d checks" % (passed, total_checks)
    lines.append(head)
    lines.append("[%s] %3d%%   elapsed %s   ETA %s" %
                 (C(bar, "36"), int(frac * 100), fmt(elapsed), C(fmt(eta), "33")))
    lines.append("")
    spin = SPIN[int(now * 8) % len(SPIN)]
    for st in states:
        e = max(expected.get(st["script"], 4.0) * sf, 0.05)
        if st["state"] == "done":
            ok = st["rc"] == 0
            mark = C("[ok]", "32") if ok else C("FAIL", "31")
            cnt = ("%d checks" % st["count"]) if st["count"] is not None else "done"
            tail = "%-11s %s" % (cnt, fmt(st["t_end"] - st["t_start"]))
        elif st["state"] == "running":
            mark = C(" " + spin + " ", "36")
            el = now - st["t_start"]
            tail = C("running %s  ~%s" % (fmt(el), fmt(max(e - el, 0))), "33")
        else:
            mark = C(" . ", "90")
            tail = C("queued", "90")
        lines.append(" %s %-44s %s" % (mark, st["label"][:44], tail))
    return lines


class LiveRenderer(threading.Thread):
    def __init__(self, states, expected, workers, t0, total_checks):
        super().__init__(daemon=True)
        self.states, self.expected = states, expected
        self.workers, self.t0, self.total_checks = workers, t0, total_checks
        self._stop = threading.Event()
        self._n = 0
        self.color = sys.stdout.isatty() or os.environ.get("VERIFY_RICH") == "1"
        if self.color:
            self.color = _enable_vt()

    def _draw(self):
        lines = render_frame(self.states, self.expected, self.workers,
                             self.t0, self.total_checks, self.color)
        out = []
        if self._n:
            out.append("\x1b[%dA" % self._n)
        for ln in lines:
            out.append("\x1b[2K" + ln + "\n")
        self._n = len(lines)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def run(self):
        while not self._stop.is_set():
            self._draw()
            time.sleep(0.12)
        self._draw()

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)


# --------------------------------------------------------------------------
def run_one(state, lock):
    state["state"] = "running"
    state["t_start"] = time.perf_counter()
    r = subprocess.run([sys.executable, state["script"]], cwd=state["cwd"],
                       capture_output=True, text=True)
    with lock:
        state["t_end"] = time.perf_counter()
        state["rc"] = r.returncode
        state["count"] = extract_count(r.stdout)
        state["out"], state["err"] = r.stdout, r.stderr
        state["state"] = "done"


def main():
    expected = load_expected()
    states = []
    for i, (label, cwd, script) in enumerate(SCRIPTS):
        states.append({"order": i, "label": label, "cwd": cwd, "script": script,
                       "state": "queued", "t_start": None, "t_end": None,
                       "rc": None, "count": None, "out": "", "err": ""})
    total_checks = int(expected.get("__checks_total__", 0) or 0) or SHIPPED_TOTAL_CHECKS
    workers = min(len(SCRIPTS), (os.cpu_count() or 2))
    rich = sys.stdout.isatty() or os.environ.get("VERIFY_RICH") == "1"
    if os.environ.get("VERIFY_RICH") == "0":
        rich = False
    json_mode = os.environ.get("VERIFY_PROGRESS") == "json"
    if json_mode:
        rich = False
    eta_log = os.environ.get("VERIFY_ETA_LOG")

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    t0 = time.perf_counter()
    lock = threading.Lock()
    renderer = None
    if rich:
        renderer = LiveRenderer(states, expected, workers, t0, total_checks)
        renderer.start()
    elif json_mode:
        emit({"t": "start", "steps_total": len(states), "workers": workers,
              "checks_total": total_checks,
              "labels": [s["label"] for s in states]})
    else:
        print("=" * 70, flush=True)
        print("Verification suite  (parallel, %d workers)" % workers, flush=True)
        print("=" * 70, flush=True)

    tick_stop = threading.Event()
    def ticker():
        while not tick_stop.is_set():
            emit(progress_snapshot(states, expected, workers, t0, total_checks))
            time.sleep(0.35)
    ticker_thread = None
    if json_mode:
        ticker_thread = threading.Thread(target=ticker, daemon=True)
        ticker_thread.start()

    log_rows = []
    log_stop = threading.Event()
    def eta_logger():
        while not log_stop.is_set():
            now = time.perf_counter()
            sf = speed_factor(states, expected, now)
            eta = predict_eta(states, expected, sf, workers, now)
            log_rows.append((round(now - t0, 3), round(eta, 3), round(now - t0 + eta, 3)))
            time.sleep(0.2)
    logger = None
    if eta_log:
        logger = threading.Thread(target=eta_logger, daemon=True)
        logger.start()

    done_seen = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, st, lock): st for st in states}
        # poll for completions so non-rich mode can print as each finishes
        while len(done_seen) < len(states):
            for st in states:
                if st["state"] == "done" and st["order"] not in done_seen:
                    done_seen.add(st["order"])
                    if json_mode:
                        emit(progress_snapshot(states, expected, workers, t0, total_checks))
                    elif not rich:
                        ok = st["rc"] == 0
                        cnt = ("%d checks" % st["count"]) if st["count"] is not None else "passed"
                        print("  %-46s %-12s %-7s %s" %
                              (st["label"], cnt, "[ok]" if ok else "[FAIL]",
                               fmt(st["t_end"] - st["t_start"])), flush=True)
            time.sleep(0.05)

    if renderer:
        renderer.stop()
    if ticker_thread:
        tick_stop.set(); ticker_thread.join(timeout=1.0)
    if logger:
        log_stop.set(); logger.join(timeout=1.0)

    actuals = {st["script"]: st["t_end"] - st["t_start"] for st in states}
    counted_total = sum(st["count"] or 0 for st in states if st["count"] is not None)
    save_expected(expected, actuals, counted_total)
    if eta_log:
        try:
            json.dump({"rows": log_rows, "actual_total": round(time.perf_counter() - t0, 3),
                       "per_step": {s["script"]: round(actuals[s["script"]], 3) for s in states}},
                      open(eta_log, "w"), indent=2)
        except OSError:
            pass

    total_pass = sum(st["count"] or 0 for st in states if st["count"] is not None)
    failed = [st for st in states if st["rc"] != 0]
    wall = time.perf_counter() - t0
    if json_mode:
        emit({"t": "end", "ok": not failed, "wall": round(wall, 1),
              "checks": total_pass, "checks_total": total_checks,
              "failed": [{"script": st["script"],
                          "tail": (st["out"][-1500:] + st["err"][-800:])}
                         for st in failed]})
        return 1 if failed else 0
    print("-" * 70, flush=True)
    print("Checks passed (where counted): %d   wall time: %s" %
          (total_pass, fmt(wall)), flush=True)
    if failed:
        for st in failed:
            sys.stdout.write(st["out"][-2000:])
            sys.stderr.write(st["err"][-2000:])
        print("FAILED: " + ", ".join(st["script"] for st in failed), flush=True)
        return 1
    print("All verification scripts passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
