"""
Top-level verification runner. Executes every check script in the repository
in parallel and reports a roll-up as each finishes. Exits nonzero if any
script fails, so it can gate CI.

Parallelism: each script runs in its own OS process; a thread pool launches
them and waits concurrently (subprocess releases the GIL), so the wall-clock
is the sum of CPU work divided across the available cores, not the serial sum.

Run: python run_all.py
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, working directory, script)
SCRIPTS = [
    ("Mode count, gap protection, decomposition", HERE, "verify.py"),
    ("Spectral radius and Born interval", HERE, "verify_spectral_bounds.py"),
    ("Krawtchouk eigenvalues and exact identities", HERE, "verify_krein.py"),
    ("Crystallographic classification", HERE, "classify_crystallographic.py"),
    ("Computational cross-checks", HERE, "verify_computational.py"),
    ("Substrate uniqueness over W(B_4)", os.path.join(HERE, "substrate-sweep"),
     "substrate_sweep.py"),
    ("Substrate uniqueness over 227 CARAT classes",
     os.path.join(HERE, "substrate-sweep"), "carat_sweep.py"),
    ("Adversarial / falsification suite",
     os.path.join(HERE, "adversarial"), "run_all.py"),
    ("Definitive value verification", HERE, "verify_all_values.py"),
]

COUNT_RE = re.compile(
    r"(\d+)\s*/\s*(\d+)\s+checks passed"
    r"|RESULTS?:\s*(\d+)\s+passed"
    r"|(\d+)\s+passed,\s*\d+\s+failed"
    r"|TOTAL:\s*(\d+)/(\d+)"
)


def extract_count(text):
    """Best-effort 'N passed' from a script's stdout; None if not found."""
    p = None
    for m in COUNT_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                p = int(g)
    return p


def run_one(item):
    label, cwd, script = item
    r = subprocess.run([sys.executable, script], cwd=cwd,
                       capture_output=True, text=True)
    return label, script, r.returncode, extract_count(r.stdout), r.stdout, r.stderr


def main():
    total_pass = 0
    failed = []
    captured = {}
    workers = min(len(SCRIPTS), (os.cpu_count() or 2))
    print("=" * 70, flush=True)
    print("Verification suite  (parallel, %d workers)" % workers, flush=True)
    print("=" * 70, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, item) for item in SCRIPTS]
        for fut in as_completed(futures):
            label, script, rc, n, out, err = fut.result()
            ok = (rc == 0)
            if n is not None:
                total_pass += n
            cnt = ("%d checks" % n) if n is not None else "passed"
            print("  %-46s %-12s [%s]" % (label, cnt, "ok" if ok else "FAIL"),
                  flush=True)
            if not ok:
                failed.append(script)
                captured[script] = (out, err)
    print("-" * 70, flush=True)
    print("Aggregate checks passed (where counted): %d" % total_pass, flush=True)
    if failed:
        for script in failed:
            out, err = captured[script]
            sys.stdout.write(out[-2000:])
            sys.stderr.write(err[-2000:])
        print("FAILED: " + ", ".join(failed))
        return 1
    print("All verification scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
