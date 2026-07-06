"""
Run every adversarial verifier in parallel and report a roll-up. Exits nonzero
if any test fails. None of the verifiers takes alpha, 137, or CODATA as
an input to any computed quantity, so a green run is itself
evidence of target-blindness across the whole chain.

Each verifier runs in its own process; a thread pool launches them and waits
concurrently, so the wall-clock is far below the serial sum.

Run: python run_all.py
"""
import subprocess
import sys
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

here = os.path.dirname(os.path.abspath(__file__))
tests = sorted(glob.glob(os.path.join(here, "verify_adversarial_*.py")))


def run_one(t):
    r = subprocess.run([sys.executable, t], capture_output=True, text=True)
    p = n = None
    for ln in r.stdout.splitlines():
        m = re.search(r"(\d+)/(\d+) checks passed", ln)
        if m:
            p, n = int(m.group(1)), int(m.group(2))
        m = re.search(r"TOTAL: (\d+) passed, (\d+) failed", ln)
        if m:
            p = int(m.group(1))
            n = p + int(m.group(2))
    return os.path.basename(t), r.returncode, p, n


total_pass = total = 0
failed = []
workers = min(len(tests), (os.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(run_one, t) for t in tests]
    for fut in as_completed(futures):
        name, rc, p, n = fut.result()
        if p is not None:
            summary = "%d/%d checks passed" % (p, n)
            total_pass += p
            total += n
        else:
            summary = "(exit %d, no count line)" % rc
        print("%-42s %s" % (name, summary), flush=True)
        if rc != 0:
            failed.append(name)

print("-" * 64)
print("TOTAL: %d/%d checks across %d tests" % (total_pass, total, len(tests)))
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
print("all adversarial tests green")
sys.exit(0)
