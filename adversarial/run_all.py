"""
Run every adversarial verifier and report a roll-up. Exits nonzero if any test
fails. None of the verifiers takes alpha / 137 / CODATA as an input (CODATA
appears only as an external end-comparison), so a green run is itself evidence
of target-blindness across the whole chain.

Run: C:\\Python313\\python.exe run_all.py
"""
import subprocess
import sys
import glob
import os

here = os.path.dirname(os.path.abspath(__file__))
tests = sorted(glob.glob(os.path.join(here, "verify_adversarial_*.py"))
               + glob.glob(os.path.join(here, "verify_substrate_*.py"))
               + glob.glob(os.path.join(here, "verify_integrality_*.py")))

total_pass = total = 0
failed = []
import re
for t in tests:
    r = subprocess.run([sys.executable, t], capture_output=True, text=True)
    name = os.path.basename(t)
    p = n = None
    for ln in r.stdout.splitlines():
        m = re.search(r"(\d+)/(\d+) checks passed", ln)
        if m:
            p, n = int(m.group(1)), int(m.group(2))
        m = re.search(r"TOTAL: (\d+) passed, (\d+) failed", ln)
        if m:
            p = int(m.group(1))
            n = p + int(m.group(2))
    if p is not None:
        summary = "%d/%d checks passed" % (p, n)
        total_pass += p
        total += n
    else:
        summary = "(exit %d, no count line)" % r.returncode
    print("%-42s %s" % (name, summary))
    if r.returncode != 0:
        failed.append(name)

print("-" * 64)
print("TOTAL: %d/%d checks across %d tests" % (total_pass, total, len(tests)))
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
print("all adversarial tests green")
sys.exit(0)
