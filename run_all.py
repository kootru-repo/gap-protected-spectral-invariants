"""
Top-level verification runner. Executes every check script in the repository
and reports a roll-up. Exits nonzero if any script fails, so it can gate CI.

Run: python run_all.py
"""
import os
import re
import subprocess
import sys

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


def main():
    total_pass = 0
    failed = []
    print("=" * 70)
    print("Verification suite")
    print("=" * 70)
    for label, cwd, script in SCRIPTS:
        r = subprocess.run([sys.executable, script], cwd=cwd,
                           capture_output=True, text=True)
        n = extract_count(r.stdout)
        ok = (r.returncode == 0)
        if n is not None:
            total_pass += n
        status = "ok" if ok else "FAIL"
        cnt = ("%d checks" % n) if n is not None else "(no count)"
        print("  %-46s %-12s [%s]" % (label, cnt, status))
        if not ok:
            failed.append(script)
            sys.stdout.write(r.stdout[-2000:])
            sys.stderr.write(r.stderr[-2000:])
    print("-" * 70)
    print("Aggregate checks passed (where counted): %d" % total_pass)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("All verification scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
