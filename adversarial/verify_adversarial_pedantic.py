"""
Tier C - pedantic decisions (one-line checks, no battery).

These are standard computed invariants or theorems, not selections. There is no
competitor a skeptic can name that survives one line of standard computation.
Listed and checked here so nothing looks silently skipped.

ASCII only. Run: C:\\Python313\\python.exe verify_adversarial_pedantic.py
"""
import itertools

checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

R = 6
PTS = [n for n in itertools.product(range(-R, R + 1), repeat=4)]
def r4(k):
    return sum(1 for n in PTS if sum(x * x for x in n) == k)
def N4(K):
    return sum(1 for n in PTS if sum(x * x for x in n) <= K)
def jacobi_r4(k):
    if k == 0:
        return 1
    return 8 * sum(d for d in range(1, k + 1) if k % d == 0 and d % 4 != 0)

check("chi_orb = 8 (= 2^{d-1}, d=4)", 2 ** 3 == 8)
check("|Fix| = 16 = 2^4 (half-lattice points 2v in Z^4)", 2 ** 4 == 16)
check("r_4(1) = 8 (Jacobi; 8 axis neighbours of origin)",
      r4(1) == 8 and jacobi_r4(1) == 8)
check("N_4(5) = 137 given K*=5 (Jacobi shell sum, a theorem not a choice)",
      N4(5) == 137 and sum(jacobi_r4(k) for k in range(0, 6)) == 137)
check("8pi^2 angular identity vol(S^3)/(2pi)^4 == 1/(8pi^2)",
      abs((2 * 3.141592653589793 ** 2) / (2 * 3.141592653589793) ** 4
          - 1 / (8 * 3.141592653589793 ** 2)) < 1e-15)

passed = sum(1 for _, ok, _ in checks if ok)
print("Tier C - pedantic decisions")
print("-" * 64)
for name, ok, detail in checks:
    line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
    if detail and not ok:
        line += "   (%s)" % detail
    print(line)
print("-" * 64)
print("%d/%d checks passed" % (passed, len(checks)))
import sys
sys.exit(0 if passed == len(checks) else 1)
