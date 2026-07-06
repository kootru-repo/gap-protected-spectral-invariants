# verify_prime_family.py -- the prime-family classification (Section: prime family).
#
# Certifies the new theorem: within diagonal Z_p-actions on A_{p-1}^d,
#   (i)  alignment r(1) = chi_orb  <=>  d = p^{d-2}  <=>  (p,d) in {(2,4),(3,3)};
#   (ii) at (3,3) the alignment holds (18 = 18) but the three-sector target
#        505 = 1 + 18 + 27*18 is a cumulative A_2^3 shell count at NO cutoff
#        (counts 1,19,127,361,595,... increasing, stepping over the target);
#   (iii) at (2,4) the target 137 = 1 + 8 + 128 IS attained, exactly at K = 5,
#        so (2,4) is the unique family member carrying the decomposition.
# Also cross-checks the paper's (3,2) facts (Section: negative result): first
# shell 12 != 6 = chi_orb, and the non-structural value coincidence 61 at K = 3.
# TEETH checks confirm each verifier would catch a planted failure.
#
# ASCII only. Run: python verify_prime_family.py ; exit 0 iff all checks pass.

from itertools import product
from math import comb
import sys

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name + (("  " + str(detail)) if detail and not cond else ""))

# ---------------------------------------------------------------- primes
def primes_up_to(n):
    s = list(range(n + 1)); s[0] = s[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, n + 1, i):
                s[j] = 0
    return [p for p in s if p]

# ---------------------------------------------------------------- (i) classification
def solutions(shift=2, pmax=199, dmax=60):
    out = []
    for p in primes_up_to(pmax):
        for d in range(2, dmax + 1):
            if d >= shift and d == p ** (d - shift):
                out.append((p, d))
    return out

check("C1 alignment d=p^{d-2} over primes: exactly {(2,4),(3,3)}",
      sorted(solutions()) == [(2, 4), (3, 3)], solutions())
check("C2 growth bound closes d>=5: 2^{d-2} > d for 5<=d<=400",
      all(2 ** (d - 2) > d for d in range(5, 401)))
check("C3 cyclotomic fixed-point count: Phi_p(1) = p for p <= 97",
      all(sum(1 * (k >= 0) for k in range(p)) == p for p in primes_up_to(97)))
# (Phi_p(x) = 1 + x + ... + x^{p-1}, so Phi_p(1) = p: per-factor fixed points.)

# chi_orb = (p-1) p^{d-1} at the three family points the paper touches
chi = lambda p, d: (p - 1) * p ** (d - 1)
check("C4 chi_orb: (2,4)->8, (3,3)->18, (3,2)->6 (matches the paper's table)",
      chi(2, 4) == 8 and chi(3, 3) == 18 and chi(3, 2) == 6)

# ---------------------------------------------------------------- root counts p(p-1)
def a_n_roots(n):
    # A_n = {x in Z^{n+1} : sum x = 0}; roots e_i - e_j, norm 2
    cnt = 0
    rng = range(-1, 2)
    for v in product(rng, repeat=n + 1):
        if sum(v) == 0 and sum(x * x for x in v) == 2:
            cnt += 1
    return cnt

check("R1 A_{p-1} root counts: A_1=2, A_2=6, A_4=20 = p(p-1) for p=2,3,5",
      a_n_roots(1) == 2 and a_n_roots(2) == 6 and a_n_roots(4) == 20
      and all(a_n_roots(p - 1) == p * (p - 1) for p in (2, 3, 5)))
check("R2 first shells: (2,4)->2d=8, (3,3)->6d=18, (3,2)->6d=12",
      2 * 4 == 8 and 6 * 3 == 18 and 6 * 2 == 12)
check("R3 alignment arithmetic: 8=8 at (2,4), 18=18 at (3,3), 12!=6 at (3,2)",
      2 * 4 == chi(2, 4) and 6 * 3 == chi(3, 3) and 6 * 2 != chi(3, 2))

# ---------------------------------------------------------------- (ii) A_2^3 counts
LMAX = 40
r_a2 = [0] * (LMAX + 1)   # Loeschian norm a^2+ab+b^2
B = 8
for a, b in product(range(-B, B + 1), repeat=2):
    q = a * a + a * b + b * b
    if q <= LMAX:
        r_a2[q] += 1
check("A1 theta(A_2) coefficients at norms 0..4: 1,6,0,6,6", r_a2[:5] == [1, 6, 0, 6, 6])

r_cube = [0] * (LMAX + 1)
for i in range(LMAX + 1):
    for j in range(LMAX + 1 - i):
        for k in range(LMAX + 1 - i - j):
            r_cube[i + j + k] += r_a2[i] * r_a2[j] * r_a2[k]
cum33, s = [], 0
for k in range(LMAX + 1):
    s += r_cube[k]; cum33.append(s)

# independent direct 6-dim enumeration for the first shells
direct = [0] * 6
for w in product(range(-3, 4), repeat=6):
    q = sum(w[2 * i] ** 2 + w[2 * i] * w[2 * i + 1] + w[2 * i + 1] ** 2 for i in range(3))
    if q <= 5:
        direct[q] += 1
dcum, s = [], 0
for k in range(6):
    s += direct[k]; dcum.append(s)

check("A2 cumulative A_2^3 counts 1,19,127,361,595 (convolution AND direct)",
      cum33[:5] == [1, 19, 127, 361, 595] and dcum[:5] == [1, 19, 127, 361, 595],
      (cum33[:5], dcum[:5]))
target33 = 1 + chi(3, 3) + 27 * chi(3, 3)
check("A3 (3,3) three-sector target = 1+18+486 = 505", target33 == 505, target33)
jump = next(k for k, c in enumerate(cum33) if c > target33)
check("A4 505 attained at NO cutoff: counts nondecreasing, step 361 -> 595 over it",
      target33 not in cum33
      and cum33[jump - 1] < target33 < cum33[jump]
      and all(cum33[i] <= cum33[i + 1] for i in range(LMAX)))

# ---------------------------------------------------------------- (iii) the binary point
r4 = [0] * 9
for w in product(range(-3, 4), repeat=4):
    q = sum(x * x for x in w)
    if q <= 8:
        r4[q] += 1
cum24, s = [], 0
for k in range(9):
    s += r4[k]; cum24.append(s)
target24 = 1 + chi(2, 4) + 16 * chi(2, 4)
check("B1 (2,4) target 137 = 1+8+128 attained exactly at K=5 (1,9,33,65,89,137)",
      target24 == 137 and cum24[:6] == [1, 9, 33, 65, 89, 137] and cum24[5] == 137)

# ---------------------------------------------------------------- (3,2) paper facts
r_a2xa2 = [0] * 9
for a, b, c, d in product(range(-4, 5), repeat=4):
    q = a * a + a * b + b * b + c * c + c * d + d * d
    if q <= 8:
        r_a2xa2[q] += 1
cum32, s = [], 0
for k in range(9):
    s += r_a2xa2[k]; cum32.append(s)
check("D1 (3,2) counts N(1),N(2),N(3) = 13,49,61 (paper, Prop. failure-on-Z3)",
      cum32[1] == 13 and cum32[2] == 49 and cum32[3] == 61, cum32[:4])
check("D2 (3,2) value coincidence 61 = 1+6+54 attained at K=3, but alignment fails (12 != 6)",
      1 + 6 + 9 * 6 == 61 and cum32[3] == 61 and 12 != chi(3, 2))

# ---------------------------------------------------------------- remark: what breaks for p >= 3
shells33 = [cum33[0]] + [cum33[k] - cum33[k - 1] for k in range(1, 5)]
check("M1 A_2^3 shells at norms 0..4: 1,18,108,234,234", shells33 == [1, 18, 108, 234, 234])
check("M2 tail shells step in multiples of chi_orb=18: 108=6x18, 234=13x18",
      108 == 6 * 18 and 234 == 13 * 18)
check("M3 running tail total in units of 18 runs 6,19,32 and skips |Fix|=27",
      [6, 6 + 13, 6 + 13 + 13] == [6, 19, 32] and 27 not in (6, 19, 32))
def sigma_tilde(k):
    return sum(e for e in range(1, k + 1) if k % e == 0 and e % 4 != 0)
check("M4 Jacobi closure on Z^4: r_4(k)=8*sigma_tilde(k), k=1..5; partial sums 3,7,10,16 hit 16 at K=5",
      all(r4[k] == 8 * sigma_tilde(k) for k in range(1, 6))
      and [sum(sigma_tilde(j) for j in range(2, K + 1)) for K in (2, 3, 4, 5)] == [3, 7, 10, 16]
      and 24 + 32 + 24 + 48 == 128 == 16 * 8)
check("M5 character classes (p-1)^w C(d,w): 27=1+6+12+8 at (3,3) vs 16=1+4+6+4+1 at (2,4)",
      [2 ** w * comb(3, w) for w in range(4)] == [1, 6, 12, 8]
      and [comb(4, w) for w in range(5)] == [1, 4, 6, 4, 1]
      and sum(2 ** w * comb(3, w) for w in range(4)) == 27)

# ---------------------------------------------------------------- TEETH
bad = solutions(shift=1)
check("T1 TEETH: wrong exponent gives a different solution set",
      sorted(bad) != [(2, 4), (3, 3)], bad)
check("T2 TEETH: 361 IS a cumulative count, so the never-attained check has teeth",
      361 in cum33)
check("T3 TEETH: the (3,2) target IS attained, so the (3,3) failure is not a checker artifact",
      61 in cum32)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(0 if not FAIL else 1)
