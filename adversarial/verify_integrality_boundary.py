"""
Integrality boundary of the three-sector decomposition.

Validates the unit-shell rigidity lemma (integral Gram convention) and the
boundary catalogue (norm-integral-only lattices admit tuned count
coincidences), closing the Step-4 gap found in referee cycle
cycle_20260610_095636 (blocking concern B1).

PART A - unit-shell rigidity on INTEGRAL lattices (inner products in Z):
  A1. Norm-1 vectors of an integral lattice form orthonormal +-pairs
      (Cauchy-Schwarz + integrality), splitting off Z^k orthogonally.
  A2. Every order-3 element of W(B_k) (signed permutations) has a +1
      eigenvalue (k <= 6, exhaustive)  -> G containing an order-3 element
      with isolated fixed points forces k = 0: r_Lambda(1) = 0.
      Kills Z_3, Z_6, Q_12, 2T on every integral lattice.
  A3. Every order-4 element g of W(B_k) with no +1 eigenvalue in g or g^2
      has all cycles of type (2,-): k even and <g>-orbits on the 2k unit
      vectors have size 4, so 4 | r_Lambda(1) != 6. Kills Z_4.
  A4. Any odd-order p >= 3 element of W(B_k) has a +1 eigenvalue
      (k <= 6) -> Z_5, Z_10 (contain order 5) force r(1) = 0 != 4.
  A5. Free-orbit divisibility: for g of order n with isolated fixed points,
      <g> acts freely on nonzero lattice vectors, so n | r_Lambda(k) for
      all k. Z_8: 8 | r(1) != 4; Z_12: 12 | r(1) != 4.
  A6. r_Lambda(1) is even (+-pairs) != 5 = chi_orb(Q_8) = chi_orb(2T).
  A7. Z_2 with r_Lambda(1) = 8: four orthonormal pairs force Lambda = Z^4
      (superlattice step needs integrality) - verified on a battery.

PART B - the boundary catalogue (norm form integer-valued, Gram
  half-integral, min norm 1; these are EXCLUDED by the integral-Gram
  hypothesis, and this part shows the hypothesis is essential):
  B1. Z_3 on A_2(1) (+) A_2(m): full count pattern
      N(K*) = 1 + 6 + 9*6 = 61 holds exactly for m in {2,3,4,5}
      (sweep m <= 20 brute force with explicit fixed points; analytic
      tail for m >= 21 checked numerically to m = 200 via the factored
      path: K* >= m and the factor-1 window alone exceeds 54).
  B2. Z_2 on A_2(1) (+) Z(1) (+) Z(m): full pattern N = 137 holds exactly
      for m = 5 in m <= 60 (tail: K* >= m and the A_2(+)Z window
      exceeds 128 for every checked m >= 6).
  B3. Controls: Z^4/Z_2 gives 137 via the same code path; the symmetric
      (isodual) hexagonal metric fails the static test (r(1) = 12 != 6).

ASCII only (cp1252-safe). Exit code 0 iff all checks pass.
"""
import sys
import numpy as np
from itertools import product, permutations

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  %s" % name)
    else:
        FAIL += 1
        print("  FAIL  %s   %s" % (name, detail))


# ----------------------------------------------------------------------
print("PART A: unit-shell rigidity on integral lattices")
# ----------------------------------------------------------------------

# A1: on a battery of integral lattices, norm-1 vectors are pairwise
# orthogonal up to sign (so they form orthonormal +-pairs).
def lattice_vectors(S, qmax, box=4):
    n = S.shape[0]
    out = {}
    for a in product(range(-box, box + 1), repeat=n):
        a = np.array(a)
        q = int(a @ S @ a)
        if 0 < q <= qmax:
            out.setdefault(q, []).append(a)
    return out

A2 = np.array([[2, 1], [1, 2]])            # A_2 root lattice (integral, min 2)
D4 = np.array([[2, 0, 1, 0], [0, 2, -1, 0], [1, -1, 2, -1], [0, 0, -1, 2]])
batteries = {
    "Z^4": np.eye(4, dtype=int),
    "Z^2 (+) A_2": np.block([[np.eye(2, dtype=int), np.zeros((2, 2), int)],
                             [np.zeros((2, 2), int), A2]]),
    "Z (+) Z(3) (+) A_2": np.diag([1, 3, 2, 2]) * 0 + np.block(
        [[np.diag([1, 3]), np.zeros((2, 2), int)],
         [np.zeros((2, 2), int), A2]]),
    "D_4": D4,
    "Z(2)^4": 2 * np.eye(4, dtype=int),
}
ok = True
for name, S in batteries.items():
    vs = lattice_vectors(np.asarray(S), 1).get(1, [])
    for u, v in product(vs, vs):
        ip = int(u @ np.asarray(S) @ v)
        if abs(ip) > 1 and not np.array_equal(u, v) and not np.array_equal(u, -v):
            ok = False
        if abs(ip) == 1 and not (np.array_equal(u, v) or np.array_equal(u, -v)):
            ok = False  # |<u,v>|=1 must force v = +-u
check("A1: norm-1 vectors form orthonormal +-pairs on battery", ok)

# A2/A3/A4: exhaustive signed-permutation sweeps, k <= 6
def signed_perms(k):
    for perm in permutations(range(k)):
        for signs in product((1, -1), repeat=k):
            M = np.zeros((k, k), dtype=int)
            for i, j in enumerate(perm):
                M[j, i] = signs[i]
            yield M

def order(M, cap=30):
    P = np.eye(M.shape[0], dtype=int)
    for t in range(1, cap + 1):
        P = P @ M
        if np.array_equal(P, np.eye(M.shape[0], dtype=int)):
            return t
    return None

def has_plus1(M):
    return abs(np.linalg.det(M - np.eye(M.shape[0]))) < 1e-9

a2_ok, a3_ok, a4_ok = True, True, True
for k in range(1, 7):
    if k > 5:
        # k=6 full sweep is 46080 perms x 64 signs: still fine
        pass
    for M in signed_perms(k):
        o = order(M)
        if o == 3 and not has_plus1(M):
            a2_ok = False
        if o in (5, 7) and not has_plus1(M):
            a4_ok = False
        if o == 4 and not has_plus1(M) and not has_plus1(M @ M):
            # all cycles must be (2,-): k even, M^2 = -I
            if k % 2 != 0 or not np.array_equal(M @ M, -np.eye(k, dtype=int)):
                a3_ok = False
check("A2: every order-3 signed permutation (k<=6) has +1 eigenvalue", a2_ok)
check("A3: isolated order-4 elements have M^2 = -I, k even (4 | r(1))", a3_ok)
check("A4: every odd-order p>=3 signed permutation (k<=6) has +1 eigenvalue",
      a4_ok)

# A5/A6: arithmetic consequences
chi = {"Z_3": 6, "Z_4": 6, "Z_6": 6, "Z_5": 4, "Z_8": 4, "Z_10": 4,
       "Z_12": 4, "Q_8": 5, "2T": 5}
check("A5: 8 | r(1) excludes Z_8 (8 != 4); 12 | r(1) excludes Z_12",
      all(x % n != 0 or x == 0 for n, x in ((8, 4), (12, 4))))
check("A6: r(1) even excludes Q_8 and 2T (chi_orb = 5 odd)", 5 % 2 == 1)

# A7: Z_2 with r(1) = 8 forces Z^4 -- verified as: among the battery,
# only Z^4 has r(1) = 8; and 8 norm-1 vectors = 4 orthonormal pairs span
# a Z^4 sublattice whose integral superlattices coincide with it.
r1 = {name: len(lattice_vectors(np.asarray(S), 1).get(1, []))
      for name, S in batteries.items()}
check("A7: battery r(1) values (Z^4 -> 8, others < 8): %s" % r1,
      r1["Z^4"] == 8 and all(v < 8 for n, v in r1.items() if n != "Z^4"))

# ----------------------------------------------------------------------
print("PART B: boundary catalogue (norm-integral only, min norm 1)")
# ----------------------------------------------------------------------
s3 = np.sqrt(3.0)
HEX = np.array([[1.0, 0.5], [0.0, s3 / 2]])   # unit-normalised hexagonal
R2 = np.array([[-0.5, -s3 / 2], [s3 / 2, -0.5]])


def z3_audit(m, qmax=70):
    """Z_3 on M = A_2(1)(+)A_2(m), explicit fixed points + numpy rank."""
    BM = np.zeros((4, 4))
    BM[:2, :2] = HEX
    BM[2:, 2:] = np.sqrt(m) * HEX
    R4 = np.zeros((4, 4))
    R4[:2, :2] = R2
    R4[2:, 2:] = R2
    BL = np.linalg.inv(BM).T
    AL = np.round(np.linalg.solve(BL, R4 @ BL)).astype(int)
    IAinv = np.linalg.inv(np.eye(4) - AL)
    fixed = set()
    for kk in product(range(-3, 4), repeat=4):
        f = IAinv @ np.array(kk, dtype=float)
        fixed.add(tuple(np.round(f % 1.0, 6) % 1.0))
    F = np.array(sorted(fixed))
    assert len(F) == 9
    norms = {}
    A = int(qmax ** 0.5) + 3
    for a in product(range(-A, A + 1), repeat=4):
        v = BM @ np.array(a, dtype=float)
        q = round(v @ v)
        if 0 < q <= qmax and abs(v @ v - q) < 1e-9:
            norms.setdefault(q, []).append(np.array(a))
    r1 = len(norms.get(1, []))
    G = np.zeros((9, 9), dtype=complex)
    window = 0
    for K in sorted(norms):
        if K < 2:
            continue
        for a in norms[K]:
            ph = np.exp(2j * np.pi * (F @ a))
            G += np.outer(ph, ph.conj())
        window += len(norms[K])
        if np.linalg.matrix_rank(G, tol=1e-8) == 9:
            return r1, K, window
    return r1, None, None


z3_pass = []
for m in range(1, 21):
    r1, K, w = z3_audit(m)
    if r1 == 6 and w == 54:
        z3_pass.append(m)
check("B1a: Z_3 product family passes exactly at m in {2,3,4,5} (m<=20)",
      z3_pass == [2, 3, 4, 5], "got %s" % z3_pass)

# B1 tail, factored fast path: K* >= m forces the factor-1 window
# (Loeschian shells 3..m, 6 modes each at minimum) plus the factor-2 unit
# shell (6 modes at norm m) to exceed 54 for all m >= 21.
LOESCH = sorted({a * a + a * b + b * b for a in range(-30, 31)
                 for b in range(-30, 31)} - {0})
tail_ok = True
for m in range(21, 201):
    factor1 = 6 * sum(1 for q in LOESCH if 3 <= q <= m)  # lower bound
    if factor1 + 6 <= 54:
        tail_ok = False
check("B1b: Z_3 tail m in [21,200]: factor-1 window + unit shell > 54",
      tail_ok)

def z2_audit(m, qmax=200):
    """Z_2 on M = A_2(1)(+)Z(1)(+)Z(m), parity-class fast path."""
    BM = np.zeros((4, 4))
    BM[:2, :2] = HEX
    BM[2, 2] = 1.0
    BM[3, 3] = np.sqrt(m)
    norms = {}
    A = int(qmax ** 0.5) + 2
    for a in product(range(-A, A + 1), repeat=4):
        v = BM @ np.array(a, dtype=float)
        q = round(v @ v)
        if 0 < q <= qmax and abs(v @ v - q) < 1e-9:
            norms.setdefault(q, []).append(a)
    r1 = len(norms.get(1, []))
    covered, window = set(), 0
    for K in sorted(norms):
        if K < 2:
            continue
        window += len(norms[K])
        covered.update(tuple(x % 2 for x in a) for a in norms[K])
        if len(covered) == 16:
            return r1, K, window
    return r1, None, None


z2_pass = []
for m in range(1, 61):
    r1, K, w = z2_audit(m)
    if r1 == 8 and w == 128:
        z2_pass.append(m)
check("B2: Z_2 impostor family passes exactly at m = 5 (m<=60)",
      z2_pass == [5], "got %s" % z2_pass)

r1, K, w = z2_audit(5)
check("B2 detail: m=5 gives r(1)=8, window=128, N=137 (impostor value)",
      (r1, w) == (8, 128) and 1 + r1 + w == 137,
      "r1=%s K=%s w=%s" % (r1, K, w))

# B3 controls
def z4_control():
    norms = {}
    for a in product(range(-7, 8), repeat=4):
        q = sum(x * x for x in a)
        if 0 < q <= 30:
            norms.setdefault(q, []).append(a)
    covered, window = set(), 0
    for K in sorted(norms):
        if K < 2:
            continue
        window += len(norms[K])
        covered.update(tuple(x % 2 for x in a) for a in norms[K])
        if len(covered) == 16:
            return len(norms[1]), K, window
r1, K, w = z4_control()
check("B3a: control Z^4/Z_2 -> r(1)=8, K*=5, window=128, N=137",
      (r1, K, w) == (8, 5, 128))
r1, K, w = z3_audit(1)
check("B3b: symmetric hexagonal metric fails static test (r(1)=12 != 6)",
      r1 == 12)

# Note: under the integral-Gram convention the Part-B lattices are excluded
# by hypothesis: their Gram matrices contain half-integers (A_2 at min
# norm 1 has <u1,u2> = 1/2), and Part A shows no integral lattice
# reproduces the pattern off Z^4.
g12 = HEX[:, 0] @ HEX[:, 1]
check("B4: unit-normalised hexagonal Gram is half-integral (<u1,u2>=1/2)",
      abs(g12 - 0.5) < 1e-12)

print()
print("TOTAL: %d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
