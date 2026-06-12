"""
Test 02 - saturation cutoff K* = 5.

K* := min{K >= 2 : rank(G^dyn(K)) = |Fix| = 16}, dynamical sector 2 <= |n|^2 <= K.
The integer 137 = N_4(5) flips if K* moves, so this is where "reverse-targeted
137" is suspected. We recompute everything from scratch - lattice point counts,
the 16x16 Gram-matrix rank (two independent ways), and the k_0 = 2 boundary -
using no alpha / 137 / CODATA as input.

ASCII only. Run: C:\\Python313\\python.exe verify_adversarial_kstar.py
"""
import itertools
import math

checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

# ---------------------------------------------------------------------------
# Lattice machinery (brute force, d=4)
# ---------------------------------------------------------------------------
R = 6
PTS = [n for n in itertools.product(range(-R, R + 1), repeat=4)]

def r4(k):
    return sum(1 for n in PTS if sum(x * x for x in n) == k)

def N4(K):
    return sum(1 for n in PTS if sum(x * x for x in n) <= K)

# r_4 sanity vs Jacobi r_4(k) = 8 * sum of divisors of k not divisible by 4
def jacobi_r4(k):
    if k == 0:
        return 1
    return 8 * sum(d for d in range(1, k + 1) if k % d == 0 and d % 4 != 0)

for k in range(0, 7):
    check("r_4(%d) matches Jacobi" % k, r4(k) == jacobi_r4(k),
          "brute=%d jacobi=%d" % (r4(k), jacobi_r4(k)))
check("r_4(1) = 8 = chi_orb", r4(1) == 8)
check("N_4(5) = 137", N4(5) == 137, "N4(5)=%d" % N4(5))
check("N_4(4) = 89 (neighbour, not special)", N4(4) == 89)
check("N_4(6) = 233 (neighbour, not special)", N4(6) == 233)

# ---------------------------------------------------------------------------
# Fixed points: 16 binary labels w in {0,1}^4 (the half-lattice points 2v).
# Dynamical Gram matrix G_ij(K) = sum_{2<=|n|^2<=K} (-1)^{p(n).(w_i xor w_j)}.
# Eigenvalues are 16 * m_c^dyn(K), m_c = #{n : 2<=|n|^2<=K, n = c mod 2}.
# rank = number of parity classes c with m_c^dyn(K) > 0.
# ---------------------------------------------------------------------------
W = list(itertools.product((0, 1), repeat=4))      # 16 fixed-point labels
CLASSES = list(itertools.product((0, 1), repeat=4))  # 16 parity classes

def occupied_classes(k0, K):
    occ = set()
    for n in PTS:
        s = sum(x * x for x in n)
        if k0 <= s <= K:
            occ.add(tuple(x % 2 for x in n))
    return occ

def rank_by_occupancy(k0, K):
    return len(occupied_classes(k0, K))

def rank_by_matrix(k0, K):
    # build 16x16 Gram and take numpy rank as an INDEPENDENT route
    try:
        import numpy as np
    except Exception:
        return None
    sel = [tuple(x % 2 for x in n) for n in PTS if k0 <= sum(x * x for x in n) <= K]
    G = np.zeros((16, 16))
    for a, wa in enumerate(W):
        for b, wb in enumerate(W):
            u = tuple((wa[t] ^ wb[t]) for t in range(4))
            G[a, b] = sum((-1) ** (sum(p[t] * u[t] for t in range(4)) % 2) for p in sel)
    return int(np.linalg.matrix_rank(G, tol=1e-6))

# ---------------------------------------------------------------------------
# E - the rank ladder for the dynamical sector k_0 = 2. Saturation first at K=5.
# ---------------------------------------------------------------------------
expected = {2: 6, 3: 10, 4: 12, 5: 16, 6: 16}
for K in range(2, 7):
    ro = rank_by_occupancy(2, K)
    check("rank G^dyn(K=%d) = %d (occupancy)" % (K, expected[K]), ro == expected[K],
          "got %d" % ro)
    rm = rank_by_matrix(2, K)
    if rm is not None:
        check("rank G^dyn(K=%d) matrix == occupancy" % K, rm == ro,
              "matrix=%s occ=%d" % (rm, ro))

kstar = min(K for K in range(2, 7) if rank_by_occupancy(2, K) == 16)
check("K* = 5 (first saturation, dynamical sector)", kstar == 5, "K*=%d" % kstar)
check("rank jumps 12 -> 16 at K=5 (not a knife-edge)",
      rank_by_occupancy(2, 4) == 12 and rank_by_occupancy(2, 5) == 16)

# weight-1 parities are the bottleneck: |n|^2=1 excluded, next is 1+4=5
w1 = (1, 0, 0, 0)
mins = min(sum(x * x for x in n) for n in PTS
           if tuple(x % 2 for x in n) == w1 and sum(x * x for x in n) >= 2)
check("weight-1 min dynamical shell = 5 (the bottleneck)", mins == 5, "min=%d" % mins)

# ---------------------------------------------------------------------------
# E/T - the k_0 = 2 boundary is forced by topological factorisation, not chosen.
# Only k_0 = 2 makes sum_{k=k0}^{K*geo} r_4(k) = |Fix| * chi_orb = 128.
# ---------------------------------------------------------------------------
def kstar_geo(k0):
    return min(K for K in range(k0, 8) if rank_by_occupancy(k0, K) == 16)

CHI, FIX = 8, 16
for k0 in (1, 2, 3):
    Kg = kstar_geo(k0)
    shellsum = sum(r4(k) for k in range(k0, Kg + 1))
    factors = (shellsum == FIX * CHI)
    check("k_0=%d: K*geo=%d, shellsum=%d, factors-to-128? %s"
          % (k0, Kg, shellsum, factors),
          (k0 == 2) == factors,
          "%d/16 = %.2f" % (shellsum, shellsum / 16))
check("only k_0=2 gives 128 = |Fix|*chi_orb",
      sum(r4(k) for k in range(2, 6)) == 128
      and sum(r4(k) for k in range(1, 5)) != 128
      and sum(r4(k) for k in range(3, 7)) != 128)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in checks if ok)
print("Test 02 - saturation cutoff K* = 5")
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
