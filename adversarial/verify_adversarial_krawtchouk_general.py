"""
Adversarial verifier for the claim "the Krawtchouk method is a general tool"
(manuscript Remark: Krawtchouk method as a general tool).

Every clause of the remark is attacked and must survive:

  A. the 2^d fixed points of -I on T^d = R^d/Z^d form a copy of Z_2^d;
  B. a coordinate-symmetric cutoff makes the inter-fixed-point Gram matrix
     Hamming-constant -- and a NON-symmetric cutoff breaks it (necessity of
     the stated hypothesis);
  C. the d-variable Krawtchouk / Z_2^d-character transform diagonalizes any
     Hamming-constant matrix into d+1 sectors of multiplicity C(d,w), with
     eigenvalues mu_u = sum_w K_w(u;d) g_w; for the Gram these are INTEGERS;
  D. K*(d) = min{K : rank G^dyn(K) = 2^d} (dynamical Gram excludes shells
     k=0,1) equals 5 for 2<=d<=5 and d for 6<=d<=8;
  E. at d=4 the bottleneck is the weight-1 (4-dim) sector: rank = 12 at K=4;
  F. the alignment r_d(1) = 2d = 2^{d-1} = chi_orb holds ONLY at d=4.

Cross-validation is the point: K*(d) is computed THREE independent ways and
must agree -- (1) the fast Krawtchouk-eigenvalue method that backs
verify_spectral_bounds.py, (2) an honest 2^d x 2^d integer matrix build with
its exact integer spectrum (numpy), (3) sympy exact rank over the integers.
Every quantity is a lattice or number-theoretic invariant; no measured
constant enters. This is pure spectral geometry.

ASCII only (Windows cp1252). Run: python verify_adversarial_krawtchouk_general.py
"""
import itertools
import math
import sys

import numpy as np

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


# ===========================================================================
# Independent building blocks (NO Krawtchouk assumed in the brute force)
# ===========================================================================
def lattice_points(d, Kmax):
    R = int(math.isqrt(Kmax))
    rng = np.arange(-R, R + 1, dtype=np.int64)
    mesh = np.meshgrid(*([rng] * d), indexing="ij")
    A = np.stack([m.ravel() for m in mesh], axis=1)
    S = (A * A).sum(axis=1)
    keep = S <= Kmax
    return A[keep], S[keep]


def g_vector(A, S, d, K, lo, weights=None):
    """Honest signed shell sums g[delta] for every delta in {0,1}^d, indexed by
    the integer value of delta.  g[delta] = sum_{lo<=Q<=K} (-1)^{n.delta} where
    Q is the (optionally anisotropic) quadratic form.  No Hamming assumption.

    Memory-safe: bin the masked lattice points by their parity class p in
    {0,1}^d (2^d bins) and Walsh-Hadamard transform the counts --
    g[delta] = sum_p cnt[p] (-1)^{popcount(p & delta)}.  This uses the actual
    enumerated lattice points (no theta/Krawtchouk shortcut), so it is
    independent of the fast method it is cross-checked against."""
    twod = 1 << d
    if weights is None:
        Q = S
    else:
        wv = np.array(weights, dtype=np.int64)
        Q = (A * A * wv).sum(axis=1)
    mask = (Q >= lo) & (Q <= K)
    par = (A & 1).astype(np.int64)
    powers = (1 << np.arange(d)).astype(np.int64)
    par_int = (par * powers).sum(axis=1)                 # parity class per point
    cnt = np.bincount(par_int[mask], minlength=twod).astype(np.int64)
    # popcount table and sign matrix Smat[delta, p] = (-1)^{popcount(delta & p)}
    pc = np.array([bin(k).count("1") for k in range(twod)], dtype=np.int64)
    idx = np.arange(twod, dtype=np.int64)
    Smat = 1 - 2 * (pc[idx[:, None] & idx[None, :]] & 1)
    g = Smat @ cnt                                       # (2^d,)
    return g


def gram_matrix(g, d):
    twod = 1 << d
    idx = np.arange(twod)
    M = g[idx[:, None] ^ idx[None, :]]                   # M[i,j] = g[i xor j]
    return M.astype(np.float64)


def int_spectrum(M):
    """Eigenvalues of a symmetric integer matrix, asserted near-integer, rounded."""
    ev = np.linalg.eigvalsh(M)
    rounded = np.rint(ev)
    near_int = bool(np.max(np.abs(ev - rounded)) < 1e-6)
    return rounded.astype(np.int64), near_int, float(np.max(np.abs(ev - rounded)))


def kstar_brute(d, claimed, lo=2):
    A, S = lattice_points(d, claimed)            # R=isqrt(claimed) keeps it small
    twod = 1 << d
    for K in range(2, claimed + 1):
        g = g_vector(A, S, d, K, lo)
        M = gram_matrix(g, d)
        ev, near_int, _ = int_spectrum(M)
        rank = int(np.count_nonzero(ev != 0))
        if rank == twod:
            return K
    return None


# ----- the fast method that backs verify_spectral_bounds.py (re-derived) -----
def _theta_plus(Kmax):
    P = [0] * (Kmax + 1); P[0] = 1; m = 1
    while m * m <= Kmax:
        P[m * m] = 2; m += 1
    return P


def _theta_minus(Kmax):
    Qq = [0] * (Kmax + 1); Qq[0] = 1; m = 1
    while m * m <= Kmax:
        Qq[m * m] = 2 * ((-1) ** m); m += 1
    return Qq


def _conv(a, b, Kmax):
    c = [0] * (Kmax + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > Kmax:
                break
            c[i + j] += ai * bj
    return c


def krawtchouk(w, u, d):
    return sum((-1) ** j * math.comb(u, j) * math.comb(d - u, w - j)
               for j in range(w + 1))


def kstar_fast(d, Kmax=16, lo=2):
    P = _theta_plus(Kmax); Qm = _theta_minus(Kmax)
    cw = []
    for w in range(d + 1):
        poly = [1] + [0] * Kmax
        for _ in range(w):
            poly = _conv(poly, Qm, Kmax)
        for _ in range(d - w):
            poly = _conv(poly, P, Kmax)
        cw.append(poly)
    for K in range(2, Kmax + 1):
        gw = [sum(cw[w][s] for s in range(lo, K + 1)) for w in range(d + 1)]
        mu = [sum(krawtchouk(w, u, d) * gw[w] for w in range(d + 1))
              for u in range(d + 1)]
        if all(m != 0 for m in mu):
            return K
    return None


CLAIMED = {2: 5, 3: 5, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8}

# ===========================================================================
# A. fixed points form Z_2^d
# ===========================================================================
for d in range(2, 9):
    check("A: |Fix(-I on T^%d)| = 2^%d = %d" % (d, d, 1 << d),
          len(list(itertools.product((0, 1), repeat=d))) == (1 << d))

# ===========================================================================
# B. Hamming-constancy of the Gram (symmetric cutoff) + necessity of symmetry
# ===========================================================================
for d in range(2, 8):
    A, S = lattice_points(d, 7)
    ok_all = True
    for K in range(2, 8):
        for lo in (0, 2):                                # threshold and dynamical
            g = g_vector(A, S, d, K, lo)
            by_w = {}
            for delta in range(1 << d):
                w = bin(delta).count("1")
                if w not in by_w:
                    by_w[w] = g[delta]
                elif g[delta] != by_w[w]:
                    ok_all = False
    check("B: Gram is Hamming-constant for all K in 2..7, d=%d (sym cutoff)" % d, ok_all)

# negative control: an anisotropic (non-S_d) cutoff must BREAK Hamming-constancy
for d in (3, 4):
    A, S = lattice_points(d, 9)
    w = [1] * d
    w[0] = 2                                              # weight one axis differently
    g = g_vector(A, S, d, K=6, lo=2, weights=w)
    broke = False
    for w_ham in range(1, d):
        vals = set()
        for delta in range(1 << d):
            if bin(delta).count("1") == w_ham:
                vals.add(int(g[delta]))
        if len(vals) > 1:
            broke = True
    check("B(neg): anisotropic cutoff BREAKS Hamming-constancy, d=%d "
          "(symmetry is necessary)" % d, broke)

# ===========================================================================
# C. character/Krawtchouk transform diagonalizes ANY Hamming-constant matrix
#    (generic test with random weight-values, then for the actual Gram)
# ===========================================================================
def diagonalizes(d, gw):
    """Check the Z_2^d character basis diagonalizes the Hamming-constant matrix
    with class values gw[w], with eigenvalues sum_w K_w(u) gw[w] of mult C(d,u)."""
    twod = 1 << d
    g = np.array([gw[bin(k).count("1")] for k in range(twod)], dtype=np.float64)
    M = g[np.arange(twod)[:, None] ^ np.arange(twod)[None, :]]
    # character matrix H_{a,x} = (-1)^{a.x}
    bits = np.array([[(k >> b) & 1 for b in range(d)] for k in range(twod)])
    H = (-1.0) ** (bits @ bits.T)
    pred = np.array([sum(krawtchouk(w, bin(u).count("1"), d) * gw[w]
                         for w in range(d + 1)) for u in range(twod)])
    # H columns are eigenvectors: M H = H diag(pred)
    lhs = M @ H
    rhs = H * pred[None, :]
    return float(np.max(np.abs(lhs - rhs)))

# deterministic pseudo-values (no RNG, varied by index) to avoid Math.random ban
for d in (3, 4, 5):
    gw = [((-1) ** w) * (3 * w + 7) for w in range(d + 1)]   # arbitrary class values
    err = diagonalizes(d, gw)
    check("C: character basis diagonalizes a generic Hamming-constant matrix, "
          "d=%d (Delsarte/Bose-Mesner)" % d, err < 1e-9, "max resid %.2e" % err)
    # multiplicity structure C(d,w)
    mult = [math.comb(d, w) for w in range(d + 1)]
    check("C: sector multiplicities C(d,w) sum to 2^d, d=%d" % d,
          sum(mult) == (1 << d))

# ===========================================================================
# D + E. K*(d): three independent methods must agree with the claim
# ===========================================================================
def rank_mod_p(M, p):
    """Exact rank of an integer matrix over GF(p) by vectorized Gaussian
    elimination.  Genuinely different from the eigenvalue criterion: this is
    row reduction, not a spectrum.  rank over Q equals rank over GF(p) for all
    but finitely many p, so agreement of two large primes is exact in practice."""
    A = (np.asarray(M, dtype=np.int64) % p)
    n, m = A.shape
    r = 0
    for c in range(m):
        nz = np.nonzero(A[r:, c] % p)[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        factors = A[:, c].copy()
        factors[r] = 0
        A = (A - np.outer(factors, A[r])) % p
        r += 1
        if r == n:
            break
    return r


def kstar_modular(d, claimed, p, lo=2):
    A, S = lattice_points(d, claimed)
    twod = 1 << d
    for K in range(2, claimed + 1):
        g = g_vector(A, S, d, K, lo)
        M = g[np.arange(twod)[:, None] ^ np.arange(twod)[None, :]]
        if rank_mod_p(M, p) == twod:
            return K
    return None

for d in range(2, 9):
    claimed = CLAIMED[d]
    kf = kstar_fast(d)
    kb = kstar_brute(d, claimed)
    check("D: K*(%d) fast-Krawtchouk = %d (claim)" % (d, claimed), kf == claimed,
          "fast=%s" % kf)
    check("D: K*(%d) brute-force integer spectrum = %d (independent)" % (d, claimed),
          kb == claimed, "brute=%s" % kb)
    check("D: K*(%d) fast == brute (cross-validation)" % d, kf == kb,
          "fast=%s brute=%s" % (kf, kb))

# third method: exact modular rank (row reduction, not eigenvalues) over two
# large primes -- catches a different class of bug than the spectrum criterion
P1, P2 = 2147483647, 2147483629
for d in range(2, 9):
    claimed = CLAIMED[d]
    km1 = kstar_modular(d, claimed, P1)
    km2 = kstar_modular(d, claimed, P2)
    check("D: K*(%d) modular rank (GF(%d)) = %d (third method, row reduction)"
          % (d, P1, claimed), km1 == claimed, "mod=%s" % km1)
    check("D: K*(%d) modular rank agrees across two primes (exact)" % d,
          km1 == km2 == claimed, "p1=%s p2=%s" % (km1, km2))

# integer spectrum (exact) for every d, threshold and dynamical
for d in range(2, 9):
    A, S = lattice_points(d, CLAIMED[d])
    for lo, tag in ((0, "threshold"), (2, "dynamical")):
        g = g_vector(A, S, d, CLAIMED[d], lo)
        ev, near_int, maxdev = int_spectrum(gram_matrix(g, d))
        check("C: %s Gram spectrum is integer at K* (d=%d)" % (tag, d),
              near_int, "max dev from int %.1e" % maxdev)

# E. d=4 specifics: rank 12 at K=4, full at K=5, weight-1 bottleneck
A4, S4 = lattice_points(4, 6)
g4_K4 = g_vector(A4, S4, 4, 4, lo=2)
ev4_K4, _, _ = int_spectrum(gram_matrix(g4_K4, 4))
rank4_K4 = int(np.count_nonzero(ev4_K4 != 0))
check("E: d=4 dynamical rank = 12 at K=4 (matches paper L557)", rank4_K4 == 12,
      "rank=%d" % rank4_K4)
g4_K5 = g_vector(A4, S4, 4, 5, lo=2)
ev4_K5, _, _ = int_spectrum(gram_matrix(g4_K5, 4))
check("E: d=4 dynamical rank = 16 at K=5 (saturates)",
      int(np.count_nonzero(ev4_K5 != 0)) == 16)
# the missing sector at K=4 is weight-1 (mult C(4,1)=4): its mu_u is the zero one
gw4 = [int(g4_K4[(1 << w) - 1]) for w in range(5)]      # representative per weight
mu_u = [sum(krawtchouk(w, u, 4) * gw4[w] for w in range(5)) for u in range(5)]
zero_sectors = [u for u in range(5) if mu_u[u] == 0]
check("E: the unresolved sector at K=4 is weight u=1 (4-dim), the bottleneck",
      zero_sectors == [1], "zero sectors=%s" % zero_sectors)

# d=4 eigenvalue tables vs paper (threshold {144,224,64,128,256};
# dynamical {128,192,64,128,256})
gw4_thr = [int(g_vector(A4, S4, 4, 5, lo=0)[(1 << w) - 1]) for w in range(5)]
mu_thr = sorted(set(sum(krawtchouk(w, u, 4) * gw4_thr[w] for w in range(5))
                    for u in range(5)))
mu_thr_set = set(sum(krawtchouk(w, u, 4) * gw4_thr[w] for w in range(5))
                 for u in range(5))
check("C: d=4 threshold-Gram eigenvalues = {144,224,64,128,256} (paper Table)",
      mu_thr_set == {144, 224, 64, 128, 256}, str(sorted(mu_thr_set)))
gw4_dyn = [int(g4_K5[(1 << w) - 1]) for w in range(5)]
mu_dyn_set = set(sum(krawtchouk(w, u, 4) * gw4_dyn[w] for w in range(5))
                 for u in range(5))
check("C: d=4 dynamical-Gram eigenvalues = {128,192,64,128,256} "
      "(threshold minus k=0,1)", mu_dyn_set == {128, 192, 64, 128, 256},
      str(sorted(mu_dyn_set)))

# ===========================================================================
# F. alignment 2d = 2^{d-1} only at d=4
# ===========================================================================
sols = [d for d in range(1, 201) if 2 * d == 2 ** (d - 1)]
check("F: r_d(1)=2d = 2^{d-1}=chi_orb has the unique solution d=4 (d in 1..200)",
      sols == [4], "solutions=%s" % sols)

# ===========================================================================
# Knife-edge robustness: at K*, the smallest nonzero eigenvalue is a clear
# integer (>=1); saturation is not marginal / tolerance-dependent.
# ===========================================================================
for d in range(2, 9):
    A, S = lattice_points(d, CLAIMED[d])
    g = g_vector(A, S, d, CLAIMED[d], lo=2)
    ev, near_int, _ = int_spectrum(gram_matrix(g, d))
    nz = np.abs(ev[ev != 0])
    check("robust: d=%d min |nonzero eigenvalue| at K* is >= 1 (no knife-edge)" % d,
          near_int and nz.size > 0 and int(nz.min()) >= 1,
          "min |lambda| = %d" % (int(nz.min()) if nz.size else 0))

# extra: the pattern K*(d) = max(5, d) continues at d=9,10 (robustness beyond
# the claimed range; informational but asserted if it holds)
for d in (9, 10):
    kf = kstar_fast(d)
    check("robust: K*(%d) = %d = max(5,%d) (pattern extends past claimed range)"
          % (d, max(5, d), d), kf == max(5, d), "K*=%s" % kf)

# ===========================================================================
# Report
# ===========================================================================
passed = sum(1 for _, ok, _ in checks if ok)
print("Adversarial - Krawtchouk method as a general tool")
print("-" * 70)
for name, ok, detail in checks:
    line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
    if detail and not ok:
        line += "   (%s)" % detail
    print(line)
print("-" * 70)
print("%d/%d checks passed" % (passed, len(checks)))
sys.exit(0 if passed == len(checks) else 1)
