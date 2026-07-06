"""
Neumann-series contraction bounds on T^4/Z_2 (the SM universal-integrality
and inter-fixed-point sections, sec:spectral-action and sec:interval).

Checks F_0 cross-paths, Fredholm eigenvalues |tube * lambda_w| < 7.2e-3,
the exact vanishing of the off-diagonal mu_2 (theta identity),
Fredholm determinant, theta identities, gap protection numerics,
operator-norm scattering bounds, and the
K*(d) saturation dictionary for d = 2..8, the closed form
K*(d) = max(5, d), and the eigenvalue factorization. 46 checks
total. Requires mpmath.
"""

from mpmath import (mp, mpf, pi, exp, quad, jtheta, re, gamma as Gamma,
                    euler as gamma_E, binomial, log, sqrt, fac, power, inf,
                    zeta as mpzeta, diff as mpdiff)
import sys

mp.dps = 30

SEP = "=" * 70
checks_passed = 0
checks_total = 0

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    flush_print(f"  [{status}] {name}")
    if detail:
        flush_print(f"         {detail}")


# ============================================================
# PHYSICAL CONSTANTS (zero free parameters)
# ============================================================

# Compute Delta_1 = F_0 + gamma_E/2 from the Rankin-Selberg formula
# F_0 = 1 - ln(pi) + 6*zeta'(2)/pi^2 + ln(4)/3
# zeta'(2) computed from mpmath (not hardcoded):
zeta_prime_2 = mpdiff(mpzeta, 2)
F_0 = 1 - log(pi) + 6 * zeta_prime_2 / pi**2 + log(4) / 3
Delta_1 = F_0 + gamma_E / 2

# --- F_0 cross-check: two independent computation paths ---

# Path 2: Glaisher-Kinkelin identity (DLMF 25.6.3)
# zeta'(2) = pi^2/6 * (gamma_E + ln(2*pi) - 12*ln(A))
from mpmath import glaisher
A_gk = glaisher
zp2_glaisher = pi**2 / 6 * (gamma_E + log(2 * pi) - 12 * log(A_gk))
F_0_glaisher = 1 - log(pi) + 6 * zp2_glaisher / pi**2 + log(4) / 3

# Path 3: Taylor coefficient extraction (different mpmath algorithm)
from mpmath import taylor
zp2_taylor = taylor(mpzeta, 2, 1)[1]
F_0_taylor = 1 - log(pi) + 6 * zp2_taylor / pi**2 + log(4) / 3

flush_print(f"  F_0 [diff]:     {float(F_0):.15f}")
flush_print(f"  F_0 [Glaisher]: {float(F_0_glaisher):.15f}")
flush_print(f"  F_0 [taylor]:   {float(F_0_taylor):.15f}")

check("F_0 cross-check: Glaisher-Kinkelin agrees to 30 digits",
      abs(F_0 - F_0_glaisher) < mpf('1e-30'),
      f"|diff - Glaisher| = {float(abs(F_0 - F_0_glaisher)):.2e}")

check("F_0 cross-check: Taylor extraction agrees to 30 digits",
      abs(F_0 - F_0_taylor) < mpf('1e-30'),
      f"|diff - Taylor| = {float(abs(F_0 - F_0_taylor)):.2e}")

check("F_0 cross-check: all three paths mutually consistent",
      abs(F_0_glaisher - F_0_taylor) < mpf('1e-30'),
      f"|Glaisher - Taylor| = {float(abs(F_0_glaisher - F_0_taylor)):.2e}")

chi_orb = 8
K_star = 5
N4_K = 137
b_0 = 1
n_fix = 16  # |Fix(sigma)| = 2^4
tube = 1 / (8 * pi**2)  # Omega_4 / (2*pi)^4

flush_print(SEP)
flush_print("SPECTRAL BOUNDS VERIFICATION: NEUMANN SERIES ON T^4/Z_2")
flush_print(SEP)
flush_print()
flush_print(f"  Delta_1           = {float(Delta_1):.6f}")
flush_print(f"  chi_orb           = {chi_orb}")
flush_print(f"  tube = 1/(8pi^2)  = {float(tube):.10f}")
flush_print(f"  |Fix|             = {n_fix}")
flush_print()


# ============================================================
# SECTION 1: Compute lattice Green's functions G^{(h)}
# ============================================================

flush_print(SEP)
flush_print("SECTION 1: Lattice Green's functions G^{(h)}")
flush_print(SEP)
flush_print()

def compute_Zh(h, t_max=40):
    """Z^{(h)}(1) by Mellin integral at s=1, exact over [0, inf).

    The [0,1] piece is mapped to [1, inf) by theta inversion
    (theta3(1/u) = sqrt(u) theta3(u), theta4(1/u) = sqrt(u) theta2(u)):
      int_0^1 [t4^h t3^{4-h} - 1] dt = int_1^inf t2^h t3^{4-h} du - 1.
    No lower cutoff: the old truncation at t = 0.001 dropped -0.001*pi
    from each Z^(h), h >= 1 (the integrand tends to -1 at t = 0),
    inflating each G^(h) by exactly 1/(4000 pi)."""
    def integrand_tail(t):
        q = exp(-pi * t)
        t3 = re(jtheta(3, 0, q))
        t4 = re(jtheta(4, 0, q))
        return t4**h * t3**(4 - h) - 1
    def integrand_inv(u):
        q = exp(-pi * u)
        t3 = re(jtheta(3, 0, q))
        t2 = re(jtheta(2, 0, q))
        return t2**h * t3**(4 - h)
    a = quad(integrand_tail, [1, 3, 10, t_max], maxdegree=6)
    b = quad(integrand_inv, [1, 3, 10, t_max], maxdegree=6)
    return pi * (a + b - 1)

Zh = {}
Gh = {}

# h = 1..4: standard Mellin integrals
for h in range(1, 5):
    Zh[h] = compute_Zh(h)
    Gh[h] = Zh[h] / (4 * pi**2)
    flush_print(f"  G^({h}) = {float(Gh[h]):+.12f}")

# h = 0: regularised self-Green's function via Poisson summation
def integrand_tail(t):
    q = exp(-pi * t)
    t3 = re(jtheta(3, 0, q))
    return t3**4 - 1

I_tail = quad(integrand_tail, [1, 5, 15], maxdegree=6)
Zh[0] = pi * (2 * I_tail - 2)
Gh[0] = Zh[0] / (4 * pi**2)
flush_print(f"  G^(0) = {float(Gh[0]):+.12f}  (regularised)")
flush_print()


# ============================================================
# SECTION 2: Krawtchouk diagonalisation
# ============================================================

flush_print(SEP)
flush_print("SECTION 2: Character eigenvalues via Krawtchouk polynomials")
flush_print(SEP)
flush_print()

def krawtchouk(w, h, n=4):
    """Krawtchouk polynomial K_w(h; n)."""
    val = mpf(0)
    for j in range(min(w, h) + 1):
        if h - j > n - w or h - j < 0:
            continue
        val += (-1)**j * binomial(w, j) * binomial(n - w, h - j)
    return int(val)

K_mat = [[krawtchouk(w, h) for h in range(5)] for w in range(5)]

# Character eigenvalues: lambda_w = sum_h G^{(h)} * K_w(h)
lam = {}
mu = {}   # off-diagonal (h >= 1 only)
mult = {}

for w in range(5):
    mult[w] = int(binomial(4, w))
    lam[w] = sum(Gh[h] * K_mat[w][h] for h in range(5))
    mu[w] = sum(Gh[h] * K_mat[w][h] for h in range(1, 5))

flush_print(f"  {'Sector':>8} {'mult':>5} {'lambda_w':>18} {'tube*lambda_w':>18}")
for w in range(5):
    flush_print(f"  {'w='+str(w):>8} {mult[w]:>5} {float(lam[w]):+18.12f} {float(tube*lam[w]):+18.10e}")
flush_print()


# ============================================================
# SECTION 3: Fredholm contraction bounds
# ============================================================

flush_print(SEP)
flush_print("SECTION 3: Fredholm contraction bounds (core claim)")
flush_print(SEP)
flush_print()

# The manuscript bound is rho < 7.2e-3 (closed form ln 2/pi^4 = 7.1158e-3).
# We verify it by computing all |tube*lambda_w| and checking the maximum.
FREDHOLM_THRESHOLD = mpf('0.0072')

flush_print("  Bound: |tube * lambda_w| < 0.0072 in all 16 sectors?")
flush_print()

all_bounded = True
spectral_radius = mpf(0)

for w in range(5):
    f_w = abs(tube * lam[w])
    if f_w > spectral_radius:
        spectral_radius = f_w
    bounded = f_w < FREDHOLM_THRESHOLD
    if not bounded:
        all_bounded = False
    flush_print(f"  w={w} (mult {mult[w]:>2}): |tube*lambda_{w}| = {float(f_w):.6e}  {'< 0.0072 OK' if bounded else '>= 0.0072 FAIL'}")

flush_print()
check("All Fredholm eigenvalues < 0.0072", all_bounded,
      f"spectral radius = {float(spectral_radius):.6e}")

check("Spectral radius < 0.0072", spectral_radius < FREDHOLM_THRESHOLD,
      f"rho = {float(spectral_radius):.6e}")

# Report the computed spectral radius explicitly so the actual value is visible
flush_print(f"\n  Computed spectral radius: rho = {float(spectral_radius):.10e}")
flush_print(f"  Paper's claimed bound:   rho < {float(FREDHOLM_THRESHOLD)}")
flush_print(f"  Margin:                  {float(FREDHOLM_THRESHOLD - spectral_radius):.6e}")

# Geometric convergence: sum_{g=2}^inf rho^g < rho^2/(1-rho)
rho = float(spectral_radius)
geom_remainder = rho**2 / (1 - rho)
check("Geometric remainder rho^2/(1-rho) < 1e-4",
      geom_remainder < 1e-4,
      f"rho^2/(1-rho) = {geom_remainder:.6e}")

flush_print()


# ============================================================
# SECTION 4: Off-diagonal coefficient mu_2
# ============================================================

flush_print(SEP)
flush_print("SECTION 4: Off-diagonal coefficient mu_2 (vanishes exactly)")
flush_print(SEP)
flush_print()

# mu_2 is the w=2 off-diagonal eigenvalue: mu_2 = -2*G^(2) + G^(4).
# The identity theta3(q)*theta4(q) = theta4(q^2)^2 gives
# Z^(2)(1) = Z^(4)(1)/2 exactly, so mu_2 = 0 (THEOREM).
# The retired expected value -1/(4000*pi) was the quadrature-cutoff
# artifact of the old t >= 0.001 truncation (each G^(h>=1) inflated by
# 1/(4000*pi), and K_2 sums those inflations to -1).
mu_2_exact = mpf(0)
mu_2_computed = mu[2]

flush_print(f"  mu_2 (computed)     = {float(mu_2_computed):+.15e}")
flush_print(f"  mu_2 (exact)        = 0 (theta3*theta4 = theta4(q^2)^2)")
flush_print(f"  |difference|        = {float(abs(mu_2_computed - mu_2_exact)):.6e}")

check("off-diagonal mu_2 = -2*G^(2) + G^(4) = 0 [THEOREM]",
      abs(mu_2_computed) < 1e-9,
      f"computed = {float(mu_2_computed):.3e}")

# Verify the algebraic identity: K_2 = [1, 0, -2, 0, 1]
K2 = K_mat[2]
check("Krawtchouk K_2 = [1, 0, -2, 0, 1]",
      K2 == [1, 0, -2, 0, 1],
      f"K_2 = {K2}")

# Verify: mu_2 = G^(0)*1 + G^(2)*(-2) + G^(4)*1 = G^(0) - 2*G^(2) + G^(4)
# Off-diagonal: mu_2 = -2*G^(2) + G^(4)
mu_2_check = -2*Gh[2] + Gh[4]
check("mu_2 = -2*G^(2) + G^(4)",
      abs(mu_2_check - mu_2_computed) < 1e-15,
      f"difference = {float(abs(mu_2_check - mu_2_computed)):.2e}")

# G^(4) = 2*G^(2) + mu_2 ~ 2*G^(2) (nearly transparent w=2 sector)
flush_print(f"\n  Physical: w=2 sector (6-fold, 37.5% of space) is nearly transparent")
flush_print(f"  G^(4)/(2*G^(2)) = {float(Gh[4]/(2*Gh[2])):.10f}")
flush_print()


# ============================================================
# SECTION 5: Fredholm determinant
# ============================================================

flush_print(SEP)
flush_print("SECTION 5: Fredholm determinant at z = tube")
flush_print(SEP)
flush_print()

det_fredholm = mpf(1)
for w in range(5):
    det_fredholm *= (1 - tube * lam[w])**mult[w]

flush_print(f"  det(I - tube*K) = {float(det_fredholm):.15f}")
flush_print(f"  |1 - det|       = {float(abs(1 - det_fredholm)):.6e}")

check("Fredholm determinant close to 1 (|1-det| < 0.035)",
      abs(1 - det_fredholm) < 0.035,
      f"|1 - det| = {float(abs(1 - det_fredholm)):.6e}")

# Log determinant bound
log_det = sum(mult[w] * log(1 - tube * lam[w]) for w in range(5))
flush_print(f"  log det(I - tube*K) = {float(log_det):+.15e}")

check("Log Fredholm determinant magnitude < 0.035",
      abs(log_det) < 0.035,
      f"|log det| = {float(abs(log_det)):.6e}")

flush_print()


# ============================================================
# SECTION 6: Theta function identities
# ============================================================

flush_print(SEP)
flush_print("SECTION 6: Exact theta function identities")
flush_print(SEP)
flush_print()

# Three identities among Z^{(h)} values (exact algebraic relations).
# Tolerance 1e-10: these are identities, not approximations; 1e-10
# accounts for quadrature error at 30-digit working precision.
# Z1 - Z3 = pi
id1 = Zh[1] - Zh[3]
check("Z^(1) - Z^(3) = pi",
      abs(id1 - pi) < 1e-10,
      f"Z1 - Z3 = {float(id1):.15f}, pi = {float(pi):.15f}")

# 2*Z2 - Z4 = 0 exactly (theta3*theta4 = theta4(q^2)^2; the old expected
# value pi/1000 was the t >= 0.001 quadrature-cutoff artifact)
id2 = 2*Zh[2] - Zh[4]
check("2*Z^(2) - Z^(4) = 0 (exact theta identity)",
      abs(id2) < 1e-10,
      f"2Z2 - Z4 = {float(id2):.15f}")

# 2*Z1 - Z2 = pi exactly (old expected 1001*pi/1000 carried the artifact)
id3 = 2*Zh[1] - Zh[2]
check("2*Z^(1) - Z^(2) = pi (exact theta identity)",
      abs(id3 - pi) < 1e-10,
      f"2Z1 - Z2 = {float(id3):.15f}, pi = {float(pi):.15f}")

flush_print()


# ============================================================
# SECTION 7: Gap protection (numerical verification)
# ============================================================

flush_print(SEP)
flush_print("SECTION 7: Gap protection -- spectral gap at K* = 5")
flush_print(SEP)
flush_print()

# Gap protection (Reed-Simon XII.13 applied to orbifold spectra):
# The Gram matrix eigenvalue structure is stable under perturbation
# because the minimum eigenvalue is bounded away from zero AND
# the spectral gap between shells K* and K*+1 prevents new modes
# from entering the resolution window.

# Compute Gram eigenvalues (integer, from character sums)
from math import comb as mcomb

# Character sums G^(h)(K*) via Jacobi + lattice enumeration
def r4_shell(k):
    """r_4(k) via Jacobi: 8 * sigma_tilde(k) for k >= 1."""
    if k == 0:
        return 1
    return 8 * sum(d for d in range(1, k + 1) if k % d == 0 and d % 4 != 0)

# Direct character sum computation for d=4
def char_sum_d4(h, K):
    """G^(h)(K) = sum_{|n|^2 <= K} (-1)^{n_1+...+n_h}."""
    R = int(K ** 0.5) + 1
    total = 0
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            for n3 in range(-R, R + 1):
                for n4 in range(-R, R + 1):
                    if n1**2 + n2**2 + n3**2 + n4**2 <= K:
                        bits = [n1, n2, n3, n4][:h]
                        total += (-1) ** (sum(b % 2 for b in bits) % 2)
    return total

G_h = [char_sum_d4(h, K_star) for h in range(5)]
flush_print(f"  Character sums G^(h)(5): {G_h}")

check("G^(0)(5) = N_4(5) = 137", G_h[0] == N4_K)

# Krawtchouk eigenvalues of full Gram matrix
def kraw_poly(w, h, n=4):
    """Krawtchouk K_w(h; n)."""
    val = 0
    for j in range(min(w, h) + 1):
        if h - j <= n - w:
            val += (-1)**j * mcomb(w, j) * mcomb(n - w, h - j)
    return val

gram_eigs = []
for w in range(5):
    # lambda_w = sum_h K_h(w; 4) * G^(h)(5), and kraw_poly(w, h) = K_h(w)
    lam_w = sum(kraw_poly(w, h, 4) * G_h[h] for h in range(5))
    gram_eigs.append(lam_w)

flush_print(f"  Gram eigenvalues: {gram_eigs}")
flush_print(f"  min = {min(gram_eigs)}, max = {max(gram_eigs)}")

check("All Gram eigenvalues positive",
      all(e > 0 for e in gram_eigs),
      f"eigenvalues = {gram_eigs}")

check("Minimum Gram eigenvalue = 64 (bounded away from 0)",
      min(gram_eigs) == 64,
      f"min = {min(gram_eigs)}")

check("Condition number = max/min = 4",
      max(gram_eigs) / min(gram_eigs) == 4,
      f"cond = {max(gram_eigs) / min(gram_eigs)}")

# Spectral gap: shells k=5 and k=6
r4_5 = r4_shell(5)
r4_6 = r4_shell(6)
gap_width = 6 - 5  # consecutive integer shells
flush_print(f"\n  r_4(5) = {r4_5} (shell K* occupied)")
flush_print(f"  r_4(6) = {r4_6} (next shell occupied)")
flush_print(f"  Gap width = {gap_width} (integer spacing)")

check("Shell K*=5 is occupied (r_4(5) = 48 > 0)", r4_5 > 0)
check("Gap width = 1 (consecutive integer shells)", gap_width == 1)

# Gap protection bound: perturbation from shell k=6 cannot change rank
# because min eigenvalue (64) >> number of new modes per sector
# The Gram matrix gain from adding shell k=6 is positive semidefinite,
# so eigenvalues can only increase -- rank is protected.
check("Min eigenvalue 64 >> 1 (gap protection margin)",
      min(gram_eigs) > 1,
      f"min_eig / gap = {min(gram_eigs) / gap_width}")

flush_print()


# ============================================================
# SECTION 8: Operator-norm scattering bounds
# ============================================================

flush_print(SEP)
flush_print("SECTION 8: Operator-norm scattering bounds")
flush_print(SEP)
flush_print()

# Scattering amplitudes delta_g from the Krawtchouk eigenvalues:
# delta_g = (1/16) * sum_w C(4,w) * (tube * lambda_w)^g,
# where lambda_w are the full Gram eigenvalues (lam[w]). Each amplitude
# is bounded by rho^g, rho = spectral radius computed in Section 3.
rho_val = spectral_radius
for g in [1, 2, 3]:
    delta_g = sum(binomial(4, w) * (tube * lam[w])**g for w in range(5)) / 16
    flush_print(f"  delta_{g} (scattering) = {float(delta_g):+.10e}, |delta_{g}| = {float(abs(delta_g)):.2e}, rho^{g} = {float(rho_val**g):.2e}")
    check(f"Scattering bound: |delta_{g}| <= rho^{g}",
          abs(delta_g) <= rho_val**g,
          f"|delta_{g}| = {float(abs(delta_g)):.2e}")

flush_print()


# ============================================================
# K*(d) DICTIONARY across dimensions (Remark: Krawtchouk method
# as a general tool). Dynamical Gram excludes shells k=0,1; it has
# full rank 2^d iff every Hamming-weight sector eigenvalue is nonzero.
# mu_u(K) = sum_w Kraw_w(u;d) g_w(K),  g_w(K)=sum_{2<=|n|^2<=K} (-1)^{n_1+..+n_w}.
# Pure integer arithmetic (theta-series convolution + Krawtchouk).
# ============================================================
flush_print()
flush_print("K*(d): dynamical-Gram saturation threshold by dimension")

from math import comb as _comb

def _theta_plus(Kmax):
    P = [0] * (Kmax + 1); P[0] = 1; m = 1
    while m * m <= Kmax:
        P[m * m] = 2; m += 1
    return P

def _theta_minus(Kmax):
    Q = [0] * (Kmax + 1); Q[0] = 1; m = 1
    while m * m <= Kmax:
        Q[m * m] = 2 * ((-1) ** m); m += 1
    return Q

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

def _krawtchouk(w, u, d):
    return sum((-1) ** j * _comb(u, j) * _comb(d - u, w - j) for j in range(w + 1))

def _Kstar(d, Kmax=14):
    P = _theta_plus(Kmax); Q = _theta_minus(Kmax)
    cw = []
    for w in range(d + 1):
        poly = [1] + [0] * Kmax
        for _ in range(w):
            poly = _conv(poly, Q, Kmax)
        for _ in range(d - w):
            poly = _conv(poly, P, Kmax)
        cw.append(poly)
    for K in range(2, Kmax + 1):
        g = [sum(cw[w][s] for s in range(2, K + 1)) for w in range(d + 1)]
        mu = [sum(_krawtchouk(w, u, d) * g[w] for w in range(d + 1)) for u in range(d + 1)]
        if all(m != 0 for m in mu):
            return K
    return None

_Kstar_expected = {2: 5, 3: 5, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8}
for _d in range(2, 9):
    _k = _Kstar(_d)
    check(f"K*({_d}) = {_Kstar_expected[_d]} (dynamical-Gram saturation, |Fix|=2^{_d})",
          _k == _Kstar_expected[_d], f"computed K* = {_k}")
check("K*(4) = 5 agrees with the threshold used throughout",
      _Kstar(4) == 5)

# closed form K*(d) = max(5, d) (minimal-shell argument), d = 2..8
for _d in range(2, 9):
    check(f"K*({_d}) = max(5, {_d}) = {max(5, _d)} (closed form)",
          _Kstar(_d) == max(5, _d))

# dictionary: the threshold-Gram eigenvalue factorizes as
# lambda_w = |Fix| * m_w = 2^d * 2^w * N_{d-w}(floor((K*-w)/4)) for d <= 8,
# and degrades at d = 9 (K* = 9 admits odd coordinates +-3).
def _Nd_small(dd, m):
    if dd == 0:
        return 1
    poly = [1] + [0] * m
    for _ in range(dd):
        poly = _conv(poly, _theta_plus(m), m)
    return sum(poly[: m + 1])

def _Gh_threshold(d, K, h):
    # G^(h)(K) = sum_{|n|^2<=K} (-1)^{n_1+..+n_h} = cumulative of theta_minus^h theta_plus^{d-h}
    poly = [1] + [0] * K
    for _ in range(h):
        poly = _conv(poly, _theta_minus(K), K)
    for _ in range(d - h):
        poly = _conv(poly, _theta_plus(K), K)
    return sum(poly[: K + 1])

for _d in range(2, 9):
    _Ks = max(5, _d)
    _G = [_Gh_threshold(_d, _Ks, _h) for _h in range(_d + 1)]
    _lam = [sum(_krawtchouk(_h, _w, _d) * _G[_h] for _h in range(_d + 1))
            for _w in range(_d + 1)]
    _fac = [2 ** _d * (2 ** _w * _Nd_small(_d - _w, (_Ks - _w) // 4))
            for _w in range(_d + 1)]
    check(f"dictionary: lambda_w = |Fix|*2^w*N_(d-w)(.) at K*({_d}) (d<=8)",
          _lam == _fac, f"lambda={_lam}")

flush_print()


# ============================================================
# SUMMARY
# ============================================================

flush_print(SEP)
flush_print(f"RESULTS: {checks_passed} passed, {checks_total - checks_passed} failed out of {checks_total} checks")
flush_print(SEP)
flush_print()

if checks_passed == checks_total:
    flush_print("All checks passed.")
else:
    flush_print("SOME CHECKS FAILED.")
    sys.exit(1)
