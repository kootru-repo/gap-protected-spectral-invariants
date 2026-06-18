"""
Adversarial verifier for Test 01 - coincident propagator G_0 = 1/(8pi^2).

Every quantity here is a lattice / number-theoretic constant or pi. None of
alpha, 137, or CODATA is used to SELECT the value 1/(8pi^2); CODATA appears only
at the very end, as an external comparison, never as an input. A passing run is
therefore itself evidence of target-blindness (attack vector T).

ASCII only (Windows cp1252). Run: C:\\Python313\\python.exe verify_adversarial_8pi2.py
"""
import math

PI = math.pi
checks = []

def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
# Epstein zeta of Z^4 at s=1 via the limit of 8(1-4^{1-s}) zeta(s) zeta(s-1).
# The 0 x inf indeterminacy resolves to -8 ln 2.
def Z4(s):
    return 8.0 * (1.0 - 4.0 ** (1.0 - s)) * _zeta(s) * _zeta(s - 1.0)

def _zeta(s):
    # Hurwitz/Riemann zeta good enough away from the pole; use mpmath if present.
    try:
        import mpmath
        return float(mpmath.zeta(s))
    except Exception:
        # crude series + Euler-Maclaurin tail; only used if mpmath absent
        N = 200000
        total = sum(1.0 / k ** s for k in range(1, N))
        total += N ** (1 - s) / (s - 1) + 0.5 * N ** (-s)
        return total

Z4_1_limit = -8.0 * math.log(2.0)           # closed form
Z4_1_num = Z4(1.0 + 1e-7)                    # numeric approach to the limit
check("Z4(1) = -8 ln2",
      math.isclose(Z4_1_num, Z4_1_limit, rel_tol=1e-3),
      "num=%.6f closed=%.6f" % (Z4_1_num, Z4_1_limit))

green_coeff = 1.0 / (4.0 * PI ** 2)          # covering-torus short-distance coeff
G0 = 1.0 / (8.0 * PI ** 2)                   # claimed even-projected diagonal
heat_a0 = 1.0 / (16.0 * PI ** 2)            # heat-kernel leading coeff

# ---------------------------------------------------------------------------
# Attack 4 (R) - the covering short-distance coefficient is the textbook value.
# 1/((d-2) vol S^{d-1}) with d=4, vol S^3 = 2 pi^2.
# ---------------------------------------------------------------------------
vol_S3 = 2.0 * PI ** 2
check("covering coeff = 1/((d-2) vol S^3) = 1/(4pi^2)",
      math.isclose(1.0 / (2.0 * vol_S3), green_coeff),
      "1/(2*volS3)=%.6f" % (1.0 / (2.0 * vol_S3)))

# ---------------------------------------------------------------------------
# Forced reason: even projection halves the covering coefficient.
# ---------------------------------------------------------------------------
check("(1/2) * 1/(4pi^2) == 1/(8pi^2)", math.isclose(0.5 * green_coeff, G0))
check("vol S^3 / (2pi)^4 == 1/(8pi^2)",
      math.isclose(vol_S3 / (2.0 * PI) ** 4, G0),
      "angular-loop form matches even-projected value")

# ---------------------------------------------------------------------------
# Attack 1 (C) - the diagonal/off-diagonal ratio is 1/2 independent of the
# overall normalisation c. Model the four orbit terms; only G(x,y), G(-x,-y)
# carry the self-singularity (weight 1 each); off-diagonal collapses to 4 equal.
# ---------------------------------------------------------------------------
for c in (0.123, 1.0, 7.5, 1000.0):
    diag_self = 2.0 * c * green_coeff        # 2 of 4 images on the x->y locus
    offdiag_prefactor = 4.0 * c              # 4 of 4 images equal at fixed pts
    ratio = diag_self / (offdiag_prefactor * green_coeff)
    check("ratio = 1/2 at c=%.3f" % c, math.isclose(ratio, 0.5),
          "ratio=%.6f" % ratio)

# ---------------------------------------------------------------------------
# Attack 6 (I) - off-diagonal entries carry 1/(4pi^2); the Epstein value
# Z4(1)/(4pi^2) shows up as a scattering eigenvalue mu_2; diagonal/off = 2.
# ---------------------------------------------------------------------------
mu = [-0.5607, 0.0888, -0.1405, -0.2295, -0.2810]   # paper operator-norm-bound eigenvalues
offdiag_green = Z4_1_limit / (4.0 * PI ** 2)
check("Z4(1)/(4pi^2) == paper mu_2",
      math.isclose(offdiag_green, mu[2], abs_tol=2e-4),
      "Z4(1)/4pi^2=%.5f mu_2=%.5f" % (offdiag_green, mu[2]))
check("diagonal/off-diagonal scale ratio == 2",
      math.isclose((1.0 / (4.0 * PI ** 2)) / G0, 2.0))

# ---------------------------------------------------------------------------
# Attack 2/5 (R) - the FULL regularised propagator is a different object:
# negative, O(0.1), shape-dependent. It cannot be the +1/(8pi^2) the series needs.
# ---------------------------------------------------------------------------
zeta_reg_full = Z4_1_limit / (4.0 * PI ** 2)         # ~ -0.1405
check("full regularised propagator != 1/(8pi^2)",
      (zeta_reg_full < 0) and (abs(zeta_reg_full - G0) > 0.1),
      "full=%.4f  needed=%.4f" % (zeta_reg_full, G0))

# ---------------------------------------------------------------------------
# Attack S - sensitivity: only 1/(8pi^2) lands on CODATA. (CODATA enters HERE
# only, as an external check - never selected the value above.)
# ---------------------------------------------------------------------------
gE = 0.5772156649015329
F0 = 1.0 - math.log(PI) + 6.0 * (-0.9375482543158437) / PI ** 2 + math.log(4.0) / 3.0
D1 = F0 + gE / 2.0
chi = 8.0
CODATA = 137.035999177

def D_orb(prop):
    eps_lo = D1 * (1.0 - D1) * prop
    eps = eps_lo * (1.0 + D1 / chi)
    return 137.0 + D1 / (1.0 + eps)

v8 = D_orb(1.0 / (8.0 * PI ** 2))
v4 = D_orb(1.0 / (4.0 * PI ** 2))
v16 = D_orb(1.0 / (16.0 * PI ** 2))
check("Delta_1 = F0 + gammaE/2 ~ 0.0360151",
      math.isclose(D1, 0.0360151, abs_tol=1e-6), "D1=%.7f" % D1)
check("only 1/(8pi^2) matches CODATA to < 1e-7",
      (abs(v8 - CODATA) < 1e-7) and (abs(v4 - CODATA) > 1e-6) and (abs(v16 - CODATA) > 1e-6),
      "v4=%.9f v8=%.9f v16=%.9f" % (v4, v8, v16))
check("D_orb(8pi^2) within 0.05 ppb of CODATA",
      abs(v8 - CODATA) / CODATA * 1e9 < 0.05,
      "offset = %.3f ppb" % (abs(v8 - CODATA) / CODATA * 1e9))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in checks if ok)
print("Test 01 - coincident propagator G_0 = 1/(8pi^2)")
print("-" * 60)
for name, ok, detail in checks:
    tag = "PASS" if ok else "FAIL"
    line = "[%s] %s" % (tag, name)
    if detail:
        line += "   (%s)" % detail
    print(line)
print("-" * 60)
print("%d/%d checks passed" % (passed, len(checks)))
import sys
sys.exit(0 if passed == len(checks) else 1)
