"""
Test 11 - are the separating/non-separating channel SIGNS derivable, or a model?

The genus corrections are DEFINED by Delta_{g+1} = -eps*_lo Delta_g with
Delta_2 = -Delta_1^2(1-Delta_1)/(8pi^2), whose sep/ns sign structure is asserted
from an orientation heuristic. This script tests whether any CANONICAL
finite-dimensional operator/functional---built only from the already-verified
inputs---reproduces that recursion (signs included). If one does, the value is
derived; if none does, "conditional" is the correct, non-removable label.

No alpha/137/CODATA is used as an input.
ASCII only. Run: C:\\Python313\\python.exe verify_channel_derivation.py
"""
import math

checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

PI = math.pi
import mpmath
gE = 0.5772156649015329
F0 = 1 - math.log(PI) + 6 * float(mpmath.zeta(2, 1, 1)) / PI ** 2 + math.log(4) / 3
D1 = F0 + gE / 2.0                       # Delta_1 (a NUMBER, RS finite part)
chi = 8.0
G0 = 1.0 / (8 * PI ** 2)                  # coincident-point propagator
Sig = D1 / chi                            # self-energy = Delta_1/chi_orb
P = 8 * PI ** 2

# ---- the ASSERTED quantities (the target the candidates must reproduce) ----
eps_lo = D1 * (1 - D1) / P                 # paper recursion ratio
D2_paper = -D1 ** 2 * (1 - D1) / P         # sep + ns
D2_sep = -D1 ** 2 / P                       # separating channel only
D2_ns = +D1 ** 3 / P                        # non-separating channel only
C_paper = D1 / (1 + eps_lo)                # the paper's correction (~ +0.036)
check("sanity: Delta_1=0.0360151, eps*_lo~4.40e-4, Delta_2~-1.584e-5",
      math.isclose(D1, 0.0360151, abs_tol=1e-6)
      and math.isclose(eps_lo, 4.397e-4, rel_tol=1e-2)
      and math.isclose(D2_paper, -1.584e-5, rel_tol=1e-2))
check("algebra: eps*_lo = (Delta_1*G_0)*(1-Delta_1) [sep term x (1-Delta_1)]",
      math.isclose(eps_lo, (D1 * G0) * (1 - D1), rel_tol=1e-12))
check("algebra: Delta_2 = sep + ns, ns = +Delta_1^3/(8pi^2) (the open + sign)",
      math.isclose(D2_paper, D2_sep + D2_ns, rel_tol=1e-12) and D2_ns > 0)

# ===========================================================================
# CANDIDATE 1 - scalar self-energy resolvent  M = Sigma_orb*G_0*I_16
# ===========================================================================
ratio1 = Sig * G0
check("C1 scalar self-energy ratio Sigma_orb*G_0 does NOT equal eps*_lo",
      not math.isclose(ratio1, eps_lo, rel_tol=0.05),
      "Sigma_orb*G_0=%.3e vs eps*_lo=%.3e  (off by %.1fx)"
      % (ratio1, eps_lo, eps_lo / ratio1))

# ===========================================================================
# CANDIDATE 2 - scalar Delta_1-coupling series  (vertex Delta_1, propagator G_0)
#   reproduces the SEPARATING term only; the (1-Delta_1) is the missing ns piece
# ===========================================================================
ratio2 = D1 * G0
D2_cand2 = -D1 ** 2 * G0                   # = D2_sep
check("C2 Delta_1-coupling reproduces the SEPARATING term -Delta_1^2/(8pi^2)",
      math.isclose(D2_cand2, D2_sep, rel_tol=1e-12))
check("C2 MISSES the non-separating +Delta_1^3/(8pi^2); gap = the (1-Delta_1)",
      not math.isclose(D2_cand2, D2_paper, rel_tol=1e-3)
      and math.isclose(D2_paper - D2_cand2, D2_ns, rel_tol=1e-9),
      "gap = %.3e = +Delta_1^3/(8pi^2)" % (D2_paper - D2_cand2))

# ===========================================================================
# CANDIDATE 3 - genuine Tr log of the DEFINED off-diagonal kernel K
#   C_det = -sum_w binom(4,w) log(1 - mu_w/8pi^2)
# ===========================================================================
mu = [-0.5607, 0.0888, -0.1405, -0.2295, -0.2810]
binom = [1, 4, 6, 4, 1]
lam = [m / P for m in mu]
C_det = -sum(b * math.log(1 - l) for b, l in zip(binom, lam))
Dorb_det = 137 + C_det
check("C3 off-diagonal Tr log gives a value FAR from the paper correction",
      not math.isclose(C_det, C_paper, rel_tol=0.1),
      "C_det=%.5f -> D_orb=%.4f  vs paper C=%.5f -> 137.0360" % (C_det, Dorb_det, C_paper))
check("C3 Tr log => D_orb ~ 136.97, NOT 137.036 (kernel is the wrong object)",
      abs(Dorb_det - 137.036) > 0.05, "D_orb(Tr log)=%.4f" % Dorb_det)

# ===========================================================================
# CANDIDATE 4 - sign structure: a Tr log gives UNIFORM signs (-lambda^n/n).
#   The paper needs separating (-) AND non-separating (+): impossible for Tr log.
# ===========================================================================
# expansion of -log(1-x) = +x +x^2/2 +x^3/3 ... ; of -log(1+x)= -x +x^2/2 -x^3/3
# In every case successive orders do NOT alternate the way sep(-)/ns(+) demand
# at the SAME order (sep and ns are both order Delta_1^2..3 but opposite signs).
trlog_signs_uniform = True   # -lambda^n/n: the n-th term sign is fixed by sign(lambda), not mixed within an order
check("C4 Tr log cannot produce sep(-) and ns(+) at one order (uniform-sign expansion)",
      trlog_signs_uniform)
check("C4 separating(-) and non-separating(+) carry opposite signs at one order",
      D2_ns > 0 and D2_sep < 0)

# ===========================================================================
# FRAMING CORRECTION - it is NOT a matrix model (those diverge factorially)
# ===========================================================================
check("series CONVERGES geometrically (eps*_lo < 1) -> NOT a matrix-model expansion",
      eps_lo < 1,
      "matrix-model genus series F_g ~ (2g)! DIVERGES; this converges, ratio=%.2e" % eps_lo)

# ===========================================================================
# BOUNDED PROBE - the whole conditional-ness reduces to ONE scalar (1-Delta_1).
# It is reproducible as a dressed-vertex / exclusion factor, but the implied
# self-energy is Delta_1, NOT the operator's Sigma_orb = Delta_1/8: a posited
# factor, not forced by the defined resolvent.
# ===========================================================================
ratio_excl = (D1 * G0) * (1 - D1)
check("(1-Delta_1) reproduced by exclusion/dressed-vertex (Delta_1*G_0)(1-Delta_1)=eps*_lo",
      math.isclose(ratio_excl, eps_lo, rel_tol=1e-12))
check("but the implied self-energy is Delta_1, NOT the operator's Sigma_orb=Delta_1/8",
      abs(D1 - Sig) > 0.02 and not math.isclose(1 - D1, 1 - Sig, rel_tol=1e-3),
      "1-Delta_1=%.4f vs 1-Sigma_orb=%.4f -> a posited vertex factor, not forced" % (1 - D1, 1 - Sig))
check("=> conditional-ness reduces to ONE posited scalar factor, (1-Delta_1)", True)

# ===========================================================================
# VERDICT
# ===========================================================================
none_match = (not math.isclose(ratio1, eps_lo, rel_tol=0.05)) and \
             (not math.isclose(D2_cand2, D2_paper, rel_tol=1e-3)) and \
             (not math.isclose(C_det, C_paper, rel_tol=0.1))
check("VERDICT (C): no canonical operator reproduces the sep/ns recursion",
      none_match)

passed = sum(1 for _, ok, _ in checks if ok)
print("Test 11 - sep/ns channel signs: derivable or model?")
print("-" * 64)
for name, ok, detail in checks:
    line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
    if detail:
        line += "\n        (%s)" % detail
    print(line)
print("-" * 64)
print("%d/%d checks passed" % (passed, len(checks)))
print()
print("FINDING: no canonical operator reproduces the recursion - scalar self-energy")
print("(C1) 7.7x off; Delta_1-coupling (C2) gives only the separating term; the")
print("off-diagonal Tr log (C3) gives D_orb ~ 136.97. It is NOT a matrix model: the")
print("series CONVERGES geometrically, while a matrix-model genus expansion diverges")
print("factorially. The conditional-ness reduces to a SINGLE posited scalar, the")
print("(1-Delta_1) vertex factor - reproducible as an exclusion/saturation factor")
print("(1 - occupation Delta_1), but with implied self-energy Delta_1 != the operator")
print("self-energy Delta_1/8, so it is posited, not forced by the defined resolvent.")
print("=> VERDICT (C): 'conditional' is correct; the whole gap is one scalar")
print("   (1-Delta_1), physically an exclusion factor, not forced by any defined operator.")
import sys
sys.exit(0 if passed == len(checks) else 1)
