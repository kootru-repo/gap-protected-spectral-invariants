# verify_theta_determination.py -- spectral determination among rank-4
# inversion quotients (SM, Theorem: theta-determination; Section: test-c).
#
# Certifies the finite step of the theorem: a rank-4 lattice with the theta
# series of Z^4 is Z^4. After polarization and the theta transformation
# formula the question is a finite one: every even positive definite
# quaternary Gram matrix of determinant 16 satisfying the Minkowski
# reduction inequalities (diagonal product <= 4 det = 64, Barnes-Cohn,
# Mathematika 23 (1976) 156-158) either is 2*I_4 or has kissing number 12.
# The script enumerates the reduced box in exact integer arithmetic,
#   (i)   113 candidates under the Barnes-Cohn product bound, inside a swept
#         box of 161 (per-entry diagonal cap 16, a strict superset holding
#         all reduced forms with margin);
#   (ii)  the same 161 at per-entry cap 24 (the caps are not load-bearing);
#   (iii) 833 presentations with the ascending-diagonal condition dropped;
#   (iv)  the unique theta survivor through norm 40 is 2*I_4 (and every
#         other candidate already fails at the first coefficient, r(2)=12);
# plus controls: Jacobi r(2k) = 8*sigmatilde(k), equality detection on
# unimodular re-presentations, the non-integral impostors A_2+Z+Z(m) of the
# integrality-boundary remark, and the quotient-multiplicity halving
# m(lambda) = r(lambda)/2. TEETH checks confirm a planted failure is caught.
#
# ASCII only. Run: python verify_theta_determination.py ; exit 0 iff all pass.

from itertools import product
import sys
import numpy as np

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name
          + (("  " + str(detail)) if detail and not cond else ""))

def sigmatilde(k):
    return sum(d for d in range(1, k + 1) if k % d == 0 and d % 4 != 0)

NORM_CAP = 40
TARGET = {2 * k: 8 * sigmatilde(k) for k in range(1, NORM_CAP // 2 + 1)}

# ------------------------------------------------ exact 4x4 symmetric det
def det_exact(G):
    a, e, f, g = G[0]
    _, b, h, i = G[1]
    _, _, c, j = G[2]
    _, _, _, d = G[3]
    return (a*b*c*d - a*b*j*j - a*h*h*d + 2*a*h*i*j - a*i*i*c
            - e*e*c*d + e*e*j*j + 2*e*h*f*d - 2*e*h*g*j - 2*e*i*f*j
            + 2*e*i*g*c - f*f*b*d + f*f*i*i + 2*f*g*b*j - 2*f*g*h*i
            - g*g*b*c + g*g*h*h)

def posdef(G):
    if G[0][0] <= 0:
        return False
    if G[0][0]*G[1][1] - G[0][1]**2 <= 0:
        return False
    M3 = (G[0][0]*G[1][1]*G[2][2] - G[0][0]*G[1][2]**2
          - G[0][1]**2*G[2][2] + 2*G[0][1]*G[1][2]*G[0][2]
          - G[0][2]**2*G[1][1])
    if M3 <= 0:
        return False
    return det_exact(G) > 0

# ------------------------------------------------ representation numbers
def theta_counts(G, maxnorm, extra=0):
    A = np.array(G, dtype=float)
    lam = float(np.linalg.eigvalsh(A)[0])
    if lam <= 1e-9:
        return None
    bound = int((maxnorm / lam) ** 0.5) + 1 + extra
    R = np.arange(-bound, bound + 1)
    X = np.array(np.meshgrid(R, R, R, R, indexing="ij")).reshape(4, -1).T
    X = X[np.any(X != 0, axis=1)]
    Gm = np.array(G, dtype=np.int64)
    norms = np.einsum("ij,jk,ik->i", X, Gm, X)
    sel = (norms > 0) & (norms <= maxnorm)
    vals, cnts = np.unique(norms[sel], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

def matches(counts, target):
    if counts is None or min(counts) != 2:
        return False
    for n in range(1, NORM_CAP + 1):
        want = target.get(n, 0) if n % 2 == 0 else 0
        if counts.get(n, 0) != want:
            return False
    return True

# ------------------------------------------------ reduced-box enumeration
def sweep(diag_cap, ordered=True):
    out = []
    if ordered:
        diags = [(2, b, c, d)
                 for b in range(2, diag_cap + 1, 2)
                 for c in range(b, diag_cap + 1, 2)
                 for d in range(c, diag_cap + 1, 2)]
    else:
        diags = [t for t in product(range(2, diag_cap + 1, 2), repeat=4)
                 if min(t) == 2]
    for dg in diags:
        a, b, c, d = dg
        rng = lambda i, j: range(-(min(dg[i], dg[j]) // 2),
                                 min(dg[i], dg[j]) // 2 + 1)
        for g12, g13, g14, g23, g24, g34 in product(
                rng(0, 1), rng(0, 2), rng(0, 3),
                rng(1, 2), rng(1, 3), rng(2, 3)):
            G = ((a, g12, g13, g14), (g12, b, g23, g24),
                 (g13, g23, c, g34), (g14, g24, g34, d))
            if det_exact(G) == 16 and posdef(G):
                out.append(G)
    return out

def min_vector_reps(G):
    A = np.array(G, dtype=float)
    lam = float(np.linalg.eigvalsh(A)[0])
    bound = int((2.0 / lam) ** 0.5) + 2
    R = np.arange(-bound, bound + 1)
    X = np.array(np.meshgrid(R, R, R, R, indexing="ij")).reshape(4, -1).T
    X = X[np.any(X != 0, axis=1)]
    Gm = np.array(G, dtype=np.int64)
    norms = np.einsum("ij,jk,ik->i", X, Gm, X)
    V = X[norms == 2]
    reps = []
    for v in V:
        if v[np.nonzero(v)[0][0]] > 0:
            reps.append(v)
    return np.array(reps, dtype=np.int64), int(len(V))

def is_2I4(G):
    reps, kiss = min_vector_reps(G)
    if kiss != 8 or len(reps) != 4:
        return False
    Gm = np.array(G, dtype=np.int64)
    if not np.array_equal(reps @ Gm @ reps.T, 2 * np.eye(4, dtype=np.int64)):
        return False
    return abs(int(round(float(np.linalg.det(reps.astype(float)))))) == 1

G0 = ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2))

# ------------------------------------------------ (i)-(ii) the reduced box
cands16 = sweep(16, ordered=True)
check("C1 swept box, diagonal cap 16: 161 candidates", len(cands16) == 161,
      len(cands16))
prod64 = [G for G in cands16
          if G[0][0] * G[1][1] * G[2][2] * G[3][3] <= 64]
check("C1b Barnes-Cohn product bound (<= 4 det = 64): 113 candidates",
      len(prod64) == 113, len(prod64))
cands24 = sweep(24, ordered=True)
check("C2 diagonal cap 24: same candidate set",
      len(cands24) == 161 and set(cands24) == set(cands16), len(cands24))

# ------------------------------------------------ (iv) unique survivor
surv = [G for G in cands16 if matches(theta_counts(G, NORM_CAP), TARGET)]
check("C3 unique theta survivor through norm 40", len(surv) == 1, len(surv))
check("C4 survivor is 2*I_4 (min vectors: Gram 2I, coordinate det +-1)",
      len(surv) == 1 and is_2I4(surv[0]))
kiss_others = [min_vector_reps(G)[1] for G in cands16 if G not in surv]
check("C5 every other candidate has kissing number 12 (fails at r(2))",
      all(k == 12 for k in kiss_others))

# ------------------------------------------------ (iii) ordering dropped
cands12u = sweep(12, ordered=False)
surv_u = [G for G in cands12u if matches(theta_counts(G, NORM_CAP), TARGET)]
check("C6 unordered sweep: 833 presentations", len(cands12u) == 833,
      len(cands12u))
check("C7 unordered sweep: every survivor is a 2*I_4 presentation",
      len(surv_u) >= 1 and all(is_2I4(G) for G in surv_u), len(surv_u))

# ------------------------------------------------ controls
c0 = theta_counts(G0, NORM_CAP)
check("C8 Jacobi: enumerated r_{2I4}(2k) = 8*sigmatilde(k), k = 1..20",
      all(c0.get(2 * k, 0) == 8 * sigmatilde(k) for k in range(1, 21))
      and all(c0.get(n, 0) == 0 for n in range(1, NORM_CAP + 1, 2)))

rs = np.random.RandomState(20260705)
ok_eq, done = True, 0
while done < 3:
    U = np.eye(4, dtype=np.int64)
    for _ in range(12):
        i, j = rs.randint(0, 4, size=2)
        if i != j:
            U[i] += int(rs.randint(-2, 3)) * U[j]
    if abs(int(round(float(np.linalg.det(U.astype(float)))))) != 1:
        continue
    Gt = tuple(map(tuple, (U @ np.array(G0, dtype=np.int64) @ U.T).tolist()))
    if theta_counts(Gt, NORM_CAP) != c0:
        ok_eq = False
    done += 1
check("C9 equality detector: theta(U^T 2I_4 U) = theta(2I_4), 3 unimodular U",
      ok_eq)

ok_imp = True
for m in (2, 3, 4, 5):
    G = ((2, 1, 0, 0), (1, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2 * m))
    cc = theta_counts(G, NORM_CAP)
    if cc.get(2, 0) != 8 or matches(cc, TARGET):
        ok_imp = False
check("C10 impostors A_2+Z+Z(m), m=2..5: first shell matches, full theta fails",
      ok_imp)

def quotient_mults(Gd, maxnorm):
    lam = float(np.linalg.eigvalsh(Gd)[0])
    bound = int((maxnorm / lam) ** 0.5) + 2
    R = np.arange(-bound, bound + 1)
    X = np.array(np.meshgrid(R, R, R, R, indexing="ij")).reshape(4, -1).T
    X = X[np.any(X != 0, axis=1)]
    out = {}
    for v in X:
        if v[np.nonzero(v)[0][0]] < 0:
            continue
        n = round(float(v @ Gd @ v), 6)
        if 0 < n <= maxnorm:
            out[n] = out.get(n, 0) + 1
    return out

q = quotient_mults(np.eye(4), 10)
ok_dual = all(q.get(float(k), 0) == 4 * sigmatilde(k) for k in range(1, 11))
Gd = np.linalg.inv(np.diag([1.0, 1.0, 1.0, 2.0]))
qm = quotient_mults(Gd, 6)
lam = float(np.linalg.eigvalsh(Gd)[0])
bound = int((6 / lam) ** 0.5) + 2
R = np.arange(-bound, bound + 1)
X = np.array(np.meshgrid(R, R, R, R, indexing="ij")).reshape(4, -1).T
X = X[np.any(X != 0, axis=1)]
full = {}
for v in X:
    n = round(float(v @ Gd @ v), 6)
    if 0 < n <= 6:
        full[n] = full.get(n, 0) + 1
for n, cnt in full.items():
    if qm.get(n, 0) * 2 != cnt:
        ok_dual = False
check("C11 quotient multiplicity halving m(lambda) = r(lambda)/2", ok_dual)

# ------------------------------------------------ manifold exclusion
# Heat-trace identity behind the manifold-exclusion remark: the quotient
# trace Q(t) = 1 + sum_k (r(k)/2) e^{-tk} equals (1/2) theta_{Z^4} + 1/2
# exactly, and as t -> 0 the torus trace approaches the pure power
# (pi/t)^2 with NO constant term while Q(t) - (1/2)(pi/t)^2 -> 1/2.
import math as _math

def _theta_z4(t, B):
    s = 0.0
    for x in product(range(-B, B + 1), repeat=4):
        s += _math.exp(-t * sum(v * v for v in x))
    return s

ok_heat = True
for t in (0.5, 1.0):
    B = int((60.0 / t) ** 0.5) + 2
    K = 4 * B * B
    th = sum((8 * sigmatilde(k) if k else 1) * _math.exp(-t * k)
             for k in range(0, K + 1))
    q = 1.0 + sum(8 * sigmatilde(k) / 2.0 * _math.exp(-t * k)
                  for k in range(1, K + 1))
    if abs(q - 0.5 * th - 0.5) > 1e-12:
        ok_heat = False
for t in (0.3, 0.2):
    B = int((80.0 / t) ** 0.5) + 3
    th = _theta_z4(t, B)
    K = int(80.0 / t) + 5
    q = 1.0 + sum(8 * sigmatilde(k) / 2.0 * _math.exp(-t * k)
                  for k in range(1, K + 1))
    if abs(th - (_math.pi / t) ** 2) > 1e-3:           # torus: no constant
        ok_heat = False
    if abs(q - 0.5 * (_math.pi / t) ** 2 - 0.5) > 1e-3:  # orbifold: +1/2
        ok_heat = False
check("C12 heat traces: quotient = theta/2 + 1/2 exactly; torus constant 0, "
      "orbifold constant 1/2", ok_heat)

# ------------------------------------------------ TEETH
GA3 = ((2, -1, -1, 0), (-1, 2, 0, 0), (-1, 0, 2, 0), (0, 0, 0, 4))
check("T1 TEETH: A_3+<4> (kissing 12) is rejected by the matcher",
      not matches(theta_counts(GA3, NORM_CAP), TARGET)
      and det_exact(GA3) == 16)
bad = dict(TARGET)
bad[4] = TARGET[4] + 8
check("T2 TEETH: corrupted target rejects 2*I_4 itself",
      not matches(c0, bad))

print()
print("SUMMARY: %d pass, %d fail" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("RESULT: ALL CHECKS PASS (15/15)")
sys.exit(0)
