"""
Exhaustive substrate sweep on the cubic lattice Z^4 (standard form).

The three-sector decomposition uses |n|^2 = sum n_i^2 (standard Euclidean form),
so any finite group G acting on (Z^4, standard form) by isometries lies in
Aut(Z^4) = O(4,Z) = the hyperoctahedral group W(B_4) of signed permutation
matrices (order 384). Other lattices/forms are excluded analytically by lattice
uniqueness r_Lambda(1)=8 => Lambda=Z^4 (SI prop:bieberbach-classification).

This script enumerates EVERY subgroup of W(B_4) that acts with ISOLATED fixed
points (every non-identity element g has det(I-g) != 0 - i.e. no +1 eigenvalue;
required for isolated conical singularities), and computes the orbifold Euler
characteristic chi_orb = (1/|G|) sum_{g!=I} |det(I-g)|. It verifies that
G = Z_2 = {I, -I} is the UNIQUE subgroup with chi_orb = 8 (the three-sector
value). Subgroups with a +1-eigenvalue element have positive-dimensional fixed
sets (non-isolated singularities) and cannot give the three-sector - the sweep
reports how many subgroups that structural criterion removes.

This is exhaustive over Aut(Z^4)=W(B_4) (the Z^4 substrate); cross-lattice
exhaustiveness is the analytic lattice-uniqueness reduction plus the CARAT
227-class carat_sweep.py. Pure exact integer arithmetic. No external data.
"""
import itertools
from collections import deque
from sympy import Matrix, eye

# ---------------------------------------------------------------------------
# 1. W(B_4): all 4x4 signed permutation matrices (384).
# ---------------------------------------------------------------------------
def signed_perm_matrices():
    mats = []
    for perm in itertools.permutations(range(4)):
        for signs in itertools.product((1, -1), repeat=4):
            M = [[0, 0, 0, 0] for _ in range(4)]
            for i in range(4):
                M[i][perm[i]] = signs[i]
            mats.append(tuple(tuple(r) for r in M))
    return mats

W = signed_perm_matrices()
assert len(W) == 384, len(W)
I4 = tuple(tuple(1 if i == j else 0 for j in range(4)) for i in range(4))

def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4))
                 for i in range(4))

idx = {m: i for i, m in enumerate(W)}
ID = idx[I4]
NEG_I = idx[tuple(tuple(-1 if i == j else 0 for j in range(4)) for i in range(4))]

# multiplication table (index-based) for fast closures
mult = [[idx[matmul(W[i], W[j])] for j in range(384)] for i in range(384)]

# det(I - g), exact, for each element
def det_I_minus(g):
    return int((eye(4) - Matrix(g)).det())
dimg = [det_I_minus(W[i]) for i in range(384)]

# "good" = isolated fixed points: g != I and det(I-g) != 0 (no +1 eigenvalue)
GOOD = frozenset(i for i in range(384) if i != ID and dimg[i] != 0)
print("W(B_4) order:", len(W))
print("elements with isolated fixed points (det(I-g)!=0):", len(GOOD), "of 383 non-identity")
print("det(I-(-I)) =", dimg[NEG_I], "(so chi_orb of {I,-I} = %g)" % (dimg[NEG_I] / 2))
print()

# ---------------------------------------------------------------------------
# 2. Enumerate EVERY subgroup whose non-identity elements are all "good"
#    (fixed-point-free / isolated-fixed-point subgroups). These are the only
#    ones that can give the three-sector; the rest are excluded structurally.
# ---------------------------------------------------------------------------
def closure(seed):
    S = {ID}
    frontier = list(seed)
    S.update(seed)
    while frontier:
        a = frontier.pop()
        for b in list(S):
            for prod in (mult[a][b], mult[b][a]):
                if prod not in S:
                    S.add(prod); frontier.append(prod)
    return frozenset(S)

fpf_subgroups = set()
fpf_subgroups.add(frozenset({ID}))
queue = deque([frozenset({ID})])
while queue:
    S = queue.popleft()
    for g in GOOD:
        if g in S:
            continue
        T = closure(set(S) | {g})
        if T in fpf_subgroups:
            continue
        # keep only if T stays fixed-point-free (all non-identity elements good)
        if all(x == ID or x in GOOD for x in T):
            fpf_subgroups.add(T)
            queue.append(T)

# ---------------------------------------------------------------------------
# 3. chi_orb for each non-trivial fixed-point-free subgroup.
# ---------------------------------------------------------------------------
def chi_orb(S):
    return sum(abs(dimg[x]) for x in S if x != ID) / len(S)

rows = []
for S in fpf_subgroups:
    if len(S) == 1:
        continue
    rows.append((len(S), chi_orb(S), S))
rows.sort(key=lambda r: (-r[1], r[0]))

print("Fixed-point-free (isolated-fixed-point) subgroups of W(B_4):",
      len(fpf_subgroups) - 1, "non-trivial")
print()
print("  |G|   chi_orb   three-sector?   is {I,-I}?")
chi8 = []
for n, chi, S in rows:
    three = "YES" if abs(chi - 8) < 1e-9 else "no"
    iszi = "yes" if S == frozenset({ID, NEG_I}) else "no"
    if abs(chi - 8) < 1e-9:
        chi8.append(S)
    if abs(chi - 8) < 1e-9 or chi > 6.0 or n <= 4:
        print("  %3d   %7.4f   %-13s   %s" % (n, chi, three, iszi))
print("  ... (subgroups with chi_orb <= 6 and |G|>4 omitted from print)")
print()

# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------
max_chi_ge3 = max((chi for n, chi, S in rows if n >= 3), default=0.0)
unique = (len(chi8) == 1 and chi8[0] == frozenset({ID, NEG_I}))
print("RESULT")
print("-" * 60)
print("subgroups with chi_orb = 8 (three-sector value):", len(chi8))
print("the unique one is {I, -I} = Z_2:", unique)
print("max chi_orb over |G| >= 3 fixed-point-free subgroups: %.4f (< 8)" % max_chi_ge3)
print("=> on Z^4, G = Z_2 = {I,-I} is the UNIQUE substrate (exhaustive over W(B_4));")
print("   all |G|>=3 isolated-fixed-point subgroups have chi_orb < 8.")
print("   (Other lattices excluded analytically by r_Lambda(1)=8 => Lambda=Z^4.)")
import sys
sys.exit(0 if unique and max_chi_ge3 < 8 else 1)
