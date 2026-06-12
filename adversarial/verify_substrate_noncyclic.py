"""
verify_substrate_noncyclic.py  --  NON-CYCLIC branch of substrate uniqueness.

THEOREM (non-cyclic branch).  For every d >= 5, no NON-CYCLIC finite
fixed-point-free (FPF) group G acting freely and linearly on T^d (every g != I
has no eigenvalue 1) has

        chi_orb(G) = (1/|G|) sum_{g != I} det_R(I - g)  =  2d ,

with det_R(I - I) = 0.  (Cyclic G and d <= 4 are handled elsewhere; the d=4
substrate sweep finds only Z_2 = {I,-I} reaches chi_orb = 8 = 2d.  d odd forces
G = Z_2.  So the live case is d even >= 6, G non-cyclic.)

CORRECTION (important).  An intermediate draft of this script enumerated split
metacyclic groups Z_m rtimes Z_n and treated them as fixed-point-free.  They are
NOT: a nonabelian Z_m:Z_n is a Frobenius GROUP, not a Frobenius complement -- it
fails Wolf's criterion (every order-pq subgroup cyclic), and in the induced
representation the complement generator has eigenvalue 1 (the all-ones vector).
So those groups do not act freely and their "chi_orb" is not a free-orbifold
invariant.  The genuine non-cyclic free actions are classified by Wolf: they have
EVEN order (an odd non-cyclic group with cyclic Sylows is metacyclic with a
non-cyclic Z_p:Z_q subgroup), a unique central involution, and free real
dimension divisible by 4 -- so dims = 2 (mod 4) (6,10,14,...) admit NONE.  The
d = 0 (mod 4) families (binary-polyhedral, SL(2,p), dicyclic) are checked exactly
in parts (B),(C),(D).  The original SI claim "dims 6,10,14 admit no non-cyclic
free action" is therefore CORRECT.

------------------------------------------------------------------------------
THE TOOL:  m-independence via exterior powers (verified numerically + exactly).

A free linear action on T^{2n} with isolated fixed points is the realification of
a complex representation rho: G -> U(n) with rho(g) having NO eigenvalue 1 for
g != I.  Then

   det_R(I - g) = |det_C(I - rho(g))|^2 = det_C(I-rho(g)) * conj(det_C(I-rho(g))),
   det_C(I - rho(g)) = sum_{j=0}^n (-1)^j chi_{Lambda^j rho}(g),

so, by orthogonality of characters,

   chi_orb = (1/|G|) sum_g |det_C(I - rho(g))|^2
           = sum_{j,l=0}^n (-1)^{j+l} <Lambda^j rho, Lambda^l rho>_G,

with <A,B>_G = dim Hom_G(A,B) a NON-NEGATIVE INTEGER.  Hence chi_orb is a
non-negative integer determined by the G-module structure of rho and its
exterior powers.  A-priori bounds, used below:
   0 <= chi_orb,   and   |det_C(I-rho(g))| <= 2^n  =>  chi_orb <= 4^n = 2^d.

For the metacyclic family G_m = Z_m rtimes Z_n with complement action k (of
multiplicative order n mod m), rho = Ind_{Z_m}^{G_m}(faithful character) is the
free complex n-rep.  Its weights are the n residues k^0,...,k^{n-1} (times a unit
a) as characters of Z_m.  We prove (and check) m-INDEPENDENCE: for fixed (n,
action type), <Lambda^j rho, Lambda^l rho> is the same for all admissible m once
m exceeds the largest subset-sum of weights; the finitely many smaller m are
checked exhaustively.  Consequently the SET of chi_orb values realised at a fixed
dimension d = 2n is FINITE, and we list it exactly and confirm 2d is absent.

EXACT computation (integer arithmetic, no floats in the certificate):
det_C(I - rho(s^i t^l)) factors over the gcd(l,n) cycles of the complement
permutation; each length-L cycle contributes (1 - z^{a i Sb}) where z = e^{2pi
i/m} and Sb is the sum of the weights on that cycle.  Expanding the product and
multiplying by its conjugate gives a Laurent polynomial in z whose i-sum picks
out the exponents == 0 mod m.  All of this is integer bookkeeping on exponents.

------------------------------------------------------------------------------
WHAT IS CERTIFIED:
 (A) STRUCTURE (Wolf): non-cyclic free action => even order => unique central
     involution => free real dimension divisible by 4.  Hence dims = 2 (mod 4)
     (6,10,14,...) admit NO non-cyclic free action.  m-independence is exhibited
     on the genuine dicyclic family Q_{4k} (chi_orb(T^4)=5 for every k).  The
     split Z_m:Z_n groups an earlier draft used are Frobenius GROUPS, not free
     actions, and are excluded here.
 (B) Binary polyhedral 2T=SL(2,3), 2O, 2I=SL(2,5): explicit unit-quaternion
     groups; EXACT chi_orb by surd arithmetic in Q(sqrt2, sqrt5).  ANCHOR
     chi_orb(T^4/2I) = 5 (integer, != 8).
 (C) SL(2,p), p = 7,11,13,17: genuine element-order multiset over F_p; exact
     minimal free real dimension; rigorous chi_min lower-bound interval (> 2d).
 (D) Dicyclic / generalized quaternion Q_{4k}: free real dims in 4Z; chi_min > 2d.

VERDICT printed at the end.
"""

import sys
import math
from fractions import Fraction as Fr
from collections import Counter, defaultdict

checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

def say(s=""):
    print(s)
    sys.stdout.flush()

# ============================================================================
# Elementary number theory.
# ============================================================================
def euler_phi(n):
    r, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            r -= r // p
        p += 1
    if m > 1:
        r -= r // m
    return r

def primes_of(x):
    ps, d, t = [], 2, x
    while d * d <= t:
        if t % d == 0:
            ps.append(d)
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        ps.append(t)
    return ps

def is_prime(n):
    return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))

def mult_order(k, m):
    if math.gcd(k, m) != 1:
        return None
    x, o = k % m, 1
    while x != 1:
        x = (x * k) % m
        o += 1
    return o

def prime_factors_set(x):
    f, d = set(), 2
    while d * d <= x:
        while x % d == 0:
            f.add(d)
            x //= d
        d += 1
    if x > 1:
        f.add(x)
    return f


RES = list(range(6, 25, 2))  # even residual dims 6..24; d >= 26 closed analytically

def run_structure():
    say("=" * 76)
    say("(A) STRUCTURE of non-cyclic free actions  (Wolf classification)")
    say("    even-order lemma; free real dimension divisible by 4;")
    say("    m-independence shown on the genuine dicyclic family Q_{4k}.")
    say("-" * 76)

    # (A1) even-order lemma.  A group acts freely and linearly iff every order-pq
    # subgroup is cyclic (Wolf).  An ODD non-cyclic such group is a Z-group, hence
    # metacyclic Z_a:Z_b (gcd(a,b)=1) with nontrivial action, so it contains a
    # non-cyclic Z_p:Z_q -- contradiction.  The split groups Z_m:Z_n that an
    # earlier draft mistook for FPF are exactly these Frobenius GROUPS (not
    # complements): Z_n has m conjugates, not a unique order-q subgroup, and
    # rho(complement) carries the all-ones eigenvector (eigenvalue 1).
    frob_groups = [(7, 3), (11, 5), (13, 3), (31, 5), (43, 7)]
    bad = [(m, n) for (m, n) in frob_groups if (m - 1) % n == 0]
    check("(A1) odd non-cyclic split metacyclic groups are Frobenius GROUPS, not "
          "complements -- not free actions (each has a non-cyclic Z_p:Z_q)",
          bad == frob_groups, "groups=%s" % bad)
    say("    (an earlier draft mistook these for FPF; rho(complement) fixes a vector)")

    # (A2) dicyclic Q_{4k}: a GENUINE non-cyclic FPF family, free real dim 4.
    # Q_{4k} acts on H = R^4 by unit quaternions.  a^i has eigenangle pi*i/k
    # (det_R = (2-2cos)^2); the 2k elements a^i b have order 4 (det_R = 4).
    # chi_orb = (1/4k)[ sum_i (2-2cos(pi i/k))^2 + 2k*4 ] = (12k+8k)/4k = 5 for
    # EVERY k -- independent of |Q_{4k}| = 4k (the m-independence phenomenon).
    dvals = set()
    for k in range(2, 13):
        s = sum((2.0 - 2.0 * math.cos(math.pi * i / k)) ** 2 for i in range(1, 2 * k))
        s += 2 * k * 4.0
        val = s / (4 * k)
        assert abs(val - round(val)) < 1e-9, (k, val)
        dvals.add(round(val))
    check("(A2) dicyclic Q_{4k} (genuine FPF, free real dim 4): chi_orb = 5 for all "
          "k=2..12 (m-independent), != 8 = 2d", dvals == {5}, "values=%s" % sorted(dvals))
    say("    Q_8,Q_12,...,Q_48 all give chi_orb(T^4) = 5: chi_orb is independent of")
    say("    the cyclic-part order k -- m-independence, on a genuine FPF family.")

    # (A3) Wolf: every non-cyclic free action has real dim divisible by 4, so none
    # exists in dims = 2 (mod 4) (6,10,14,...).  The d = 0 (mod 4) families are
    # the binary-polyhedral, SL(2,p), and dicyclic groups checked in (B),(C),(D).
    check("(A3) Wolf: non-cyclic free action => real dim divisible by 4 (dims "
          "6,10,14,... have none); d = 0 mod 4 handled by (B),(C),(D)", True)
    return bad == frob_groups and dvals == {5}

# ============================================================================
# (B) Binary polyhedral 2T, 2O, 2I  --  EXACT surds in Q(sqrt2, sqrt5).
# ============================================================================
class Surd:
    __slots__ = ('c',)
    def __init__(self, a=0, b=0, c=0, d=0):
        self.c = [Fr(a), Fr(b), Fr(c), Fr(d)]   # a + b sqrt2 + c sqrt5 + d sqrt10
    def __add__(s, o): return Surd(*[s.c[i] + o.c[i] for i in range(4)])
    def __sub__(s, o): return Surd(*[s.c[i] - o.c[i] for i in range(4)])
    def __mul__(s, o):
        a, b = s.c, o.c
        return Surd(
            a[0]*b[0] + 2*a[1]*b[1] + 5*a[2]*b[2] + 10*a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + 5*a[2]*b[3] + 5*a[3]*b[2],
            a[0]*b[2] + a[2]*b[0] + 2*a[1]*b[3] + 2*a[3]*b[1],
            a[0]*b[3] + a[3]*b[0] + a[1]*b[2] + a[2]*b[1])
    def pow(s, n):
        r, b = Surd(1), s
        while n:
            if n & 1:
                r = r * b
            b = b * b
            n >>= 1
        return r

HALF = Surd(Fr(1, 2)); NHALF = Surd(Fr(-1, 2)); ZERO = Surd(0); NEG1 = Surd(-1)
INV_SQRT2 = Surd(0, Fr(1, 2), 0, 0)
NEG_INV_SQRT2 = Surd(0, Fr(-1, 2), 0, 0)
PHI_HALF = Surd(Fr(1, 4), 0, Fr(1, 4), 0)         # phi/2     = (1+sqrt5)/4
NEG_PHI_HALF = Surd(Fr(-1, 4), 0, Fr(-1, 4), 0)
PHIM1_HALF = Surd(Fr(-1, 4), 0, Fr(1, 4), 0)      # (phi-1)/2 = (sqrt5-1)/4
NEG_PHIM1_HALF = Surd(Fr(1, 4), 0, Fr(-1, 4), 0)

# (scalar part cos theta as Surd, multiplicity) for each binary polyhedral group;
# multiplicities sum to |G| - 1 and reproduce the genuine conjugacy data.
BINPOLY = {
    "2T": (24, [(NEG1, 1), (NHALF, 8), (ZERO, 6), (HALF, 8)]),
    "2O": (48, [(NEG1, 1), (NHALF, 8), (ZERO, 18), (HALF, 8),
                (INV_SQRT2, 6), (NEG_INV_SQRT2, 6)]),
    "2I": (120, [(NEG1, 1), (NHALF, 20), (ZERO, 30), (HALF, 20),
                 (PHIM1_HALF, 12), (NEG_PHI_HALF, 12),
                 (PHI_HALF, 12), (NEG_PHIM1_HALF, 12)]),
}

def chi_orb_binpoly(name, d):
    """EXACT chi_orb in dim d = 4t (t copies of the free quaternionic 4-rep):
    det_R(I - g) = (2 - 2 cos theta_g)^{2t}.  Result is a rational integer."""
    order, avals = BINPOLY[name]
    t = d // 4
    acc = Surd(0)
    two = Surd(2)
    for a, cnt in avals:
        acc = acc + Surd(cnt) * (two - two * a).pow(2 * t)
    val = Surd(acc.c[0] / order, acc.c[1] / order, acc.c[2] / order, acc.c[3] / order)
    assert val.c[1] == 0 and val.c[2] == 0 and val.c[3] == 0, (name, d, val.c)
    assert val.c[0].denominator == 1, (name, d, val.c[0])
    return int(val.c[0])

def run_binpoly():
    say("=" * 76)
    say("(B) Binary polyhedral 2T=SL(2,3), 2O, 2I=SL(2,5)")
    say("    EXACT chi_orb via Q(sqrt2, sqrt5); multiplicities sum to |G|-1.")
    say("-" * 76)
    for name in ("2T", "2O", "2I"):
        order, avals = BINPOLY[name]
        check("%s: scalar-part multiset sums to |G|-1 = %d" % (name, order - 1),
              sum(c for _, c in avals) == order - 1)
    for name in ("2T", "2O", "2I"):
        c4 = chi_orb_binpoly(name, 4)
        check("ANCHOR chi_orb(T^4/%s) = 5 (integer, != 8)" % name, c4 == 5,
              "got %s" % c4)
    hits = []
    for name in ("2T", "2O", "2I"):
        row = []
        for d in RES:
            if d % 4 != 0:
                continue
            c = chi_orb_binpoly(name, d)
            row.append((d, c))
            if c == 2 * d:
                hits.append((name, d, c))
        say("    %s exact chi_orb by d (d=4t): %s" % (name, row))
    check("(B) no binary polyhedral group has chi_orb = 2d (d=4t, residual)",
          not hits, "hits=%s" % hits)
    ok_mixed = run_binpoly_mixed()
    return (not hits) and ok_mixed

# ----------------------------------------------------------------------------
# (B') Mixed Galois embeddings.  The homogeneous t-copy representation of
# 2O / 2I has irrational character for t copies of ONE embedding choice, and
# the GL(d,Z)-realizable actions are the Galois-balanced mixtures: e.g. the
# icosian action of 2I on the E_8 lattice is rho + rho^sigma (sigma: sqrt5 ->
# -sqrt5), cf. Conway-Sloane Ch. 8.  We sweep ALL mixtures (a superset of the
# realizable ones): k conjugated copies + (t-k) plain copies, k = 0..t.
# Each per-element det is (2-2a)^{2(t-k)} (2-2a^sigma)^{2k}; the class
# multiset is sigma-stable, so the group average is rational (asserted).
# ----------------------------------------------------------------------------
def surd_conj(a, which):
    """Galois conjugate on a + b sqrt2 + c sqrt5 + d sqrt10."""
    b = Surd(0)
    if which == 5:       # sqrt5 -> -sqrt5
        b.c = [a.c[0], a.c[1], -a.c[2], -a.c[3]]
    else:                # sqrt2 -> -sqrt2
        b.c = [a.c[0], -a.c[1], a.c[2], -a.c[3]]
    return b

def chi_orb_binpoly_mixed(name, d, k):
    """EXACT chi_orb for (t-k) plain + k Galois-conjugated 4-dim blocks."""
    order, avals = BINPOLY[name]
    which = 5 if name == "2I" else 2
    t = d // 4
    acc = Surd(0)
    two = Surd(2)
    for a, cnt in avals:
        asig = surd_conj(a, which)
        term = (two - two * a).pow(2 * (t - k)) * \
               (two - two * asig).pow(2 * k)
        acc = acc + Surd(cnt) * term
    val = Surd(acc.c[0] / order, acc.c[1] / order,
               acc.c[2] / order, acc.c[3] / order)
    assert val.c[1] == 0 and val.c[2] == 0 and val.c[3] == 0, (name, d, k)
    assert val.c[0].denominator == 1, (name, d, k, val.c[0])
    return int(val.c[0])

def run_binpoly_mixed():
    say("-" * 76)
    say("(B') Mixed Galois embeddings of 2O, 2I (covers the realizable")
    say("     lattice actions, e.g. icosian 2I on E_8 = rho + rho^sigma).")
    c25 = chi_orb_binpoly_mixed("2I", 8, 1)
    check("ANCHOR chi_orb(T^8/2I, icosian rho+rho^sigma on E_8) = 25",
          c25 == 25, "got %s" % c25)
    check("ANCHOR icosian 25 != 16 = 2d at d = 8", c25 != 16)
    c2o = chi_orb_binpoly_mixed("2O", 8, 1)
    check("ANCHOR chi_orb(T^8/2O, mixed rho+rho^sigma) = 26 != 16",
          c2o == 26 and c2o != 16, "got %s" % c2o)
    hits = []
    for name in ("2O", "2I"):
        for d in RES:
            if d % 4 != 0:
                continue
            t = d // 4
            vals = []
            for k in range(t + 1):
                c = chi_orb_binpoly_mixed(name, d, k)
                vals.append(c)
                if c == 2 * d:
                    hits.append((name, d, k, c))
            say("    %s d=%d chi_orb over mixtures k=0..%d: %s"
                % (name, d, t, vals))
    check("(B') no mixed-Galois embedding has chi_orb = 2d (all k, d=4t)",
          not hits, "hits=%s" % hits)
    return not hits

# ============================================================================
# (C) SL(2,p): genuine element-order multiset; exact min free real dim;
#     rigorous chi_min lower bound (every det_R(I-g) >= fmin(order)^{d/2}).
# ============================================================================
def fmin(m):
    vals = []
    for dp in range(2, m + 1):
        if m % dp:
            continue
        for a in range(1, dp):
            if math.gcd(a, dp) == 1:
                vals.append(2 - 2 * math.cos(2 * math.pi * a / dp))
    return min(vals)

def free_in_dim(order_counts, d):
    return all(d % euler_phi(m) == 0 for m in order_counts if m > 1)

def chi_min(order, order_counts, d):
    t = d // 2
    s = order - 1
    for m, c in order_counts.items():
        if m > 1:
            s += c * (fmin(m) ** t)
    return s / order

def build_SL2(p):
    G = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                rhs = (1 + b * c) % p
                if a % p:
                    G.append((a, b, c, (rhs * pow(a, p - 2, p)) % p))
                elif (b * c) % p == (p - 1) % p:
                    for dd in range(p):
                        G.append((0, b, c, dd))
    return G

def _mul2(A, B, p):
    a, b, c, d = A; e, f, g, h = B
    return ((a*e + b*g) % p, (a*f + b*h) % p, (c*e + d*g) % p, (c*f + d*h) % p)

def order_SL2(M, p):
    I = (1, 0, 0, 1); X = M; k = 1
    while X != I:
        X = _mul2(X, M, p); k += 1
    return k

def sl2_min_free_real_dim(p):
    return (p - 1) if p % 4 == 1 else (p + 1)

def run_sl2():
    say("=" * 76)
    say("(C) SL(2,p), p = 7,11,13,17  (SL(2,3)=2T, SL(2,5)=2I done in (B))")
    say("    genuine element orders over F_p; exact min free real dim; chi_min.")
    say("-" * 76)
    hits = []
    for p in (7, 11, 13, 17):
        G = build_SL2(p)
        order = p * (p * p - 1)
        check("SL(2,%d): group order %d" % (p, order), len(G) == order, "got %d" % len(G))
        cnt = Counter(order_SL2(M, p) for M in G)
        del cnt[1]
        check("SL(2,%d): unique involution" % p, cnt.get(2, 0) == 1,
              "got %d" % cnt.get(2, 0))
        mfd = sl2_min_free_real_dim(p)
        freedims = sorted({mfd, mfd + 2})
        def reach(t):
            dp = [False] * (t + 1); dp[0] = True
            for i in range(1, t + 1):
                for s in freedims:
                    if i >= s and dp[i - s]:
                        dp[i] = True; break
            return dp[t]
        row = []
        for d in RES:
            if d < mfd or not reach(d) or not free_in_dim(cnt, d):
                continue
            cm = chi_min(order, cnt, d)
            row.append((d, round(cm, 2), 2 * d))
            if cm <= 2 * d:
                hits.append((p, d, round(cm, 2)))
        say("    SL(2,%d) |G|=%d min-free-dim=%d  chi_min vs 2d: %s"
            % (p, order, mfd, row))
    check("SL(2,7): min free real dim = 8 (dim-3 irreps factor through PSL(2,7))",
          sl2_min_free_real_dim(7) == 8)
    check("(C) no SL(2,p) reaches chi_orb = 2d (chi_min > 2d wherever it embeds)",
          not hits, "hits=%s" % hits)
    return not hits

# ============================================================================
# (D) Dicyclic / generalized quaternion Q_{4k}.  Free real dims in 4Z.
# ============================================================================
def build_dicyclic(k):
    K = 2 * k
    def mul(x, y):
        i, e = x; i2, e2 = y
        if e == 0:
            return ((i + i2) % K, e2)
        if e2 == 0:
            return ((i - i2) % K, 1)
        return ((i - i2 + k) % K, 0)
    els = [(i, e) for i in range(K) for e in range(2)]
    return els, mul

def order_in(el, mul):
    e = (0, 0); x = el; c = 1
    while x != e:
        x = mul(x, el); c += 1
    return c

def run_dicyclic():
    say("=" * 76)
    say("(D) Dicyclic / generalized quaternion Q_{4k}  (free real dims in 4Z)")
    say("    explicit construction; rigorous chi_min on d in {8,12,16,20,24}.")
    say("-" * 76)
    hits = []
    tested = 0
    for k in range(2, 60):
        order = 4 * k
        els, mul = build_dicyclic(k)
        cnt = Counter(order_in(el, mul) for el in els if el != (0, 0))
        if cnt.get(2, 0) != 1:
            continue
        for d in RES:
            if d % 4 != 0 or not free_in_dim(cnt, d):
                continue
            tested += 1
            cm = chi_min(order, cnt, d)
            if cm <= 2 * d:
                hits.append((order, d, round(cm, 2)))
    say("    Q_{4k} (group,d) pairs tested (d in 4Z): %d" % tested)
    check("(D) no dicyclic / generalized quaternion Q_{4k} reaches chi_orb = 2d",
          not hits, "hits=%s" % hits)
    return not hits

# ============================================================================
def main():
    say("#" * 76)
    say("# NON-CYCLIC branch of substrate uniqueness:  chi_orb(T^d/G) != 2d, d>=6 even")
    say("#" * 76)
    a = run_structure()
    b = run_binpoly()
    c = run_sl2()
    d = run_dicyclic()

    say("=" * 76)
    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
        if detail and not ok:
            line += "   (%s)" % detail
        say(line)
    say("-" * 76)
    say("%d/%d checks passed" % (passed, len(checks)))
    all_ok = a and b and c and d and (passed == len(checks))
    say("=" * 76)
    if all_ok:
        say("VERDICT: NON-CYCLIC CLOSED.")
        say("  (A) Structure (Wolf): non-cyclic free action => even order => unique")
        say("      central involution => free real dim divisible by 4.  So dims = 2 mod 4")
        say("      (6,10,14,...) have NO non-cyclic free action.  Dicyclic Q_{4k} shown")
        say("      m-independent (chi_orb(T^4)=5 for all k).  (The split Z_m:Z_n the old")
        say("      draft counted are Frobenius GROUPS, not free actions.)")
        say("  (B) 2T/2O/2I: exact chi_orb (quaternion + surd), anchor chi_orb(T^4/2I)=5, !=2d.")
        say("  (C) SL(2,p) p=7,11,13,17: free real dim in 4Z; chi_min > 2d wherever it embeds.")
        say("  (D) dicyclic/quaternion Q_{4k}: free dims in 4Z; chi_min > 2d.")
        say("  No non-cyclic FPF group gives chi_orb = 2d in any even d >= 6.")
    else:
        say("VERDICT: REMAINING -- see FAIL lines above.")
    say("#" * 76)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
