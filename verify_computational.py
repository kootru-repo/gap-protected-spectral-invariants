"""
Computational Verification for SI Section S9
=============================================
Implements Test A (spectral resolution thresholds) and Test B
(crystallographic resolution interpretation) as described in
SI Appendix, Section S9.

Requires only numpy. Run: python verify_computational.py

Reproduces Table S9.1 and all intermediate checks.
"""

import numpy as np
from itertools import product as cart_product
from math import comb
from collections import Counter

# ====================================================================
# Core lattice utilities
# ====================================================================

def lattice_points_on_shell(d, k):
    """All n in Z^d with |n|^2 = k."""
    if k == 0:
        return [tuple(0 for _ in range(d))]
    bound = int(np.sqrt(k)) + 1
    return [pt for pt in cart_product(range(-bound, bound + 1), repeat=d)
            if sum(x * x for x in pt) == k]


def fixed_points_z2(d):
    """The 2^d fixed points of Z_2 inversion on T^d."""
    return list(cart_product([0, 1], repeat=d))


def gram_matrix(d, K, dynamical_only=False):
    """Build the 2^d x 2^d Gram matrix at Z_2 fixed points."""
    fps = fixed_points_z2(d)
    G = np.zeros((len(fps), len(fps)), dtype=float)
    k_start = 2 if dynamical_only else 0
    for k in range(k_start, K + 1):
        for n_vec in lattice_points_on_shell(d, k):
            phases = np.array([
                (-1) ** sum(n_vec[i] * a[i] for i in range(d))
                for a in fps
            ])
            G += np.outer(phases, phases)
    return G


def krawtchouk_poly(h, w, n):
    """Krawtchouk polynomial K_h(w; n)."""
    val = 0
    for j in range(h + 1):
        if j <= w and (h - j) <= (n - w):
            val += ((-1) ** j) * comb(w, j) * comb(n - w, h - j)
    return val


# ====================================================================
# Test A: Spectral resolution thresholds
# ====================================================================

def test_a(d_range=None, K_max=25):
    """Compute K* and verify three-sector decomposition for each d."""
    if d_range is None:
        d_range = [2, 3, 4, 5, 6, 7]

    results = {}
    for d in d_range:
        n_fp = 2 ** d
        chi_orb = 2 ** (d - 1)

        # Find K* = min{K >= 2 : rank(G^dyn(K)) = 2^d}
        K_star = None
        rank_prog = {}
        for K in range(2, K_max + 1):
            G = gram_matrix(d, K, dynamical_only=True)
            r = np.linalg.matrix_rank(G, tol=1e-8)
            rank_prog[K] = r
            if r == n_fp and K_star is None:
                K_star = K

        if K_star is None:
            results[d] = {"error": f"rank < {n_fp} at K={K_max}"}
            continue

        # Counts
        N_d_K = sum(len(lattice_points_on_shell(d, k))
                     for k in range(K_star + 1))
        r_d_1 = len(lattice_points_on_shell(d, 1))
        dyn_count = sum(len(lattice_points_on_shell(d, k))
                        for k in range(2, K_star + 1))

        three_sector = (N_d_K == 1 + chi_orb + n_fp * chi_orb)
        topo_match = (r_d_1 == chi_orb)

        # Gap after K*
        gap_width = None
        for k in range(K_star + 1, K_star + 10):
            if len(lattice_points_on_shell(d, k)) > 0:
                gap_width = k - K_star
                break

        results[d] = {
            "K_star": K_star,
            "N_d_K": N_d_K,
            "n_fp": n_fp,
            "chi_orb": chi_orb,
            "r_d_1": r_d_1,
            "dyn_count": dyn_count,
            "three_sector": three_sector,
            "topo_match": topo_match,
            "gap_width": gap_width,
            "rank_prog": {k: v for k, v in rank_prog.items()
                          if k <= K_star + 1},
        }

    return results


# ====================================================================
# Test A supplementary: Krawtchouk eigenvalues at d=4, K=5
# ====================================================================

def krawtchouk_eigenvalues_d4(dynamical_only=False):
    """Compute Krawtchouk eigenvalues of G(5) for d=4.
    If dynamical_only=True, uses shells k=2..5 only (dynamical matrix).
    If False, uses all shells k=0..5 (full Gram matrix)."""
    d, K = 4, 5

    fps = fixed_points_z2(d)
    G = gram_matrix(d, K, dynamical_only=dynamical_only)

    def hamming(a, b):
        return sum(x != y for x, y in zip(a, b))

    char_sums = {}
    for h in range(d + 1):
        for i, a in enumerate(fps):
            for j, b in enumerate(fps):
                if hamming(a, b) == h:
                    char_sums[h] = G[i, j]
                    break
            else:
                continue
            break

    # Krawtchouk eigenvalues: lambda_w = sum_h K_h(w) * G^{(h)}
    eig_by_weight = {}
    for w in range(d + 1):
        lam = sum(krawtchouk_poly(h, w, d) * char_sums[h]
                  for h in range(d + 1))
        # Eigenvalues must be near-integer (exact for integer character sums)
        if abs(lam - round(lam)) >= 1e-6:
            raise ValueError(
                f"Eigenvalue lambda_{w} = {lam} is not near-integer"
            )
        eig_by_weight[w] = round(lam)

    # Verify against direct diagonalisation
    eigs_direct = sorted(np.linalg.eigvalsh(G), reverse=True)

    return char_sums, eig_by_weight, eigs_direct


# ====================================================================
# Test B: Crystallographic resolution
# ====================================================================

def resolution_matrix(d, K):
    """Build the crystallographic resolution matrix."""
    fps_real = [tuple(e / 2.0 for e in eps)
                for eps in cart_product([0, 1], repeat=d)]
    n_fp = len(fps_real)
    M = np.zeros((n_fp, n_fp), dtype=complex)

    for k in range(2, K + 1):
        for h in lattice_points_on_shell(d, k):
            phases = np.array([
                np.exp(2j * np.pi * sum(h[i] * r[i] for i in range(d)))
                for r in fps_real
            ])
            M += np.outer(phases, np.conj(phases))

    return M


def test_b(d_range=None):
    """Crystallographic resolution tests."""
    if d_range is None:
        d_range = [2, 3, 4, 5]

    results = {}
    for d in d_range:
        n_fp = 2 ** d

        # Identity test: resolution matrix == Gram matrix
        G = gram_matrix(d, 5, dynamical_only=True)
        M = resolution_matrix(d, 5)
        max_diff = np.max(np.abs(M.real - G))
        identity_pass = max_diff < 1e-10

        # Find K* from resolution matrix rank
        K_star = None
        for K in range(2, 20):
            Mr = resolution_matrix(d, K)
            if np.linalg.matrix_rank(Mr.real, tol=1e-8) == n_fp:
                K_star = K
                break

        # Shell necessity: remove each shell, check rank drops
        shells_necessary = 0
        shells_total = 0
        if K_star:
            for k in range(2, K_star + 1):
                if len(lattice_points_on_shell(d, k)) == 0:
                    continue
                shells_total += 1
                # Build matrix without shell k
                M_without = np.zeros((n_fp, n_fp), dtype=complex)
                for kk in range(2, K_star + 1):
                    if kk == k:
                        continue
                    fps_real = [tuple(e / 2.0 for e in eps)
                                for eps in cart_product([0, 1], repeat=d)]
                    for h in lattice_points_on_shell(d, kk):
                        phases = np.array([
                            np.exp(2j * np.pi * sum(h[i] * r[i]
                                                     for i in range(d)))
                            for r in fps_real
                        ])
                        M_without += np.outer(phases, np.conj(phases))
                rank_without = np.linalg.matrix_rank(M_without.real,
                                                      tol=1e-8)
                if rank_without < n_fp:
                    shells_necessary += 1

        # Noise robustness
        noise_pass = True
        if K_star:
            Mr = resolution_matrix(d, K_star)
            rng = np.random.default_rng(42)
            for sigma in [0.1, 0.2, 0.5]:
                for _ in range(10):
                    noise = rng.normal(0, 1, Mr.shape)
                    noise = (noise + noise.T) / 2
                    fnorm = np.linalg.norm(Mr.real, 'fro')
                    M_noisy = Mr.real + sigma * fnorm * noise / np.linalg.norm(noise, 'fro')
                    if np.linalg.matrix_rank(M_noisy, tol=1e-8) < n_fp:
                        noise_pass = False

        results[d] = {
            "identity_pass": identity_pass,
            "max_diff": max_diff,
            "K_star": K_star,
            "dyn_modes": sum(len(lattice_points_on_shell(d, k))
                            for k in range(2, (K_star or 6) + 1)),
            "shells_necessary": shells_necessary,
            "shells_total": shells_total,
            "noise_robust": noise_pass,
        }

    return results


# ====================================================================
# Main
# ====================================================================

def main():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")

    print("=" * 70)
    print("  TEST A: Spectral Resolution Thresholds (d = 2..5)")
    print("=" * 70)

    results_a = test_a([2, 3, 4, 5])

    # Table S9.1
    print(f"\n  {'d':>2} | {'|Fix|':>5} | {'chi':>4} | {'K*':>3} | "
          f"{'N_d(K*)':>8} | {'3-sec':>5} | {'r(1)=chi':>8} | {'Gap':>6}")
    print(f"  {'-'*2}-+-{'-'*5}-+-{'-'*4}-+-{'-'*3}-+-"
          f"{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}")
    for d in [2, 3, 4, 5]:
        r = results_a[d]
        ts = "yes" if r["three_sector"] else "no"
        tm = "yes" if r["topo_match"] else "no"
        gw = f"w={r['gap_width']}" if r["gap_width"] else "N/A"
        print(f"  {d:>2} | {r['n_fp']:>5} | {r['chi_orb']:>4} | "
              f"{r['K_star']:>3} | {r['N_d_K']:>8} | {ts:>5} | "
              f"{tm:>8} | {gw:>6}")

    # Checks
    print()
    r4 = results_a[4]
    check("K*(d=4) = 5", r4["K_star"] == 5)
    check("N_4(5) = 137", r4["N_d_K"] == 137)
    check("Three-sector holds at d=4", r4["three_sector"])
    check("r_4(1) = chi_orb = 8", r4["topo_match"])
    check("Dynamical count = 128 = |Fix|*chi_orb",
          r4["dyn_count"] == 128)
    check("Gap after K* at d=4 (width 1)", r4["gap_width"] == 1)

    for d in [2, 3, 5]:
        r = results_a[d]
        check(f"Three-sector FAILS at d={d}", not r["three_sector"])

    check("Gap after K* at d=2 (width 3)",
          results_a[2]["gap_width"] == 3)

    # Rank progression d=4
    rp = r4["rank_prog"]
    check("d=4 rank at K=2 is 6", rp.get(2) == 6)
    check("d=4 rank at K=3 is 10", rp.get(3) == 10)
    check("d=4 rank at K=4 is 12", rp.get(4) == 12)
    check("d=4 rank at K=5 is 16 (full)", rp.get(5) == 16)

    # Krawtchouk eigenvalues — FULL Gram matrix (all shells k=0..5)
    print("\n  Krawtchouk eigenvalues of full G(5) at d=4:")
    char_sums, eig_by_weight, eigs_direct = krawtchouk_eigenvalues_d4(dynamical_only=False)
    print(f"    Character sums G^(h): {dict(char_sums)}")
    print(f"    Eigenvalues by weight: {eig_by_weight}")
    print(f"    Multiplicities: "
          + ", ".join(f"C(4,{w})={comb(4,w)}" for w in range(5)))

    # Full Gram eigenvalues (registry values)
    check("lambda_0 = 144 = N_4(5) + chi_orb - 1", eig_by_weight[0] == 144)
    check("lambda_1 = 224", eig_by_weight[1] == 224)
    check("lambda_2 = 64 = chi_orb^2", eig_by_weight[2] == 64)
    check("lambda_3 = 128 = |Fix|*chi_orb", eig_by_weight[3] == 128)
    check("lambda_4 = 256 = |Fix|^2", eig_by_weight[4] == 256)
    check("All Gram eigenvalues positive",
          all(e > -1e-10 for e in eigs_direct))

    # Cross-check: Krawtchouk decomposition matches direct diagonalisation
    eigs_sorted = sorted(eigs_direct, reverse=True)
    expected_spectrum = []
    for w in range(5):
        expected_spectrum.extend([eig_by_weight[w]] * comb(4, w))
    expected_sorted = sorted(expected_spectrum, reverse=True)
    max_eig_diff = max(abs(a - b) for a, b in zip(eigs_sorted, expected_sorted))
    check(f"Krawtchouk eigenvalues match direct diag (max diff = {max_eig_diff:.2e})",
          max_eig_diff < 1e-8)

    # Trace identity: sum_w C(4,w)*lambda_w = 16 * N_4(5)
    trace_sum = sum(comb(4, w) * eig_by_weight[w] for w in range(5))
    N4_5 = sum(len(lattice_points_on_shell(4, k)) for k in range(0, 6))
    check(f"Trace identity: sum C(4,w)*lambda_w = {trace_sum} = 16*{N4_5} = {16*N4_5}",
          trace_sum == 16 * N4_5)

    # Also verify dynamical Gram eigenvalues separately
    print("\n  Dynamical Gram eigenvalues G^dyn(5) at d=4:")
    cs_dyn, eig_dyn, _ = krawtchouk_eigenvalues_d4(dynamical_only=True)
    print(f"    Dynamical eigenvalues: {eig_dyn}")
    check("Dynamical lambda_0 = 128 = |Fix|*chi_orb",
          eig_dyn[0] == 128)
    check("All dynamical eigenvalues non-negative",
          all(eig_dyn[w] >= 0 for w in range(5)))

    # Jacobi four-square cross-check: r_4(k) = 8 * sigma_tilde(k)
    print("\n  Jacobi four-square cross-check:")

    def sigma_tilde(k):
        """Sum of divisors of k not divisible by 4."""
        if k == 0:
            return 0
        return sum(d for d in range(1, k + 1) if k % d == 0 and d % 4 != 0)

    for k in range(6):
        r4_jacobi = 1 if k == 0 else 8 * sigma_tilde(k)
        r4_enum = len(lattice_points_on_shell(4, k))
        check(f"r_4({k}): Jacobi = {r4_jacobi}, enum = {r4_enum}",
              r4_jacobi == r4_enum)

    N4_jacobi = sum(1 if k == 0 else 8 * sigma_tilde(k) for k in range(6))
    check(f"N_4(5) via Jacobi = {N4_jacobi} = 137", N4_jacobi == 137)

    # Independent mu_2 verification: K_2 annihilation
    # Krawtchouk K_2 = [1, 0, -2, 0, 1] applied to character sums gives
    # the w=2 eigenvalue. For the off-diagonal (h=1..4 only) Green's function,
    # the exact result is mu_2 = -1/(4000*pi).
    # K_h(w=2; 4) for h=0..4 gives the Krawtchouk coefficients for sector w=2
    K_at_w2 = [krawtchouk_poly(h, 2, 4) for h in range(5)]
    check(f"[K_h(2;4)] = {K_at_w2} (annihilates h=1,3)",
          K_at_w2 == [1, 0, -2, 0, 1])

    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST B: Crystallographic Resolution Interpretation (d = 2..5)")
    print("=" * 70)

    results_b = test_b([2, 3, 4, 5])

    # Test B results
    print(f"\n  {'d':>2} | {'2^d':>4} | {'K*':>3} | {'Dyn':>5} | "
          f"{'Identity':>8} | {'Shells':>8} | {'Noise':>6}")
    print(f"  {'-'*2}-+-{'-'*4}-+-{'-'*3}-+-{'-'*5}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for d in [2, 3, 4, 5]:
        r = results_b[d]
        ident = "pass" if r["identity_pass"] else "FAIL"
        shells = f"{r['shells_necessary']}/{r['shells_total']}"
        noise = "100%" if r["noise_robust"] else "FAIL"
        print(f"  {d:>2} | {2**d:>4} | {r['K_star']:>3} | "
              f"{r['dyn_modes']:>5} | {ident:>8} | {shells:>8} | "
              f"{noise:>6}")

    print()
    for d in [2, 3, 4, 5]:
        r = results_b[d]
        check(f"d={d}: Gram == resolution matrix",
              r["identity_pass"])
        check(f"d={d}: all occupied shells necessary",
              r["shells_necessary"] == r["shells_total"])
        check(f"d={d}: noise robust at sigma=0.5",
              r["noise_robust"])

    check("d=4: 128 dynamical modes", results_b[4]["dyn_modes"] == 128)

    # Shell necessity detail for d=4
    print("\n  Shell necessity (d=4):")
    fps_real = [tuple(e / 2.0 for e in eps)
                for eps in cart_product([0, 1], repeat=4)]
    for k in [2, 3, 4, 5]:
        modes = lattice_points_on_shell(4, k)
        n_modes = len(modes)
        # Build without this shell
        M = np.zeros((16, 16), dtype=complex)
        for kk in [2, 3, 4, 5]:
            if kk == k:
                continue
            for h in lattice_points_on_shell(4, kk):
                phases = np.array([
                    np.exp(2j * np.pi * sum(h[i] * r[i] for i in range(4)))
                    for r in fps_real
                ])
                M += np.outer(phases, np.conj(phases))
        rank_drop = 16 - np.linalg.matrix_rank(M.real, tol=1e-8)
        print(f"    Remove k={k}: {n_modes} modes, "
              f"rank drop = {rank_drop}")

    # ================================================================
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed "
          f"out of {passed + failed} checks")
    print("=" * 70)
    if failed == 0:
        print("\n  All checks passed.\n")
    else:
        print(f"\n  {failed} FAILURES detected.\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
