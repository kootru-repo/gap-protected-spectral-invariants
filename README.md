# Gap-protected spectral invariants on T^d/Z_2 (verification code)

[![verify](https://github.com/kootru-repo/gap-protected-spectral-invariants/actions/workflows/verify.yml/badge.svg)](https://github.com/kootru-repo/gap-protected-spectral-invariants/actions/workflows/verify.yml)
[![executed notebook](https://img.shields.io/badge/notebook-view%20executed-2ea44f)](https://kootru-repo.github.io/gap-protected-spectral-invariants/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/gap-protected-spectral-invariants/blob/main/reproduce.ipynb)

Reproducible verification code for the paper *Gap-protected spectral
invariants on T^d/Z_2: dimensional rigidity at d = 4*
(archived at [doi:10.5281/zenodo.20597196](https://doi.org/10.5281/zenodo.20597196)).

**Always test against `main`.** This branch is the canonical, continuously updated version: it is the code the paper points to, and CI re-runs the full suite on it on every commit. There is deliberately no pinned release or tagged version to track down; a fixed tag would freeze you to older, possibly superseded code, so `main` is the single source of the current, corrected checks. The Zenodo DOI above archives the paper itself (PDFs), not a code snapshot to run.

Every numerical and structural claim in the paper is recomputed here from
first principles: exact lattice enumeration, closed forms, high-precision
`mpmath`, cyclotomic roots, and an exhaustive crystallographic sweep. Among
the verified results: the mode count `N_4(5) = 137 = 1 + 8 + 128`, the
dimensional rigidity `2d = 2^{d-1}` (unique at `d = 4`), the spectral
radius `rho < 7.2e-3`, and the `K*(d)` saturation thresholds for `d = 2..8`.

Three ways to check the results, from least to most involvement. The
computation is never on your machine unless you want it to be.

## 1. Look (no login, no launch)

The **view executed** badge opens the notebook *already run* on GitHub's
servers:
[kootru-repo.github.io/gap-protected-spectral-invariants](https://kootru-repo.github.io/gap-protected-spectral-invariants/).
Every headline number sits beside the value printed in the paper, with its
equation or theorem number and an `OK` match. GitHub Actions re-executes the
notebook on every commit and republishes it; the page updates only when all
checks pass, so it is a machine-run result, not a saved screenshot. It is a
static render -- it shows the computation, it does not perform it as you
read.

## 2. Re-run the whole thing live (one click, no install)

The **Open in Colab** badge runs `reproduce.ipynb` on Google's servers.
Choose *Runtime -> Run all*: it clones this repository, installs the
dependencies, recomputes each headline number against the paper (equation
and theorem numbers shown next to each), and then runs the **full
919-check suite** -- the same `run_all.py` the CI runs. This is the live,
from-source re-run, including the parts too heavy for an in-browser kernel
(the crystallographic sweep over the bundled CARAT catalogue and the
adversarial battery). Nothing runs on your machine.

## 3. Run locally

The two commands under **Run it** below reproduce the same checks on your
own machine.

## For reviewers

- The strongest evidence is the **continuous integration**: the full suite
  runs on every commit on fresh Linux and Windows machines across Python
  3.11-3.13 (the green badge above), with public per-step logs.
- For a live, from-source re-run of the entire suite, the **Open in Colab**
  badge needs only a browser.
- A clean clone reproduces everything offline: `pip install -r requirements.txt && python run_all.py` exits `0`, with no network or
  external data needed (the CARAT catalogue is bundled).
- Every checked value derives from closed-form lattice and orbifold data
  by exact arithmetic; no measured or fitted input enters any computed
  quantity.

## Run it

> **A full run takes about a minute** on a typical multi-core machine (longer on 2-4 cores, where the heavy checks queue rather than run side by side). The live progress bar shows an accurate ETA the whole way, so you always know how much is left.

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` executes every script below and exits nonzero if any fails.
Each script is also runnable on its own.

On an interactive terminal it shows a live progress bar with an accurate
estimated time to completion (the steps run in parallel; the ETA simulates the
remaining schedule across your cores and recalibrates to your machine as it
goes). Under CI, pipes, or
redirects it prints one plain line per script as it finishes, keeping logs
clean. Force the rich view anywhere with `VERIFY_RICH=1`.

## What is checked

| Script | Verifies |
|---|---|
| `verify.py` | Mode counts `r_4(k)`, `N_4(5)=137=1+8+128`, the Gram cascade, `K*=5` |
| `verify_spectral_bounds.py` | Spectral radius `rho < 7.2e-3`, the operator-norm interval, and the `K*(d)` saturation dictionary |
| `verify_krein.py` | Krawtchouk eigenvalues `mu_w` and the exact theta identities |
| `classify_crystallographic.py` | Kissing numbers and the integrality-boundary lattices |
| `verify_computational.py` | Gram ranks, shell counts, dynamical eigenvalues |
| `substrate-sweep/substrate_sweep.py` | `T^4/Z_2` unique over the signed-permutation group `W(B_4)` |
| `substrate-sweep/carat_sweep.py` | `T^4/Z_2` unique over all 227 four-dimensional crystal classes |
| `adversarial/run_all.py` | Falsification battery over the decomposition and substrate claims |
| `verify_all_values.py` | Independent recomputation of every printed value |

## Layout

```
.
|- verify*.py, classify_crystallographic.py   primary checks
|- verify_all_values.py                       definitive value recomputation
|- run_all.py                                 top-level runner (CI entry point)
|- adversarial/                               falsification battery
|- substrate-sweep/                           227-class and W(B_4) sweeps
|  |- carat/                                  bundled dim-4 CARAT subset (GPL-2)
|- .github/workflows/verify.yml               CI matrix (Linux + Windows)
```

## Dependencies

`numpy`, `sympy`, `mpmath` (see `requirements.txt`). Python 3.11 or newer.

## License

The verification code is released under the MIT License (see `LICENSE`). The
`substrate-sweep/carat/` subtree is a subset of the CARAT catalogue
(github.com/lbfm-rwth/carat, commit 65619d6), redistributed under the GNU
GPL v2; see `substrate-sweep/carat/LICENSE` and `substrate-sweep/carat/NOTICE.txt`.
