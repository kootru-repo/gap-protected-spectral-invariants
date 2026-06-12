# Spectral rigidity verification

[![verify](https://github.com/kootru-repo/gap-protected-spectral-invariants/actions/workflows/verify.yml/badge.svg)](https://github.com/kootru-repo/gap-protected-spectral-invariants/actions/workflows/verify.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/gap-protected-spectral-invariants/blob/main/reproduce.ipynb)

Reproducible verification code for the paper *Gap-protected spectral
invariants on T^d/Z_2: dimensional rigidity at d = 4*
(archived at [doi:10.5281/zenodo.20597196](https://doi.org/10.5281/zenodo.20597196)).

Every numerical and structural claim in the paper is recomputed here from
first principles: exact lattice enumeration, closed forms, high-precision
`mpmath`, cyclotomic roots, and an exhaustive crystallographic sweep. Among
the verified results: the mode count `N_4(5) = 137 = 1 + 8 + 128`, the
spectral radius `rho < 7.2e-3`, the Born interval `[137.03596, 137.03607]`,
and the smooth spectral action `137.036015074`.

## Run it in the browser

Click **Open in Colab** above. It opens `reproduce.ipynb`; choose
*Runtime -> Run all*. The notebook clones this repository, installs the
dependencies, prints the headline numbers, and runs the full suite. No
local install.

## For reviewers

- One click, no setup: the **Open in Colab** badge runs `reproduce.ipynb`
  end to end in the browser.
- Locally: the two commands under **Run it** below reproduce the same checks.
- No script takes the fine-structure constant, 137, or any measured
  constant as an input to a computed quantity; the measured value appears
  only as an external comparison target in the agreement checks.
- The green badge above is the same suite run on every commit, on Linux and
  Windows across Python 3.11-3.13.

## Run it

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` executes every script below and exits nonzero if any fails.
Each script is also runnable on its own.

## What is checked

| Script | Verifies |
|---|---|
| `verify.py` | Mode counts `r_4(k)`, `N_4(5)=137=1+8+128`, the Gram cascade, `K*=5` |
| `verify_spectral_bounds.py` | Spectral radius `rho < 7.2e-3` and the Born interval |
| `verify_krein.py` | Krawtchouk eigenvalues `mu_w` and the exact theta identities |
| `classify_crystallographic.py` | Kissing numbers and the integrality-boundary lattices |
| `verify_computational.py` | Gram ranks, shell counts, dynamical eigenvalues |
| `substrate-sweep/substrate_sweep.py` | `T^4/Z_2` unique over the signed-permutation group `W(B_4)` |
| `substrate-sweep/carat_sweep.py` | `T^4/Z_2` unique over all 227 four-dimensional crystal classes |
| `adversarial/run_all.py` | Falsification battery (no input takes the comparison constant) |
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
