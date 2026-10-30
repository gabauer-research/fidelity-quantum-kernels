# Spectral Analysis of Fidelity Quantum Kernels

**Reproducible numerical experiments on the spectral structure and effective complexity of quantum-kernel Gram matrices.**

This repository investigates how quantum feature-map architecture and circuit depth influence the spectrum of finite-sample fidelity quantum kernels. The implementation uses **Qiskit statevector simulation** to compare separable angle encodings with interleaved **CNOT** and **CZ** feature maps, without finite-shot or hardware noise.

The current release contains the analytical-reference validation and the entangling-depth study. Additional parameter studies are not included in this version.

## Research question

How do quantum feature-map structure and input geometry determine the spectral complexity of fidelity quantum kernels?

## Mathematical setting

For classical inputs $\mathbf{x}$ and $\mathbf{y}$, the fidelity quantum kernel is

$$
k_Q(\mathbf{x},\mathbf{y})
=
\left|
\langle
\psi(\mathbf{x})
\mid
\psi(\mathbf{y})
\rangle
\right|^2,
$$

where $\lvert\psi(\mathbf{x})\rangle$ is prepared by an input-dependent quantum feature map. Given samples $\{\mathbf{x}_i\}_{i=1}^n$, the associated Gram matrix is

$$
K_{ij}
=
k_Q(\mathbf{x}_i,\mathbf{x}_j).
$$

The experiments examine numerical rank, spectral concentration, and low-rank approximability. In particular, **effective rank** refers here to the participation ratio

$$
r_{\mathrm{eff}}(K)
=
\frac{
\left(\sum_i \lambda_i\right)^2
}{
\sum_i \lambda_i^2
},
$$

rather than the entropy-based effective rank.

The relative error of the optimal rank-$r$ spectral approximation is measured in Frobenius norm:

$$
\varepsilon_r
=
\sqrt{
\frac{
\sum_{i>r}\lambda_i^2
}{
\sum_i\lambda_i^2
}
}.
$$

Here $\lambda_i$ are the eigenvalues of $K$, ordered from largest to smallest. The implementation uses an absolute numerical-rank threshold of $10^{-10}$.

## Implemented experiments

### 1. Analytical-reference validation

[`experiments/reference_grid.py`](experiments/reference_grid.py) evaluates two-qubit feature maps on the deterministic Cartesian product grid

$$
\Theta_5 \times \Theta_5,
\qquad
\Theta_5
=
\left\{
0,
\frac{\pi}{4},
\frac{\pi}{2},
\frac{3\pi}{4},
\pi
\right\}.
$$

The script computes fidelity Gram matrices for the separable reference, interleaved CNOT, and interleaved CZ constructions. It reports eigenvalues, numerical rank, effective rank, leading spectral weight, off-diagonal kernel statistics, and relative low-rank approximation errors.

For the separable construction, the analytically derived Cartesian-product structure predicts

$$
K_{\mathrm{sep}}
=
K_{\mathrm{1D}}
\otimes
K_{\mathrm{1D}}.
$$

The implementation verifies this relation explicitly by comparing the directly simulated two-dimensional Gram matrix with the Kronecker product of the corresponding one-dimensional Gram matrices. On the $5\times5$ reference grid, the observed deviations are

$$
\max_{ij}
\left|
(K_{\mathrm{sep}})_{ij}
-
(K_{\mathrm{1D}}\otimes K_{\mathrm{1D}})_{ij}
\right|
=
5.551\times10^{-16},
$$

and

$$
\left\|
K_{\mathrm{sep}}
-
K_{\mathrm{1D}}\otimes K_{\mathrm{1D}}
\right\|_F
=
2.265\times10^{-15},
$$

confirming agreement up to numerical precision.

The separable and single-CNOT constructions reproduce the following reference values on this grid:

| Feature map | Analytical rank bound | Numerical rank | Effective rank |
| :-- | --: | --: | --: |
| Separable | 9 | 9 | 4.0555 |
| Interleaved CNOT ($L=1$) | 10 | 10 | 4.3967 |

The analytical rank bounds follow from the respective fidelity-feature representations; the script provides numerical consistency checks rather than symbolic proofs.

The complete compact numerical summary is written to

[`results/reference_grid/reference_grid_summary.csv`](results/reference_grid/reference_grid_summary.csv).

### 2. Entangling-depth study

[`experiments/depth.py`](experiments/depth.py) compares interleaved CNOT and CZ constructions for

$$
L\in\{1,2,3,4,5\},
$$

using the separable map at $L=0$ as a common reference.

Two input settings are evaluated:

- **Deterministic grid:** $5\times5$ Cartesian product grid, with $n=25$ and $q=d=2$.
- **Uniform random inputs:** 20 independently seeded datasets (seeds 0–19), each containing 25 samples in $[0,\pi]^2$. Within a seed, the same dataset is used for every circuit configuration.

Each entangling layer is followed by another input-dependent $R_y$ encoding layer. Consequently, changing $L$ changes **both** the entangling depth and the number of data re-uploading layers; the experiment does not isolate the effect of entanglement alone.

The study produces per-configuration spectral diagnostics, raw results, across-seed means and standard deviations, eigenvalue spectra for the grid experiment, and PDF figures.

## Selected findings

- **Analytical reference results are reproduced numerically.** For the separable and single-CNOT feature maps, the numerical Gram-matrix ranks are 9 and 10, respectively, matching the analytically derived upper bounds. Their effective ranks are approximately 4.0555 and 4.3967.
- **The separable Cartesian-grid Gram matrix exhibits the predicted Kronecker structure.** The directly simulated matrix and the corresponding Kronecker-product construction differ by only $5.551\times10^{-16}$ in maximum absolute entrywise deviation and $2.265\times10^{-15}$ in Frobenius norm.
- **Effective complexity varies without a corresponding change in numerical rank.** On the deterministic grid, both entangling circuit families have numerical rank 10 across the investigated depths, while their effective ranks depend non-monotonically on depth. For CNOT, for example, the effective rank changes from 2.8482 at $L=3$ to 5.7479 at $L=4$.
- **The non-monotonic depth dependence persists for random inputs.** Across 20 uniformly sampled datasets, the mean CNOT effective rank is $3.0277 \pm 0.3081$ at $L=3$ and $4.5204 \pm 0.5166$ at $L=5$ (mean $\pm$ standard deviation). The quantitative behaviour differs from the deterministic grid.

These observations concern the spectra of the tested finite-sample Gram matrices. Because each additional entangling layer is accompanied by another encoding layer, they do not establish a separate causal effect of entanglement or imply an improvement in downstream predictive performance.

## Repository structure

```text
.
├── qfk/
│   ├── __init__.py
│   ├── data.py            # Synthetic input data
│   ├── circuits.py        # Ry feature maps with optional CNOT/CZ layers
│   ├── kernels.py         # Statevectors and fidelity Gram matrices
│   └── spectral.py        # Eigenvalue-based diagnostics
├── experiments/
│   ├── reference_grid.py  # Analytical-reference validation
│   └── depth.py           # Entangling-depth experiments
├── results/
│   ├── reference_grid/
│   │   └── reference_grid_summary.csv
│   └── depth/             # CSV results and PDF figures
├── pyproject.toml
└── README.md
```

## Installation

Use **Python 3.11 or newer** in a dedicated environment. The project specifies Qiskit 2.x and its numerical dependencies in [`pyproject.toml`](pyproject.toml).

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

Editable installation allows the experiment scripts to import `qfk` when executed directly from an IDE as well as from the command line.

## Reproducing the experiments

Run the following commands from the repository root:

```bash
python -m experiments.reference_grid
python -m experiments.depth
```

### Reference-grid outputs

The reference-grid experiment prints the detailed diagnostics to the console and writes the compact summary

[`results/reference_grid/reference_grid_summary.csv`](results/reference_grid/reference_grid_summary.csv).

The summary contains the principal spectral diagnostics for the separable, CNOT, and CZ constructions together with the Kronecker-product validation errors for the separable reference case.

### Depth-study outputs

The depth experiment writes its outputs to [`results/depth/`](results/depth/), including:

| File | Description |
| :-- | :-- |
| `depth_raw.csv` | Per-configuration metrics on the deterministic grid |
| `depth_spectra.csv` | Eigenvalue spectra on the deterministic grid |
| `depth_random_raw.csv` | Per-seed metrics for uniformly sampled inputs |
| `depth_random_summary.csv` | Across-seed means and standard deviations |
| `effective_rank_vs_depth.pdf` | Grid-based effective rank versus depth |
| `effective_rank_vs_depth_random.pdf` | Random-input effective rank with standard-deviation error bars |
| `largest_weight_vs_depth.pdf` | Grid-based leading spectral weight versus depth |
| `largest_weight_vs_depth_random.pdf` | Random-input leading spectral weight with standard-deviation error bars |

The output directories are resolved relative to the repository rather than the shell's working directory. Figures are saved as PDF; whether they also appear in a separate window depends on the active Matplotlib backend.

## Methodological scope

- All kernels are computed from **exact statevectors**, without sampling noise, decoherence, or hardware-specific effects.
- The reported ranks are **numerical estimates** based on an absolute eigenvalue threshold of $10^{-10}$.
- Random-input experiments use fixed seeds for reproducible comparisons.
- Identical random input realisations are used across circuit variants whenever direct comparisons are made.
- The current experiments concern deliberately small, controlled systems and should not be interpreted as hardware-performance benchmarks or evidence of quantum advantage.
- Spectral broadening, larger effective rank, or larger numerical rank are treated as structural properties and are not assumed to imply better predictive performance.

## Research status

This repository documents a defined subset of an ongoing investigation into the spectral and effective complexity of fidelity quantum kernels. The current version contains the analytical-reference validation and the entangling-depth study.

The repository is intended to provide a transparent connection between the numerical experiment code, the generated result files, and the quantitative results reported in the associated Master's thesis.