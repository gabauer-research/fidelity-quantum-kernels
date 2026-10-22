# Spectral Analysis of Fidelity Quantum Kernels

**Reproducible numerical experiments on the spectral structure and effective complexity of quantum-kernel Gram matrices.**

This repository investigates how quantum feature-map architecture and circuit depth influence the spectrum of finite-sample fidelity quantum kernels. The implementation uses **Qiskit statevector simulation** to compare separable angle encodings with interleaved **CNOT** and **CZ** feature maps, without finite-shot or hardware noise.

The current release contains the analytical-reference validation and the entangling-depth study. Additional parameter studies are not included in this version.

## Mathematical setting

For classical inputs $\mathbf{x}$ and $\mathbf{y}$, the fidelity quantum kernel is

$$
k_Q(\mathbf{x},\mathbf{y})
= \left|\langle\psi(\mathbf{x})\mid\psi(\mathbf{y})\rangle\right|^2,
$$

where $\lvert\psi(\mathbf{x})\rangle$ is prepared by an input-dependent quantum feature map. Given samples $\{\mathbf{x}_i\}_{i=1}^n$, the associated Gram matrix is

$$
K_{ij} = k_Q(\mathbf{x}_i,\mathbf{x}_j).
$$

The experiments examine numerical rank, spectral concentration, and low-rank approximability. In particular, **effective rank** refers here to the *participation ratio*

$$
r_{\mathrm{eff}}(K)
= \frac{\left(\sum_i\lambda_i\right)^2}{\sum_i\lambda_i^2},
$$

rather than the entropy-based effective rank. The relative error of the optimal rank-$r$ spectral approximation is measured in Frobenius norm:

$$
\varepsilon_r
= \sqrt{\frac{\sum_{i>r}\lambda_i^2}{\sum_i\lambda_i^2}}.
$$

Here $\lambda_i$ are the eigenvalues of $K$, ordered from largest to smallest. The implementation uses an absolute numerical-rank threshold of $10^{-10}$.

## Implemented experiments

### 1. Analytical-reference validation

[`experiments/reference_grid.py`](experiments/reference_grid.py) evaluates a two-qubit feature map on the deterministic Cartesian product grid

$$
\Theta_5\times\Theta_5,\qquad
\Theta_5=\left\{0,\frac{\pi}{4},\frac{\pi}{2},\frac{3\pi}{4},\pi\right\}.
$$

The script computes fidelity Gram matrices for the separable reference, interleaved CNOT, and interleaved CZ constructions. It reports eigenvalues, numerical rank, effective rank, leading spectral weight, off-diagonal kernel statistics, and relative low-rank approximation errors.

The separable and single-CNOT constructions reproduce the following reference values on this grid:

| Feature map | Analytical rank bound | Numerical rank | Effective rank |
| :-- | --: | --: | --: |
| Separable | 9 | 9 | 4.0555 |
| Interleaved CNOT ($L=1$) | 10 | 10 | 4.3967 |

The analytical rank bounds follow from the respective fidelity-feature representations; the script provides numerical consistency checks, not symbolic proofs.

### 2. Entangling-depth study

[`experiments/depth.py`](experiments/depth.py) compares interleaved CNOT and CZ constructions for $L\in\{1,2,3,4,5\}$, using the separable map at $L=0$ as a common reference.

Two input settings are evaluated:

- **Deterministic grid:** $5\times5$ Cartesian product grid, with $n=25$ and $q=d=2$.
- **Uniform random inputs:** 20 independently seeded datasets (seeds 0–19), each containing 25 samples in $[0,\pi]^2$. Within a seed, the same dataset is used for every circuit configuration.

Each entangling layer is followed by another input-dependent $R_y$ encoding layer. Consequently, changing $L$ changes **both** the entangling depth and the number of data re-uploading layers; the experiment does not isolate the effect of entanglement alone.

The study produces per-configuration spectral diagnostics, raw results, across-seed means and standard deviations, eigenvalue spectra for the grid experiment, and PDF figures.

## Repository structure

```text
.
├── qfk/
│   ├── __init__.py
│   ├── data.py            # Synthetic input data
│   ├── circuits.py        # RY feature maps with optional CNOT/CZ layers
│   ├── kernels.py         # Statevectors and fidelity Gram matrices
│   └── spectral.py        # Eigenvalue-based diagnostics
├── experiments/
│   ├── reference_grid.py  # Analytical-reference validation
│   └── depth.py           # Entangling-depth experiments
├── results/
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

The reference-grid script prints its numerical diagnostics to the console. The depth experiment writes its outputs to [`results/depth/`](results/depth/), including:

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

The output directory is resolved relative to the repository, not the shell's working directory. Figures are saved as PDF; whether they also appear in a separate window depends on the active Matplotlib backend.

## Methodological scope

- All kernels are computed from **exact statevectors**, without sampling noise, decoherence, or hardware-specific effects.
- The reported ranks are **numerical estimates** based on an absolute eigenvalue threshold of $10^{-10}$.
- Random-input experiments use fixed seeds for reproducible comparisons.
- The current experiments concern deliberately small, controlled systems and should not be interpreted as hardware-performance benchmarks or evidence of quantum advantage.

## Research status

This repository documents a defined subset of an ongoing research investigation into the spectral and effective complexity of fidelity quantum kernels. The code and included numerical outputs are intended to make the reported reference and depth experiments reproducible.
