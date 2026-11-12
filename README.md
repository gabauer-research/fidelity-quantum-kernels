# Spectral Analysis of Fidelity Quantum Kernels

**Reproducible numerical experiments on the spectral structure and effective complexity of finite-sample fidelity quantum kernels.**

This repository investigates how quantum feature-map structure and classical input geometry affect the spectra of fidelity-kernel Gram matrices. Exact Qiskit statevector simulations compare separable angle encodings with interleaved CNOT and CZ feature maps.

## Research question

How do quantum feature-map structure and input geometry determine the spectral complexity of fidelity quantum kernels?

## Mathematical setting

For classical inputs $\mathbf{x}$ and $\mathbf{y}$, the pure-state fidelity kernel is

```math
k_Q(\mathbf{x},\mathbf{y})
=
\left|\langle\psi(\mathbf{x})\mid\psi(\mathbf{y})\rangle\right|^2.
```

For a finite sample, its Gram matrix has entries $K_{ij}=k_Q(\mathbf{x}_i,\mathbf{x}_j)$. The numerical analysis considers eigenvalue decay, numerical rank, spectral concentration and low-rank approximability.

The **effective rank** is defined by the participation ratio, rather than the entropy-based definition:

```math
r_{\mathrm{eff}}(K)
=
\frac{\left(\sum_i\lambda_i\right)^2}{\sum_i\lambda_i^2}.
```

The relative Frobenius error of the optimal rank-$r$ approximation is

```math
\varepsilon_r
=
\sqrt{\frac{\sum_{i>r}\lambda_i^2}{\sum_i\lambda_i^2}}.
```

Here the eigenvalues $\lambda_i$ are ordered from largest to smallest. Numerical rank uses an absolute eigenvalue threshold of $10^{-10}$.

## Experiments

All circuit families use local $R_y$ angle encoding. The CNOT and CZ variants interleave encoding layers with nearest-neighbour entangling gates. Increasing entangling depth also increases the number of encoding layers.

| Script | Investigation |
| --- | --- |
| [`reference_grid.py`](experiments/reference_grid.py) | Analytical-reference validation on a 5 × 5 Cartesian product grid, including the separable Kronecker structure |
| [`depth.py`](experiments/depth.py) | Entangling-depth dependence for CNOT and CZ, using deterministic and uniformly sampled inputs |
| [`scaling.py`](experiments/scaling.py) | Joint and separate variation of qubit number and independently encoded input dimension |
| [`input_geometry.py`](experiments/input_geometry.py) | Comparison of uniform, Gaussian, clustered and correlated inputs |
| [`sample_size.py`](experiments/sample_size.py) | Dependence of numerical and effective rank on sample size |
| [`diagnostics.py`](experiments/diagnostics.py) | Representative spectra, Gram matrices, fidelity distributions and low-rank approximation errors |

The reference-grid experiment checks the separable identity

```math
K_{\mathrm{sep}}=K_{\mathrm{1D}}\otimes K_{\mathrm{1D}}.
```

Randomised parameter studies use 20 fixed seeds (0–19), with the same input realisation shared across feature-map variants in each comparison. The representative diagnostic plots use seed 42.

## Repository structure

```text
qfk/                 Reusable circuit, data, kernel and spectral-analysis functions
experiments/         Six executable numerical studies
results/             Exported CSV data and PDF figures, organised by experiment
pyproject.toml       Package metadata and dependencies
README.md            Project documentation
```

## Installation

Python **3.11 or later** is required. Dependencies, including Qiskit 2.x, are specified in [`pyproject.toml`](pyproject.toml).

From the repository root, create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
# Activate .venv using the appropriate command for your shell.
python -m pip install -e .
```

## Reproducing the results

Run the scripts from the repository root, individually or sequentially:

```bash
python -m experiments.reference_grid
python -m experiments.depth
python -m experiments.scaling
python -m experiments.input_geometry
python -m experiments.sample_size
python -m experiments.diagnostics
```

There is no required execution order. Each experiment writes its output to a corresponding subdirectory of `results/`. CSV files contain numerical results; PDF files contain the generated figures. The results are produced using exact statevectors, without finite-shot or hardware noise.

This repository characterises finite-sample spectral structure. It does not evaluate downstream predictive performance or claim quantum advantage.
