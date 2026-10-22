"""Spectral comparison of fidelity quantum kernels on a Cartesian grid.

Evaluates separable, CNOT-entangling, and CZ-entangling feature maps on the
same two-dimensional reference dataset. Reports eigenvalue spectra, matrix
rank, effective rank, spectral concentration, and low-rank approximation
errors for each circuit architecture.
"""

import numpy as np

from qfk.data import cartesian_grid
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


POINTS_PER_DIM = 5
INPUT_DIMENSION = 2
CIRCUIT_VARIANTS = (
    ("Separable", None, 0),
    ("CNOT", "cnot", 1),
    ("CZ", "cz", 1),
)


def main():
    """Evaluate kernel spectra for three circuit architectures on one grid."""
    X = cartesian_grid(points_per_dim=POINTS_PER_DIM, d=INPUT_DIMENSION)

    print(f"Reference grid: {X.shape[0]} samples, {X.shape[1]} dimensions")

    for name, entangler, depth in CIRCUIT_VARIANTS:
        K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)
        metrics = spectral_metrics(K)

        # Numerical sanity checks for a pure-state fidelity Gram matrix.
        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(K).min() >= -1e-10

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Rank:             {metrics['rank']}")
        print(f"Effective rank:   {metrics['effective_rank']:.4f}")
        print(f"Largest weight:   {metrics['largest_weight']:.4f}")
        print(f"Off-diag mean:    {metrics['offdiag_mean']:.4f}")
        print(f"Off-diag std:     {metrics['offdiag_std']:.4f}")
        print("Eigenvalues:      ", np.round(metrics["eigenvalues"], 6))
        print("Low-rank errors:")
        for rank, error in metrics["low_rank_errors"].items():
            print(f"  r={rank}: {error:.4f}")


if __name__ == "__main__":
    main()
