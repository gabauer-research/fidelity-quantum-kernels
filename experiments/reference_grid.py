"""Spectral comparison of fidelity quantum kernels on a Cartesian grid.

Evaluates separable, CNOT-entangling, and CZ-entangling feature maps on the
same two-dimensional reference dataset. Reports eigenvalue spectra, matrix
rank, effective rank, spectral concentration, and low-rank approximation
errors for each circuit architecture.

For the separable reference case, the analytically predicted Kronecker-product
structure of the Gram matrix is additionally verified numerically.
"""

import csv
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "reference_grid"
SUMMARY_FILE = RESULTS_DIR / "reference_grid_summary.csv"


def main():
    """Evaluate and validate reference-grid fidelity-kernel spectra."""
    X = cartesian_grid(
        points_per_dim=POINTS_PER_DIM,
        d=INPUT_DIMENSION,
    )

    print(
        f"Reference grid: {X.shape[0]} samples, "
        f"{X.shape[1]} dimensions"
    )

    results = []

    for name, entangler, depth in CIRCUIT_VARIANTS:
        K = fidelity_gram_matrix(
            X,
            entangler=entangler,
            depth=depth,
        )

        metrics = spectral_metrics(K)

        # Numerical sanity checks for a pure-state fidelity Gram matrix.
        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(K).min() >= -1e-10

        print(f"\n{name}")
        print("-" * len(name))

        kron_max_error = ""
        kron_fro_error = ""

        # --------------------------------------------------
        # Kronecker-product validation
        # --------------------------------------------------
        #
        # For the separable feature map on the Cartesian grid,
        # the analytical result predicts
        #
        #     K = K_1d ⊗ K_1d.
        #
        # The directly simulated two-dimensional Gram matrix is
        # therefore compared with the Kronecker product of the
        # corresponding one-dimensional Gram matrices.
        if name == "Separable":
            X_1d = cartesian_grid(
                points_per_dim=POINTS_PER_DIM,
                d=1,
            )

            K_1d = fidelity_gram_matrix(
                X_1d,
                entangler=None,
                depth=0,
            )

            K_kron = np.kron(K_1d, K_1d)

            kron_max_error = np.max(
                np.abs(K - K_kron)
            )

            kron_fro_error = np.linalg.norm(
                K - K_kron,
                ord="fro",
            )

            assert np.allclose(
                K,
                K_kron,
                atol=1e-12,
                rtol=1e-12,
            )

            print(
                f"Kronecker max error: "
                f"{kron_max_error:.3e}"
            )

            print(
                f"Kronecker Fro error: "
                f"{kron_fro_error:.3e}"
            )

        # --------------------------------------------------
        # Console output
        # --------------------------------------------------

        print(f"Rank:             {metrics['rank']}")
        print(
            f"Effective rank:   "
            f"{metrics['effective_rank']:.4f}"
        )
        print(
            f"Largest weight:   "
            f"{metrics['largest_weight']:.4f}"
        )
        print(
            f"Off-diag mean:    "
            f"{metrics['offdiag_mean']:.4f}"
        )
        print(
            f"Off-diag std:     "
            f"{metrics['offdiag_std']:.4f}"
        )

        print(
            "Eigenvalues:      ",
            np.round(metrics["eigenvalues"], 6),
        )

        print("Low-rank errors:")

        for rank, error in metrics["low_rank_errors"].items():
            print(f"  r={rank}: {error:.4f}")

        # --------------------------------------------------
        # Collect summary results
        # --------------------------------------------------

        results.append(
            {
                "feature_map": name,
                "depth": depth,
                "rank": metrics["rank"],
                "effective_rank": metrics["effective_rank"],
                "largest_weight": metrics["largest_weight"],
                "offdiag_mean": metrics["offdiag_mean"],
                "offdiag_std": metrics["offdiag_std"],
                "epsilon_1": metrics["low_rank_errors"].get(1, ""),
                "epsilon_2": metrics["low_rank_errors"].get(2, ""),
                "epsilon_3": metrics["low_rank_errors"].get(3, ""),
                "epsilon_4": metrics["low_rank_errors"].get(4, ""),
                "epsilon_5": metrics["low_rank_errors"].get(5, ""),
                "epsilon_6": metrics["low_rank_errors"].get(6, ""),
                "kron_max_error": kron_max_error,
                "kron_fro_error": kron_fro_error,
            }
        )

    # --------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "feature_map",
        "depth",
        "rank",
        "effective_rank",
        "largest_weight",
        "offdiag_mean",
        "offdiag_std",
        "epsilon_1",
        "epsilon_2",
        "epsilon_3",
        "epsilon_4",
        "epsilon_5",
        "epsilon_6",
        "kron_max_error",
        "kron_fro_error",
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nSummary saved to: "
        f"{SUMMARY_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()