"""Representative spectral diagnostics for fidelity quantum kernels.

Compare separable, CNOT, and CZ feature maps on uniform and clustered inputs
at q = d = 4, n = 25, and seed 42. The entangling maps use depth L = 1.
Export Gram matrices, spectra, pairwise fidelities, low-rank errors, and plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import clustered_data, uniform_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


Q = 4
D = 4
N = 25
SEED = 42

VARIANTS = (
    ("Separable", None, 0),
    ("CNOT", "cnot", 1),
    ("CZ", "cz", 1),
)

MARKERS = {
    "Separable": "^",
    "CNOT": "o",
    "CZ": "s",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"


def make_dataset(geometry):
    """Generate the specified input geometry with a fixed random seed."""
    if geometry == "Uniform":
        return uniform_data(n=N, d=D, seed=SEED, low=0.0, high=np.pi)

    if geometry == "Clustered":
        return clustered_data(n=N, d=D, seed=SEED, low=0.0, high=np.pi)

    raise ValueError(f"Unknown geometry: {geometry}")


def lowrank_curve(eigenvalues):
    """Compute relative Frobenius errors for truncated spectral approximations."""
    denominator = np.sum(eigenvalues ** 2)
    ranks = np.arange(1, len(eigenvalues) + 1)

    errors = [
        np.sqrt(np.sum(eigenvalues[r:] ** 2) / denominator)
        for r in ranks
    ]

    return pd.DataFrame({"rank": ranks, "relative_error": errors})


def run_geometry(geometry):
    """Evaluate all feature maps on shared inputs and export diagnostics."""
    output_dir = RESULTS_DIR / geometry.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    X = make_dataset(geometry)
    pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(D)]).to_csv(
        output_dir / "input_data.csv", index=False
    )

    data = {}
    summary_rows = []

    for name, entangler, depth in VARIANTS:
        K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)

        metrics = spectral_metrics(K)
        eigenvalues = metrics["eigenvalues"]
        normalised = eigenvalues / N

        upper = np.triu_indices(N, k=1)
        offdiag = K[upper]

        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert eigenvalues.min() >= -1e-10

        data[name] = {
            "K": K,
            "eigenvalues": eigenvalues,
            "normalised": normalised,
            "offdiag": offdiag,
        }

        safe_name = name.lower()
        pd.DataFrame(K).to_csv(
            output_dir / f"gram_{safe_name}.csv", index=False, header=False
        )
        pd.DataFrame(
            {
                "index": np.arange(1, len(eigenvalues) + 1),
                "eigenvalue": eigenvalues,
                "normalised_eigenvalue": normalised,
                "cumulative_weight": np.cumsum(normalised),
            }
        ).to_csv(output_dir / f"spectrum_{safe_name}.csv", index=False)
        pd.DataFrame({"fidelity": offdiag}).to_csv(
            output_dir / f"offdiag_{safe_name}.csv", index=False
        )

        curve = lowrank_curve(eigenvalues)
        curve.to_csv(output_dir / f"lowrank_{safe_name}.csv", index=False)

        summary_rows.append(
            {
                "geometry": geometry,
                "feature_map": name,
                "rank": metrics["rank"],
                "effective_rank": metrics["effective_rank"],
                "largest_weight": metrics["largest_weight"],
                "offdiag_mean": metrics["offdiag_mean"],
                "offdiag_std": metrics["offdiag_std"],
                "epsilon_3": metrics["low_rank_errors"].get(3),
                "epsilon_5": metrics["low_rank_errors"].get(5),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "parameter": [
                "input_geometry", "q", "d", "n", "seed",
                "entangling_depth", "encoding_interval",
            ],
            "value": [geometry, Q, D, N, SEED, 1, "[0, pi]"],
        }
    ).to_csv(output_dir / "settings.csv", index=False)

    plot_eigenvalue_decay(data, output_dir)
    plot_gram_heatmaps(data, output_dir)
    plot_offdiagonal_distribution(data, output_dir)
    plot_lowrank_error(data, output_dir)

    return summary


def plot_eigenvalue_decay(data, output_dir):
    """Plot normalised Gram-matrix eigenvalues on a log scale."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for name, values in data.items():
        indices = np.arange(1, len(values["normalised"]) + 1)
        ax.plot(
            indices,
            values["normalised"],
            marker=MARKERS[name],
            markersize=4,
            label=name,
        )

    ax.set_xlabel("Eigenvalue index $j$")
    ax.set_ylabel("Normalised eigenvalue $\\lambda_j/n$")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "eigenvalue_decay.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gram_heatmaps(data, output_dir):
    """Export one Gram-matrix heatmap for each feature-map family."""
    for name, values in data.items():
        fig, ax = plt.subplots(figsize=(5.0, 4.3))
        image = ax.imshow(values["K"], vmin=0, vmax=1, aspect="equal")

        ax.set_xlabel("Sample index $j$")
        ax.set_ylabel("Sample index $i$")
        ax.set_title(name)
        fig.colorbar(image, ax=ax, label="$K_{ij}$")

        fig.tight_layout()
        fig.savefig(
            output_dir / f"gram_heatmap_{name.lower()}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_offdiagonal_distribution(data, output_dir):
    """Compare the distributions of off-diagonal fidelity values."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.linspace(0.0, 1.0, 21)

    for name, values in data.items():
        ax.hist(
            values["offdiag"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )

    ax.set_xlabel("Off-diagonal fidelity $K_{ij}$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "offdiagonal_distribution.pdf", bbox_inches="tight"
    )
    plt.close(fig)


def plot_lowrank_error(data, output_dir):
    """Plot relative low-rank approximation errors by truncation rank."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for name, values in data.items():
        curve = lowrank_curve(values["eigenvalues"])
        ax.plot(
            curve["rank"],
            curve["relative_error"],
            marker=MARKERS[name],
            markersize=4,
            label=name,
        )

    ax.set_xlabel("Approximation rank $r$")
    ax.set_ylabel("Relative Frobenius error $\\varepsilon_r$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "lowrank_error.pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(summary):
    """Print a compact summary of spectral metrics for both input geometries."""
    for geometry in ("Uniform", "Clustered"):
        print(f"\n{geometry}")
        print("-" * len(geometry))
        subset = summary[summary["geometry"] == geometry]

        for feature_map in ("Separable", "CNOT", "CZ"):
            row = subset[subset["feature_map"] == feature_map].iloc[0]
            print(
                f"{feature_map:<9} "
                f"rank={int(row['rank']):2d}  "
                f"reff={row['effective_rank']:.4f}  "
                f"largest={row['largest_weight']:.4f}  "
                f"offdiag={row['offdiag_mean']:.4f}  "
                f"eps3={row['epsilon_3']:.4f}  "
                f"eps5={row['epsilon_5']:.4f}"
            )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running representative spectral diagnostics")
    print("-------------------------------------------")
    print(f"Qubits:           q = {Q}")
    print(f"Input dimension:  d = {D}")
    print(f"Samples:          n = {N}")
    print(f"Seed:             {SEED}")
    print("Entangling depth: L = 1")

    summaries = [run_geometry("Uniform"), run_geometry("Clustered")]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print_summary(summary)
    print("\nResults saved to: " f"{RESULTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
