"""Spectral analysis of fidelity quantum kernels across input geometries.

Compare uniform, Gaussian, clustered, and correlated inputs using separable,
CNOT, and CZ feature maps. Each of 20 seeds supplies the same samples to all
circuit variants (q = d = 4, n = 25, L = 1 for entangling maps). Representative
two-dimensional input samples are also exported for visualisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import uniform_data, clustered_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


N_SAMPLES = 25
Q = 4
D = 4
SEEDS = range(20)

ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

CIRCUIT_VARIANTS = (
    ("Separable", None, 0),
    ("CNOT", "cnot", 1),
    ("CZ", "cz", 1),
)

GEOMETRY_ORDER = (
    "Clustered",
    "Correlated",
    "Gaussian",
    "Uniform",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "input_geometry"


def gaussian_data(n, d, seed=42, low=0.0, high=np.pi, std_fraction=1 / 6):
    """Sample independent Gaussian coordinates, clipped to ``[low, high]``."""
    rng = np.random.default_rng(seed)

    mean = (low + high) / 2
    std = std_fraction * (high - low)

    X = rng.normal(loc=mean, scale=std, size=(n, d))

    return np.clip(X, low, high)


def correlated_data(n, d, seed=42, low=0.0, high=np.pi, relative_noise=0.05):
    """Sample noisy, correlated coordinates around the main diagonal."""
    rng = np.random.default_rng(seed)

    span = high - low

    z = rng.uniform(low=low, high=high, size=(n, 1))

    X = np.repeat(z, d, axis=1)

    X += rng.normal(loc=0.0, scale=relative_noise * span, size=(n, d))

    return np.clip(X, low, high)


def make_datasets(n, d, seed):
    """Generate all four input geometries for one random seed."""
    return {
        "Uniform": uniform_data(
            n=n, d=d, seed=seed, low=ANGLE_MIN, high=ANGLE_MAX
        ),
        "Gaussian": gaussian_data(
            n=n, d=d, seed=seed, low=ANGLE_MIN, high=ANGLE_MAX
        ),
        "Clustered": clustered_data(
            n=n, d=d, seed=seed, low=ANGLE_MIN, high=ANGLE_MAX
        ),
        "Correlated": correlated_data(
            n=n, d=d, seed=seed, low=ANGLE_MIN, high=ANGLE_MAX
        ),
    }


def evaluate_configuration(X, seed, geometry):
    """Evaluate all feature-map families on the same classical input sample."""
    rows = []

    for feature_map, entangler, depth in CIRCUIT_VARIANTS:
        K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)
        metrics = spectral_metrics(K)

        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(K).min() >= -1e-10

        rows.append(
            {
                "seed": seed,
                "geometry": geometry,
                "feature_map": feature_map,
                "rank": metrics["rank"],
                "effective_rank": metrics["effective_rank"],
                "largest_weight": metrics["largest_weight"],
                "offdiag_mean": metrics["offdiag_mean"],
                "offdiag_std": metrics["offdiag_std"],
                "epsilon_3": metrics["low_rank_errors"].get(3),
                "epsilon_5": metrics["low_rank_errors"].get(5),
            }
        )

    return rows


def run_geometry_experiment():
    """Compute per-seed metrics and aggregate them by geometry and feature map."""
    rows = []

    for seed in SEEDS:
        datasets = make_datasets(n=N_SAMPLES, d=D, seed=seed)

        for geometry, X in datasets.items():
            rows.extend(evaluate_configuration(X=X, seed=seed, geometry=geometry))

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(["geometry", "feature_map"], sort=True)
        .agg(
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            effective_rank_mean=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            largest_weight_mean=("largest_weight", "mean"),
            largest_weight_std=("largest_weight", "std"),
            offdiag_mean_mean=("offdiag_mean", "mean"),
            offdiag_mean_std=("offdiag_mean", "std"),
            offdiag_std_mean=("offdiag_std", "mean"),
            offdiag_std_std=("offdiag_std", "std"),
            epsilon_3_mean=("epsilon_3", "mean"),
            epsilon_3_std=("epsilon_3", "std"),
            epsilon_5_mean=("epsilon_5", "mean"),
            epsilon_5_std=("epsilon_5", "std"),
        )
        .reset_index()
    )

    return raw, summary


def plot_effective_rank(summary):
    """Plot effective rank by input geometry and feature-map family."""
    x = np.arange(len(GEOMETRY_ORDER))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for offset, feature_map in zip(
        (-width, 0.0, width), ("Separable", "CNOT", "CZ")
    ):
        means = []
        stds = []

        for geometry in GEOMETRY_ORDER:
            row = summary[
                (summary["geometry"] == geometry)
                & (summary["feature_map"] == feature_map)
            ].iloc[0]

            means.append(row["effective_rank_mean"])
            stds.append(row["effective_rank_std"])

        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=feature_map)

    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_ORDER)
    ax.set_xlabel("Input geometry")
    ax.set_ylabel("Effective rank")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "effective_rank_vs_input_geometry.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def export_representative_inputs():
    """Export and plot representative two-dimensional samples with seed 42."""
    seed = 42
    n = 100
    d = 2

    datasets = make_datasets(n=n, d=d, seed=seed)

    for geometry, X in datasets.items():
        safe_name = geometry.lower()

        pd.DataFrame(X, columns=["x1", "x2"]).to_csv(
            RESULTS_DIR / f"input_{safe_name}.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(4.5, 4.0))

        ax.scatter(X[:, 0], X[:, 1], s=20)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, np.pi)
        ax.set_title(geometry)

        fig.tight_layout()

        fig.savefig(RESULTS_DIR / f"input_{safe_name}.pdf", bbox_inches="tight")

        plt.close(fig)


def print_effective_rank_summary(summary):
    """Print effective-rank means and standard deviations."""
    print("\nEffective rank by input geometry")
    print("================================")

    for geometry in GEOMETRY_ORDER:
        print(f"\n{geometry}")

        data = summary[summary["geometry"] == geometry]

        for feature_map in ("Separable", "CNOT", "CZ"):
            row = data[data["feature_map"] == feature_map].iloc[0]

            mean = row["effective_rank_mean"]
            std = row["effective_rank_std"]

            print(f"  {feature_map:<9} " f"{mean:.4f} +/- {std:.4f}")


def main():
    """Run the input-geometry study and save results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running input-geometry experiment")
    print("---------------------------------")
    print(f"Qubits:              q = {Q}")
    print(f"Input dimension:     d = {D}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print(f"Random seeds:        {len(list(SEEDS))}")
    print("Entangling depth:    L = 1")

    raw, summary = run_geometry_experiment()

    raw.to_csv(RESULTS_DIR / "geometry_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "geometry_summary.csv", index=False)

    plot_effective_rank(summary)
    export_representative_inputs()

    print_effective_rank_summary(summary)

    print("\nResults saved to: " f"{RESULTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
