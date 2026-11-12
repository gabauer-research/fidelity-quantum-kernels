"""Sample-size dependence of fidelity quantum-kernel spectra.

Compare separable, CNOT, and CZ feature maps for uniformly sampled inputs
at q = d = 4 and n in {10, 25, 50, 75, 100, 125, 150, 200}. The entangling
maps use depth L = 1. Each of 20 fixed seeds supplies the same inputs to
all three feature-map families at a given sample size.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import uniform_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


Q = 4
D = 4

N_VALUES = (10, 25, 50, 75, 100, 125, 150, 200)
SEEDS = range(20)

CIRCUIT_VARIANTS = (
    ("Separable", None, 0),
    ("CNOT", "cnot", 1),
    ("CZ", "cz", 1),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "sample_size"


def evaluate_configuration(X, n, seed):
    """Compute spectral diagnostics for all feature maps on shared inputs."""
    rows = []

    for feature_map, entangler, depth in CIRCUIT_VARIANTS:
        K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)
        metrics = spectral_metrics(K)

        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.min(metrics["eigenvalues"]) >= -1e-10

        rows.append(
            {
                "n": n,
                "seed": seed,
                "feature_map": feature_map,
                "rank": metrics["rank"],
                "effective_rank": metrics["effective_rank"],
                "relative_effective_rank": metrics["effective_rank"] / n,
                "largest_weight": metrics["largest_weight"],
                "offdiag_mean": metrics["offdiag_mean"],
                "offdiag_std": metrics["offdiag_std"],
                "epsilon_3": metrics["low_rank_errors"].get(3),
                "epsilon_5": metrics["low_rank_errors"].get(5),
            }
        )

    return rows


def run_sample_size_experiment():
    """Evaluate all sample sizes and summarise metrics across 20 seeds."""
    rows = []

    for n in N_VALUES:
        for seed in SEEDS:
            X = uniform_data(n=n, d=D, seed=seed, low=0.0, high=np.pi)
            rows.extend(evaluate_configuration(X=X, n=n, seed=seed))

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(["feature_map", "n"], sort=True)
        .agg(
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            effective_rank_mean=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            relative_effective_rank_mean=("relative_effective_rank", "mean"),
            relative_effective_rank_std=("relative_effective_rank", "std"),
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


def plot_rank(summary):
    """Plot mean numerical rank against sample size."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for feature_map, marker in markers.items():
        data = summary[summary["feature_map"] == feature_map].sort_values("n")
        ax.plot(data["n"], data["rank_mean"], marker=marker, label=feature_map)

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Numerical rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rank_vs_sample_size.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_effective_rank(summary):
    """Plot mean effective rank with standard deviations across seeds."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for feature_map, marker in markers.items():
        data = summary[summary["feature_map"] == feature_map].sort_values("n")
        ax.errorbar(
            data["n"],
            data["effective_rank_mean"],
            yerr=data["effective_rank_std"],
            marker=marker,
            capsize=3,
            label=feature_map,
        )

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Effective rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "effective_rank_vs_sample_size.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_relative_effective_rank(summary):
    """Plot effective rank divided by sample size, including error bars."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for feature_map, marker in markers.items():
        data = summary[summary["feature_map"] == feature_map].sort_values("n")
        ax.errorbar(
            data["n"],
            data["relative_effective_rank_mean"],
            yerr=data["relative_effective_rank_std"],
            marker=marker,
            capsize=3,
            label=feature_map,
        )

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Relative effective rank $r_{\\mathrm{eff}}/n$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "relative_effective_rank_vs_sample_size.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def print_effective_rank_summary(summary):
    """Print effective-rank means and standard deviations by sample size."""
    print("\nEffective rank by sample size")
    print("=============================")

    for n in N_VALUES:
        print(f"\nn = {n}")
        data = summary[summary["n"] == n]

        for feature_map in ("Separable", "CNOT", "CZ"):
            row = data[data["feature_map"] == feature_map].iloc[0]
            mean = row["effective_rank_mean"]
            std = row["effective_rank_std"]
            print(
                f"  {feature_map:<9} "
                f"{mean:.4f} +/- {std:.4f}"
            )


def print_rank_summary(summary):
    """Print mean numerical ranks for each feature map and sample size."""
    print("\nNumerical rank by sample size")
    print("=============================")

    table = summary.pivot(index="n", columns="feature_map", values="rank_mean")
    table = table[["Separable", "CNOT", "CZ"]]
    print(table.round(4))


def main():
    """Run the study and export numerical summaries and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running sample-size experiment")
    print("------------------------------")
    print(f"Qubits:              q = {Q}")
    print(f"Input dimension:     d = {D}")
    print(f"Sample sizes:        {list(N_VALUES)}")
    print(f"Random seeds:        {len(list(SEEDS))}")
    print("Entangling depth:    L = 1")

    raw, summary = run_sample_size_experiment()

    raw.to_csv(RESULTS_DIR / "sample_size_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "sample_size_summary.csv", index=False)

    plot_rank(summary)
    plot_effective_rank(summary)
    plot_relative_effective_rank(summary)

    print_effective_rank_summary(summary)
    print_rank_summary(summary)

    print(
        "\nResults saved to: "
        f"{RESULTS_DIR.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
