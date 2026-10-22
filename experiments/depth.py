"""Depth-dependent spectra of statevector-based fidelity quantum kernels.

Compare a separable feature map with CNOT- and CZ-entangling feature maps at
entangling depths one through five. The study uses a fixed Cartesian grid and
an ensemble of uniformly sampled datasets to assess the sensitivity of the
spectral diagnostics to input sampling.

The same input dataset is used for all circuit variants within each experiment
or random seed. The separable architecture at depth zero provides a reference
rather than an entangling-depth measurement.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import cartesian_grid, uniform_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


# Experiment settings
POINTS_PER_DIM = 5
INPUT_DIMENSION = 2
N_SAMPLES = 25
DEPTHS = range(1, 6)
SEEDS = range(20)
ENTANGLERS = ("cnot", "cz")

# Resolve output paths relative to the repository, not the working directory.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "depth"


def circuit_variants():
    """Yield the separable reference and all entangling circuit variants."""
    yield "Separable", None, 0

    for entangler in ENTANGLERS:
        for depth in DEPTHS:
            yield entangler.upper(), entangler, depth


def evaluate_variant(X, name, entangler, depth, seed=None):
    """Evaluate one kernel and return its spectral metrics and eigenvalues."""
    K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)
    metrics = spectral_metrics(K)

    row = {
        "entangler": name,
        "depth": depth,
        "rank": metrics["rank"],
        "effective_rank": metrics["effective_rank"],
        "largest_weight": metrics["largest_weight"],
        "offdiag_mean": metrics["offdiag_mean"],
        "offdiag_std": metrics["offdiag_std"],
    }

    if seed is not None:
        row = {"seed": seed, **row}

    return row, metrics


def grid_depth_study():
    """Compute depth-dependent metrics and spectra on a 5-by-5 input grid.

    Returns
    -------
    pandas.DataFrame
        One row for each of the eleven circuit configurations.
    pandas.DataFrame
        Eigenvalue spectra in long format, indexed by configuration and
        descending eigenvalue position.
    """
    X = cartesian_grid(points_per_dim=POINTS_PER_DIM, d=INPUT_DIMENSION)
    rows = []
    spectrum_rows = []

    for name, entangler, depth in circuit_variants():
        row, metrics = evaluate_variant(X, name, entangler, depth)
        row.update({
            f"epsilon_{r}": metrics["low_rank_errors"].get(r)
            for r in (1, 2, 3, 5)
        })
        rows.append(row)

        for position, eigenvalue in enumerate(metrics["eigenvalues"], start=1):
            spectrum_rows.append({
                "entangler": name,
                "depth": depth,
                "index": position,
                "eigenvalue": eigenvalue,
            })

    return pd.DataFrame(rows), pd.DataFrame(spectrum_rows)


def random_depth_study():
    """Repeat the depth comparison on twenty seeded uniform datasets.

    For each seed, the same 25 two-dimensional inputs are evaluated for the
    separable reference and all entangling configurations. Standard deviations
    use pandas' sample standard deviation (``ddof=1``).

    Returns
    -------
    pandas.DataFrame
        Metrics for every seed and circuit configuration.
    pandas.DataFrame
        Mean and standard deviation grouped by circuit type and depth.
    """
    rows = []

    for seed in SEEDS:
        X = uniform_data(
            n=N_SAMPLES,
            d=INPUT_DIMENSION,
            seed=seed,
            low=0.0,
            high=np.pi,
        )

        for name, entangler, depth in circuit_variants():
            row, metrics = evaluate_variant(
                X, name, entangler, depth, seed=seed
            )
            row.update({
                f"epsilon_{r}": metrics["low_rank_errors"].get(r)
                for r in (3, 5)
            })
            rows.append(row)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["entangler", "depth"])
        .agg(
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            effective_rank_mean=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            largest_weight_mean=("largest_weight", "mean"),
            largest_weight_std=("largest_weight", "std"),
            offdiag_mean_mean=("offdiag_mean", "mean"),
            offdiag_mean_std=("offdiag_mean", "std"),
            epsilon_3_mean=("epsilon_3", "mean"),
            epsilon_3_std=("epsilon_3", "std"),
            epsilon_5_mean=("epsilon_5", "mean"),
            epsilon_5_std=("epsilon_5", "std"),
        )
        .reset_index()
    )

    return raw, summary


def plot_grid_metrics(raw, output_dir):
    """Save depth-dependent effective-rank and spectral-weight figures."""
    baseline = raw.loc[raw["entangler"] == "Separable"].iloc[0]

    plot_specs = (
        ("effective_rank", "Effective rank", "effective_rank_vs_depth.pdf"),
        (
            "largest_weight",
            "Largest spectral weight",
            "largest_weight_vs_depth.pdf",
        ),
    )

    for column, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        for name, marker in (("CNOT", "o"), ("CZ", "s")):
            subset = raw.loc[raw["entangler"] == name]
            ax.plot(subset["depth"], subset[column], marker=marker, label=name)

        ax.axhline(
            baseline[column], linestyle="--", label="Separable reference"
        )
        ax.set(xlabel=r"Entangling depth $L$", ylabel=ylabel)
        ax.set_xticks(list(DEPTHS))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_random_metrics(summary, output_dir):
    """Save across-seed mean curves with standard-deviation error bars."""
    baseline = summary.loc[
        summary["entangler"] == "Separable"
    ].iloc[0]

    plot_specs = (
        (
            "effective_rank",
            "Effective rank",
            "effective_rank_vs_depth_random.pdf",
        ),
        (
            "largest_weight",
            "Largest spectral weight",
            "largest_weight_vs_depth_random.pdf",
        ),
    )

    for metric, ylabel, filename in plot_specs:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        for name, marker in (("CNOT", "o"), ("CZ", "s")):
            subset = summary.loc[summary["entangler"] == name]
            ax.errorbar(
                subset["depth"],
                subset[mean_key],
                yerr=subset[std_key],
                marker=marker,
                capsize=3,
                label=name,
            )

        mean = baseline[mean_key]
        std = baseline[std_key]
        ax.axhline(mean, linestyle="--", label="Separable mean")
        ax.axhspan(mean - std, mean + std, alpha=0.15)
        ax.set(xlabel=r"Entangling depth $L$", ylabel=ylabel)
        ax.set_xticks(list(DEPTHS))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def main():
    """Run both depth studies and save their numerical and visual outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_raw, grid_spectra = grid_depth_study()
    grid_raw.to_csv(OUTPUT_DIR / "depth_raw.csv", index=False)
    grid_spectra.to_csv(OUTPUT_DIR / "depth_spectra.csv", index=False)
    plot_grid_metrics(grid_raw, OUTPUT_DIR)

    random_raw, random_summary = random_depth_study()
    random_raw.to_csv(OUTPUT_DIR / "depth_random_raw.csv", index=False)
    random_summary.to_csv(OUTPUT_DIR / "depth_random_summary.csv", index=False)
    plot_random_metrics(random_summary, OUTPUT_DIR)

    print("\nCartesian grid depth study:")
    print(grid_raw.round(4).to_string(index=False))
    print("\nUniform random depth study (20 seeds):")
    print(random_summary.round(4).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
