"""Spectral scaling experiments for fidelity quantum kernels.

Compare joint qubit/input-dimension scaling, input-dimension scaling at
fixed qubit number, and qubit-number scaling at fixed input dimension.
All experiments use 25 uniform samples and 20 fixed seeds. Inactive input
coordinates are assigned pi/2; CNOT and CZ maps use one entangling layer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import uniform_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


N_SAMPLES = 25
SEEDS = range(20)

Q_VALUES = range(2, 7)
D_VALUES = range(1, 7)

Q_FIXED = 6
D_FIXED = 2

FIXED_ANGLE = np.pi / 2

CIRCUIT_VARIANTS = (
    ("Separable", None, 0),
    ("CNOT", "cnot", 1),
    ("CZ", "cz", 1),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "scaling"


def embed_active_dimensions(X_active, q, fixed_angle=FIXED_ANGLE):
    """Fill inactive qubit coordinates with a fixed encoding angle."""
    n, d = X_active.shape

    if d > q:
        raise ValueError("Input dimension d must not exceed the number of qubits q.")

    X = np.full((n, q), fixed_angle, dtype=float)
    X[:, :d] = X_active

    return X


def evaluate_configuration(X, q, d, seed):
    """Compute spectral diagnostics for each feature map on shared inputs."""
    rows = []

    for name, entangler, depth in CIRCUIT_VARIANTS:
        K = fidelity_gram_matrix(X, entangler=entangler, depth=depth)
        metrics = spectral_metrics(K)

        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(K).min() >= -1e-10

        rows.append(
            {
                "q": q,
                "d": d,
                "seed": seed,
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

    return rows


def summarise(df, grouping_columns):
    """Aggregate spectral metrics as means and standard deviations over seeds."""
    return (
        df.groupby([*grouping_columns, "feature_map"], sort=True)
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


def run_joint_scaling():
    """Vary qubit number and input dimension jointly (q = d)."""
    rows = []

    for q in Q_VALUES:
        d = q

        for seed in SEEDS:
            X = uniform_data(n=N_SAMPLES, d=d, seed=seed, low=0.0, high=np.pi)
            rows.extend(evaluate_configuration(X=X, q=q, d=d, seed=seed))

    raw = pd.DataFrame(rows)
    summary = summarise(raw, grouping_columns=["q", "d"])

    return raw, summary


def run_fixed_qubit_scaling():
    """Vary active input dimension at fixed q = 6."""
    rows = []

    for d in D_VALUES:
        for seed in SEEDS:
            X_active = uniform_data(n=N_SAMPLES, d=d, seed=seed, low=0.0, high=np.pi)
            X = embed_active_dimensions(X_active, q=Q_FIXED)
            rows.extend(evaluate_configuration(X=X, q=Q_FIXED, d=d, seed=seed))

    raw = pd.DataFrame(rows)
    summary = summarise(raw, grouping_columns=["d", "q"])

    return raw, summary


def run_fixed_dimension_scaling():
    """Vary qubit number at fixed input dimension d = 2."""
    rows = []

    for q in Q_VALUES:
        for seed in SEEDS:
            X_active = uniform_data(
                n=N_SAMPLES, d=D_FIXED, seed=seed, low=0.0, high=np.pi
            )
            X = embed_active_dimensions(X_active, q=q)
            rows.extend(evaluate_configuration(X=X, q=q, d=D_FIXED, seed=seed))

    raw = pd.DataFrame(rows)
    summary = summarise(raw, grouping_columns=["q", "d"])

    return raw, summary


def plot_joint_scaling(summary):
    """Plot effective rank under joint q = d scaling."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[summary["feature_map"] == name].sort_values("q")
        plt.plot(
            data["q"],
            data["effective_rank_mean"],
            marker=markers[name],
            label=name,
        )

    plt.xlabel("Number of qubits / input dimension $q=d$")
    plt.ylabel("Effective rank $r_{\\mathrm{eff}}$")
    plt.xticks(list(Q_VALUES))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "effective_rank_joint_scaling.pdf", bbox_inches="tight")
    plt.close()


def plot_fixed_qubit_scaling(summary):
    """Plot effective rank versus active input dimension at fixed q = 6."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[summary["feature_map"] == name].sort_values("d")
        plt.plot(
            data["d"],
            data["effective_rank_mean"],
            marker=markers[name],
            label=name,
        )

    plt.xlabel("Input dimension $d$")
    plt.ylabel("Effective rank $r_{\\mathrm{eff}}$")
    plt.xticks(list(D_VALUES))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "effective_rank_fixed_q.pdf", bbox_inches="tight")
    plt.close()


def plot_fixed_dimension_scaling(summary):
    """Plot effective rank versus qubit number at fixed d = 2."""
    markers = {"Separable": "^", "CNOT": "o", "CZ": "s"}

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[summary["feature_map"] == name].sort_values("q")
        plt.plot(
            data["q"],
            data["effective_rank_mean"],
            marker=markers[name],
            label=name,
        )

    plt.xlabel("Number of qubits $q$")
    plt.ylabel("Effective rank $r_{\\mathrm{eff}}$")
    plt.xticks(list(Q_VALUES))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "effective_rank_fixed_d.pdf", bbox_inches="tight")
    plt.close()


def print_effective_rank_summary(title, summary, varying_column):
    """Print mean effective ranks for the three feature maps."""
    print(f"\n{title}")
    print("=" * len(title))

    table = summary.pivot(
        index=varying_column,
        columns="feature_map",
        values="effective_rank_mean",
    )

    print(table.round(4))


def main():
    """Run the scaling studies and save tables and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running scaling experiments")
    print("---------------------------")
    print(f"Samples per dataset: {N_SAMPLES}")
    print(f"Random seeds:        {len(list(SEEDS))}")
    print("Entangling depth:    L = 1")

    joint_raw, joint_summary = run_joint_scaling()
    joint_raw.to_csv(RESULTS_DIR / "joint_raw.csv", index=False)
    joint_summary.to_csv(RESULTS_DIR / "joint_summary.csv", index=False)
    print_effective_rank_summary(
        title="Joint scaling: q = d",
        summary=joint_summary,
        varying_column="q",
    )

    fixed_q_raw, fixed_q_summary = run_fixed_qubit_scaling()
    fixed_q_raw.to_csv(RESULTS_DIR / "fixed_q_raw.csv", index=False)
    fixed_q_summary.to_csv(RESULTS_DIR / "fixed_q_summary.csv", index=False)
    print_effective_rank_summary(
        title="Input-dimension scaling: fixed q = 6",
        summary=fixed_q_summary,
        varying_column="d",
    )

    fixed_d_raw, fixed_d_summary = run_fixed_dimension_scaling()
    fixed_d_raw.to_csv(RESULTS_DIR / "fixed_d_raw.csv", index=False)
    fixed_d_summary.to_csv(RESULTS_DIR / "fixed_d_summary.csv", index=False)
    print_effective_rank_summary(
        title="Qubit scaling: fixed d = 2",
        summary=fixed_d_summary,
        varying_column="q",
    )

    plot_joint_scaling(joint_summary)
    plot_fixed_qubit_scaling(fixed_q_summary)
    plot_fixed_dimension_scaling(fixed_d_summary)

    print(
        "\nResults saved to: "
        f"{RESULTS_DIR.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
