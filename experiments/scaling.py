"""System-size and input-dimension scaling experiments.

This script reproduces the three controlled scaling experiments used in
Section 4.4 of the thesis:

1. Joint scaling of qubit number and input dimension:
       q = d in {2, 3, 4, 5, 6}

2. Input-dimension scaling at fixed system size:
       q = 6
       d in {1, 2, 3, 4, 5, 6}

3. Qubit-number scaling at fixed input dimension:
       d = 2
       q in {2, 3, 4, 5, 6}

All experiments use n = 25 uniformly sampled inputs, entangling depth L = 1
for the CNOT and CZ feature maps, and 20 fixed random seeds.

For configurations with d < q, only the first d qubits receive
sample-dependent input coordinates. The remaining qubits receive the fixed
rotation angle pi/2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qfk.data import uniform_data
from qfk.kernels import fidelity_gram_matrix
from qfk.spectral import spectral_metrics


# ---------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def embed_active_dimensions(
    X_active,
    q,
    fixed_angle=FIXED_ANGLE,
):
    """Embed d sample-dependent coordinates into a q-qubit input array."""
    n, d = X_active.shape

    if d > q:
        raise ValueError(
            "Input dimension d must not exceed the number of qubits q."
        )

    X = np.full(
        (n, q),
        fixed_angle,
        dtype=float,
    )

    X[:, :d] = X_active

    return X


def evaluate_configuration(X, q, d, seed):
    """Evaluate all three feature-map families on one input sample."""
    rows = []

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
    """Compute across-seed means and standard deviations."""
    return (
        df.groupby(
            [*grouping_columns, "feature_map"],
            sort=True,
        )
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


# ---------------------------------------------------------------------
# Experiment 1: joint scaling q = d
# ---------------------------------------------------------------------

def run_joint_scaling():
    """Vary qubit number and input dimension jointly."""
    rows = []

    for q in Q_VALUES:
        d = q

        for seed in SEEDS:
            X = uniform_data(
                n=N_SAMPLES,
                d=d,
                seed=seed,
                low=0.0,
                high=np.pi,
            )

            rows.extend(
                evaluate_configuration(
                    X=X,
                    q=q,
                    d=d,
                    seed=seed,
                )
            )

    raw = pd.DataFrame(rows)

    summary = summarise(
        raw,
        grouping_columns=["q", "d"],
    )

    return raw, summary


# ---------------------------------------------------------------------
# Experiment 2: varying d at fixed q = 6
# ---------------------------------------------------------------------

def run_fixed_qubit_scaling():
    """Vary the number of active input dimensions at fixed q = 6."""
    rows = []

    for d in D_VALUES:
        for seed in SEEDS:
            X_active = uniform_data(
                n=N_SAMPLES,
                d=d,
                seed=seed,
                low=0.0,
                high=np.pi,
            )

            X = embed_active_dimensions(
                X_active,
                q=Q_FIXED,
            )

            rows.extend(
                evaluate_configuration(
                    X=X,
                    q=Q_FIXED,
                    d=d,
                    seed=seed,
                )
            )

    raw = pd.DataFrame(rows)

    summary = summarise(
        raw,
        grouping_columns=["d", "q"],
    )

    return raw, summary


# ---------------------------------------------------------------------
# Experiment 3: varying q at fixed d = 2
# ---------------------------------------------------------------------

def run_fixed_dimension_scaling():
    """Vary the number of qubits at fixed input dimension d = 2."""
    rows = []

    for q in Q_VALUES:
        for seed in SEEDS:
            X_active = uniform_data(
                n=N_SAMPLES,
                d=D_FIXED,
                seed=seed,
                low=0.0,
                high=np.pi,
            )

            X = embed_active_dimensions(
                X_active,
                q=q,
            )

            rows.extend(
                evaluate_configuration(
                    X=X,
                    q=q,
                    d=D_FIXED,
                    seed=seed,
                )
            )

    raw = pd.DataFrame(rows)

    summary = summarise(
        raw,
        grouping_columns=["q", "d"],
    )

    return raw, summary


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def plot_joint_scaling(summary):
    """Plot effective rank for joint q = d scaling."""
    markers = {
        "Separable": "^",
        "CNOT": "o",
        "CZ": "s",
    }

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[
            summary["feature_map"] == name
        ].sort_values("q")

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

    plt.savefig(
        RESULTS_DIR / "effective_rank_joint_scaling.pdf",
        bbox_inches="tight",
    )

    plt.close()


def plot_fixed_qubit_scaling(summary):
    """Plot effective rank versus input dimension at fixed q = 6."""
    markers = {
        "Separable": "^",
        "CNOT": "o",
        "CZ": "s",
    }

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[
            summary["feature_map"] == name
        ].sort_values("d")

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

    plt.savefig(
        RESULTS_DIR / "effective_rank_fixed_q.pdf",
        bbox_inches="tight",
    )

    plt.close()


def plot_fixed_dimension_scaling(summary):
    """Plot effective rank versus qubit number at fixed d = 2."""
    markers = {
        "Separable": "^",
        "CNOT": "o",
        "CZ": "s",
    }

    plt.figure(figsize=(6.5, 4.2))

    for name in markers:
        data = summary[
            summary["feature_map"] == name
        ].sort_values("q")

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

    plt.savefig(
        RESULTS_DIR / "effective_rank_fixed_d.pdf",
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

def print_effective_rank_summary(
    title,
    summary,
    varying_column,
):
    """Print compact effective-rank results to the console."""
    print(f"\n{title}")
    print("=" * len(title))

    table = summary.pivot(
        index=varying_column,
        columns="feature_map",
        values="effective_rank_mean",
    )

    print(table.round(4))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Run all system-size and input-dimension scaling experiments."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Running scaling experiments")
    print("---------------------------")
    print(f"Samples per dataset: {N_SAMPLES}")
    print(f"Random seeds:        {len(list(SEEDS))}")
    print("Entangling depth:    L = 1")

    # Joint q = d scaling
    joint_raw, joint_summary = run_joint_scaling()

    joint_raw.to_csv(
        RESULTS_DIR / "joint_raw.csv",
        index=False,
    )

    joint_summary.to_csv(
        RESULTS_DIR / "joint_summary.csv",
        index=False,
    )

    print_effective_rank_summary(
        title="Joint scaling: q = d",
        summary=joint_summary,
        varying_column="q",
    )

    # Fixed q = 6, varying d
    fixed_q_raw, fixed_q_summary = run_fixed_qubit_scaling()

    fixed_q_raw.to_csv(
        RESULTS_DIR / "fixed_q_raw.csv",
        index=False,
    )

    fixed_q_summary.to_csv(
        RESULTS_DIR / "fixed_q_summary.csv",
        index=False,
    )

    print_effective_rank_summary(
        title="Input-dimension scaling: fixed q = 6",
        summary=fixed_q_summary,
        varying_column="d",
    )

    # Fixed d = 2, varying q
    fixed_d_raw, fixed_d_summary = run_fixed_dimension_scaling()

    fixed_d_raw.to_csv(
        RESULTS_DIR / "fixed_d_raw.csv",
        index=False,
    )

    fixed_d_summary.to_csv(
        RESULTS_DIR / "fixed_d_summary.csv",
        index=False,
    )

    print_effective_rank_summary(
        title="Qubit scaling: fixed d = 2",
        summary=fixed_d_summary,
        varying_column="q",
    )

    # Figures
    plot_joint_scaling(joint_summary)
    plot_fixed_qubit_scaling(fixed_q_summary)
    plot_fixed_dimension_scaling(fixed_d_summary)

    print(
        "\nResults saved to: "
        f"{RESULTS_DIR.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
