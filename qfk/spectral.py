"""Spectral diagnostics for finite fidelity quantum-kernel Gram matrices."""

import numpy as np


def spectral_metrics(
    K,
    rank_tol=1e-10,
    low_rank_values=(1, 2, 3, 4, 5, 6)
):
    """Compute spectral and low-rank diagnostics of a fidelity Gram matrix.

    Effective rank is the participation ratio,
    ``(sum(lambda))**2 / sum(lambda**2)``. Low-rank errors are relative
    Frobenius norms of truncated eigenvalue tails.

    Parameters
    ----------
    K : numpy.ndarray
        Symmetric positive semidefinite Gram matrix of shape ``(n, n)``.
    rank_tol : float, optional
        Absolute eigenvalue tolerance (default: ``1e-10``).
    low_rank_values : iterable of int, optional
        Approximation ranks; values greater than ``n`` are skipped.

    Returns
    -------
    dict
        Spectral quantities, low-rank errors, and off-diagonal statistics.

    Notes
    -----
    Eigenvalues are sorted in descending order. Values with absolute
    magnitude below ``rank_tol`` are set to zero; numerical rank counts
    eigenvalues greater than ``rank_tol``. ``normalised_eigenvalues`` uses
    division by ``n``; ``spectral_weights`` uses division by the trace.
    Assumes a nonzero Gram matrix with at least two samples.
    """

    n = K.shape[0]
    eigenvalues = np.linalg.eigvalsh(K)[::-1]

    # Suppress eigenvalues indistinguishable from zero at the chosen tolerance.
    eigenvalues[np.abs(eigenvalues) < rank_tol] = 0.0

    trace = np.sum(eigenvalues)
    numerical_rank = np.sum(eigenvalues > rank_tol)

    effective_rank = trace**2 / np.sum(eigenvalues**2)
    spectral_weights = eigenvalues / trace
    largest_weight = spectral_weights[0]
    cumulative_weights = np.cumsum(spectral_weights)

    denominator = np.sum(eigenvalues**2)
    low_rank_errors = {}
    for r in low_rank_values:
        if r <= n:
            residual = np.sum(eigenvalues[r:]**2)
            low_rank_errors[r] = np.sqrt(residual / denominator)

    upper = np.triu_indices(n, k=1)
    off_diagonal = K[upper]
    offdiag_mean = np.mean(off_diagonal)
    offdiag_std = np.std(off_diagonal)

    return {
        "n": n,
        "rank": int(numerical_rank),
        "effective_rank": effective_rank,
        "eigenvalues": eigenvalues,
        "normalised_eigenvalues": eigenvalues / n,
        "spectral_weights": spectral_weights,
        "largest_weight": largest_weight,
        "cumulative_weights": cumulative_weights,
        "low_rank_errors": low_rank_errors,
        "offdiag_mean": offdiag_mean,
        "offdiag_std": offdiag_std
    }
