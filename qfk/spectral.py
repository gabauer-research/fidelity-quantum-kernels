"""Spectral diagnostics for fidelity quantum-kernel Gram matrices.

This module computes eigenvalue-based measures of matrix rank, spectral
concentration, and low-rank approximation quality, together with basic
statistics of off-diagonal kernel entries.
"""

import numpy as np


def spectral_metrics(
    K,
    rank_tol=1e-10,
    low_rank_values=(1, 2, 3, 4, 5, 6)
):
    """Compute spectral diagnostics for a kernel Gram matrix.

    Effective rank is the participation ratio,
    ``(sum(lambda))**2 / sum(lambda**2)``. Low-rank errors are relative
    Frobenius norms of the eigenvalue tails.

    Parameters
    ----------
    K : numpy.ndarray
        Symmetric positive semidefinite Gram matrix of shape ``(n, n)``.
    rank_tol : float, optional
        Absolute eigenvalue threshold for numerical rank (default ``1e-10``).
    low_rank_values : iterable of int, optional
        Approximation ranks; values greater than ``n`` are skipped.

    Returns
    -------
    dict
        ``n``, ``rank``, ``effective_rank``, ``eigenvalues``,
        ``normalised_eigenvalues``, ``spectral_weights``, ``largest_weight``,
        ``cumulative_weights``, ``low_rank_errors``, ``offdiag_mean``, and
        ``offdiag_std``.

    Notes
    -----
    Eigenvalues are in descending order and values with absolute magnitude
    below ``rank_tol`` are set to zero. ``normalised_eigenvalues`` is scaled
    by ``n``; ``spectral_weights`` is scaled by the eigenvalue sum.
    Assumes a valid, nonzero Gram matrix with at least two samples.
    """

    n = K.shape[0]
    eigenvalues = np.linalg.eigvalsh(K)[::-1]

    # Suppress eigenvalues indistinguishable from zero at the chosen tolerance.
    eigenvalues[np.abs(eigenvalues) < rank_tol] = 0.0

    trace = np.sum(eigenvalues)
    numerical_rank = np.sum(eigenvalues > rank_tol)

    # Participation ratio of the eigenvalue spectrum.
    effective_rank = trace**2 / np.sum(eigenvalues**2)
    spectral_weights = eigenvalues / trace
    largest_weight = spectral_weights[0]
    cumulative_weights = np.cumsum(spectral_weights)

    # Relative Frobenius errors of truncated spectral approximations.
    denominator = np.sum(eigenvalues**2)
    low_rank_errors = {}
    for r in low_rank_values:
        if r <= n:
            residual = np.sum(eigenvalues[r:]**2)
            low_rank_errors[r] = np.sqrt(residual / denominator)

    # Statistics over distinct pairs of input samples.
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
