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

    The effective rank is the participation ratio of the eigenvalue spectrum,
    ``(sum(lambda))**2 / sum(lambda**2)``. The rank-r approximation error is
    the relative Frobenius norm of the eigenvalue tail, as appropriate for a
    positive semidefinite Gram matrix.

    Parameters
    ----------
    K : numpy.ndarray
        Square, symmetric, positive semidefinite kernel Gram matrix of shape
        ``(n, n)``. For a fidelity kernel, diagonal entries equal one.
    rank_tol : float, optional
        Absolute threshold for identifying numerically zero eigenvalues and
        calculating the numerical rank. Defaults to ``1e-10``.
    low_rank_values : iterable of int, optional
        Approximation ranks for which to calculate relative Frobenius errors.
        Values exceeding the number of samples are skipped.

    Returns
    -------
    dict
        ``n`` : int
            Number of samples.
        ``rank`` : int
            Number of eigenvalues exceeding ``rank_tol``.
        ``effective_rank`` : float
            Participation ratio of the eigenvalue spectrum.
        ``eigenvalues`` : numpy.ndarray
            Eigenvalues in descending order after numerical zeroing.
        ``normalised_eigenvalues`` : numpy.ndarray
            Eigenvalues divided by the sample count ``n``.
        ``spectral_weights`` : numpy.ndarray
            Eigenvalues divided by their sum.
        ``largest_weight`` : float
            Fraction of the spectrum carried by the largest eigenvalue.
        ``cumulative_weights`` : numpy.ndarray
            Cumulative sums of the spectral weights.
        ``low_rank_errors`` : dict[int, float]
            Relative Frobenius errors for the specified approximation ranks.
        ``offdiag_mean``, ``offdiag_std`` : float
            Mean and standard deviation of upper-triangular off-diagonal
            kernel entries.

    Notes
    -----
    The matrix is assumed to be a valid, nonzero Gram matrix; the routine does
    not validate its shape or positive semidefiniteness. Off-diagonal statistics
    require at least two samples.
    """

    n = K.shape[0]

    eigenvalues = np.linalg.eigvalsh(K)[::-1]

    # Suppress eigenvalues indistinguishable from zero at the chosen tolerance.
    eigenvalues[
        np.abs(eigenvalues) < rank_tol
    ] = 0.0

    trace = np.sum(eigenvalues)

    numerical_rank = np.sum(
        eigenvalues > rank_tol
    )

    # Participation ratio of the eigenvalue spectrum.
    effective_rank = (
        trace ** 2
        /
        np.sum(eigenvalues ** 2)
    )

    spectral_weights = eigenvalues / trace

    largest_weight = spectral_weights[0]

    cumulative_weights = np.cumsum(
        spectral_weights
    )

    # Relative Frobenius errors of truncated spectral approximations.
    denominator = np.sum(
        eigenvalues ** 2
    )

    low_rank_errors = {}

    for r in low_rank_values:

        if r <= n:

            residual = np.sum(
                eigenvalues[r:] ** 2
            )

            low_rank_errors[r] = np.sqrt(
                residual / denominator
            )

    # Statistics over distinct pairs of input samples.
    upper = np.triu_indices(
        n,
        k=1
    )

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
