"""Synthetic input distributions for fidelity quantum-kernel experiments.

Provides deterministic sampling routines and dimension embeddings for controlled
studies of kernel spectra, effective rank, and input-data geometry.
"""

import numpy as np


def cartesian_grid(points_per_dim, d, low=0.0, high=np.pi):
    """Construct a regular Cartesian grid over a bounded input domain.

    Parameters
    ----------
    points_per_dim : int
        Number of equally spaced grid points along each coordinate axis.
    d : int
        Number of input dimensions.
    low, high : float
        Lower and upper bounds of the grid on each coordinate axis.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(points_per_dim**d, d)`` containing all grid points.
    """
    axes = [
        np.linspace(low, high, points_per_dim)
        for _ in range(d)
    ]

    mesh = np.meshgrid(*axes, indexing="ij")

    X = np.stack(
        [axis.ravel() for axis in mesh],
        axis=1
    )

    return X


def uniform_data(n, d, seed=42, low=0.0, high=np.pi):
    """Sample input vectors uniformly from a bounded hyperrectangle.

    Parameters
    ----------
    n : int
        Number of input samples.
    d : int
        Number of input dimensions.
    seed : int
        Random seed for reproducible sampling.
    low, high : float
        Lower and upper bounds for each coordinate.

    Returns
    -------
    numpy.ndarray
        Uniformly distributed input samples of shape ``(n, d)``.
    """
    rng = np.random.default_rng(seed)

    X = rng.uniform(
        low=low,
        high=high,
        size=(n, d)
    )

    return X


def clustered_data(
    n,
    d,
    seed=42,
    low=0.0,
    high=np.pi,
    relative_std=0.08
):
    """Generate a two-cluster dataset within the input domain.

    Cluster centers lie at one quarter and three quarters of the coordinate
    interval on every axis. Each sample is assigned to a cluster with equal
    probability and perturbed by independent Gaussian noise; coordinates are
    clipped to the specified bounds.

    Parameters
    ----------
    n, d : int
        Number of samples and input dimensions.
    seed : int
        Random seed for reproducibility.
    low, high : float
        Coordinate bounds of the input domain.
    relative_std : float
        Noise standard deviation as a fraction of ``high - low``.

    Returns
    -------
    numpy.ndarray
        Clustered input samples of shape ``(n, d)``.
    """
    rng = np.random.default_rng(seed)

    span = high - low

    centres = np.array([
        np.full(d, low + 0.25 * span),
        np.full(d, low + 0.75 * span)
    ])

    cluster_ids = rng.integers(
        low=0,
        high=2,
        size=n
    )

    X = centres[cluster_ids]

    X = X + rng.normal(
        loc=0.0,
        scale=relative_std * span,
        size=(n, d)
    )

    return np.clip(X, low, high)


def correlated_data(
    n,
    d,
    seed=42,
    low=0.0,
    high=np.pi,
    relative_noise=0.05
):
    """Generate correlated samples concentrated near the main diagonal.

    A single uniformly sampled latent coordinate is replicated across all
    dimensions, then independent Gaussian perturbations are added. Values
    outside the input domain are clipped to its bounds.

    Parameters
    ----------
    n, d : int
        Number of samples and input dimensions.
    seed : int
        Random seed for reproducibility.
    low, high : float
        Coordinate bounds of the input domain.
    relative_noise : float
        Perturbation standard deviation relative to ``high - low``.

    Returns
    -------
    numpy.ndarray
        Correlated input samples of shape ``(n, d)``.
    """

    rng = np.random.default_rng(seed)

    span = high - low

    # One latent coordinate per sample
    z = rng.uniform(
        low=low,
        high=high,
        size=(n, 1)
    )

    # Repeat it across all dimensions
    X = np.repeat(z, d, axis=1)

    # Small perturbations
    X += rng.normal(
        loc=0.0,
        scale=relative_noise * span,
        size=(n, d)
    )

    return np.clip(X, low, high)


def gaussian_data(
    n,
    d,
    seed=42,
    low=0.0,
    high=np.pi,
    std_fraction=1/6
):
    """Sample Gaussian-distributed inputs within the encoding interval.

    Coordinates are independently sampled with mean ``(low + high) / 2`` and
    standard deviation ``std_fraction * (high - low)``. Samples outside the
    interval are clipped to its boundaries.

    Parameters
    ----------
    n, d : int
        Number of samples and input dimensions.
    seed : int
        Random seed for reproducibility.
    low, high : float
        Bounds of the encoding interval.
    std_fraction : float
        Standard deviation as a fraction of the interval width.

    Returns
    -------
    numpy.ndarray
        Bounded Gaussian input samples of shape ``(n, d)``.
    """

    rng = np.random.default_rng(seed)

    mean = (low + high) / 2
    std = std_fraction * (high - low)

    X = rng.normal(
        loc=mean,
        scale=std,
        size=(n, d)
    )

    return np.clip(X, low, high)


def embed_active_dimensions(
    X_active,
    q,
    fixed_angle=np.pi / 2
):
    """Embed active input dimensions into a fixed-size qubit register.

    The first ``d`` coordinates retain sample-dependent values. The remaining
    ``q - d`` coordinates are assigned a constant encoding angle, allowing
    changes in input dimension to be studied independently of register size.

    Parameters
    ----------
    X_active : numpy.ndarray
        Active input features with shape ``(n, d)``.
    q : int
        Number of qubits, satisfying ``q >= d``.
    fixed_angle : float
        Encoding angle assigned to inactive coordinates.

    Returns
    -------
    numpy.ndarray
        Embedded inputs of shape ``(n, q)``.
    """

    n, d = X_active.shape

    if d > q:
        raise ValueError("Input dimension d must not exceed q.")

    X = np.full(
        (n, q),
        fixed_angle,
        dtype=float
    )

    X[:, :d] = X_active

    return X
