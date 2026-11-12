"""Synthetic input distributions and fixed-angle dimension embeddings."""

import numpy as np


def cartesian_grid(points_per_dim, d, low=0.0, high=np.pi):
    """Construct a Cartesian product grid with ``points_per_dim`` points per axis.

    Returns an array of shape ``(points_per_dim**d, d)`` with indexing
    consistent with ``numpy.meshgrid(indexing="ij")``.
    """
    axes = [np.linspace(low, high, points_per_dim) for _ in range(d)]

    mesh = np.meshgrid(*axes, indexing="ij")

    X = np.stack([axis.ravel() for axis in mesh], axis=1)

    return X


def uniform_data(n, d, seed=42, low=0.0, high=np.pi):
    """Sample independent uniform coordinates on ``[low, high]``.

    Returns an array of shape ``(n, d)`` using the specified random seed.
    """
    rng = np.random.default_rng(seed)

    X = rng.uniform(low=low, high=high, size=(n, d))

    return X


def clustered_data(n, d, seed=42, low=0.0, high=np.pi, relative_std=0.08):
    """Sample two equally weighted clusters within ``[low, high]``.

    Cluster centres lie at one quarter and three quarters of the interval
    along every axis. Independent Gaussian perturbations have standard
    deviation ``relative_std * (high - low)``; outputs are clipped to bounds.
    """
    rng = np.random.default_rng(seed)

    span = high - low

    centres = np.array(
        [np.full(d, low + 0.25 * span), np.full(d, low + 0.75 * span)]
    )

    cluster_ids = rng.integers(low=0, high=2, size=n)

    X = centres[cluster_ids]

    X = X + rng.normal(loc=0.0, scale=relative_std * span, size=(n, d))

    return np.clip(X, low, high)


def correlated_data(n, d, seed=42, low=0.0, high=np.pi, relative_noise=0.05):
    """Sample correlated coordinates around the main diagonal.

    Each sample shares one uniformly distributed latent coordinate across
    dimensions. Independent Gaussian noise has standard deviation
    ``relative_noise * (high - low)``; outputs are clipped to bounds.
    """
    rng = np.random.default_rng(seed)

    span = high - low

    z = rng.uniform(low=low, high=high, size=(n, 1))

    X = np.repeat(z, d, axis=1)

    X += rng.normal(loc=0.0, scale=relative_noise * span, size=(n, d))

    return np.clip(X, low, high)


def gaussian_data(n, d, seed=42, low=0.0, high=np.pi, std_fraction=1 / 6):
    """Sample Gaussian coordinates and clip them to ``[low, high]``.

    Coordinates have mean ``(low + high) / 2`` and standard deviation
    ``std_fraction * (high - low)`` before clipping.
    """
    rng = np.random.default_rng(seed)

    mean = (low + high) / 2
    std = std_fraction * (high - low)

    X = rng.normal(loc=mean, scale=std, size=(n, d))

    return np.clip(X, low, high)


def embed_active_dimensions(X_active, q, fixed_angle=np.pi / 2):
    """Pad active input coordinates with a sample-independent angle.

    Maps an ``(n, d)`` array to ``(n, q)`` by filling inactive coordinates
    with ``fixed_angle``. Requires ``d <= q``.
    """
    n, d = X_active.shape

    if d > q:
        raise ValueError("Input dimension d must not exceed q.")

    X = np.full((n, q), fixed_angle, dtype=float)

    X[:, :d] = X_active

    return X
