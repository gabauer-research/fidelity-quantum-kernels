"""Exact statevector fidelity kernels and Gram matrices."""

import numpy as np
from qiskit.quantum_info import Statevector

from .circuits import feature_map_circuit


def compute_statevectors(X, entangler=None, depth=0):
    """Evaluate feature-map states using exact statevector simulation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input angles, one per qubit.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour entangling gate; ``None`` omits entanglement.
    depth : int, optional
        Number of interleaved entangling and re-encoding layers.

    Returns
    -------
    numpy.ndarray of complex, shape (n_samples, 2**n_features)
        Encoded pure-state vectors, one per row.
    """
    states = []

    for x in X:
        qc = feature_map_circuit(
            x,
            entangler=entangler,
            depth=depth,
        )

        state = Statevector.from_instruction(qc)
        states.append(state.data)

    return np.asarray(states)


def fidelity_gram_matrix(X, entangler=None, depth=0):
    """Compute pairwise squared state overlaps for an input dataset.

    The fidelity kernel is ``K[i, j] = |<psi(x_i)|psi(x_j)>|**2``.
    Its Gram matrix is real, symmetric, and positive semidefinite up to
    floating-point precision, with unit diagonal for normalized states.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input angles, one per qubit.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour entangling gate; ``None`` omits entanglement.
    depth : int, optional
        Number of interleaved entangling and re-encoding layers.

    Returns
    -------
    numpy.ndarray of float, shape (n_samples, n_samples)
        Fidelity Gram matrix.
    """
    states = compute_statevectors(
        X,
        entangler=entangler,
        depth=depth,
    )

    overlaps = states @ states.conj().T
    K = np.abs(overlaps) ** 2

    # Symmetrize to remove floating-point asymmetries.
    K = (K + K.T) / 2.0

    return K
