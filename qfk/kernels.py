"""Statevector-based fidelity quantum kernels.

Constructs fidelity Gram matrices from parameterized quantum feature maps.
The implementation uses exact statevector simulation and computes squared
state overlaps for all pairs of input samples.
"""

import numpy as np
from qiskit.quantum_info import Statevector

from .circuits import feature_map_circuit


def compute_statevectors(X, entangler=None, depth=0):
    """Evaluate quantum feature-map states for an input dataset.

    Each input vector is encoded into a pure quantum state using the specified
    feature-map circuit. Statevectors are computed without sampling noise.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input angles for the quantum feature map. Each feature corresponds to
        one qubit.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour entangling gate type. ``None`` omits entanglement.
    depth : int, optional
        Number of entangling and data-reuploading repetitions.

    Returns
    -------
    numpy.ndarray of complex, shape (n_samples, 2**n_features)
        Row-wise statevectors of the encoded input samples.
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
    """Compute the fidelity Gram matrix for a quantum feature map.

    For the encoded pure states ``phi(x_i)``, the kernel is defined by
    ``K[i, j] = |<phi(x_i) | phi(x_j)>|**2``. The returned matrix is real,
    symmetric, and positive semidefinite up to floating-point precision.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input angles used to prepare the feature-map states.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour entangling gate type.
    depth : int, optional
        Number of entangling and data-reuploading repetitions.

    Returns
    -------
    numpy.ndarray of float, shape (n_samples, n_samples)
        Pairwise fidelity Gram matrix. Its diagonal entries equal one up to
        numerical precision because the encoded states are normalized.
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
