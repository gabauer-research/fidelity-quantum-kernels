"""Quantum feature-map circuits for fidelity-kernel experiments.

Constructs parameter-dependent state-preparation circuits using RY angle
encoding, optional nearest-neighbour entangling gates, and data re-uploading.
"""

from qiskit import QuantumCircuit


def add_encoding_layer(qc: QuantumCircuit, x) -> None:
    """Apply an RY angle-encoding layer in place.

    Each input coordinate controls the Y rotation of the corresponding qubit.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to which the encoding operations are appended.
    x : array-like of float
        Input angles, ordered by qubit index. The input is expected to
        contain one angle per qubit.

    Returns
    -------
    None
        The input circuit is modified in place.
    """
    for qubit, angle in enumerate(x):
        qc.ry(float(angle), qubit)


def add_entangling_layer(qc: QuantumCircuit, entangler: str | None) -> None:
    """Append a nearest-neighbour CNOT or CZ chain in place.

    Adjacent qubits are coupled in ascending index order. For CNOT gates,
    qubit ``j`` is the control and qubit ``j + 1`` the target.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to which the entangling operations are appended.
    entangler : {'cnot', 'cz', None}
        Entangling gate type. ``None`` leaves the circuit unchanged.

    Raises
    ------
    ValueError
        If the entangler is not one of the supported choices.
    """
    q = qc.num_qubits

    if entangler == "cnot":
        for j in range(q - 1):
            qc.cx(j, j + 1)

    elif entangler == "cz":
        for j in range(q - 1):
            qc.cz(j, j + 1)

    elif entangler is None:
        pass

    else:
        raise ValueError("entangler must be None, 'cnot', or 'cz'")


def feature_map_circuit(
    x,
    entangler: str | None = None,
    depth: int = 0,
    barriers: bool = False,
) -> QuantumCircuit:
    """Build a data-reuploading quantum feature map.

    An initial RY encoding is followed by ``depth`` repetitions of an
    entangling layer and another RY encoding of the same input. Thus,
    ``depth=0`` applies one encoding layer, whereas ``depth=L`` applies
    ``L + 1`` encoding layers and ``L`` entangling layers.

    Parameters
    ----------
    x : array-like of float
        Input angles. The number of coordinates determines the qubit count.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour entangling gate; defaults to no entangling gates.
    depth : int, optional
        Number of entangling/re-uploading repetitions. Defaults to zero.
    barriers : bool, optional
        Insert circuit barriers around each entangling layer for readability.
        Barriers do not change the ideal state prepared by the circuit.

    Returns
    -------
    QuantumCircuit
        Parameter-dependent state-preparation circuit.
    """
    q = len(x)
    qc = QuantumCircuit(q)

    add_encoding_layer(qc, x)

    for _ in range(depth):
        if barriers:
            qc.barrier()

        add_entangling_layer(qc, entangler)

        if barriers:
            qc.barrier()

        add_encoding_layer(qc, x)

    return qc
