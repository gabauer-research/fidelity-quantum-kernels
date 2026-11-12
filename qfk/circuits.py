"""RY-based quantum feature maps with optional interleaved entangling layers."""

from qiskit import QuantumCircuit


def add_encoding_layer(qc: QuantumCircuit, x) -> None:
    """Apply input-dependent RY rotations in qubit order."""
    for qubit, angle in enumerate(x):
        qc.ry(float(angle), qubit)


def add_entangling_layer(qc: QuantumCircuit, entangler: str | None) -> None:
    """Apply a nearest-neighbour CNOT or CZ chain, or no gates for ``None``."""
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
    """Construct a feature map with repeated RY data encoding.

    The circuit starts with one RY encoding layer. Each of ``depth``
    repetitions adds an entangling layer followed by another encoding
    layer. For ``entangler=None``, no entangling gates are inserted.

    Parameters
    ----------
    x : array-like of float
        Input angles, one per qubit.
    entangler : {'cnot', 'cz', None}, optional
        Nearest-neighbour gate used between encoding layers.
    depth : int, optional
        Number of re-uploading repetitions; ``depth=0`` uses one encoding.
    barriers : bool, optional
        Insert circuit barriers around entangling layers.

    Returns
    -------
    QuantumCircuit
        Data-dependent state-preparation circuit.
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
