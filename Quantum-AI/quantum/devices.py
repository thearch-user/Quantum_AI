import pennylane as qml

def get_device(n_qubits: int, shots: int | None):
    """
    Prefer fast C++ backends; fall back to default.
    - lightning.qubit is very fast on CPU simulators.
    - shots=None uses analytic mode (no sampling noise) for training stability.
    """
    try:
        return qml.device("lightning.qubit", wires=n_qubits, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits, shots=shots)
