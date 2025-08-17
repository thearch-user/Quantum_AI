import pennylane as qml
from pennylane import numpy as pnp
import torch
from quantum.devices import get_device

def _feature_map(features, wires):
    """
    Encode a small number of summary features into rotations.
    features: (B, F) mapped to rotations on qubits.
    """
    # simple angle spreading; you can design richer maps
    for w, f in zip(wires, features):
        qml.RY(f, wires=w)
        qml.RZ(0.5 * f, wires=w)

def _entangling_block(wires):
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    qml.CNOT(wires=[wires[-1], wires[0]])

def _variational_block(params, wires):
    # params shape: (layers, len(wires), 3) → (Rx,Ry,Rz) per qubit
    L = params.shape[0]
    for l in range(L):
        for i, w in enumerate(wires):
            rx, ry, rz = params[l, i]
            qml.RX(rx, wires=w)
            qml.RY(ry, wires=w)
            qml.RZ(rz, wires=w)
        _entangling_block(wires)

class QuantumEmbedding(torch.nn.Module):
    """
    Wrap a PennyLane circuit as a PyTorch module for end-to-end training.
    We aggregate the spatial grid into a compact summary vector → quantum.
    """
    def __init__(self, n_qubits: int, layers: int, shots: int|None):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots

        self.dev = get_device(n_qubits, shots)
        self.wires = list(range(n_qubits))

        # Trainable parameters in quantum variational circuit
        init = 0.01 * torch.randn(layers, n_qubits, 3)
        self.theta = torch.nn.Parameter(init)

        # Build a QNode in Torch interface
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(feature_vec, theta):
            _feature_map(feature_vec, wires=self.wires)
            _variational_block(theta, wires=self.wires)
            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def forward(self, feature_vec: torch.Tensor) -> torch.Tensor:
        """
        feature_vec: (B, n_qubits) compact summary per batch item.
        Returns: (B, n_qubits) expectation values in [-1,1].
        """
        # PennyLane circuits accept per-sample calls; vectorize via vmap-like loop.
        # For performance, keep B reasonably small per forward.
        outs = []
        for i in range(feature_vec.shape[0]):
            outs.append(self.circuit(feature_vec[i], self.theta))
        return torch.stack(outs, dim=0)
