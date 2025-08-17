import torch, torch.nn as nn, torch.nn.functional as F
from models.classical_blocks import ConvEncoder1D, ResidualMLP
from models.quantum_layers import QuantumEmbedding

class HybridQuantumModel(nn.Module):
    """
    (B, 2, G) → encoder → compress → map to n_qubits → quantum layer
    → concat back with classical → head → predict density (G points)
    """
    def __init__(self, grid_points: int, n_qubits=6, q_layers=2,
                 embed_dim=64, mlp_hidden=128, shots=None):
        super().__init__()
        self.grid_points = grid_points
        self.encoder = ConvEncoder1D(in_ch=2, embed_dim=embed_dim)

        self.proj_to_qubits = nn.Linear(embed_dim, n_qubits)
        self.quantum = QuantumEmbedding(n_qubits=n_qubits, layers=q_layers, shots=shots)

        # Fuse quantum outputs with classical embedding
        self.fuse = ResidualMLP(in_dim=embed_dim + n_qubits,
                                hidden=mlp_hidden, out_dim=mlp_hidden, depth=3)

        # Upsample to grid with a lightweight deconv (learned “shape”)
        self.head = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.grid_points),
            nn.Softplus()  # ensure non-negative density
        )

    def forward(self, x):  # x: (B, 2, G)
        B = x.shape[0]
        enc = self.encoder(x)                  # (B, embed_dim)
        q_in = self.proj_to_qubits(enc)       # (B, n_qubits)
        q_out = self.quantum(q_in)            # (B, n_qubits) in [-1,1]
        fused = torch.cat([enc, q_out], dim=-1)
        h = self.fuse(fused)                  # (B, hidden)
        y = self.head(h)                      # (B, G), non-negative
        # Normalize as a probability density across grid
        y = y / (y.sum(dim=1, keepdim=True) + 1e-9)
        return y
