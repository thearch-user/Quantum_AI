import torch, torch.nn as nn, torch.nn.functional as F
import math

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=3, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)
        ) for _ in range(depth)])
        self.out = nn.Linear(hidden, out_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) if hasattr(__import__('math'),'sqrt') else None

    def forward(self, x):
        x = F.gelu(self.inp(x))
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)

class ConvEncoder1D(nn.Module):
    """
    Encodes (2, grid) → compact embed_dim using strided convs + pooling.
    """
    def __init__(self, in_ch=2, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):        # x: (B, 2, G)
        h = self.net(x).squeeze(-1)  # (B, 64)
        return self.fc(h)             # (B, embed_dim)
