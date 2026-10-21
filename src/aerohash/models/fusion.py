"""Gated fusion of time and frequency tokens."""
import torch
from torch import nn

class GatedFusion(nn.Module):
    """Time-frequency gated fusion: g=σ(Conv1d([H_t;H_f])); H=g⊙H_t+(1-g)⊙H_f."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, H_t, H_f):                   # both (B, hidden, W)
        g = torch.sigmoid(self.gate(torch.cat([H_t, H_f], dim=1)))
        return g * H_t + (1.0 - g) * H_f
