"""Time-domain and frequency-domain window encoders."""
import torch
import torch.nn.functional as F
from torch import nn

class ECALayer(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020).

    For (B, C, W): temporal GAP → cross-channel 1D convolution → sigmoid gating.
    The kernel size k is determined adaptively from the channel count (odd, ≥3), with O(k) parameters.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((torch.log2(torch.tensor(float(channels))).item() + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):                       # x: (B, C, W)
        y = x.mean(dim=-1, keepdim=True)         # (B, C, 1)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        return x * torch.sigmoid(y)

class TSEncoder(nn.Module):
    """Time-domain encoder: (B, W, D) → H_t (B, C, W).

    Transpose → (optional) concatenate first-order differences → (optional) ECA → Conv1d projection →
    dilated residual convolution stack (dilation=1/2/4).
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.0,
                 use_deriv=False, use_ch_attn=False):
        super().__init__()
        self.use_deriv = use_deriv
        self.use_ch_attn = use_ch_attn
        proj_in_dim = in_dim * 2 if use_deriv else in_dim
        if use_ch_attn:
            self.ch_attn = ECALayer(proj_in_dim)
        self.proj = nn.Conv1d(proj_in_dim, hidden_dim, kernel_size=3, padding=1)
        self.proj_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for d in (1, 2, 4)
        ])

    def forward(self, x):
        # x: (B, W, D) -> (B, D, W) -> (B, hidden, W)
        x = x.transpose(1, 2)
        if self.use_deriv:
            deriv = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
            x = torch.cat([x, deriv], dim=1)      # (B, 2D, W)
        if self.use_ch_attn:
            x = self.ch_attn(x)
        h = F.relu(self.proj_bn(self.proj(x)))
        for blk in self.blocks:
            h = h + blk(h)
        return h

class FreqEncoder(nn.Module):
    """Frequency-domain encoder: raw x → H_f (B, C, W).

    rfft magnitude spectrum → remove DC → log1p → flatten → BN+MLP → h_f,
    then broadcast to W tokens (a frequency-domain signature of the entire window).
    """

    def __init__(self, in_dim, hidden_dim, n_freq):
        super().__init__()
        feat_dim = (n_freq - 1) * in_dim          # Remove the DC bin at index 0
        self.bn = nn.BatchNorm1d(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):                          # x: (B, W, D)
        if self.training and x.size(0) < 2:
            raise ValueError('FreqEncoder training requires at least 2 samples per batch for BatchNorm; '
                             'adjust batch_size/num_samples (including the final batch) or disable use_tf')
        mag = torch.fft.rfft(x, dim=1).abs()[:, 1:, :]
        h_f = self.mlp(self.bn(torch.log1p(mag).flatten(1)))  # (B, hidden)
        return h_f.unsqueeze(-1).expand(-1, -1, x.size(1))    # (B, hidden, W)
