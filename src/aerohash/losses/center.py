"""Contribution ①: Severity-aware hyperspherical hash centers + adaptive additive cosine margin.

Multi-label soft-proxy loss (multi-label margin-BCE): learn one center μ_c for each of K classes
(8 weather classes + CLEAR); soft labels Y associate each sample with multiple positive centers. Apply
a per-class adaptive additive cosine margin m_c to positive-class logits (cosθ − m_c, rather than cos(θ+m)),
giving rare, high-severity classes larger margins.

    m_c = m0 · (1 + λ·w_c) · (1 + γ·log(N_max / N_c))

Here w_c is the severity weight (config.json severity_weight), and N_c is the number of training windows
containing that class. The loss operates on 96-dimensional continuous hash codes F=pred (L2-normalized onto the hypersphere).
"""
import torch
from torch import nn
import torch.nn.functional as F


class SeverityCenterLoss(nn.Module):
    """Multi-label hyperspherical center loss with four margin modes (none/fixed/severity/severity_freq).

    Args:
        num_classes (int): Number of classes K (including CLEAR).
        code_length (int): Hash-code dimension M (GAP output dimension).
        severity (Tensor): (K,) Per-class severity weights w_c.
        class_freq (Tensor): (K,) Per-class occurrence counts N_c in the training set.
        m0 (float): Base additive cosine margin.
        lam (float): Severity amplification factor λ.
        gamma (float): Long-tail frequency amplification factor γ.
        s (float): Logit scale.
        margin_mode (str): 'none' | 'fixed' | 'severity' | 'severity_freq'.
    """

    def __init__(self, num_classes, code_length, severity, class_freq,
                 m0=0.15, lam=0.5, gamma=0.3, s=12.0,
                 margin_mode='severity_freq'):
        super().__init__()
        self.num_classes = num_classes
        self.code_length = code_length
        self.s = float(s)
        self.margin_mode = margin_mode

        # Learnable hyperspherical centers C (K, M), row-wise L2-normalized to μ_c during the forward pass
        self.centers = nn.Parameter(torch.randn(num_classes, code_length))
        nn.init.kaiming_uniform_(self.centers, a=5 ** 0.5)

        w = torch.as_tensor(severity, dtype=torch.float32).clone()
        n = torch.as_tensor(class_freq, dtype=torch.float32).clone().clamp_min(1.0)
        n_max = n.max()

        if margin_mode == 'none':
            m_c = torch.zeros(num_classes)
        elif margin_mode == 'fixed':
            m_c = torch.full((num_classes,), float(m0))
        elif margin_mode == 'severity':
            m_c = m0 * (1.0 + lam * w)
        elif margin_mode == 'severity_freq':
            m_c = m0 * (1.0 + lam * w) * (1.0 + gamma * torch.log(n_max / n))
        else:
            raise ValueError('Invalid margin_mode: {}'.format(margin_mode))

        # The margin applies only to positive-class logits and is stored as a buffer without gradients
        self.register_buffer('m_c', m_c)

    def forward(self, codes, targets):
        """
        Args:
            codes (Tensor): (B, M) Continuous hash codes F=pred.
            targets (Tensor): (B, K) Multi-label soft targets Y (>0 indicates class presence).
        Returns:
            Tensor: Scalar loss.
        """
        f = F.normalize(codes, p=2, dim=1)                  # (B, M)
        mu = F.normalize(self.centers, p=2, dim=1)          # (K, M)
        cos = f @ mu.t()                                    # (B, K) ∈ [-1, 1]

        p = (targets > 0).float()                           # (B, K) Positive-class mask

        # Additive cosine margin: s*(cosθ − m_c) for positive classes; negative classes remain unchanged
        logits = self.s * (cos - p * self.m_c.unsqueeze(0))
        loss = F.binary_cross_entropy_with_logits(logits, p)
        return loss
