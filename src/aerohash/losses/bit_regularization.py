"""bit regularization objective."""
import torch


def bit_balance_loss(codes):
    """P0-②: Bit-balance regularization: drive each bit's batch mean toward 0.

    Suppress collapse modes such as constant bits / identical codes, encouraging each bit to take ±1 evenly across the dataset.
    """
    return codes.mean(dim=0).pow(2).mean()

def bit_decorrelation_loss(codes, eps=1e-6):
    """P0-②: Bit-decorrelation regularization: make code-matrix columns (bits) nearly orthogonal.

    Center continuous codes per bit and normalize to unit norm, then compute the Pearson correlation matrix and penalize off-diagonal entries,
    reducing redundant/highly correlated bits, mitigating Hamming ties, and improving q_uniq.
    """
    n, m = codes.shape
    if n < 2 or m < 2:
        return codes.sum() * 0.0
    c = codes - codes.mean(dim=0, keepdim=True)
    c = c / (c.norm(dim=0, keepdim=True) + eps)
    corr = c.t() @ c                                    # (M, M) Pearson correlation
    off = corr - torch.diag(torch.diagonal(corr))
    return off.pow(2).sum() / (m * (m - 1))
