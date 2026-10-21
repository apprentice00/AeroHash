"""similarity objective."""
import torch


def weighted_rbf_similarity(t, r, w, sigma):
    """Severity-weighted RBF similarity S[i,j]=exp(-sum_k w_k (t_ik-r_jk)^2 / 2 sigma^2).

    Consistent with the data-side supervision matrix: sensitive to magnitude/intensity, with CLEAR substantially downweighted by w
    to suppress clear-clear dominance. Return continuous S∈[0,1] (ADSH_Loss supports continuous targets).
    """
    a2 = (t * t) @ w                    # (n,)
    b2 = (r * r) @ w                    # (m,)
    cross = (t * w) @ r.t()             # (n, m)
    d2 = (a2.unsqueeze(1) + b2.unsqueeze(0) - 2.0 * cross).clamp_min(0.0)
    return torch.exp(-d2 / (2.0 * sigma * sigma))
