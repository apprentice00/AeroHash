"""ordinal objective."""
import torch
from .similarity import weighted_rbf_similarity

def ordinal_relational_loss(codes, targets, severity, sigma, delta, s_margin,
                            neg_boost=2.0, eps=1e-8):
    """Contribution ③ (enhanced A+B+D): Severity-weighted ordinal ranking supervision anchored on weather samples.

    Constrain relative code-similarity rankings to match ground-truth soft similarities without regressing absolute similarity (robust to soft-label noise).
    Ground-truth within-batch soft similarities S use the same severity-weighted RBF as training; code similarity C=normalize(F)·ᵀ.
    For anchor i and pair (j,k), when S_ij - S_ik > s_margin (j should be more similar than k), penalize
    relu(C_ik - C_ij + δ), requiring C_ij ≥ C_ik + δ. Three enhancements for long-tailed data:

    - A Weather anchors: apply constraints only to samples i containing weather. Ranking clear queries only tightens
      the already 0.95+ head cluster and crowds out tail-class code space, without helping rare-weather retrieval (previous results: clear↑/weather↓).
    - B Prioritize weather negatives + severity weighting: multiply weights by (1+neg_boost) when negative k also contains weather, emphasizing
      severity gradients within rare weather (equally severe > weak weather > clear); also weight each anchor by its severity.
    - D Per-anchor normalization: normalize valid triplets within each anchor, then average across anchors with anchor_sev weights,
      preventing anchors with many triplets from dominating the gradients.
    """
    sev_w = severity[:-1]                                    # Weather-class severity (excluding CLEAR)
    wmask = (targets[:, :-1] > 0).float()                    # (B, K-1) Weather-class mask
    is_w = wmask.sum(dim=1) > 0                              # (B,) Whether weather is present
    if int(is_w.sum()) < 2:
        return codes.sum() * 0.0
    anchor_sev = (wmask * sev_w.unsqueeze(0)).amax(dim=1)    # (B,) Anchor severity = maximum severity

    S = weighted_rbf_similarity(targets, targets, severity, sigma)   # (B,B) ∈[0,1]
    f = torch.nn.functional.normalize(codes, p=2, dim=1)
    C = f @ f.t()                                                    # (B,B) ∈[-1,1]

    valid = (S.unsqueeze(2) - S.unsqueeze(1)) > s_margin            # [i,j,k]
    valid = valid & is_w.view(-1, 1, 1)                            # A: Weather anchors only
    negw = 1.0 + neg_boost * is_w.float().view(1, 1, -1)           # B: Weight weather negatives k
    w_tri = valid.float() * negw
    viol = torch.relu(C.unsqueeze(1) - C.unsqueeze(2) + delta)     # C_ik - C_ij + δ

    num_i = (w_tri * viol).sum(dim=(1, 2))                         # (B,) Weighted sum of violations per anchor
    den_i = w_tri.sum(dim=(1, 2))                                  # (B,) Sum of weights per anchor
    has = den_i > 0
    if int(has.sum()) == 0:
        return codes.sum() * 0.0
    L_i = num_i[has] / (den_i[has] + eps)                          # D: Per-anchor normalization
    a = anchor_sev[has]                                            # B: Severity weighting across anchors
    return (a * L_i).sum() / (a.sum() + eps)
