"""tfc objective."""
import torch


def tfc_descriptor(x, eps=1e-8):
    """Equal-feature spectral descriptor: (B,W,D) → (B,(F-1)*D).

    rFFT along time, remove DC, log1p magnitude, flatten, then L2 normalize.
    Every input feature has equal weight. Weather-class severity weights have
    no mapping to physical input channels and must not enter this descriptor.
    A window with no non-DC energy keeps the historical zero descriptor.
    """
    mag = torch.log1p(torch.fft.rfft(x, dim=1).abs()[:, 1:, :])   # (B, F-1, D)
    phi = mag.flatten(1)                                    # (B, (F-1)*D)
    return phi / (phi.norm(dim=1, keepdim=True) + eps)

def tfc_relational_loss(codes, x):
    """Align code-similarity structure with TFC spectral consistency (within-batch relational supervision).

    C = normalize(F)·normalize(F)^T ∈[-1,1]; target T = 2·(φφ^T)-1 ∈[-1,1].
    Windows with similar spectral shapes → high target similarity (pull together); dissimilar windows → push apart. Label-independent,
    providing frequency-domain consistency constraints for rare severe weather. Compute MSE on off-diagonal entries only.
    """
    if codes.shape[0] < 2:
        return codes.sum() * 0.0
    phi = tfc_descriptor(x)
    T = 2.0 * (phi @ phi.t()) - 1.0
    f = torch.nn.functional.normalize(codes, p=2, dim=1)
    C = f @ f.t()
    n = C.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=C.device)
    return ((C - T)[off] ** 2).mean()
