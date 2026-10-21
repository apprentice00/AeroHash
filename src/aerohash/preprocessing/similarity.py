"""similarity stages of the weather preparation pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .constants import (DROP_DESCRIPTORS, INTENSITY_WEIGHT, VICINITY_WEIGHT, CONT_FEATURES, FEATURE_ORDER, WEATHER_CLASSES, CLASS_MAP, SEVERITY)

def weighted_sqdist(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized weighted squared distance D2[i,j] = sum_k w_k (A_ik - B_jk)^2."""
    a2 = (A * A) @ w                       # (nA,)
    b2 = (B * B) @ w                       # (nB,)
    cross = (A * w) @ B.T                  # (nA, nB)
    d2 = a2[:, None] + b2[None, :] - 2.0 * cross
    return np.maximum(d2, 0.0)

def estimate_sigma(Y: np.ndarray, w: np.ndarray, rng, sample: int = 2000) -> float:
    """Estimate RBF bandwidth sigma from the median weighted pairwise distance of training windows."""
    n = Y.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    Ys = Y[idx]
    d2 = weighted_sqdist(Ys, Ys, w)
    m = ~np.eye(d2.shape[0], dtype=bool)
    dist = np.sqrt(d2[m])
    med = float(np.median(dist))
    return med if med > 1e-6 else 1.0

def calibrate_tau(S_sim: np.ndarray, weather_rows: np.ndarray,
                  target_density: float) -> float:
    """Calibrate the threshold on weather queries so that positive-pair density ≈ target_density."""
    sub = S_sim[weather_rows] if weather_rows.size else S_sim
    return float(np.quantile(sub, 1.0 - target_density))
