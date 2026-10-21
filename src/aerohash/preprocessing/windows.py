"""windows stages of the weather preparation pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .constants import (DROP_DESCRIPTORS, INTENSITY_WEIGHT, VICINITY_WEIGHT, CONT_FEATURES, FEATURE_ORDER, WEATHER_CLASSES, CLASS_MAP, SEVERITY)

def make_splits(n: int, train_frac: float, db_frac: float,
                buffer_steps: int) -> dict[str, tuple[int, int]]:
    """Split the time axis into [train][buf][database][buf][query] and return each [start,end) interval."""
    usable = n - 2 * buffer_steps
    if usable <= 0:
        raise ValueError("Buffer too large or data too short; no usable interval")
    n_train = int(train_frac * usable)
    n_db = int(db_frac * usable)
    n_query = usable - n_train - n_db
    if min(n_train, n_db, n_query) <= 0:
        raise ValueError("Split fractions produce an empty subset; adjust frac")

    train = (0, n_train)
    db = (n_train + buffer_steps, n_train + buffer_steps + n_db)
    query = (db[1] + buffer_steps, db[1] + buffer_steps + n_query)
    return {"train": train, "database": db, "query": query}

def segment_windows(span: tuple[int, int], W: int, S: int,
                    feat: np.ndarray, soft: np.ndarray, row_valid: np.ndarray,
                    interp: np.ndarray, times: pd.DatetimeIndex,
                    max_interp_ratio: float):
    """Extract windows within one time interval, discarding those with invalid rows or excessive interpolation.

    Window labels = mean-pooled values of 8 weather classes + CLEAR (fraction of timesteps without weather).
    Return X(n,W,D), Y(n,K=9), meta(list of dict).
    """
    start, end = span
    X, Y, meta = [], [], []
    for s in range(start, end - W + 1, S):
        e = s + W
        if not row_valid[s:e].all():
            continue
        interp_ratio = float(interp[s:e].mean())
        if interp_ratio > max_interp_ratio:
            continue
        win_soft = soft[s:e]                       # (W, 8)
        yw = win_soft.mean(axis=0)                 # Mean-pool weather classes
        step_has_weather = (win_soft.sum(axis=1) > 0)
        clear_frac = float(1.0 - step_has_weather.mean())
        y_full = np.concatenate([yw, [clear_frac]]).astype(np.float32)
        X.append(feat[s:e])
        Y.append(y_full)
        meta.append({
            "start_idx": s,
            "start_time": times[s].isoformat(),
            "end_time": times[e - 1].isoformat(),
            "interp_ratio": round(interp_ratio, 4),
            "n_weather_atoms": int((yw > 0).sum()),
            "is_clear": int(yw.sum() == 0),
        })
    K = len(WEATHER_CLASSES) + 1
    if not X:
        return (np.empty((0, W, feat.shape[1]), np.float32),
                np.empty((0, K), np.float32), meta)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), meta

def downsample_clear(split: dict, keep_frac: float, rng) -> dict:
    """Randomly downsample clear windows in one subset, retaining all weather windows."""
    meta = split["meta"]
    if keep_frac >= 1.0 or not meta:
        return split
    is_clear = np.array([m["is_clear"] for m in meta], dtype=bool)
    clear_idx = np.where(is_clear)[0]
    n_keep = int(round(len(clear_idx) * keep_frac))
    keep_clear = rng.choice(clear_idx, size=n_keep, replace=False) if n_keep else \
        np.array([], dtype=int)
    keep = np.sort(np.concatenate([np.where(~is_clear)[0], keep_clear]))
    return {
        "X": split["X"][keep],
        "Y": split["Y"][keep],
        "meta": [meta[i] for i in keep],
    }
