"""features stages of the weather preparation pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .constants import (DROP_DESCRIPTORS, INTENSITY_WEIGHT, VICINITY_WEIGHT, CONT_FEATURES, FEATURE_ORDER, WEATHER_CLASSES, CLASS_MAP, SEVERITY)

def canonicalize_token(token: str) -> tuple[float, list[str]]:
    """Split a single METAR phenomenon token into (intensity weight, [atomic phenomena...]).

    Examples: '-TSRA' -> (0.5, ['TS','RA']); '+TSGSRA' -> (1.5, ['TS','GS','RA']);
        'VCTS' -> (0.5, ['TS']); 'BCFG' -> (1.0, ['FG']) (BC is discarded).
    """
    weight = 1.0
    if token.startswith("+"):
        weight, token = INTENSITY_WEIGHT["+"], token[1:]
    elif token.startswith("-"):
        weight, token = INTENSITY_WEIGHT["-"], token[1:]
    elif token.startswith("VC"):
        weight, token = VICINITY_WEIGHT, token[2:]

    atoms: list[str] = []
    for i in range(0, len(token) - 1, 2):
        atom = token[i:i + 2]
        if atom in DROP_DESCRIPTORS:
            continue
        if atom == "GR":  # Merge hail into graupel/small hail
            atom = "GS"
        atoms.append(atom)
    return weight, atoms

def parse_wxcodes(value) -> dict[str, float]:
    """Parse a wxcodes string into {atomic phenomenon: soft-label weight} (take max for duplicates)."""
    out: dict[str, float] = {}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return out
    s = str(value).strip()
    if s == "" or s.lower() == "null":
        return out
    for tok in s.split():
        weight, atoms = canonicalize_token(tok)
        for atom in atoms:
            out[atom] = max(out.get(atom, 0.0), weight)
    return out

def load_and_resample(input_path: Path, freq: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = df.replace("null", np.nan)
    df["valid"] = pd.to_datetime(df["valid"])
    df = df.sort_values("valid").drop_duplicates(subset="valid", keep="first")
    df = df.set_index("valid")

    for col in ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "valid"
    return df

def handle_missing(df: pd.DataFrame, max_interp_steps: int) -> pd.DataFrame:
    """Handle missing values by gap length: linearly interpolate short gaps in continuous variables; retain NaN for long gaps (windows are discarded later).

    Do not interpolate drct (missing = variable wind/calm, handled through wind components).
    Return df with an `interpolated` flag column.
    """
    cont = ["tmpf", "dwpf", "relh", "sknt", "alti", "vsby"]
    was_nan = df[cont].isna()
    df[cont] = df[cont].interpolate(method="linear", limit=max_interp_steps,
                                    limit_area="inside")
    df["interpolated"] = (was_nan & df[cont].notna()).any(axis=1)
    df["row_valid"] = df[cont].notna().all(axis=1)
    return df

def build_wind(df: pd.DataFrame) -> pd.DataFrame:
    rad = np.deg2rad(df["drct"])
    u = df["sknt"] * np.sin(rad)
    v = df["sknt"] * np.cos(rad)
    var = df["drct"].isna()  # Variable wind/calm
    df["u"] = u.where(~var, 0.0)
    df["v"] = v.where(~var, 0.0)
    df["wind_var_flag"] = var.astype(float)
    return df

def build_vsby_flag(df: pd.DataFrame, low_vsby_mi: float) -> pd.DataFrame:
    df["low_vsby_flag"] = (df["vsby"] < low_vsby_mi).astype(float)
    return df

def build_soft_labels(df: pd.DataFrame) -> np.ndarray:
    """Per-timestep soft-label matrix (T, 8), with columns ordered by WEATHER_CLASSES.

    CLASS_MAP maps atomic phenomena into 8 fixed classes (unknown -> OTHER); take the maximum weight for duplicates.
    CLEAR is determined during window aggregation from the presence of weather at each timestep.
    """
    idx = {c: i for i, c in enumerate(WEATHER_CLASSES)}
    other_i = idx["OTHER"]
    mat = np.zeros((len(df), len(WEATHER_CLASSES)), dtype=np.float32)
    for r, value in enumerate(df["wxcodes"].values):
        for atom, w in parse_wxcodes(value).items():
            j = idx.get(CLASS_MAP.get(atom, "OTHER"), other_i)
            mat[r, j] = max(mat[r, j], w)
    return mat

def count_tokens(df_train: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in df_train["wxcodes"].values:
        for atom in parse_wxcodes(value):
            counts[atom] = counts.get(atom, 0) + 1
    return counts

def fit_norm_stats(df_train_valid: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for col in CONT_FEATURES:
        vals = df_train_valid[col].to_numpy(dtype=float)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        stats[col] = {"mean": mu, "std": sd if sd > 1e-8 else 1.0}
    return stats

def fit_vsby_cap(df_train_valid: pd.DataFrame) -> float:
    """vsby cap = maximum over valid training rows (the CAVOK ceiling)."""
    cap = float(np.nanmax(df_train_valid["vsby"].to_numpy(dtype=float)))
    return cap if cap > 1e-8 else 1.0

def build_feature_matrix(df: pd.DataFrame, norm: dict, vsby_cap: float) -> np.ndarray:
    """Per-timestep feature matrix (T, D).

    Z-score continuous variables using training statistics; cap vsby to [0,cap] and normalize to [0,1] (without Z-score normalization);
    leave flags unchanged.
    """
    cols = []
    for f in FEATURE_ORDER:
        if f == "vsby":
            v = df["vsby"].to_numpy(dtype=float)
            cols.append(np.clip(v, 0.0, vsby_cap) / vsby_cap)
        elif f in norm:
            mu, sd = norm[f]["mean"], norm[f]["std"]
            cols.append((df[f].to_numpy(dtype=float) - mu) / sd)
        else:
            cols.append(df[f].to_numpy(dtype=float))
    return np.stack(cols, axis=1).astype(np.float32)
