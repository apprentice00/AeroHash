"""Prepare windows, soft labels and retrieval ground truth from CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from aerohash.paths import PROJECT_ROOT
from .constants import WEATHER_CLASSES, SEVERITY, FEATURE_ORDER, CLASS_MAP, INTENSITY_WEIGHT, VICINITY_WEIGHT, DROP_DESCRIPTORS
from .features import (load_and_resample, handle_missing, build_wind, build_vsby_flag, fit_norm_stats, fit_vsby_cap, count_tokens, build_feature_matrix, build_soft_labels)
from .windows import make_splits, segment_windows, downsample_clear
from .similarity import estimate_sigma, weighted_sqdist, calibrate_tau

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weather time-series retrieval data preprocessing pipeline v2")
    p.add_argument("--input", default=str(PROJECT_ROOT / "data_processed/ZBAA_2015-2025_clean.csv"))
    p.add_argument("--outdir", default=str(PROJECT_ROOT / "data_processed/processed"))
    p.add_argument("--freq", default="30min")
    p.add_argument("--window", type=int, default=12, help="Window length W (timesteps)")
    p.add_argument("--stride", type=int, default=3, help="Sliding stride S (timesteps)")
    p.add_argument("--buffer-mult", type=int, default=2,
                   help="Buffer multiplier B; buffer length = B*W timesteps (must be >=1 to ensure a leakage-free gap of >=W)")
    p.add_argument("--max-interp-steps", type=int, default=3, help="Maximum timesteps to interpolate in a short gap")
    p.add_argument("--max-interp-ratio", type=float, default=0.3, help="Maximum allowed fraction of interpolated timesteps per window")
    p.add_argument("--low-vsby-mi", type=float, default=3.1, help="Low-visibility threshold (miles, ~5km)")
    p.add_argument("--tau", type=float, default=None,
                   help="Similarity binarization threshold; calibrated automatically using --target-density if omitted")
    p.add_argument("--target-density", type=float, default=0.15,
                   help="Target positive-pair density for weather queries (used to calibrate tau automatically)")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--db-frac", type=float, default=0.2)
    p.add_argument("--clear-keep-frac", type=float, default=0.35,
                   help="Fraction of clear training windows to retain (mitigates code collapse)")
    p.add_argument("--db-clear-keep-frac", type=float, default=0.5,
                   help="Fraction of clear retrieval-database windows to retain")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)

def main(argv=None) -> None:
    run(parse_args(argv))


def run(args) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    W, S = args.window, args.stride
    buffer_steps = args.buffer_mult * W

    # --- Stage 1 ---
    df = load_and_resample(Path(args.input), args.freq)
    df = handle_missing(df, args.max_interp_steps)
    print(f"[1] Resampled rows: {len(df)}; valid rows: {int(df['row_valid'].sum())}; "
          f"interpolated rows: {int(df['interpolated'].sum())}")

    # --- Stage 2 (no fitting required) ---
    df = build_wind(df)
    df = build_vsby_flag(df, args.low_vsby_mi)

    # --- Stage 3: Split ---
    splits = make_splits(len(df), args.train_frac, args.db_frac, buffer_steps)
    print(f"[3] Splits (timesteps) train={splits['train']} database={splits['database']} "
          f"query={splits['query']} buffer={buffer_steps}")

    # --- Stage 4: Fit normalization (including vsby_cap) and token statistics on train only ---
    tr0, tr1 = splits["train"]
    df_train = df.iloc[tr0:tr1]
    df_train_valid = df_train[df_train["row_valid"]]
    norm = fit_norm_stats(df_train_valid)
    vsby_cap = fit_vsby_cap(df_train_valid)
    counts = count_tokens(df_train)
    print(f"[4] Fixed weather classes K_weather={len(WEATHER_CLASSES)}: {WEATHER_CLASSES}")
    print(f"    vsby_cap(train)={vsby_cap:.3f}; train token counts={counts}")

    # --- Stage 2 (requires normalization/class mapping) ---
    feat = build_feature_matrix(df, norm, vsby_cap)
    soft = build_soft_labels(df)
    row_valid = df["row_valid"].to_numpy()
    interp = df["interpolated"].to_numpy()
    times = df.index

    # --- Stage 5: Extract windows and aggregate within each subset ---
    data = {}
    for name, span in splits.items():
        X, Y, meta = segment_windows(span, W, S, feat, soft, row_valid,
                                     interp, times, args.max_interp_ratio)
        data[name] = {"X": X, "Y": Y, "meta": meta}
        n_clear = sum(m["is_clear"] for m in meta)
        print(f"[5] {name}: windows {len(meta)} (clear {n_clear} / "
              f"weather {len(meta) - n_clear})")

    # --- Stage 5b: Downsample clear windows (train / database; query retains the natural distribution) ---
    data["train"] = downsample_clear(data["train"], args.clear_keep_frac, rng)
    data["database"] = downsample_clear(data["database"], args.db_clear_keep_frac, rng)
    for name in ("train", "database"):
        meta = data[name]["meta"]
        n_clear = sum(m["is_clear"] for m in meta)
        print(f"[5b] {name} after downsampling: windows {len(meta)} (clear {n_clear} / "
              f"weather {len(meta) - n_clear})")

    full_vocab = WEATHER_CLASSES + ["CLEAR"]
    K = len(full_vocab)
    severity = np.array([SEVERITY[c] for c in full_vocab], dtype=np.float32)

    # --- Stage 6: query×database weighted RBF + target-density calibration ---
    Yq, Yd = data["query"]["Y"], data["database"]["Y"]
    q_is_clear = np.array([m["is_clear"] for m in data["query"]["meta"]],
                          dtype=np.uint8)
    sigma = estimate_sigma(data["train"]["Y"], severity, rng)
    if Yq.shape[0] and Yd.shape[0]:
        d2 = weighted_sqdist(Yq, Yd, severity)
        S_sim = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        weather_rows = np.where(q_is_clear == 0)[0]
        tau = args.tau if args.tau is not None else \
            calibrate_tau(S_sim, weather_rows, args.target_density)
        S_qd = (S_sim >= tau).astype(np.uint8)
        overall_density = float(S_qd.mean())
        weather_density = float(S_qd[weather_rows].mean()) if weather_rows.size else 0.0
        print(f"[6] sigma={sigma:.4f} tau={tau:.4f} "
              f"S(query×database)={S_qd.shape} overall density={overall_density:.4f} "
              f"weather density={weather_density:.4f}")
    else:
        tau = args.tau if args.tau is not None else 0.5
        overall_density = weather_density = 0.0
        S_qd = np.empty((Yq.shape[0], Yd.shape[0]), np.uint8)

    # --- Stage 7: Save ---
    for name in splits:
        np.save(outdir / f"X_{name}.npy", data[name]["X"])
        np.save(outdir / f"Y_{name}.npy", data[name]["Y"])
        pd.DataFrame(data[name]["meta"]).to_csv(
            outdir / f"meta_{name}.csv", index=False)
    np.save(outdir / "S_query_db.npy", S_qd)
    np.save(outdir / "query_is_clear.npy", q_is_clear)

    with open(outdir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab": full_vocab,
            "weather_vocab": WEATHER_CLASSES,
            "clear_index": K - 1,
            "class_map": CLASS_MAP,
            "intensity_weight": INTENSITY_WEIGHT,
            "vicinity_weight": VICINITY_WEIGHT,
            "severity_weight": severity.tolist(),
            "train_token_counts": counts,
            "drop_descriptors": sorted(DROP_DESCRIPTORS),
        }, f, ensure_ascii=False, indent=2)

    with open(outdir / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump({"feature_order": FEATURE_ORDER, "stats": norm,
                   "vsby_cap": vsby_cap}, f, ensure_ascii=False, indent=2)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        cfg = vars(args).copy()
        cfg.update({"buffer_steps": buffer_steps, "K": K,
                    "D": len(FEATURE_ORDER), "feature_order": FEATURE_ORDER,
                    "weather_classes": WEATHER_CLASSES,
                    "severity_weight": severity.tolist(),
                    "vsby_cap": vsby_cap, "sigma": sigma, "tau": tau,
                    "pos_density_overall": overall_density,
                    "pos_density_weather": weather_density,
                    "similarity": "weighted_rbf", "supervision": "binary",
                    "aggregation": "mean_pool"})
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[7] Saved to {outdir.resolve()}")
