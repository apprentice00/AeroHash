"""Create an independent validation split from the tail of Train (scheme T) for leakage-free model selection.

Motivation
----
The original pipeline used the test protocol (X_query × X_database, ground truth S_query_db) for both training-time model selection/
early stopping and final reporting → selection bias (test-set peeking). This script creates an independent retrieval validation set
chronologically within each station's **Train** split for training-time selection; test is run only once after training.

Splits (ordered by start_idx, all within Train, before database/query, and therefore isolated from test)
    [ ...train_core... ] <buf> [ ...val_db... ] <buf> [ ...val_query... ]
    - train_core: Samples used for training/building database B (= original Train minus the trailing validation pool).
    - val_db    : Validation gallery.
    - val_query : Validation queries.
The buffer spans window+buffer_steps (config.json) start_idx units to ensure non-overlapping windows without leakage.

S_val uses exactly the same similarity definition as test: severity-weighted RBF ≥ tau (verified to reproduce S_query_db with
99.999% agreement). val_is_clear uses the same definition as query_is_clear: Y[:, :-1].sum <= 1e-6.

Outputs (written to each station directory)
    X_train_core.npy / Y_train_core.npy
    X_val_db.npy     / Y_val_db.npy
    X_val_query.npy  / Y_val_query.npy
    S_val.npy (uint8, [n_val_query, n_val_db])
    val_is_clear.npy (uint8, [n_val_query])

Usage::
    aerohash build-val                       # Four stations by default
    aerohash build-val --stations weather
    STATIONS=weather,weather_ZBAD aerohash build-val
    aerohash build-val --val-frac 0.15 --query-frac 0.34
"""
import argparse
import csv
import json
import os

import numpy as np

from aerohash.paths import PROJECT_ROOT

HERE = str(PROJECT_ROOT)
DEFAULT_STATIONS = ["weather", "weather_ZBAD", "weather_ZSPD", "weather_ZSSS"]


def weighted_rbf(t, r, w, sigma):
    """Severity-weighted RBF: S[i,j]=exp(-sum_k w_k (t_ik-r_jk)^2 / 2 sigma^2), float64."""
    a2 = (t * t) @ w
    b2 = (r * r) @ w
    cross = (t * w) @ r.T
    d2 = np.clip(a2[:, None] + b2[None, :] - 2.0 * cross, 0.0, None)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def read_start_idx(meta_csv):
    si = []
    with open(meta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            si.append(int(row["start_idx"]))
    return np.asarray(si, dtype=np.int64)


def build_one(root, val_frac, query_frac, is_clear_eps=1e-6):
    cfg = json.load(open(os.path.join(root, "config.json"), encoding="utf-8"))
    sev = np.asarray(cfg["severity_weight"], dtype=np.float64)
    sigma = float(cfg["sigma"])
    tau = float(cfg["tau"])
    window = int(cfg["window"])
    buffer_steps = int(cfg.get("buffer_steps", 24))
    buf = window + buffer_steps  # start_idx distance threshold ensuring non-overlapping windows

    X = np.load(os.path.join(root, "X_train.npy")).astype(np.float32)
    Y = np.load(os.path.join(root, "Y_train.npy")).astype(np.float32)
    si = read_start_idx(os.path.join(root, "meta_train.csv"))
    N = X.shape[0]
    assert N == len(si) == Y.shape[0], f"{root}: X/Y/meta row counts do not match"
    order = np.argsort(si, kind="stable")  # Sort by start_idx again as a precaution
    X, Y, si = X[order], Y[order], si[order]

    # 1) Validation pool at the tail
    n_pool = int(round(N * val_frac))
    if n_pool < 20:
        raise SystemExit(f"{root}: val_frac too small; validation pool has only {n_pool} samples")
    pool_lo = N - n_pool
    pool_start = int(si[pool_lo])

    # 2) train_core = samples before the pool, separated from its start by a buffer of buf
    core_mask = si < (pool_start - buf)
    core_mask[pool_lo:] = False
    Xc, Yc = X[core_mask], Y[core_mask]

    # 3) Split the pool into val_db (earlier) / val_query (later), separated by buf
    p_si = si[pool_lo:]
    p_X, p_Y = X[pool_lo:], Y[pool_lo:]
    k = int(round(n_pool * (1.0 - query_frac)))
    k = max(1, min(k, n_pool - 1))
    q_start = int(p_si[k])
    db_mask = p_si < (q_start - buf)          # val_db: separated from query start by buf
    q_mask = np.zeros(n_pool, dtype=bool)
    q_mask[k:] = True
    Xvd, Yvd = p_X[db_mask], p_Y[db_mask]
    Xvq, Yvq = p_X[q_mask], p_Y[q_mask]
    if len(Xvd) == 0 or len(Xvq) == 0:
        raise SystemExit(f"{root}: val_db/val_query is empty; increase val_frac or adjust query_frac")

    # 4) S_val (same definition as test: weighted RBF >= tau) and val_is_clear
    S = (weighted_rbf(Yvq.astype(np.float64), Yvd.astype(np.float64), sev, sigma)
         >= tau).astype(np.uint8)
    val_is_clear = (Yvq[:, :-1].sum(axis=1) <= is_clear_eps).astype(np.uint8)

    np.save(os.path.join(root, "X_train_core.npy"), Xc)
    np.save(os.path.join(root, "Y_train_core.npy"), Yc)
    np.save(os.path.join(root, "X_val_db.npy"), Xvd)
    np.save(os.path.join(root, "Y_val_db.npy"), Yvd)
    np.save(os.path.join(root, "X_val_query.npy"), Xvq)
    np.save(os.path.join(root, "Y_val_query.npy"), Yvq)
    np.save(os.path.join(root, "S_val.npy"), S)
    np.save(os.path.join(root, "val_is_clear.npy"), val_is_clear)

    return {
        "station": os.path.basename(root),
        "N_train_orig": N,
        "train_core": int(Xc.shape[0]),
        "val_db": int(Xvd.shape[0]),
        "val_query": int(Xvq.shape[0]),
        "dropped_buffer": int(N - Xc.shape[0] - Xvd.shape[0] - Xvq.shape[0]),
        "val_density": round(float(S.mean()), 4),
        "val_query_clear_frac": round(float(val_is_clear.mean()), 4),
        "buf_startidx": buf,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Build independent validation split (scheme T)")
    env_st = os.environ.get("STATIONS")
    default_st = [s.strip() for s in env_st.split(",")] if env_st else DEFAULT_STATIONS
    p.add_argument("--stations", nargs="+", default=default_st,
                   help="Station directory names under dataset/")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of the Train tail to use as the validation pool")
    p.add_argument("--query-frac", type=float, default=0.34,
                   help="Fraction of the validation pool used as val_query (the remainder is val_db)")
    args = p.parse_args(argv)

    print(f"[build_val] stations={args.stations} val_frac={args.val_frac} "
          f"query_frac={args.query_frac}")
    for st in args.stations:
        root = os.path.normpath(os.path.join(HERE, "dataset", st))
        if not os.path.isdir(root):
            print(f"[skip] Directory does not exist: {root}")
            continue
        info = build_one(root, args.val_frac, args.query_frac)
        print("[ok] " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
