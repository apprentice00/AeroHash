"""Hamming ranking, ordinary/tie-aware AP and stratified retrieval metrics."""
from __future__ import annotations


import numpy as np


def to_pm1(codes: np.ndarray) -> np.ndarray:
    """Convert to {-1,+1} for Hamming ranking via inner products."""
    codes = np.asarray(codes)
    uniq = np.unique(codes)
    if set(uniq.tolist()) <= {0, 1}:
        codes = codes * 2 - 1
    return np.sign(codes).astype(np.float32)


def rank_by_hamming(codes_q: np.ndarray, codes_db: np.ndarray) -> np.ndarray:
    """Return database ranking indices for each query (nearest first)."""
    q = to_pm1(codes_q)
    d = to_pm1(codes_db)
    return np.argsort(-(q @ d.T), axis=1)


def rank_and_dist_by_hamming(codes_q: np.ndarray,
                             codes_db: np.ndarray,
                             batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return (rank, dist_sorted).

    rank[i]:        Database indices sorted by ascending Hamming distance for query i (nearest first).
    dist_sorted[i]: Hamming distances (int16) aligned with rank[i], used to identify tied-distance buckets.

    Process queries in batches using float32/int16 throughout to avoid float64 memory spikes from large matrices (container memory is limited).
    """
    q = to_pm1(codes_q)
    d = to_pm1(codes_db)
    bits = q.shape[1]
    nq, ndb = q.shape[0], d.shape[0]
    d_t = d.T
    rank = np.empty((nq, ndb), dtype=np.int32)
    dist_sorted = np.empty((nq, ndb), dtype=np.int16)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        dist = (bits - q[s:e] @ d_t) * np.float32(0.5)  # (b, ndb) float32
        r = np.argsort(dist, axis=1)
        rank[s:e] = r.astype(np.int32)
        dist_sorted[s:e] = np.rint(
            np.take_along_axis(dist, r, axis=1)).astype(np.int16)
    return rank, dist_sorted


def _average_precision(rel_sorted: np.ndarray) -> float:
    n_rel = rel_sorted.sum()
    if n_rel == 0:
        return np.nan
    cum = np.cumsum(rel_sorted)
    ranks = np.arange(1, len(rel_sorted) + 1)
    prec = cum / ranks
    return float((prec * rel_sorted).sum() / n_rel)


def _tie_aware_average_precision(rel_sorted: np.ndarray,
                                 dist_sorted: np.ndarray) -> float:
    """Expected AP under random ordering within tied-distance buckets (McSherry & Najork, 2008, closed form).

    Equivalent to averaging infinitely many random tie-breakings among samples at the same Hamming distance,
    computed in one pass and thus independent of the original temporal index order. rel_sorted is binary relevance.

    Vectorized implementation: form buckets of equal distance and sum the closed form via segment reductions.
    No Python loop per query; suitable for long codes with few ties (many size-1 buckets).
    """
    rel = (np.asarray(rel_sorted) > 0).astype(np.float64)
    dist = np.asarray(dist_sorted)
    R = rel.sum()
    if R == 0:
        return np.nan

    n = len(rel)
    ranks = np.arange(1, n + 1, dtype=np.float64)      # Global 1-based ranks
    recip = 1.0 / ranks
    # Start index of each tied-distance bucket (0-based)
    starts = np.flatnonzero(
        np.concatenate(([True], dist[1:] != dist[:-1])))
    n_before = starts.astype(np.float64)               # Number of samples before each bucket
    n_g = np.diff(np.append(starts, n)).astype(np.float64)  # Bucket size
    r_g = np.add.reduceat(rel, starts)                 # Relevant samples in each bucket
    s1 = np.add.reduceat(recip, starts)                # Sum of 1/rank within each bucket
    r_before = np.concatenate(([0.0], np.cumsum(r_g)[:-1]))  # Relevant samples before each bucket
    slope = np.where(n_g > 1, (r_g - 1.0) / np.where(n_g > 1, n_g - 1.0, 1.0), 0.0)
    contrib = r_g / n_g * (
        (r_before + 1.0) * s1 + slope * (n_g - (n_before + 1.0) * s1))
    return float(contrib.sum() / R)


def bucket_stats(codes_db: np.ndarray, codes_q: np.ndarray | None = None) -> dict:
    """Diagnose code-space collapse: bucket sizes / collision rate / unique-code ratio / bit balance and entropy."""
    keys = (to_pm1(codes_db) > 0).astype(np.uint8)
    n, bits = keys.shape
    _, counts = np.unique(keys, axis=0, return_counts=True)
    p = keys.mean(axis=0)  # Fraction of ones for each bit
    eps = 1e-12
    ent = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))

    out = {
        "n_db": int(n),
        "bits": int(bits),
        "n_unique_codes": int(len(counts)),
        "unique_code_ratio": round(float(len(counts) / n), 4),
        "max_bucket": int(counts.max()),
        "mean_bucket": round(float(counts.mean()), 2),
        "median_bucket": round(float(np.median(counts)), 2),
        "collision_rate": round(float(counts[counts > 1].sum() / n), 4),
        "bit_balance_mean": round(float(p.mean()), 4),
        "bit_balance_min": round(float(p.min()), 4),
        "bit_balance_max": round(float(p.max()), 4),
        "bit_entropy_mean": round(float(ent.mean()), 4),
    }
    if codes_q is not None:
        kq = (to_pm1(codes_q) > 0).astype(np.uint8)
        out["n_unique_query_codes"] = int(len(np.unique(kq, axis=0)))
        out["n_query"] = int(kq.shape[0])
    return out


def compute_metrics(rank: np.ndarray, S: np.ndarray, k_list: list[int],
                    idx: np.ndarray, dist_sorted: np.ndarray | None = None) -> dict:
    """Compute mAP and P@k / R@k for the query subset specified by idx.

    If dist_sorted is provided, also report tie-aware mAP (expected ordering within tied-distance buckets),
    to diagnose whether the original mAP is affected by temporal index ordering within ties.
    """
    if len(idx) == 0:
        return {"n_query": 0}
    aps, tie_aps, pk = [], [], {k: [] for k in k_list}
    rk = {k: [] for k in k_list}
    for i in idx:
        rel = S[i, rank[i]].astype(np.float32)
        ap = _average_precision(rel)
        if not np.isnan(ap):
            aps.append(ap)
        if dist_sorted is not None:
            tap = _tie_aware_average_precision(rel, dist_sorted[i])
            if not np.isnan(tap):
                tie_aps.append(tap)
        n_rel = rel.sum()
        for k in k_list:
            pk[k].append(rel[:k].mean())
            rk[k].append(rel[:k].sum() / n_rel if n_rel > 0 else np.nan)
    out = {
        "n_query": int(len(idx)),
        "n_query_with_pos": int(len(aps)),
        "mAP": round(float(np.mean(aps)), 4) if aps else None,
    }
    if dist_sorted is not None:
        out["mAP_tie_aware"] = round(float(np.mean(tie_aps)), 4) if tie_aps else None
    for k in k_list:
        out[f"P@{k}"] = round(float(np.mean(pk[k])), 4)
        valid = [x for x in rk[k] if not np.isnan(x)]
        out[f"R@{k}"] = round(float(np.mean(valid)), 4) if valid else None
    return out


def stratified_report(rank, S, q_is_clear, k_list, dist_sorted=None) -> dict:
    all_idx = np.arange(S.shape[0])
    weather_idx = np.where(q_is_clear == 0)[0]
    clear_idx = np.where(q_is_clear == 1)[0]
    return {
        "overall": compute_metrics(rank, S, k_list, all_idx, dist_sorted),
        "weather_query": compute_metrics(rank, S, k_list, weather_idx, dist_sorted),
        "clear_query": compute_metrics(rank, S, k_list, clear_idx, dist_sorted),
    }
