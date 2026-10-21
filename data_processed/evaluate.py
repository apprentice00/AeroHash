"""Compatibility interface for baseline evaluators. Install with pip install -e . first."""
import argparse
import json
from pathlib import Path
import numpy as np

from aerohash.evaluation.metrics import (
    to_pm1, rank_by_hamming, _average_precision,
    compute_metrics as _compute_metrics, stratified_report as _stratified_report,
)


def compute_metrics(rank, S, k_list, idx):
    return _compute_metrics(rank, S, k_list, idx)


def stratified_report(rank, S, q_is_clear, k_list):
    return _stratified_report(rank, S, q_is_clear, k_list)

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified retrieval evaluation")
    p.add_argument("--datadir", default="processed")
    p.add_argument("--codes-query", help="(n_query, bits) binary codes in .npy format")
    p.add_argument("--codes-db", help="(n_db, bits) binary codes in .npy format")
    p.add_argument("--k", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--demo", action="store_true",
                   help="Use label vectors as pseudo-codes to check the metric implementation without a model")
    return p.parse_args(argv)

def main(argv=None) -> None:
    args = parse_args(argv)
    d = Path(args.datadir)
    S = np.load(d / "S_query_db.npy")
    q_is_clear = np.load(d / "query_is_clear.npy")

    if args.demo:
        Yq = np.load(d / "Y_query.npy")
        Yd = np.load(d / "Y_database.npy")
        codes_q = (Yq > 0).astype(np.int8)
        codes_db = (Yd > 0).astype(np.int8)
        print("[demo] Using binarized label vectors as pseudo-hash codes (only to verify the metric implementation)")
    else:
        if not (args.codes_query and args.codes_db):
            raise SystemExit("Provide --codes-query and --codes-db, or use --demo")
        codes_q = np.load(args.codes_query)
        codes_db = np.load(args.codes_db)

    assert codes_q.shape[0] == S.shape[0], "Query count does not match the number of rows in S"
    assert codes_db.shape[0] == S.shape[1], "Database count does not match the number of columns in S"

    rank = rank_by_hamming(codes_q, codes_db)
    report = stratified_report(rank, S, q_is_clear, args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
