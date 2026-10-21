"""Evaluate a full-model checkpoint with ordinary and tie-aware metrics."""
import glob
import json
import os
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import yaml
from aerohash.paths import PROJECT_ROOT
from aerohash.config import parse_config
from aerohash.checkpoints import load_checkpoint
from aerohash.evaluation.encoding import encode_array
from aerohash.evaluation import metrics as wx_eval

def latest_ckpt(root="checkpoints"):
    """Return the most recently modified model.t under checkpoints/, or None if absent."""
    cands = glob.glob(os.path.join(root, "**", "model.t"), recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None



def main(argv=None):
    config = parse_config('evaluate', argv)
    args = config.evaluation
    args.datadir = args.datadir if args.datadir is not None else str(PROJECT_ROOT/'dataset/weather')
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    args.ckpt = args.ckpt or latest_ckpt(config.output.ckpt_root)
    if args.ckpt is None:
        raise SystemExit('model.t not found; train a model first or specify --ckpt')
    if args.outdir is None:
        args.outdir = os.path.dirname(args.ckpt) or '.'
    device = args.device

    model = load_checkpoint(args.ckpt, device)
    model.to(device)
    print(f"[ckpt] {args.ckpt} -> device={device}")

    Xq = np.load(os.path.join(args.datadir, "X_query.npy")).astype(np.float32)
    Xd = np.load(os.path.join(args.datadir, "X_database.npy")).astype(np.float32)

    codes_q = encode_array(model.eval(), Xq, device, args.batch_size)
    codes_d = encode_array(model, Xd, device, args.batch_size)
    print(f"[encode] query={codes_q.shape} database={codes_d.shape} "
          f"(bits={codes_q.shape[1]})")

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "codes_query.npy"), codes_q)
    np.save(os.path.join(args.outdir, "codes_db.npy"), codes_d)

    S = np.load(os.path.join(args.datadir, "S_query_db.npy"))
    q_is_clear = np.load(os.path.join(args.datadir, "query_is_clear.npy"))
    assert codes_q.shape[0] == S.shape[0]
    assert codes_d.shape[0] == S.shape[1]

    rank, dist_sorted = wx_eval.rank_and_dist_by_hamming(codes_q, codes_d)
    report = wx_eval.stratified_report(rank, S, q_is_clear, args.k, dist_sorted)
    report["bucket_stats"] = wx_eval.bucket_stats(codes_d, codes_q)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    with open(os.path.join(args.outdir, "metrics_weather.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[save] {os.path.abspath(args.outdir)}")

    Path(args.outdir, 'resolved_evaluation.yaml').write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding='utf-8')
