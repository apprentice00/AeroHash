"""Summarize repeated runs across multiple seeds (with scripts/run_multiseed.sh).

Read runs_multiseed/manifest.tsv (one run and its checkpoint directory per row).
For each checkpoint:
    - Read weather/overall/clear mAP and weather R@100 from metrics_weather.json;
    - Compute q_uniq from codes_db.npy (the number of unique database codewords);
then group by tag and report mean ± sample standard deviation (n = valid seeds per group).

Usage:
  aerohash collect                                   # Default manifest
  aerohash collect --manifest runs_multiseed/manifest.tsv --code-length 96
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def q_uniq_from_codes(ckpt_dir):
    """Return the number of unique database codewords, or None if unavailable."""
    path = os.path.join(ckpt_dir, "codes_db.npy")
    if not os.path.isfile(path):
        return None
    codes = np.load(path)
    return int(np.unique(codes, axis=0).shape[0])


def load_one(ckpt_dir):
    mpath = os.path.join(ckpt_dir, "metrics_weather.json")
    if not os.path.isfile(mpath):
        return None
    with open(mpath, encoding="utf-8") as f:
        rep = json.load(f)
    return {
        "weather_mAP": rep.get("weather_query", {}).get("mAP"),
        "overall_mAP": rep.get("overall", {}).get("mAP"),
        "clear_mAP": rep.get("clear_query", {}).get("mAP"),
        "weather_R@100": rep.get("weather_query", {}).get("R@100"),
        "q_uniq": q_uniq_from_codes(ckpt_dir),
    }


def fmt_stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  —  "
    arr = np.array(vals, dtype=float)
    if len(arr) == 1:
        return f"{arr[0]:.4f} (n=1)"
    return f"{arr.mean():.4f}±{arr.std(ddof=1):.4f} (n={len(arr)})"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs_multiseed/manifest.tsv")
    p.add_argument("--code-length", type=int, default=96)
    args = p.parse_args(argv)

    if not os.path.isfile(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest} (run scripts/run_multiseed.sh first)")

    groups = defaultdict(list)  # tag -> list of metric dicts
    with open(args.manifest, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            m = load_one(row["ckpt_dir"])
            if m is None:
                print(f"[skip] Missing metrics_weather.json: {row['ckpt_dir']}")
                continue
            m["_seed"] = row.get("seed")
            groups[row["tag"]].append(m)

    if not groups:
        raise SystemExit("No valid runs with status=ok in the manifest.")

    metrics = ["weather_mAP", "overall_mAP", "clear_mAP", "weather_R@100", "q_uniq"]
    header = ["tag"] + metrics
    widths = [max(len(h), 22) for h in header]

    print("\n=== Multi-seed summary (mean ± sample std) ===")
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for tag in sorted(groups):
        rows = groups[tag]
        cells = [tag.ljust(widths[0])]
        for i, mkey in enumerate(metrics, start=1):
            cells.append(fmt_stat([r[mkey] for r in rows]).ljust(widths[i]))
        print(" | ".join(cells))

    print("\n=== Details (per run) ===")
    for tag in sorted(groups):
        for r in groups[tag]:
            print(f"  {tag} seed={r['_seed']}: "
                  f"weather={r['weather_mAP']} overall={r['overall_mAP']} "
                  f"R@100={r['weather_R@100']} q_uniq={r['q_uniq']}")


if __name__ == "__main__":
    main()
