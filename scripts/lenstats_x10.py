#!/usr/bin/env python3
"""Sequence-length statistics for preprocessed XERON-1.0 item files.

Loads each `--items` file **one at a time** (so a 4096 + 8192 comparison never holds both in
RAM), and reports the token-length distribution plus the long-context tail:

  items, p50/p90/p99/max/mean ids, how many items exceed a threshold (4096 by default),
  markers/target length mismatches and target-sum violations (cheap integrity counters).

Writes atomically.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/lenstats_x10.py \
        --items 4096=train_items_x10.pt \
        --items 8192=train_items_x10_8192.pt \
        --json-out data/xeron10_lenstats_8192.json
"""
import argparse
import json
import os
import sys

import torch


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def stats_for(path, thresholds):
    items = torch.load(path, weights_only=False)
    lens = []
    mismatch = bad_sum = empty = 0
    over = {t: 0 for t in thresholds}
    for it in items:
        L = len(it["ids"])
        lens.append(L)
        for t in thresholds:
            if L > t:
                over[t] += 1
        if len(it["markers"]) != len(it["target"]):
            mismatch += 1
        if not it["markers"]:
            empty += 1
        s = sum(it["target"])
        if abs(s - 1.0) > 1e-6:
            bad_sum += 1
    lens.sort()
    del items
    n = len(lens)
    out = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "items": n,
        "ids_p50": percentile(lens, 0.50),
        "ids_p90": percentile(lens, 0.90),
        "ids_p99": percentile(lens, 0.99),
        "ids_p999": percentile(lens, 0.999),
        "ids_max": lens[-1] if lens else 0,
        "ids_mean": round(sum(lens) / n, 1) if n else 0.0,
        "items_gt": {str(t): over[t] for t in thresholds},
        "items_gt_share": {str(t): round(over[t] / (n or 1), 6) for t in thresholds},
        "markers_target_mismatch": mismatch,
        "empty_markers": empty,
        "target_sum_violations": bad_sum,
    }
    del lens
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", action="append", required=True,
                    help="label=path (repeatable); processed one at a time")
    ap.add_argument("--thresholds", default="4096,8192")
    ap.add_argument("--json-out", default="data/xeron10_lenstats_8192.json")
    args = ap.parse_args()

    thresholds = [int(t) for t in args.thresholds.split(",") if t.strip()]
    res = {"thresholds": thresholds, "files": []}
    for spec in args.items:
        label, _, path = spec.partition("=")
        s = stats_for(path, thresholds)
        s["label"] = label
        res["files"].append(s)
        print(f"[{label}] items={s['items']:,} size={s['size_bytes'] / 1e9:.2f}GB "
              f"p50={s['ids_p50']} p90={s['ids_p90']} p99={s['ids_p99']} "
              f"p99.9={s['ids_p999']} max={s['ids_max']} mean={s['ids_mean']} "
              f">4096={s['items_gt'].get('4096', 0):,} "
              f"({s['items_gt_share'].get('4096', 0):.4%}) "
              f"target_violations={s['target_sum_violations']} "
              f"marker_mismatch={s['markers_target_mismatch']}", flush=True)

    tmp = f"{args.json_out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.json_out)
    print(f"-> {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
