#!/usr/bin/env python3
"""Merge preprocess shards into `train_items_x10.pt`.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/merge_shards_x10.py \
        --shards "data/x10_shards/shard_*.pt" --output train_items_x10.pt \
        --stats data/xeron10_preprocess_stats.json
"""
import argparse
import glob
import json
import os
import re
import sys

import torch


def shard_index(path):
    m = re.search(r"shard_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/x10_shards/shard_*.pt")
    ap.add_argument("--output", default="train_items_x10.pt")
    ap.add_argument("--stats", default="data/xeron10_preprocess_stats.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards), key=shard_index)
    if not paths:
        raise SystemExit(f"no shards matched {args.shards}")

    merged, per_shard, max_ids = [], [], 0
    qt = {}
    for p in paths:
        items = torch.load(p, weights_only=False)
        per_shard.append({"shard": shard_index(p), "file": p, "items": len(items)})
        for it in items:
            max_ids = max(max_ids, len(it["ids"]))
            qt[it["qtype"]] = qt.get(it["qtype"], 0) + 1
        merged.extend(items)
        print(f"  + {os.path.basename(p):16} {len(items):>9,} items (total {len(merged):,})",
              flush=True)

    torch.save(merged, args.output)
    stats = {"output": args.output, "shards": per_shard, "items": len(merged),
             "qtype_counts": {str(k): v for k, v in sorted(qt.items())},
             "max_ids_len": max_ids, "size_bytes": os.path.getsize(args.output)}
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"merged {len(merged):,} items -> {args.output} "
          f"({stats['size_bytes'] / 1e9:.2f} GB, max_ids={max_ids})")
    print(f"-> {args.stats}")


if __name__ == "__main__":
    main()