#!/usr/bin/env python3
"""Build the long-context (>1024 tokens) XERON-1.0 subset, length-sorted.

collate_train_batch pads to the max length *inside* each micro-batch, so sorting by
length makes every batch pad-efficient (uniform-8192 padding wastes 97.9%).

Usage: .venv/bin/python scripts/build_long1024_subset.py \
         --src train_items_x10_8192.pt --out train_items_x10_long1024.pt --min-len 1024
"""
import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="train_items_x10_8192.pt")
    ap.add_argument("--out", default="train_items_x10_long1024.pt")
    ap.add_argument("--min-len", type=int, default=1024)
    ap.add_argument("--stats", default="data/xeron10_long1024_stats.json")
    args = ap.parse_args()

    items = torch.load(args.src, weights_only=False)
    print(f"[load] {len(items):,} items from {args.src}", flush=True)

    long = [it for it in items if len(it["ids"]) > args.min_len]
    del items
    long.sort(key=lambda it: len(it["ids"]))
    print(f"[filter] >{args.min_len} tokens: {len(long):,} items", flush=True)

    lens = [len(it["ids"]) for it in long]
    tok = sum(lens)
    buckets = {}
    for lo, hi in ((1024, 2048), (2048, 4096), (4096, 8192)):
        sel = [L for L in lens if lo < L <= hi]
        buckets[f"{lo+1}-{hi}"] = {"items": len(sel), "real_tokens": sum(sel),
                                   "padded_to_hi": len(sel) * hi}
        print(f"  {lo+1}-{hi}: {len(sel):,} items  tokens={sum(sel):,}", flush=True)

    torch.save(long, args.out)
    stats = {
        "source": args.src, "min_len": args.min_len, "items": len(long),
        "real_tokens": tok, "mean_len": round(tok / max(1, len(long)), 1),
        "p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)],
        "max_len": max(lens), "sorted_by": "len(ids) ascending",
        "buckets": buckets,
        "bytes": os.path.getsize(args.out),
    }
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    with open(args.stats, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    print("BUILD_DONE ->", args.out)


if __name__ == "__main__":
    main()
