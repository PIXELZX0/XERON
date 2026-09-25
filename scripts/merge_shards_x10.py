#!/usr/bin/env python3
"""Merge preprocess shards into `train_items_x10.pt`.

Writes atomically (`<out>.tmp.<pid>` + `os.replace`) and records sha256/size in the stats
file. A re-run whose `<out>` is newer than every shard and matches the recorded size/items is
a no-op (idempotent; override with `--force`).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/merge_shards_x10.py \
        --shards "data/x10_shards/shard_*.pt" --output train_items_x10.pt \
        --stats data/xeron10_preprocess_stats.json
"""
import argparse
import glob
import hashlib
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
    ap.add_argument("--force", action="store_true",
                    help="re-merge even when --output already matches --stats")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards), key=shard_index)
    if not paths:
        raise SystemExit(f"no shards matched {args.shards}")

    if not args.force and os.path.exists(args.output) and os.path.exists(args.stats):
        try:
            prev = json.load(open(args.stats))
            newest_shard = max(os.path.getmtime(p) for p in paths)
            complete = (prev.get("size_bytes") == os.path.getsize(args.output)
                        and prev.get("items") == sum(s.get("items", 0)
                                                     for s in prev.get("shards", []))
                        and prev.get("shards") and len(prev["shards"]) == len(paths)
                        and os.path.getmtime(args.output) > newest_shard)
            if complete:
                print(f"[skip] {args.output} already merged from {len(paths)} shards "
                      f"({prev['items']:,} items, {prev['size_bytes']:,} B) — nothing to do")
                return
            print(f"[merge] {args.output} is stale vs {args.stats} — re-merging", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[merge] cannot validate {args.output}: {type(e).__name__}: {e}", flush=True)

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

    tmp = f"{args.output}.tmp.{os.getpid()}"
    torch.save(merged, tmp)
    os.replace(tmp, args.output)           # atomic: a half-written items file never appears
    h = hashlib.sha256()
    with open(args.output, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)

    stats = {"output": args.output, "shards": per_shard, "items": len(merged),
             "qtype_counts": {str(k): v for k, v in sorted(qt.items())},
             "max_ids_len": max_ids, "size_bytes": os.path.getsize(args.output),
             "sha256": h.hexdigest(), "atomic_write": "tmp + os.replace"}
    tmp_js = f"{args.stats}.tmp.{os.getpid()}"
    with open(tmp_js, "w") as f:
        json.dump(stats, f, indent=1)
    os.replace(tmp_js, args.stats)
    print(f"merged {len(merged):,} items -> {args.output} "
          f"({stats['size_bytes'] / 1e9:.2f} GB, max_ids={max_ids})")
    print(f"sha256 {stats['sha256']}")
    print(f"-> {args.stats}")


if __name__ == "__main__":
    main()