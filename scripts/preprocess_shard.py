#!/usr/bin/env python3
"""Sharded driver for XERON-1.0 preprocessing (wraps `preprocess.build_training_item`).

Splits the input JSONL(s) into `--num-shards` **equal-row-count** windows using a cached
line-offset index (`--index-file`, built once with `--build-index-only`), so every row is
processed exactly once and no worker gets a long-context tail by accident. All item
construction is `preprocess.build_training_item` — imported, never re-implemented.

Build the index (one pass, ~1 min for 3.4 M rows):

    PYTHONPATH=scripts .venv/bin/python scripts/preprocess_shard.py \
        --data-files data/xeron10_mix.jsonl --index-file data/x10_shards/mix.lineidx.npz \
        --build-index-only

Run 10 shards on 10 of 12 cores:

    cd ~/XERON && mkdir -p data/x10_shards
    seq 0 9 | xargs -P 10 -I{} env PYTHONPATH=scripts MAX_LEN=4096 HEAD_MAX_LEN=256 \
        .venv/bin/python scripts/preprocess_shard.py \
        --data-files data/xeron10_mix.jsonl --index-file data/x10_shards/mix.lineidx.npz \
        --shard {} --num-shards 10 \
        --output data/x10_shards/shard_{}.pt --model-id ~/laya-models/xeron-0.9-base
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

from preprocess import build_training_item                   # noqa: E402  (shared logic)
from laya.agent import _fix_tokenizer_config                 # noqa: E402
from ctx_extend import ensure_long_context                    # noqa: E402


def build_index(files, index_path):
    """Record the byte offset of every line start, per file, into an .npz."""
    all_off, counts = [], []
    for p in files:
        n = 0
        with open(p, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():          # empty lines are not rows
                    continue
                all_off.append(pos)
                n += 1
        counts.append(n)
        print(f"  indexed {os.path.basename(p)}: {n:,} rows", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    np.savez(index_path, offsets=np.asarray(all_off, dtype=np.int64),
             counts=np.asarray(counts, dtype=np.int64),
             sizes=np.asarray([os.path.getsize(p) for p in files], dtype=np.int64))
    print(f"index -> {index_path} ({sum(counts):,} rows total)")


def load_index(index_path, files):
    z = np.load(index_path)
    counts, sizes = z["counts"], z["sizes"]
    if len(counts) != len(files) or not np.array_equal(
            sizes, np.asarray([os.path.getsize(p) for p in files], dtype=np.int64)):
        raise SystemExit("index is stale (file list/sizes changed) — rebuild it")
    return z["offsets"], counts


def iter_rows(files, offsets, counts, lo, hi):
    """Yield the raw lines of rows [lo, hi) (global row index across `files`).

    Reads sequentially from the first row's offset (the index only stores line starts), so a
    shard costs one pass over its own byte span.
    """
    base = 0
    for p, cnt in zip(files, counts):
        a, b = max(lo, base), min(hi, base + int(cnt))
        if a < b:
            with open(p, "rb") as f:
                f.seek(int(offsets[a]))
                for _ in range(b - a):
                    line = f.readline()
                    while line and not line.strip():      # index skips empty lines
                        line = f.readline()
                    if not line:
                        break
                    yield line
        base += int(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-files", nargs="+", required=True)
    ap.add_argument("--index-file", default=None)
    ap.add_argument("--build-index-only", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output", default=None)
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--model-id", default=os.path.expanduser("~/laya-models/xeron-0.9-base"))
    args = ap.parse_args()

    if not args.index_file:
        raise SystemExit("--index-file is required (line-offset index)")
    if args.build_index_only:
        build_index(args.data_files, args.index_file)
        return

    if args.output is None:
        raise SystemExit("--output is required")
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("shard must be in [0, num_shards)")

    offsets, counts = load_index(args.index_file, args.data_files)
    total = int(counts.sum())
    lo = total * args.shard // args.num_shards
    hi = total * (args.shard + 1) // args.num_shards

    model_dir = args.model_id
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    max_len = int(os.environ.get("MAX_LEN") or 4096)
    head_max_len = int(os.environ.get("HEAD_MAX_LEN") or 256)
    cfg = ensure_long_context(model_dir, max_len, head_max_len=head_max_len)
    cfg["max_len"] = max_len
    cfg["head_max_len"] = head_max_len

    items = []
    stats = {"shard": args.shard, "num_shards": args.num_shards, "model_id": model_dir,
             "max_len": max_len, "head_max_len": head_max_len,
             "row_lo": lo, "row_hi": hi, "rows": 0, "items": 0, "bad_json": 0,
             "no_gold": 0, "item_none": 0, "max_ids": 0}
    for raw in iter_rows(args.data_files, offsets, counts, lo, hi):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            state = json.loads(row["state"])
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
        except Exception:                                      # noqa: BLE001
            stats["bad_json"] += 1
            continue
        stats["rows"] += 1
        for qid, q in questions.items():
            if qid not in gold:
                stats["no_gold"] += 1
                continue
            it = build_training_item(tok, cfg, state, q, gold[qid])
            if it is None:
                stats["item_none"] += 1
                continue
            items.append(it)
            stats["max_ids"] = max(stats["max_ids"], len(it["ids"]))
    stats["items"] = len(items)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(items, args.output)
    stats_path = args.stats_json or (os.path.splitext(args.output)[0] + ".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[shard {args.shard}/{args.num_shards}] rows={stats['rows']:,} "
          f"items={stats['items']:,} item_none={stats['item_none']} "
          f"bad_json={stats['bad_json']} max_ids={stats['max_ids']} -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()