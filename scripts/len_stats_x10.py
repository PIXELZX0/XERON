#!/usr/bin/env python3
"""Compute the exact token-length distribution of train_items_x10.pt (via data/x10_shards/*.pt).

Writes data/xeron10_len_stats.json: count, mean/percentiles, histogram, and a
compact length-sampling table usable by scripts/bench_xeron10.py DATA_MODE=synth.

Shards are loaded one at a time and dropped, so peak RSS stays ~2.5 GB.
"""
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN_W = 32  # length-histogram bin width


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main():
    shards = sorted(glob.glob(os.path.join(ROOT, "data", "x10_shards", "shard_*.pt")))
    if not shards:
        raise SystemExit("no shards found")
    n = 0
    total_len = 0
    total_markers = 0
    hist = {}
    lens = []            # sampled lengths for percentiles (every 97th item)
    at_cap = {4096: 0, 2048: 0}
    per_shard = []
    for sp in shards:
        items = torch.load(sp, weights_only=False)
        s_n = len(items)
        s_sum = 0
        s_max = 0
        for i, it in enumerate(items):
            L = len(it["ids"])
            s_sum += L
            if L > s_max:
                s_max = L
            total_markers += len(it["markers"])
            b = (L // BIN_W) * BIN_W
            hist[b] = hist.get(b, 0) + 1
            if L >= 4096:
                at_cap[4096] += 1
            if L >= 2048:
                at_cap[2048] += 1
            if i % 97 == 0:
                lens.append(L)
        n += s_n
        total_len += s_sum
        per_shard.append({"shard": os.path.basename(sp), "items": s_n,
                          "sum_len": s_sum, "mean_len": round(s_sum / s_n, 2),
                          "max_len": s_max})
        print(f"[len] {os.path.basename(sp)} items={s_n} mean={s_sum/s_n:.1f} max={s_max}",
              flush=True)
        del items

    lens.sort()
    out = {
        "source": "data/x10_shards/shard_*.pt (train_items_x10.pt, max_len=4096 cap)",
        "items": n,
        "mean_len": round(total_len / n, 3),
        "total_tokens": total_len,
        "sampled": len(lens),
        "percentiles": {f"p{q}": pct(lens, q) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)},
        "at_or_over_2048": at_cap[2048],
        "at_or_over_4096": at_cap[4096],
        "mean_markers": round(total_markers / n, 3),
        "bin_width": BIN_W,
        "hist": {str(k): v for k, v in sorted(hist.items())},
        "per_shard": per_shard,
    }
    p = os.path.join(ROOT, "data", "xeron10_len_stats.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, p)
    print(f"[len] wrote {p}")
    print(f"[len] items={n} mean={out['mean_len']} tokens={total_len:,} "
          f"p50={out['percentiles']['p50']} p90={out['percentiles']['p90']} "
          f"p99={out['percentiles']['p99']} >=4096={at_cap[4096]}")


if __name__ == "__main__":
    sys.exit(main())
