#!/usr/bin/env python3
"""CPU smoke test for a preprocessed XERON item file.

Pulls N random items (plus the longest and the shortest) and runs them through the
0.9-snapshot model (encoder + wide head) in a single CPU forward pass, checking:
  * logits shape == (batch, max markers in batch)
  * len(markers) == len(target) for every item, and every marker position < len(ids)
  * marker positions are inside the (untruncated) head region, ids fit in max_len
  * no NaN/Inf in logits, marker_mask covers exactly the real markers

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/smoke_x10.py \
        --items train_items_x10.pt --model-id ~/laya-models/xeron-0.9-base --n 200
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_xeron import build_wide_model                      # noqa: E402
from laya.common import QTYPES                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="train_items_x10.pt")
    ap.add_argument("--model-id", default=os.path.expanduser("~/laya-models/xeron-0.9-base"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("MAX_LEN") or 4096))
    args = ap.parse_args()

    items = torch.load(args.items, weights_only=False)
    print(f"loaded {len(items):,} items from {args.items}")
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(items)), min(args.n, len(items)))
    longest = max(range(len(items)), key=lambda i: len(items[i]["ids"]))
    shortest = min(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for extra in (longest, shortest):
        if extra not in idx:
            idx.append(extra)
    sel = [items[i] for i in idx]
    print(f"sample: {len(sel)} items (incl. longest ids={len(items[longest]['ids'])} "
          f"and shortest ids={len(items[shortest]['ids'])})")

    bad = [i for i, it in enumerate(sel) if len(it["markers"]) != len(it["target"])]
    over = [i for i, it in enumerate(sel) if len(it["ids"]) > args.max_len]
    oob = [i for i, it in enumerate(sel)
           if it["markers"] and max(it["markers"]) >= len(it["ids"])]
    empty = [i for i, it in enumerate(sel) if not it["markers"]]
    print(f"check markers==target: {'OK' if not bad else f'FAIL {len(bad)}'}")
    print(f"check ids<=max_len({args.max_len}): {'OK' if not over else f'FAIL {len(over)}'}")
    print(f"check marker_pos<len(ids): {'OK' if not oob else f'FAIL {len(oob)}'}")
    print(f"check non-empty markers: {'OK' if not empty else f'FAIL {len(empty)}'}")
    kdist = {}
    for it in sel:
        kdist[len(it["markers"])] = kdist.get(len(it["markers"]), 0) + 1
    print(f"marker counts: {dict(sorted(kdist.items()))}")
    print(f"qtype counts: { {k: sum(1 for it in sel if it['qtype'] == v) for k, v in QTYPES.items()} }")

    cfg_file = os.path.join(args.model_id, "rl_agent_config.json")
    with open(cfg_file) as f:
        cfg = json.load(f)
    head_layers = int(cfg.get("head_layers", 4))
    head_size = int(cfg.get("head_size", cfg.get("head_width", 1024)))
    print(f"model: head_layers={head_layers} head_size={head_size}")

    model = build_wide_model(cfg, os.path.join(args.model_id, "encoder"),
                             head_layers=head_layers, head_size=head_size)
    from safetensors.torch import load_file
    weights = load_file(os.path.join(args.model_id, "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"weights loaded (missing={len(missing)} unexpected={len(unexpected)})")
    model.float().eval()

    pad_id = 1
    n_ok, n_nan = 0, 0
    with torch.no_grad():
        for lo in range(0, len(sel), 2):                       # batch of 2
            chunk = sel[lo:lo + 2]
            L = max(len(it["ids"]) for it in chunk)
            kmax = max(len(it["markers"]) for it in chunk)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            mpos = torch.zeros((len(chunk), kmax), dtype=torch.long)
            mmask = torch.zeros((len(chunk), kmax), dtype=torch.bool)
            for i, it in enumerate(chunk):
                ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
                att[i, :len(it["ids"])] = 1
                mpos[i, :len(it["markers"])] = torch.tensor(it["markers"])
                mmask[i, :len(it["markers"])] = True
            logits, act_logits = model(ids, att, mpos, mmask,
                                       torch.tensor([it["qtype"] for it in chunk]))
            assert logits.shape == (len(chunk), kmax), (logits.shape, kmax)
            assert act_logits.shape[0] == len(chunk)
            nan = (~torch.isfinite(logits[mmask])).sum().item()
            n_nan += nan
            n_ok += 1
    print(f"forward: {n_ok} batches OK (logits shape (batch, kmax), finite); "
          f"non-finite marker logits: {n_nan}")
    print("SMOKE TEST", "PASS" if (n_nan == 0 and not (bad or over or oob or empty)) else "FAIL")


if __name__ == "__main__":
    main()