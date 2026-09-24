#!/usr/bin/env python3
"""Diagnose WHERE XERON-0.4's overconfidence comes from.

0.4 was trained on T4 in fp16 and its fitted temperatures came out [2.25, 1.81, 3.19]
(0.2, trained in bf16, sat at ~1.0). Two competing explanations:

  (A) fp16 storage/arithmetic inflated the logits scale at *inference* time.
      -> reloading the same weights and running in fp32 should restore ~1.0.
  (B) the fp16 *training* itself pushed the learned logits scale up (weights are
      what they are).
      -> fp32 inference keeps the temperatures high (~2+).

This runs the same temperature fit train_ddp.py does at the end of training, but on a
fixed snapshot and in a chosen dtype, so the answer is directly comparable.

Usage:
    python scripts/diag_logit_scale.py <snapshot_dir> [--dtype fp32|fp16] [--n 400]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from safetensors.torch import load_file                      # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from laya.common import build_model                           # noqa: E402
from ctx_extend import ensure_long_context                    # noqa: E402
from train_ddp import collate_train_batch, fit_one_temp       # noqa: E402

QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--items", default="train_items_x3.pt")
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()

    snap = args.snapshot
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    cfg = ensure_long_context(snap, 4096, 32768, 256)
    cfg["max_len"] = 4096
    cfg["head_max_len"] = 256
    cfg["gradient_checkpointing"] = False

    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(snap, "encoder"))
    model.load_state_dict(load_file(os.path.join(snap, "model.safetensors")), strict=True)
    model.to(dtype).eval()
    print(f"[diag] loaded {snap} as {args.dtype}", flush=True)

    all_items = torch.load(args.items, weights_only=False)
    calib = all_items[:: args.stride][: args.n]
    print(f"[diag] fitting temperature on {len(calib)} items", flush=True)

    preds = []
    with torch.no_grad():
        for i in range(0, len(calib), 16):
            chunk = calib[i : i + 16]
            b = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cpu", dtype=dtype, enabled=(dtype != torch.float32)):
                logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                  b["marker_mask"], b["qtype"])
            ln = logits.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                preds.append((it["qtype"], ln[r, :k], it["target"]))

    # raw logit spread is the direct evidence for (A) vs (B)
    import statistics
    spreads = [max(z) - min(z) for _qt, z, _t in preds if len(z) > 1]
    print(f"[diag] mean logit spread (max-min): {statistics.mean(spreads):.3f}", flush=True)

    temps = []
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps.append(round(fit_one_temp(sel), 3))
        else:
            temps.append(None)
    print(f"[diag] RESULT dtype={args.dtype} fitted_temperature={temps}", flush=True)


if __name__ == "__main__":
    main()
