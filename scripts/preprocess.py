#!/usr/bin/env python3
"""Preprocess a Laya-format dataset into tokenized training items (train_items.pt).

Usage:
    python scripts/preprocess.py \
        --dataset LocalLLaMA/typed-decisions \
        --config-name all \
        --split train \
        --output train_items.pt

Custom datasets must follow the Laya native format — each row contains:
    state     : string (JSON)  — the input state (text/email/ticket/JSON)
    questions : string (JSON)  — {qid: {"type": choice|score|noul, "instructions": ..., "criteria": ...}}
    gold      : string (JSON)  — {qid: {"probabilities": {...}}}
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES

MODEL_ID = "convaiinnovations/laya"


def build_training_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5),
                  gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))

    seq, markers = build_sequence(
        tok, state,
        {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != k:
        return None
    return {
        "ids": list(seq),
        "markers": list(markers),
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LocalLLaMA/typed-decisions")
    ap.add_argument("--config-name", default="all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-files", default=None,
                    help="Local JSON/JSONL file(s) to load instead of a HF dataset "
                         "(e.g. data/korean_typed.jsonl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of rows processed (0 = all)")
    ap.add_argument("--output", default="train_items.pt")
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    print(f"Fetching tokenizer and config from {args.model_id}...")
    if os.path.isdir(args.model_id):
        model_dir = args.model_id
    else:
        model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    # Context extension: override max lengths via env (must match train_ddp.py)
    if os.environ.get("MAX_LEN"):
        cfg["max_len"] = int(os.environ["MAX_LEN"])
    if os.environ.get("HEAD_MAX_LEN"):
        cfg["head_max_len"] = int(os.environ["HEAD_MAX_LEN"])
    print(f"Context: max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    if args.data_files:
        print(f"Loading local data from {args.data_files}...")
        ds = load_dataset("json", data_files=args.data_files, split="train")
    else:
        print(f"Loading dataset {args.dataset} ({args.config_name} / {args.split})...")
        ds = load_dataset(args.dataset, args.config_name, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limited to {len(ds)} rows")

    items = []
    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(tok, cfg, state, q, gold[qid])
                if it:
                    items.append(it)

    print(f"Preprocessed {len(items)} training sequences.")
    torch.save(items, args.output)
    print(f"Saved items to {args.output}")


if __name__ == "__main__":
    main()