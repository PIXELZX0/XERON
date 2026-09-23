#!/usr/bin/env python3
"""Build a Laya-format commonsense-continuation dataset from HellaSwag (Rowan/hellaswag).

Why: HellaSwag is the canonical 4-way "which ending is plausible?" task — a direct
match for the Laya choice head, and a dense source of the pragmatic/commonsense
decisions JevBench's reasoning tier probes. Not present in JevBench at all.

Split used: validation (10,042 rows). The `ctx` / `endings` fields are the only
inputs copied into `state`; `label` (the gold index) is never written.

Output rows follow the Laya native format consumed by scripts/preprocess.py.

Usage:
    python scripts/build_hellaswag_dataset.py --output data/hellaswag_typed.jsonl [--target 10000]
"""
import argparse
import os
import random

from datasets import load_dataset

from laya_choice_utils import build_row, dedupe_key, write_jsonl

REPO = "Rowan/hellaswag"
SPLIT = "validation"
WORKFLOW = "hellaswag-continuation"
INSTRUCTIONS = (
    "Which ending most plausibly continues the given context? "
    "Pick the single most likely continuation."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/hellaswag_typed.jsonl")
    ap.add_argument("--target", type=int, default=10000, help="rows to write (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    try:
        ds = load_dataset(REPO, split=SPLIT)
    except Exception as e:                           # noqa: BLE001
        raise SystemExit(f"could not load {REPO}/{SPLIT}: {type(e).__name__}: {e}")
    print(f"  {SPLIT}: {len(ds):,} rows")

    pool = []
    for row in ds:
        ctx = row.get("ctx") or f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}"
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            continue
        pool.append((row.get("activity_label") or "", ctx, list(row["endings"]), label))
    print(f"pool: {len(pool):,} rows")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    target = args.target or len(pool)
    rows, skipped, seen = [], 0, set()
    for activity, ctx, endings, label in pool:
        rec = build_row(
            guid=f"hellaswag/validation/{len(rows)}",
            workflow=WORKFLOW,
            state={"activity": activity, "context": " ".join(str(ctx).split())},
            instructions=INSTRUCTIONS,
            options=endings,
            correct_idx=label,
            rng=rng,
            choices_key="endings",
        )
        if rec is None:
            skipped += 1
            continue
        k = dedupe_key(rec["state"])
        if k in seen:
            skipped += 1
            continue
        seen.add(k)
        rows.append(rec)
        if len(rows) >= target:
            break

    write_jsonl(args.output, rows)
    print(f"\nWrote {len(rows):,} rows -> {args.output} (skipped {skipped})")
    print("example:", rows[0]["state"][:300])


if __name__ == "__main__":
    main()
