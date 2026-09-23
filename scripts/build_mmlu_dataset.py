#!/usr/bin/env python3
"""Build a Laya-format knowledge-choice dataset from MMLU (cais/mmlu, config `all`).

Why: JevBench's `mmlu` source only covers the 285-row dev split. MMLU is the single
densest source of "pick the right option" knowledge decisions, and the Laya choice
head already handles 4 options natively.

Splits used: test (14,042) + validation (1,531). The dev split is deliberately
skipped because it is already inside data/jevbench_full.jsonl (no state overlap).

Output rows follow the Laya native format consumed by scripts/preprocess.py.

Usage:
    python scripts/build_mmlu_dataset.py --output data/mmlu_typed.jsonl [--target 15000]
"""
import argparse
import json
import os
import random

from datasets import load_dataset

from laya_choice_utils import build_row, dedupe_key, write_jsonl

REPO = "cais/mmlu"
CONFIG = "all"
SPLITS = ("test", "validation")          # dev = already in jevbench_full.jsonl
WORKFLOW = "mmlu-choice"
INSTRUCTIONS = "Which option is the correct answer to `question`?"
LETTERS = "ABCDEFGHIJ"


def to_index(answer, n):
    """MMLU ships the gold as an int index or a letter (A..D)."""
    if isinstance(answer, str):
        a = answer.strip()
        if a.isalpha():
            return LETTERS.find(a.upper())
        if a.isdigit():
            return int(a)
        return -1
    try:
        return int(answer)
    except (TypeError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/mmlu_typed.jsonl")
    ap.add_argument("--target", type=int, default=15000, help="rows to write (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    pool = []
    for split in SPLITS:
        try:
            ds = load_dataset(REPO, CONFIG, split=split)
        except Exception as e:                       # noqa: BLE001
            print(f"  {split}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        for row in ds:
            choices = row["choices"]
            if isinstance(choices, str):             # older mirrors store a python-list repr
                try:
                    choices = json.loads(choices.replace("'", '"'))
                except Exception:                    # noqa: BLE001
                    continue
            pool.append((split, row["subject"], row["question"], list(choices), row["answer"]))
        print(f"  {split}: {len(ds):,} rows", flush=True)
    print(f"pool: {len(pool):,} questions")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    target = args.target or len(pool)
    rows, skipped, seen = [], 0, set()
    for split, subject, question, choices, answer in pool:
        idx = to_index(answer, len(choices))
        rec = build_row(
            guid=f"mmlu/{split}/{len(rows)}",
            workflow=WORKFLOW,
            state={"subject": subject, "question": " ".join(str(question).split())},
            instructions=INSTRUCTIONS,
            options=choices,
            correct_idx=idx,
            rng=rng,
            choices_key="choices",
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
