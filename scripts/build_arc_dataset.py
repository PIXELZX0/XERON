#!/usr/bin/env python3
"""Build a Laya-format science-reasoning dataset from ARC (allenai/ai2_arc).

Why: ARC is the 4-way science multiple-choice task JevBench uses for its reasoning
tier — but JevBench only ships ARC-Challenge *train* (1,119 rows). Here we add
ARC-Challenge test/validation plus all of ARC-Easy.

Splits used (ARC-Challenge train is deliberately skipped — already in
data/jevbench_full.jsonl, so reusing it would duplicate states):
    ARC-Challenge : validation (299) + test (1,172)
    ARC-Easy      : train (2,251) + validation (570) + test (2,376)
    -> 6,668 available rows; `--target` samples uniformly at random from the pool.

`answerKey` is mapped to an option index and never written into `state`.

Output rows follow the Laya native format consumed by scripts/preprocess.py.

Usage:
    python scripts/build_arc_dataset.py --output data/arc_typed.jsonl [--target 3000]
"""
import argparse
import os
import random

from datasets import load_dataset

from laya_choice_utils import build_row, dedupe_key, write_jsonl

WORKFLOW = "arc-choice"
INSTRUCTIONS = "Which option is the correct answer to `question`?"
LETTERS = "ABCDEFGHIJ"

# (config, split) — ARC-Challenge train excluded on purpose (jevbench overlap)
SOURCES = (
    ("ARC-Challenge", "validation"),
    ("ARC-Challenge", "test"),
    ("ARC-Easy", "train"),
    ("ARC-Easy", "validation"),
    ("ARC-Easy", "test"),
)


def resolve_answer(answer_key, choice_labels, n_texts):
    """ARC answerKey is a label letter, a 1-based digit, or a bare letter."""
    key = str(answer_key).strip()
    if choice_labels and key in [str(l).strip() for l in choice_labels]:
        return [str(l).strip() for l in choice_labels].index(key)
    if key.isalpha() and len(key) == 1:
        return LETTERS.find(key.upper())
    if key.isdigit():
        v = int(key)
        return v - 1 if v >= 1 else 0
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/arc_typed.jsonl")
    ap.add_argument("--target", type=int, default=3000, help="rows to write (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    pool = []
    for config, split in SOURCES:
        try:
            ds = load_dataset("allenai/ai2_arc", config, split=split)
        except Exception as e:                       # noqa: BLE001
            print(f"  {config}/{split}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        for row in ds:
            ch = row["choices"]
            texts = list(ch["text"])
            labels = ch.get("label")
            pool.append((f"{config}/{split}", row["question"], texts, labels, row["answerKey"]))
        print(f"  {config}/{split}: {len(ds):,} rows", flush=True)
    print(f"pool: {len(pool):,} questions")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    target = args.target or len(pool)
    rows, skipped, seen = [], 0, set()
    for src, question, texts, labels, answer_key in pool:
        idx = resolve_answer(answer_key, labels, len(texts))
        rec = build_row(
            guid=f"arc/{src}/{len(rows)}",
            workflow=WORKFLOW,
            state={"question": " ".join(str(question).split()), "source": src.split("/")[0]},
            instructions=INSTRUCTIONS,
            options=texts,
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
