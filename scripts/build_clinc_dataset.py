#!/usr/bin/env python3
"""Build a Laya-format intent-routing dataset from CLINC150 (clinc/clinc_oos, config `plus`).

Why: JevBench's weak spot is *routing / intent* selection — deciding which one of a
small set of labels a short user utterance maps to. JevBench ships clinc150 only with
k=151 options (skipped by build_jevbench_dataset.py because the Laya choice head
budget is 256 tokens), so the routing signal is missing from the mix entirely.

Here every utterance becomes a small-arity choice: the gold intent + 3..5 randomly
sampled distractor intents (4..6 options total), which is what a router actually sees.

Splits used: train + validation + test (23,850 utterances, 150 in-scope intents).
The `oos` (out-of-scope) class is excluded — it is not a routable intent and would
make distractors degenerate.

Output rows follow the Laya native format consumed by scripts/preprocess.py.

Usage:
    python scripts/build_clinc_dataset.py --output data/clinc_typed.jsonl [--target 20000]
"""
import argparse
import json
import os
import random

from datasets import load_dataset

from laya_choice_utils import build_row, dedupe_key, write_jsonl

REPO = "clinc/clinc_oos"
CONFIG = "plus"
SPLITS = ("train", "validation", "test")
WORKFLOW = "clinc150-intent"
INSTRUCTIONS = (
    "Which intent label best matches the user utterance? "
    "Pick the single best-matching label from the candidates."
)
NEG_MIN, NEG_MAX = 3, 5          # distractors -> 4..6 total options


def load_pool():
    """All (utterance, intent_name) pairs from train/validation/test, in split order."""
    pool = []
    names = None
    for split in SPLITS:
        try:
            ds = load_dataset(REPO, CONFIG, split=split)
        except Exception as e:                       # noqa: BLE001
            print(f"  {split}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        if names is None:
            names = list(ds.features["intent"].names)
        for row in ds:
            pool.append((row["text"], names[int(row["intent"])]))
        print(f"  {split}: {len(ds):,} rows", flush=True)
    return pool, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/clinc_typed.jsonl")
    ap.add_argument("--target", type=int, default=20000, help="rows to write (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--distractor-mode", choices=("random", "hard"), default="random",
                    help="random = uniformly sampled wrong intents (as specified); "
                         "hard = distractors sharing name tokens with the gold intent, "
                         "which removes the lexical 'pick the label whose words appear "
                         "in the utterance' shortcut")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    pool, names = load_pool()
    print(f"pool: {len(pool):,} utterances | intent labels: {len(names)}")

    # in-scope intents only (drop `oos`)
    in_scope = [n for n in names if n.lower() != "oos"]
    print(f"in-scope intents: {len(in_scope)} (dropped: {[n for n in names if n.lower() == 'oos']})")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    def _toks(name):
        return {t for t in name.split("_") if len(t) > 2}

    # hard-negative pools are precomputed once per intent (name-token overlap, descending)
    hard_pool = {}
    for intent in in_scope:
        t0 = _toks(intent)
        ranked = sorted((n for n in in_scope if n != intent),
                        key=lambda n: (-len(_toks(n) & t0), n))
        hard_pool[intent] = ranked[:30]

    target = args.target or len(pool)
    rows, skipped, seen = [], 0, set()
    for text, intent in pool:
        if intent not in in_scope:
            skipped += 1
            continue
        if args.distractor_mode == "hard":
            distractors = rng.sample(hard_pool[intent], rng.randint(NEG_MIN, NEG_MAX))
        else:
            distractors = rng.sample([n for n in in_scope if n != intent],
                                     rng.randint(NEG_MIN, NEG_MAX))
        options = [intent] + distractors
        rec = build_row(
            guid=f"clinc150/{len(rows)}",
            workflow=WORKFLOW,
            state={"utterance": " ".join(str(text).split())},
            instructions=INSTRUCTIONS,
            options=options,
            correct_idx=0,
            rng=rng,
            choices_key="candidate_intents",
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
    print(f"\nWrote {len(rows):,} rows -> {args.output} (skipped {skipped}, mode={args.distractor_mode})")
    print("example:", json.dumps(json.loads(rows[0]["state"]), ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
