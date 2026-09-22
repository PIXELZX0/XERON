#!/usr/bin/env python3
"""Build a Laya-format JSONL from ANLI (Adversarial NLI, facebook/anli).

Why: JevBench's hard tier is dominated by traps, ambiguous cases and multi-hop
reasoning. ANLI is *adversarially collected* NLI — examples written specifically to
break models — which is the closest public proxy for that style.

Output rows follow the Laya native format consumed by scripts/preprocess.py.

Usage:
    python scripts/build_anli_dataset.py --output data/anli_typed.jsonl [--per-round 8000]
"""
import argparse
import json
import os

from datasets import load_dataset

CRITERIA = {
    "entailment": "The premise entails the hypothesis: if the premise is true, the hypothesis must be true.",
    "neutral": "The premise neither entails nor contradicts the hypothesis; the hypothesis may or may not hold.",
    "contradiction": "The premise contradicts the hypothesis: they cannot both be true.",
}
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
INSTRUCTIONS = (
    "Given `premise` and `hypothesis`, what is their logical relation? "
    "Judge only from the premise; do not use outside knowledge."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/anli_typed.jsonl")
    ap.add_argument("--per-round", type=int, default=8000, help="rows sampled per ANLI round (r1/r2/r3)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    written = 0
    per_round = {}
    with open(args.output, "w") as out:
        for rnd in ("r1", "r2", "r3"):
            try:
                ds = load_dataset("facebook/anli", "plain_text", split=f"train_{rnd}")
            except Exception as e:                       # noqa: BLE001
                print(f"  {rnd}: skipped ({type(e).__name__}: {str(e)[:120]})")
                continue
            ds = ds.shuffle(seed=args.seed + int(rnd[1]))
            n = 0
            for row in ds:
                lab = LABELS.get(int(row["label"]))
                if lab is None:
                    continue
                probs = {k: (1.0 if k == lab else 0.0) for k in CRITERIA}
                rec = {
                    "id": f"anli-{rnd}-{n}",
                    "workflow": "anli-adversarial-nli",
                    "state": json.dumps({"premise": row["premise"], "hypothesis": row["hypothesis"]},
                                        ensure_ascii=False),
                    "questions": json.dumps({"decision": {"type": "choice",
                                                          "instructions": INSTRUCTIONS,
                                                          "criteria": CRITERIA}}, ensure_ascii=False),
                    "gold": json.dumps({"decision": {"probabilities": probs}}, ensure_ascii=False),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                written += 1
                if n >= args.per_round:
                    break
            per_round[rnd] = n
            print(f"  {rnd}: {n:,} rows", flush=True)

    print(f"\nWrote {written:,} rows -> {args.output}")
    print("By round:", per_round)


if __name__ == "__main__":
    main()
