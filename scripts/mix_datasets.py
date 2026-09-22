#!/usr/bin/env python3
"""Mix several Laya-format JSONL sources into one training set with explicit proportions.

Every source is sampled to an exact row budget (deterministic, seeded), so the mix is
reproducible and sized for the compute window it has to fit in.

`--soft-boost` oversamples rows that carry a non-degenerate gold distribution (i.e. rows
whose gold is not one-hot). JevBench's Calibration axis scores fidelity to exact gold
distributions, and a purely single-label diet sharpens the model and hurts that axis.

Usage:
    python scripts/mix_datasets.py --out data/xeron3_mix.jsonl \
        --source data/jevbench_full.jsonl:45000 \
        --source data/anli_typed.jsonl:8000 \
        --source data/long_typed.jsonl:4000 \
        --source data/korean_typed.jsonl:5000 \
        --source data/browser_typed.jsonl:3000 \
        --source data/mind2web_typed.jsonl:2000 \
        --soft-boost 1.5
"""
import argparse
import hashlib
import json
import os
import random
import re


def _norm_state(state):
    try:
        s = json.loads(state)
    except Exception:                                    # noqa: BLE001
        s = state
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def is_soft(row):
    try:
        probs = json.loads(row["gold"])
    except Exception:                                    # noqa: BLE001
        return False
    for q in probs.values():
        p = q.get("probabilities") or {}
        if p and max(p.values()) < 0.999:
            return True
    return False


def reservoir(path, budget, rng, soft_boost=1.0):
    """Weighted reservoir sample (A-Res) of `budget` rows from a JSONL file.

    Each row gets key = u ** (1/w); the `budget` largest keys win. w > 1 for rows with a
    non-degenerate gold distribution when soft_boost > 1. Deterministic given the seed.
    """
    import heapq

    heap = []            # min-heap of (key, seq, row)
    seen = 0
    for row in iter_rows(path):
        seen += 1
        w = soft_boost if (soft_boost != 1.0 and is_soft(row)) else 1.0
        key = rng.random() ** (1.0 / w)
        if len(heap) < budget:
            heapq.heappush(heap, (key, seen, row))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, seen, row))
    return [r for _, _, r in heap], seen


def iter_rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="path:budget (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--soft-boost", type=float, default=1.0,
                    help=">1 oversamples rows with non-one-hot gold (calibration fidelity)")
    ap.add_argument("--max-soft-share", type=float, default=0.0,
                    help="if >0, cap the share of soft rows in the output")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop rows whose (state, questions) pair was already emitted")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    report = []
    for spec in args.source:
        path, _, budget = spec.rpartition(":")
        if not path:
            path, budget = budget, None
        if not os.path.exists(path):
            report.append((path, 0, 0, "MISSING"))
            continue
        got, seen = reservoir(path, int(budget), rng, args.soft_boost)
        rows.extend(got)
        report.append((path, len(got), seen, ""))
        print(f"  {os.path.basename(path):28} kept {len(got):7,} / {seen:8,} rows", flush=True)

    rng.shuffle(rows)

    if args.max_soft_share > 0:
        soft = [r for r in rows if is_soft(r)]
        hard = [r for r in rows if not is_soft(r)]
        cap = int(args.max_soft_share * len(rows))
        if len(soft) > cap:
            rng.shuffle(soft)
            soft = soft[:cap]
        rows = soft + hard
        rng.shuffle(rows)
        print(f"  soft-label share capped: {len(soft):,} soft rows kept")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_soft = 0
    dropped = 0
    seen_exact = set()
    with open(args.out, "w") as f:
        for r in rows:
            if args.dedupe:
                k = hashlib.sha256((_norm_state(r["state"]) + "###" + r["questions"]).encode()).hexdigest()
                if k in seen_exact:
                    dropped += 1
                    continue
                seen_exact.add(k)
            n_soft += is_soft(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows) - dropped:,} rows -> {args.out}")
    if args.dedupe:
        print(f"Exact (state, questions) duplicates dropped: {dropped:,}")
    print(f"Soft-label rows: {n_soft:,} ({100 * n_soft / max(1, len(rows) - dropped):.1f}%)")
    missing = [p for p, _, _, s in report if s]
    if missing:
        print("MISSING SOURCES:", missing)


if __name__ == "__main__":
    main()
