#!/usr/bin/env python3
"""Duplicate / leakage audit for a Laya-format mix.

Checks three things:
  1. exact duplicate states inside the mix itself
  2. overlap between the mix and an evaluation set (contamination)
  3. per-source contribution of any duplicate state (so the source of a clash is visible)

Usage:
    python scripts/audit_duplicates.py --mix data/xeron3_mix.jsonl \
        [--eval data/...jsonl] [--eval <path>] [--report out.json]
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict


def norm(state: str) -> str:
    """Canonical text form of a state for duplicate detection."""
    try:
        s = json.loads(state)
    except Exception:                                     # noqa: BLE001
        s = state
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def key_of(row) -> str:
    return hashlib.sha256(norm(row["state"]).encode()).hexdigest()[:16]


def load(path):
    for line in open(path):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--eval", action="append", default=[])
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    seen = {}
    dup_pairs = defaultdict(int)
    by_source = defaultdict(int)
    n = 0
    for row in load(args.mix):
        n += 1
        k = key_of(row)
        src = row.get("workflow", "?")
        by_source[src] += 1
        if k in seen:
            dup_pairs[(seen[k], src)] += 1
        else:
            seen[k] = src
    print(f"mix rows: {n:,} | unique states: {len(seen):,} | duplicate rows: {n - len(seen):,} "
          f"({100 * (n - len(seen)) / max(1, n):.2f}%)")
    if dup_pairs:
        print("duplicate state pairs (first_seen_source -> later_source: count):")
        for (a, b), c in sorted(dup_pairs.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {a} -> {b}: {c:,}")

    report = {"mix": args.mix, "mix_rows": n, "unique_states": len(seen),
              "duplicate_rows": n - len(seen),
              "duplicate_pairs": {f"{a} -> {b}": c for (a, b), c in dup_pairs.items()},
              "by_source": dict(by_source), "eval_overlap": {}}

    for ev in args.eval:
        ev_keys, ev_n, hits = {}, 0, 0
        for row in load(ev):
            ev_n += 1
            ev_keys[key_of(row)] = row.get("id") or row.get("guid")
        for k in ev_keys:
            if k in seen:
                hits += 1
        print(f"eval {ev}: {ev_n:,} rows | overlapping with mix: {hits:,}")
        report["eval_overlap"][ev] = {"rows": ev_n, "overlap": hits,
                                      "overlap_ids": [ev_keys[k] for k in ev_keys if k in seen][:20]}

    if args.report:
        json.dump(report, open(args.report, "w"), indent=2, ensure_ascii=False)
        print("report ->", args.report)


if __name__ == "__main__":
    main()
