#!/usr/bin/env python3
"""Hard gate: no JevBench public-231 / typed-decisions-test state may appear in the mix.

Both eval sets are compared on the same normalized state key the builders/
`verify_bulk` use (`mix_datasets._norm_state`).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/check_eval_leak.py \
        --mix data/xeron10_mix.jsonl \
        --public "/tmp/jevbench/datasets/public/*.jsonl"

typed-decisions `test` is loaded from the HF cache (`datasets.load_dataset`); if the cache
is missing, the run is marked SKIPPED instead of silently passing.
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix_datasets import _norm_state as norm_state                # noqa: E402


def public_keys(pattern):
    keys, rows = set(), 0
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rows += 1
            keys.add(norm_state(json.loads(line)["state"]))
    return keys, rows


def typed_decisions_keys():
    try:
        from datasets import load_dataset
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    except Exception as e:                                        # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:160]}"
    keys = {norm_state(r["state"]) for r in ds}
    return keys, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/xeron10_mix.jsonl")
    ap.add_argument("--public", default="/tmp/jevbench/datasets/public/*.jsonl")
    ap.add_argument("--json-out", default="data/xeron10_eval_leak.json")
    args = ap.parse_args()

    pub, pub_rows = public_keys(args.public)
    print(f"jelbench public: {pub_rows} rows -> {len(pub)} state keys", flush=True)
    td, td_info = typed_decisions_keys()
    if td is None:
        print(f"typed-decisions test: SKIPPED ({td_info})", flush=True)
    else:
        print(f"typed-decisions test: {td_info} rows -> {len(td)} state keys", flush=True)

    hits = collections.Counter()
    n = 0
    with open(args.mix) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            k = norm_state(json.loads(line)["state"])
            if k in pub:
                hits["public"] += 1
            if td is not None and k in td:
                hits["typed_decisions_test"] += 1
    res = {"mix": args.mix, "mix_rows": n, "public_rows": pub_rows,
           "public_unique_states": len(pub),
           "typed_decisions_test_rows": td_info if td is None else len(td),
           "typed_decisions_test_status": "skipped" if td is None else "checked",
           "leaks": dict(hits)}
    with open(args.json_out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"mix rows {n:,} | leaks: {dict(hits) or 'NONE'}")
    print(f"-> {args.json_out}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())