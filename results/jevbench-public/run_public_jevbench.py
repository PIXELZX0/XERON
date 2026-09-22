#!/usr/bin/env python3
"""Run the public JevBench items (231) through the JevBench harness with a local laya checkpoint.

Usage: run_pub.py <model_path> <label> <out_prefix>
Model is loaded and warmed up BEFORE any timing, matching the harness convention
("model loaded before timing"). Serial, one request at a time, threads=4 (same as the
published Laya row).
"""
import json
import os
import sys
import time

sys.path.insert(0, "/tmp/jevbench")

from jevbench.adapters import LayaLocalAdapter          # noqa: E402
from jevbench.budget import Ledger                      # noqa: E402
from jevbench.runner import Runner                      # noqa: E402
from jevbench.tasks import load_jsonl                   # noqa: E402

PUB = "/tmp/jevbench/datasets/public"
ORDER = [("original", "standard"), ("easy", "easy"), ("hard", "hard")]


def main():
    model_path, label, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    tasks = []
    for fname, _tier in ORDER:
        tasks.extend(load_jsonl(f"{PUB}/{fname}.jsonl"))
    print(f"[{label}] {len(tasks)} public tasks", flush=True)

    ad = LayaLocalAdapter(endpoint=model_path, model=label, threads=4)
    t0 = time.perf_counter()
    ad.load()
    print(f"[{label}] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    for t in tasks[:2]:                                  # warm-up, untimed by the runner
        ad.run(t)
    print(f"[{label}] warm-up done", flush=True)

    ledger = Ledger(f"{out_prefix}.ledger.jsonl")
    runner = Runner(ad, ledger, f"{out_prefix}.raw")
    t0 = time.perf_counter()
    recs = runner.run_all(tasks, progress_every=25, results_path=f"{out_prefix}.results.jsonl")
    print(f"[{label}] done in {time.perf_counter() - t0:.1f}s "
          f"({sum(1 for r in recs if r['ok'])}/{len(recs)} ok)", flush=True)


if __name__ == "__main__":
    main()
