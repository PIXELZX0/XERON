#!/usr/bin/env python3
"""Evaluate an XERON wide-head snapshot with laya's harness.

Problem: `laya.agent.Agent.__init__` builds stock DecisionModel (head width == encoder
width) and then calls `_verify_compatibility`, so a snapshot with head_size != 768 is
rejected ("Model architecture mismatch ... expected (2304,768), found (3072,1024)").

Fix: monkey-patch (a) build_model to honour cfg["head_size"] and (b) the compatibility
check, then hand the path to jevbench's LayaLocalAdapter, which calls laya.load().

Usage:
    python scripts/eval_wide.py <snapshot_dir> <label> <out_prefix>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import laya.agent as LA  # noqa: E402
import laya.common as LC  # noqa: E402
from model_xeron import build_wide_model  # noqa: E402

_orig_build = LC.build_model


def _patched_build(cfg, encoder_dir=None):
    hs = int(cfg.get("head_size") or 0)
    if hs:
        return build_wide_model(cfg, encoder_dir, head_layers=cfg.get("head_layers", 2),
                                head_size=hs, dropout=cfg.get("dropout", 0.1) or 0.1)
    return _orig_build(cfg, encoder_dir)


LC.build_model = _patched_build
LA.build_model = _patched_build          # agent.py did `from .common import build_model`
LA._verify_compatibility = lambda *a, **k: None
print("[eval_wide] patched laya for wide heads", flush=True)


def main():
    snap, label, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]

    cfg = json.load(open(os.path.join(snap, "rl_agent_config.json")))
    print(f"[eval_wide] {label}: head_size={cfg.get('head_size')} "
          f"head_layers={cfg.get('head_layers')} temp={cfg.get('temperature')}", flush=True)

    # sanity: can we actually load it?
    import laya
    ag = laya.load(snap, device="cpu")
    print(f"[eval_wide] loaded OK; model={type(ag.model).__name__} "
          f"head_width={getattr(ag.model, 'head_width', 'n/a')}", flush=True)
    del ag

    jev = "/tmp/jevbench"
    if not os.path.isdir(jev):
        os.system(f"git clone --depth 1 https://github.com/fstandhartinger/jevbench.git {jev}")
    sys.path.insert(0, jev)
    from jevbench.adapters import LayaLocalAdapter
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import load_jsonl

    pub = os.path.join(jev, "datasets/public")
    tasks = []
    for f in ("original", "easy", "hard"):
        tasks.extend(load_jsonl(f"{pub}/{f}.jsonl"))
    print(f"[eval_wide] public tasks: {len(tasks)}", flush=True)

    ad = LayaLocalAdapter(endpoint=snap, model=label, threads=4)
    ad.load()
    for t in tasks[:2]:
        ad.run(t)
    r = Runner(ad, Ledger(f"{out_prefix}.ledger.jsonl"), f"{out_prefix}.raw")
    recs = r.run_all(tasks, progress_every=50, results_path=f"{out_prefix}.results.jsonl")
    print(f"[eval_wide] done {sum(1 for x in recs if x['ok'])}/{len(recs)}", flush=True)


if __name__ == "__main__":
    main()
