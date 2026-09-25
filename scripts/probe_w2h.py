#!/usr/bin/env python3
import json, sys
from datasets import load_dataset, load_dataset_builder

def show(path, cfg=None, split=None):
    print(f"### {path} cfg={cfg} split={split}", flush=True)
    try:
        b = load_dataset_builder(path, cfg) if cfg else load_dataset_builder(path)
        print("  splits:", {k: (v.num_examples or -1) for k, v in (b.info.splits or {}).items()}, flush=True)
        print("  feats:", {k: str(v) for k, v in (b.info.features or {}).items()}, flush=True)
    except Exception as e:
        print("  meta ERR", type(e).__name__, str(e)[:180], flush=True)
    try:
        it = load_dataset(path, cfg, split=split, streaming=True) if cfg else \
            load_dataset(path, split=split, streaming=True)
        r = next(iter(it))
        print("  row:", json.dumps({k: (str(v)[:150] if not isinstance(v, (int, float)) else v)
                                    for k, v in r.items()}, ensure_ascii=False)[:900], flush=True)
    except Exception as e:
        print("  row ERR", type(e).__name__, str(e)[:200], flush=True)

show("li-lab/MMLU-ProX", "ko", "test")
show("CohereLabs/include-lite-44", "Korean", "test")
show("chiayewken/m3exam", "english", "test")
