#!/usr/bin/env python3
"""W2 probe part 7: extra multilingual QA / sentiment candidates."""
import json
import sys
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder

def meta(path, cfg=None):
    try:
        b = load_dataset_builder(path, cfg) if cfg else load_dataset_builder(path)
        return {k: (v.num_examples if v.num_examples else -1) for k, v in (b.info.splits or {}).items()}
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:150]}"

def row(path, cfg=None, split="train"):
    try:
        it = load_dataset(path, cfg, split=split, streaming=True) if cfg else \
            load_dataset(path, split=split, streaming=True)
        r = next(iter(it))
        return {k: (str(v)[:170] if not isinstance(v, (int, float)) else v) for k, v in r.items()}
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:200]}"

for p, cfg in [("li-lab/MMLU-ProX", "ko"), ("CohereLabs/include-lite-44", None),
               ("masakhane/afrisenti", "hau"), ("masakhane/masakhanews", "hau"),
               ("chiayewken/m3exam", None)]:
    print(f"### {p} cfg={cfg}", flush=True)
    try:
        cfgs = get_dataset_config_names(p)
        print("  n_configs:", len(cfgs), "|", (cfgs if len(cfgs) <= 60 else cfgs[:60]), flush=True)
    except Exception as e:
        print("  cfg ERR", type(e).__name__, str(e)[:150], flush=True); continue
    print("  splits:", meta(p, cfg), flush=True)
    print("  row:", json.dumps(row(p, cfg), ensure_ascii=False)[:800], flush=True)
