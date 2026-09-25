#!/usr/bin/env python3
"""W2 probe part 5b: streaming, no full downloads."""
import json
import sys
from datasets import load_dataset, get_dataset_config_names

def show(path, cfg=None, split="train"):
    print(f"### {path} cfg={cfg} split={split}", flush=True)
    try:
        cfgs = get_dataset_config_names(path)
        c = cfgs if len(cfgs) <= 70 else cfgs[:70]
        print("  n_configs:", len(cfgs), "|", c, flush=True)
    except Exception as e:
        print("  configs ERR", type(e).__name__, str(e)[:160], flush=True); return
    if split is None:
        return
    try:
        it = load_dataset(path, cfg, split=split, streaming=True) if cfg else \
            load_dataset(path, split=split, streaming=True)
        n = 0
        feats = None
        first = None
        for r in it:
            if first is None:
                first = r
            n += 1
            if n >= 2000:
                break
        ds0 = load_dataset(path, cfg, split=split, streaming=True) if cfg else \
            load_dataset(path, split=split, streaming=True)
        try:
            feats = {k: str(v) for k, v in ds0.features.items()}
        except Exception:
            feats = "?"
        print("  feats:", json.dumps(feats, ensure_ascii=False), flush=True)
        print("  sample:", json.dumps({k: (str(v)[:170] if not isinstance(v, (int, float)) else v)
                                       for k, v in (first or {}).items()}, ensure_ascii=False)[:900],
              flush=True)
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:220], flush=True)

show("Davlan/sib200", "kor_Hang", "train")
show("mteb/sib200", "kor_Hang", "test")
show("mteb/amazon_massive_scenario", "ko", "train")
show("Muennighoff/xwinograd", "en", "test")
show("MoritzLaurer/multilingual-NLI-26lang-2mil7", None, "train")
