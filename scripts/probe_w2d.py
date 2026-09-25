#!/usr/bin/env python3
"""W2 probe part 4: extra multilingual candidates."""
import json
import sys
from datasets import load_dataset, get_dataset_config_names

def show(path, cfg=None, split="train", ncfg=True):
    print(f"### {path} cfg={cfg}")
    try:
        cfgs = get_dataset_config_names(path)
        c = cfgs if len(cfgs) <= 60 else cfgs[:60]
        print("  n_configs:", len(cfgs), "|", c)
    except Exception as e:
        print("  configs ERR", type(e).__name__, str(e)[:160])
        return
    if split is None:
        return
    try:
        ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
        print("  rows:", len(ds))
        print("  feats:", json.dumps({k: str(v) for k, v in ds.features.items()}, ensure_ascii=False))
        print("  sample:", json.dumps({k: (str(v)[:160] if not isinstance(v, (int, float)) else v)
                                       for k, v in ds[0].items()}, ensure_ascii=False)[:800])
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:220])
    sys.stdout.flush()

show("mteb/amazon_massive_scenario", "ko-KR", "train")
show("mteb/mtop_intent", "en", "train")
show("Muennighoff/xwinograd", "en", "train")
show("MoritzLaurer/multilingual-NLI-26lang-2mil7", None, None)
show("tyqiangz/multilingual-sentiments", "Korean", "train")
show("cointegrated/ru-posts-casual", None, None)
show("mteb/amazon_reviews_multi", None, None)
