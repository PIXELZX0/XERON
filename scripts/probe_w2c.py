#!/usr/bin/env python3
"""W2 probe part 3: PAWS-X mirror + multilingual review mirrors + extras."""
import json
import sys
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder

def show(path, cfg=None, split=None, feat_only=False):
    print(f"### {path} cfg={cfg}")
    try:
        cfgs = get_dataset_config_names(path)
        print("  n_configs:", len(cfgs) if isinstance(cfgs, list) else cfgs, "|", (cfgs[:12] if isinstance(cfgs, list) else cfgs))
    except Exception as e:
        print("  configs ERR", type(e).__name__, str(e)[:150])
        return
    if feat_only:
        return
    try:
        ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
        print("  rows:", len(ds))
        print("  feats:", json.dumps({k: str(v) for k, v in ds.features.items()}, ensure_ascii=False))
        print("  sample:", json.dumps({k: (str(v)[:200] if not isinstance(v, (int, float)) else v)
                                       for k, v in ds[0].items()}, ensure_ascii=False)[:900])
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:250])
    sys.stdout.flush()

show("google-research-datasets/paws-x", "de", "train")
show("SetFit/amazon_reviews_multi_de", None, "train")
show("SetFit/amazon_reviews_multi_ja", None, "train")
show("SetFit/amazon_reviews_multi_zh", None, "train")
show("facebook/mlqa", "en", "train")
show("tydiqa", None, None, feat_only=True)
show("papluca/language-identification", None, "train")
