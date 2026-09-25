#!/usr/bin/env python3
"""W2 probe: multilingual candidates. Metadata-first (load_dataset_builder), then 1-2 real rows."""
import json
import sys
from datasets import load_dataset_builder, get_dataset_config_names, load_dataset

CAND = [
    ("belebele", "facebook/belebele", None),
    ("global_mmlu", "CohereForAI/Global-MMLU", None),
    ("xcopa", "cambridgeltl/xcopa", None),
    ("xstory_cloze", "juletxara/xstory_cloze", None),
    ("paws_x", "google/paws-x", None),
    ("paws_x_alt", "paws-x", None),
]

def meta(name, path, config=None):
    out = {"name": name, "path": path, "config": config}
    try:
        if config:
            b = load_dataset_builder(path, config)
        else:
            b = load_dataset_builder(path)
        out["splits"] = {k: (v.num_examples if v.num_examples else -1)
                         for k, v in (b.info.splits or {}).items()}
        feats = {}
        for k, v in (b.info.features or {}).items():
            d = str(getattr(v, "dtype", type(v).__name__))
            names = getattr(v, "names", None)
            if names:
                d += f"[{len(names)}]"
                out.setdefault("label_names", {})[k] = list(names)[:40]
            feats[k] = d
        out["features"] = feats
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:250]}"
    return out

def try_configs(path):
    try:
        cfgs = get_dataset_config_names(path)
        return cfgs
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:200]}"

def main():
    which = sys.argv[1:] or [c[0] for c in CAND]
    rep = {}
    for name, path, config in CAND:
        if name not in which:
            continue
        e = meta(name, path, config)
        cfgs = try_configs(path)
        e["n_configs"] = len(cfgs) if isinstance(cfgs, list) else cfgs
        e["configs_sample"] = cfgs[:8] if isinstance(cfgs, list) else cfgs
        rep[name] = e
        print(f"[{'OK ' if e.get('ok') else 'ERR'}] {name} {path}")
        print(f"    n_configs={e['n_configs']} sample={e['configs_sample']}")
        print(f"    splits={e.get('splits')}")
        print(f"    feats={e.get('features')}")
        if not e.get("ok"):
            print(f"    error={e.get('error')}")
        sys.stdout.flush()
    with open("/tmp/probe_w2.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("-> /tmp/probe_w2.json")

if __name__ == "__main__":
    main()
