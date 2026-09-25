#!/usr/bin/env python3
"""W2 probe part 2: concrete configs -> features + 2 real rows (streaming)."""
import json
import sys
from datasets import load_dataset

TARGETS = [
    ("belebele/kor_Hang", "facebook/belebele", "kor_Hang", "test"),
    ("belebele/eng_Latn", "facebook/belebele", "eng_Latn", "test"),
    ("global_mmlu/ko", "CohereForAI/Global-MMLU", "ko", "test"),
    ("global_mmlu/am", "CohereForAI/Global-MMLU", "am", "test"),
    ("xcopa/et", "cambridgeltl/xcopa", "et", "validation"),
    ("xcopa/tr", "cambridgeltl/xcopa", "tr", "validation"),
    ("xstory_cloze/en", "juletxara/xstory_cloze", "en", "train"),
    ("xstory_cloze/ar", "juletxara/xstory_cloze", "ar", "train"),
]

def main():
    rep = {}
    for tag, path, cfg, split in TARGETS:
        e = {"tag": tag, "path": path, "config": cfg}
        try:
            ds = load_dataset(path, cfg, split=split)
            e["rows"] = len(ds)
            e["features"] = {k: str(v) for k, v in ds.features.items()}
            rows = []
            for i in range(min(2, len(ds))):
                r = ds[i]
                rows.append({k: (str(v)[:220] if not isinstance(v, (int, float)) else v)
                             for k, v in r.items()})
            e["sample"] = rows
            e["ok"] = True
        except Exception as ex:
            e["ok"] = False
            e["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        rep[tag] = e
        print(f"=== {tag} ok={e.get('ok')} rows={e.get('rows')}")
        print(json.dumps(e.get("sample") or e.get("error"), ensure_ascii=False, indent=1)[:2500])
        sys.stdout.flush()
    with open("/tmp/probe_w2b.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("-> /tmp/probe_w2b.json")

if __name__ == "__main__":
    main()
