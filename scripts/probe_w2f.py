#!/usr/bin/env python3
"""W2 probe part 6: split lists + row counts (metadata only where possible)."""
import json
import sys
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder

def meta(path, cfg=None):
    try:
        b = load_dataset_builder(path, cfg) if cfg else load_dataset_builder(path)
        return {k: (v.num_examples if v.num_examples else -1) for k, v in (b.info.splits or {}).items()}
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:160]}"

print("sib200(Davlan)/kor_Hang splits:", meta("Davlan/sib200", "kor_Hang"), flush=True)
print("sib200(Davlan)/eng_Latn splits:", meta("Davlan/sib200", "eng_Latn"), flush=True)
print("sib200(mteb)/kor_Hang splits:", meta("mteb/sib200", "kor_Hang"), flush=True)
print("belebele/kor_Hang splits:", meta("facebook/belebele", "kor_Hang"), flush=True)
print("global_mmlu/ko splits:", meta("CohereForAI/Global-MMLU", "ko"), flush=True)
print("xcopa/et splits:", meta("cambridgeltl/xcopa", "et"), flush=True)
print("xstory/en splits:", meta("juletxara/xstory_cloze", "en"), flush=True)
print("pawsx/de splits:", meta("google-research-datasets/paws-x", "de"), flush=True)
print("massive_scenario/ko splits:", meta("mteb/amazon_massive_scenario", "ko"), flush=True)
print("xwinograd/en splits:", meta("Muennighoff/xwinograd", "en"), flush=True)
print("aml_de splits:", meta("SetFit/amazon_reviews_multi_de"), flush=True)

cfgs = get_dataset_config_names("facebook/belebele")
print("belebele n_configs:", len(cfgs), flush=True)
xc = get_dataset_config_names("cambridgeltl/xcopa")
print("xcopa configs:", xc, flush=True)
gm = get_dataset_config_names("CohereForAI/Global-MMLU")
print("global_mmlu configs:", gm, flush=True)

# MoritzLaurer splits + fields
try:
    b = load_dataset_builder("MoritzLaurer/multilingual-NLI-26lang-2mil7")
    sp = list((b.info.splits or {}).keys())
    print("mlnli n_splits:", len(sp), flush=True)
    print("mlnli splits:", sp, flush=True)
    print("mlnli feats:", {k: str(v) for k, v in (b.info.features or {}).items()}, flush=True)
except Exception as e:
    print("mlnli ERR", type(e).__name__, str(e)[:200], flush=True)

# one real row from a mlnli split
try:
    it = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split="de_mnli", streaming=True)
    r = next(iter(it))
    print("mlnli de_mnli sample:", json.dumps({k: (str(v)[:180] if not isinstance(v, (int, float)) else v)
                                               for k, v in r.items()}, ensure_ascii=False)[:900], flush=True)
except Exception as e:
    print("mlnli row ERR", type(e).__name__, str(e)[:200], flush=True)
