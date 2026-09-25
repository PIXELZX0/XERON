#!/usr/bin/env python3
"""Probe candidate HF datasets for XERON-1.0 bulk build: features/splits WITHOUT downloading data.

Uses `load_dataset_builder` (metadata only). Prints a compact, grep-able report so the
registry in scripts/build_bulk_choice.py can be written against real schemas.

Usage:
    python scripts/probe_candidates.py [--configs name=path@config,...]
"""
import argparse
import json
import sys

from datasets import load_dataset_builder

# (name, hf path, config or None)
CANDIDATES = [
    # ---- multilingual ----
    ("massive", "mteb/amazon_massive_intent", None),
    ("xnli", "facebook/xnli", "en"),
    ("xnli_all", "xnli", None),
    ("belebele", "facebook/belebele", None),
    ("global_mmlu", "CohereForAI/Global-MMLU", None),
    ("mtop_intent", "mteb/mtop_intent", None),
    ("mtop_domain", "mteb/mtop_domain", None),
    ("amazon_multi", "mteb/amazon_reviews_multi", None),
    ("paws_x", "google/paws-x", None),
    ("xcopa", "cambridgeltl/xcopa", None),
    ("xstorycloze", "juletxara/xstory_cloze", None),
    # ---- english knowledge / reasoning ----
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", None),
    ("snli", "stanfordnlp/snli", None),
    ("mnli", "nyu-mll/multi_nli", None),
    ("fever", "fever/fever", "v1.0"),
    ("vitaminc", "tals/vitaminc", None),
    ("scitail", "allenai/scitail", None),
    ("anli", "facebook/anli", None),
    ("race", "ehovy/race", "high"),
    ("race_all", "ehovy/race", None),
    ("qasc", "allenai/qasc", None),
    ("winogrande", "allenai/winogrande", "winogrande_xl"),
    ("social_i_qa", "allenai/social_i_qa", None),
    ("cosmos_qa", "allenai/cosmos_qa", None),
    ("logiqa", "lucasmccabe/logiqa", None),
    ("logiqa2", "tasksource/logiqa2", None),
    ("reclor", "metaeval/reclor", None),
    ("math_qa", "allenai/math_qa", None),
    ("aqua_rat", "deepmind/aqua_rat", None),
    ("boolq", "google/boolq", None),
    ("super_glue_boolq", "aps/super_glue", "boolq"),
    ("super_glue_all", "aps/super_glue", "copa"),
    ("anli_all", "facebook/anli", None),
    # ---- sentiment / topic / toxicity (volume) ----
    ("imdb", "stanfordnlp/imdb", None),
    ("yelp_polarity", "fancyzhx/yelp_polarity", None),
    ("amazon_polarity", "fancyzhx/amazon_polarity", None),
    ("sst5", "SetFit/sst5", None),
    ("ag_news", "fancyzhx/ag_news", None),
    ("dbpedia_14", "fancyzhx/dbpedia_14", None),
    ("yahoo_topics", "fancyzhx/yahoo_answers_topics", None),
    ("tweet_eval_emoji", "cardiffnlp/tweet_eval", "emoji"),
    ("tweet_eval_emotion", "cardiffnlp/tweet_eval", "emotion"),
    ("tweet_eval_offensive", "cardiffnlp/tweet_eval", "offensive"),
    ("tweet_eval_hate", "cardiffnlp/tweet_eval", "hate"),
    ("civil_comments", "google/civil_comments", None),
    ("implicit_hate", "ucberkeley-dlab/measuring-hate-speech", None),
    ("emotion", "dair-ai/emotion", None),
    ("go_emotions", "google/go_emotions", None),
    ("poem_sentiment", "google/poem_sentiment", None),
    ("financial_phrasebank", "takala/financial_phrasebank", "sentences_50agree"),
    # ---- korean / japanese / chinese ----
    ("nsmc", "e9t/nsmc", None),
    ("kmmlu", "HAERAE-HUB/KMMLU", "Accounting"),
    ("kobest_boolq", "skt/kobest_v1", "boolq"),
    ("klue_ynat", "klue/klue", "ynat"),
    ("klue_all", "klue/klue", None),
    ("jglue_jnli", "shunk031/JGLUE", "JNLI"),
    ("jglue_marc", "shunk031/JGLUE", "MARC-ja"),
    ("jglue_jsts", "shunk031/JGLUE", "JSTS"),
    ("c3", "shibing624/C3", None),
    ("clue_tnews", "clue", "tnews"),
    ("clue_iflytek", "clue", "iflytek"),
    ("clue_ocnli", "clue", "ocnli"),
    ("clue_cmnli", "clue", "cmnli"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated names to probe")
    ap.add_argument("--out", default="data/candidate_probe.json")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    report = {}
    for name, path, config in CANDIDATES:
        if only and name not in only:
            continue
        entry = {"path": path, "config": config}
        try:
            b = load_dataset_builder(path, config) if config else load_dataset_builder(path)
            entry["splits"] = {k: (v.num_examples if v.num_examples else -1)
                               for k, v in (b.info.splits or {}).items()}
            feats = {}
            for k, v in (b.info.features or {}).items():
                d = str(getattr(v, "dtype", type(v).__name__))
                names = getattr(v, "names", None)
                if names:
                    d += f"[{len(names)}]"
                    entry.setdefault("label_names", {})[k] = list(names)[:80]
                feats[k] = d
            entry["features"] = feats
            entry["ok"] = True
        except Exception as e:                             # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        report[name] = entry
        flag = "OK " if entry.get("ok") else "ERR"
        print(f"[{flag}] {name:22} {path}"
              f"{'/' + config if config else ''} "
              f"| splits={entry.get('splits')} | feats={entry.get('features')}"
              f"{'' if entry.get('ok') else ' | ' + entry.get('error', '')}",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    ok = sum(1 for v in report.values() if v.get("ok"))
    print(f"\nprobed {len(report)}  ok={ok}  err={len(report) - ok} -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
