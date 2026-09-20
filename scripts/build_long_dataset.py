#!/usr/bin/env python3
"""Build a LONG-CONTEXT Laya-format decision dataset from public HF datasets.

Long documents that actually exercise the extended context (MAX_LEN 4096+):

  1. SCOTUS (coastalcph/lex_glue scotus) — US Supreme Court opinions
     -> "which justice authored this opinion?" (choice, 14 classes)
     token len: p50 ~6.1k, p90 ~18k  (docs mostly truncate at MAX_LEN -> real test)
  2. 20 Newsgroups (SetFit/20_newsgroups) — full newsgroup posts
     -> topic classification (choice, 20 classes) with long posts included

Output: JSONL in Laya native format (guid / workflow / state / questions / gold).

Usage:
    python scripts/build_long_dataset.py --output data/long_typed.jsonl --limit 0
"""
import argparse
import json

from datasets import load_dataset

WORKFLOW = "long-context"

SCOTUS_N = 14  # justices 0..13


def scotus_rows(limit):
    ds = load_dataset("coastalcph/lex_glue", "scotus", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    rows = []
    for i in range(n):
        r = ds[i]
        q = {
            "author_justice": {
                "type": "choice",
                "instructions": "이 미국 대법원 판례를 작성한 판사(justice)는 누구입니까?",
                "criteria": {str(k): f"justice_{k}" for k in range(SCOTUS_N)},
            }
        }
        gold = {str(k): (1.0 if k == r["label"] else 0.0) for k in range(SCOTUS_N)}
        rows.append({
            "guid": f"scotus-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"court_document": r["text"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"author_justice": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def newsgroups_rows(limit):
    ds = load_dataset("SetFit/20_newsgroups", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    labels = sorted(set(ds["label_text"][:n]))
    rows = []
    for i in range(n):
        r = ds[i]
        lt = r["label_text"]
        q = {
            "topic": {
                "type": "choice",
                "instructions": "이 뉴스그룹 게시물의 주제는 무엇입니까?",
                "criteria": {k: k for k in labels},
            }
        }
        gold = {k: (1.0 if k == lt else 0.0) for k in labels}
        rows.append({
            "guid": f"ng20-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"article": r["text"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"topic": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/long_typed.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows per task (0 = all)")
    args = ap.parse_args()

    rows = scotus_rows(args.limit) + newsgroups_rows(args.limit)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} long-context rows -> {args.output} "
          f"(scotus={args.limit or 5000} + newsgroups={args.limit or 11314})")


if __name__ == "__main__":
    main()