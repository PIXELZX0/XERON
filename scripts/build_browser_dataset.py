#!/usr/bin/env python3
"""Build a browser-use Laya-format decision dataset from public HF datasets.

Covered browser-use decisions:
  - webpage_topic: classify a visited webpage's topic (BBC News 5cls + AG News 4cls) -> choice
  - is_spam:       is this message spam? (UCI SMS spam)                                -> noul
  - is_phishing:   is this URL phishing? (phishing-site-classification)                -> noul

Output: JSONL in Laya native format (guid / workflow / state / questions / gold).

Usage:
    python scripts/build_browser_dataset.py --output data/browser_typed.jsonl --limit 0
"""
import argparse
import json

from datasets import load_dataset

WORKFLOW = "browser-use"

BBC_LABELS = ["business", "entertainment", "politics", "sport", "tech"]
BBC_CRITERIA = {
    "business": "비즈니스, 금융, 경제 관련 웹 페이지",
    "entertainment": "엔터테인먼트, 연예, 문화 관련 웹 페이지",
    "politics": "정치 관련 웹 페이지",
    "sport": "스포츠 관련 웹 페이지",
    "tech": "기술, IT 관련 웹 페이지",
}

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
AG_CRITERIA = {
    "World": "세계 뉴스, 국제 사건 관련 웹 페이지",
    "Sports": "스포츠 관련 웹 페이지",
    "Business": "비즈니스, 경제 관련 웹 페이지",
    "Sci/Tech": "과학, 기술 관련 웹 페이지",
}


def webpage_topic_rows(limit):
    """BBC News (5 classes) + AG News (4 classes) -> choice."""
    rows = []

    ds = load_dataset("SetFit/bbc-news", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    for i in range(n):
        r = ds[i]
        q = {
            "topic": {
                "type": "choice",
                "instructions": "브라우저에서 방문한 이 웹 페이지의 주제는 무엇입니까?",
                "criteria": BBC_CRITERIA,
            }
        }
        gold = {k: (1.0 if k == r["label_text"] else 0.0) for k in BBC_LABELS}
        rows.append({
            "guid": f"bbc-{r['label']}-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"webpage_text": r["text"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"topic": {"probabilities": gold}}, ensure_ascii=False),
        })

    ds = load_dataset("fancyzhx/ag_news", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    for i in range(n):
        r = ds[i]
        q = {
            "topic": {
                "type": "choice",
                "instructions": "브라우저에서 방문한 이 웹 페이지의 주제는 무엇입니까?",
                "criteria": AG_CRITERIA,
            }
        }
        gold = {k: (1.0 if k == AG_LABELS[r["label"]] else 0.0) for k in AG_LABELS}
        rows.append({
            "guid": f"agnews-{r['label']}-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"webpage_text": r["text"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"topic": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def sms_spam_rows(limit):
    """UCI SMS spam -> noul (P(true) = spam)."""
    ds = load_dataset("ucirvine/sms_spam", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    rows = []
    for i in range(n):
        r = ds[i]
        q = {
            "is_spam": {
                "type": "noul",
                "instructions": "이 메시지는 스팸입니까?",
            }
        }
        true_p = 1.0 if r["label"] == 1 else 0.0
        rows.append({
            "guid": f"sms-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"message": r["sms"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"is_spam": {"probabilities": {"false": 1 - true_p, "true": true_p}}},
                               ensure_ascii=False),
        })
    return rows


def phishing_rows(limit):
    """phishing-site-classification -> noul (P(true) = phishing)."""
    ds = load_dataset("shawhin/phishing-site-classification", split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    rows = []
    for i in range(n):
        r = ds[i]
        q = {
            "is_phishing": {
                "type": "noul",
                "instructions": "이 URL은 피싱(사기) 사이트입니까?",
            }
        }
        true_p = 1.0 if r["labels"] == 1 else 0.0
        rows.append({
            "guid": f"phish-{i}",
            "workflow": WORKFLOW,
            "state": json.dumps({"url": r["text"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"is_phishing": {"probabilities": {"false": 1 - true_p, "true": true_p}}},
                               ensure_ascii=False),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/browser_typed.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit rows per task (0 = all; AG News defaults to 20k of 120k)")
    args = ap.parse_args()

    ag_limit = args.limit if args.limit else 20000  # keep AG News manageable
    rows = webpage_topic_rows(ag_limit) + sms_spam_rows(args.limit) + phishing_rows(args.limit)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} browser-use rows -> {args.output}")
    print(f"  bbc={min(args.limit or 1225, 1225)} ag_news={ag_limit} sms={args.limit or 5574} "
          f"phishing={args.limit or 2100}")


if __name__ == "__main__":
    main()