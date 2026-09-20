#!/usr/bin/env python3
"""Build a Korean Laya-format decision dataset from KLUE (klue/klue).

Tasks converted to Laya native format (state / questions / gold):
  - ynat  : topic classification (7 classes)  -> choice, 45.6k rows
  - nli   : natural language inference (3)      -> choice, 25.0k rows
  - sts   : semantic similarity (0-5)           -> score,  11.7k rows

Output: JSONL, one row per example with `guid`, `workflow`, `state`,
`questions`, `gold` (all state/questions/gold as JSON strings, matching the
Laya native dataset format expected by scripts/preprocess.py).

Usage:
    python scripts/build_korean_dataset.py --output data/korean_typed.jsonl --limit 0
"""
import argparse
import json

from datasets import load_dataset

YNAT_LABELS = ["IT과학", "경제", "사회", "생활문화", "세계", "정치", "스포츠"]

YMAT_CRITERIA = {
    "IT과학": "과학, IT, 기술, 인터넷, 컴퓨터, 우주, 연구 관련 뉴스",
    "경제": "금융, 증시, 부동산, 산업, 무역, 기업 경영 관련 뉴스",
    "사회": "교육, 환경, 복지, 노동, 법원, 사건사고 관련 뉴스",
    "생활문화": "음식, 여행, 패션, 문화, 예술, 건강, 연예 관련 뉴스",
    "세계": "국제 정치, 외교, 해외 사건, 글로벌 경제 관련 뉴스",
    "정치": "국내 정치, 정당, 선거, 국회, 정부 정책 관련 뉴스",
    "스포츠": "축구, 야구, 농구, 올림픽 등 스포츠 경기와 선수 관련 뉴스",
}

NLI_CRITERIA = {
    "entailment": "전제(premise)가 가설(hypothesis)을 논리적으로 함의한다",
    "contradiction": "전제(premise)가 가설(hypothesis)과 모순된다",
    "neutral": "전제(premise)와 가설(hypothesis)이 관련 없거나 중립적이다",
}

STS_CRITERIA = [
    "전혀 유사하지 않다",
    "대부분 유사하지 않다",
    "조금 유사하지 않다",
    "조금 유사하다",
    "대부분 유사하다",
    "거의 동일하다",
]

WORKFLOW = "korean-general"


def ynat_rows(limit):
    ds = load_dataset("klue/klue", "ynat", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for i, r in enumerate(ds):
        q = {
            "topic": {
                "type": "choice",
                "instructions": "이 뉴스 기사는 어떤 주제에 속합니까?",
                "criteria": YMAT_CRITERIA,
            }
        }
        gold = {k: (1.0 if k == YNAT_LABELS[r["label"]] else 0.0) for k in YNAT_LABELS}
        rows.append({
            "guid": r["guid"],
            "workflow": WORKFLOW,
            "state": json.dumps({"title": r["title"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"topic": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def nli_rows(limit):
    ds = load_dataset("klue/klue", "nli", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    labels = ["entailment", "contradiction", "neutral"]
    rows = []
    for i, r in enumerate(ds):
        q = {
            "relation": {
                "type": "choice",
                "instructions": "전제와 가설의 관계는 무엇입니까?",
                "criteria": NLI_CRITERIA,
            }
        }
        gold = {k: (1.0 if k == labels[r["label"]] else 0.0) for k in labels}
        rows.append({
            "guid": r["guid"],
            "workflow": WORKFLOW,
            "state": json.dumps({"premise": r["premise"], "hypothesis": r["hypothesis"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"relation": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def sts_rows(limit):
    ds = load_dataset("klue/klue", "sts", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for i, r in enumerate(ds):
        lvl = int(round(float(r["labels"]["label"])))
        lvl = max(0, min(5, lvl))
        q = {
            "similarity": {
                "type": "score",
                "instructions": "두 문장의 의미적 유사도 수준은 어느 정도입니까?",
                "criteria": STS_CRITERIA,
            }
        }
        gold = {str(k): (1.0 if k == lvl else 0.0) for k in range(6)}
        rows.append({
            "guid": r["guid"],
            "workflow": WORKFLOW,
            "state": json.dumps({"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, ensure_ascii=False),
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps({"similarity": {"probabilities": gold}}, ensure_ascii=False),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/korean_typed.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit rows per task (0 = all)")
    args = ap.parse_args()

    all_rows = ynat_rows(args.limit) + nli_rows(args.limit) + sts_rows(args.limit)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} Korean rows -> {args.output}")
    print(f"  ynat={args.limit or 45678} nli={args.limit or 24998} sts={args.limit or 11668} "
          f"(limit={args.limit or 'all'})")


if __name__ == "__main__":
    main()