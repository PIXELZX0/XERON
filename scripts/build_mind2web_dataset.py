#!/usr/bin/env python3
"""Build a browser-agent decision dataset from Mind2Web (osunlp/Mind2Web).

For each step of each web task we emit TWO Laya decisions:
  1. action_op   (choice): which operation to perform next
                          (CLICK / TYPE / SELECT / HOVER / PRESS_ENTER / ...)
  2. element     (choice): which candidate element to operate on
                          (pos_candidates + sampled neg_candidates)

Output: JSONL in Laya native format (guid / workflow / state / questions / gold).

Usage:
    python scripts/build_mind2web_dataset.py --output data/mind2web_typed.jsonl --limit 0
"""
import argparse
import json
import random

from datasets import load_dataset

WORKFLOW = "browser-agent"

# Common Mind2Web operations (op field of each step)
OP_CRITERIA = {
    "CLICK": "클릭",
    "TYPE": "입력 (텍스트 타이핑)",
    "SELECT": "드롭다운/목록에서 값 선택",
    "HOVER": "마우스 호버",
    "PRESS_ENTER": "엔터 키 입력",
    "SCROLL": "스크롤",
    "PRESS": "키 입력",
}
OPS = list(OP_CRITERIA.keys())

MAX_NEG = 25          # max negative candidates per step
MAX_ELEMENTS = 26     # max candidate elements shown to the model (1 pos + 25 neg)


def elem_text(cand, idx):
    """Compact text representation of a candidate element."""
    tag = cand.get("tag", "?")
    attrs = {}
    try:
        attrs = json.loads(cand.get("attributes", "{}"))
    except Exception:
        pass
    text = (
        attrs.get("placeholder")
        or attrs.get("aria-label")
        or attrs.get("text")
        or attrs.get("value")
        or attrs.get("title")
        or attrs.get("alt")
        or attrs.get("content")
        or ""
    )
    text = " ".join(str(text).split())[:60]
    return f"[{idx}] <{tag}> {text}".strip()


def build_rows(limit):
    ds = load_dataset("osunlp/Mind2Web", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    skipped = 0
    for t_idx, task in enumerate(ds):
        instruction = task["confirmed_task"]
        history = list(task.get("action_reprs") or [])
        steps = task.get("actions") or []

        for s_idx, step in enumerate(steps):
            op = (step.get("operation") or {}).get("op")
            if op not in OPS:
                op = "CLICK" if op else None
            if op is None:
                skipped += 1
                continue

            pos = step.get("pos_candidates") or []
            neg = step.get("neg_candidates") or []
            if not pos:
                skipped += 1
                continue

            target = next((c for c in pos if c.get("is_original_target")), pos[0])

            # Build candidate pool: target + sampled negatives (shuffled)
            neg_sample = random.Random(f"{t_idx}-{s_idx}").sample(
                neg, min(MAX_NEG, len(neg))
            ) if neg else []
            pool = pos[:1] + neg_sample
            random.Random(f"{t_idx}-{s_idx}").shuffle(pool)
            pool = pool[:MAX_ELEMENTS]

            # Map back to pool index of the target element
            try:
                target_idx = pool.index(target)
            except ValueError:
                # fall back: exact dict match failed -> compare backend_node_id
                tid = target.get("backend_node_id")
                target_idx = next(
                    (i for i, c in enumerate(pool) if c.get("backend_node_id") == tid),
                    0,
                )

            element_criteria = {
                str(i): elem_text(c, i)
                for i, c in enumerate(pool)
            }

            # Question 1: which operation
            q_op = {
                "action_op": {
                    "type": "choice",
                    "instructions": "브라우저에서 다음으로 수행할 동작은 무엇입니까?",
                    "criteria": OP_CRITERIA,
                }
            }
            gold_op = {k: (1.0 if k == op else 0.0) for k in OPS}

            # Question 2: which element
            q_el = {
                "element": {
                    "type": "choice",
                    "instructions": "이 동작을 어느 요소에 수행해야 합니까?",
                    "criteria": element_criteria,
                }
            }
            gold_el = {k: (1.0 if k == str(target_idx) else 0.0) for k in element_criteria}

            state = {
                "instruction": instruction,
                "history": history[:s_idx][-4:],  # recent action summaries only
                "website": task.get("website"),
            }

            rows.append({
                "guid": f"{task['annotation_id']}-step{s_idx}-op",
                "workflow": WORKFLOW,
                "state": json.dumps(state, ensure_ascii=False),
                "questions": json.dumps(q_op, ensure_ascii=False),
                "gold": json.dumps({"action_op": {"probabilities": gold_op}}, ensure_ascii=False),
            })
            rows.append({
                "guid": f"{task['annotation_id']}-step{s_idx}-el",
                "workflow": WORKFLOW,
                "state": json.dumps(state, ensure_ascii=False),
                "questions": json.dumps(q_el, ensure_ascii=False),
                "gold": json.dumps({"element": {"probabilities": gold_el}}, ensure_ascii=False),
            })

    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/mind2web_typed.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0 = all 1009)")
    args = ap.parse_args()

    rows, skipped = build_rows(args.limit)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} browser-agent rows -> {args.output} (skipped {skipped} steps)")


if __name__ == "__main__":
    main()