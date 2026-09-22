#!/usr/bin/env python3
"""Evaluate a fine-tuned XERON (or base Laya) checkpoint on typed-decisions test split.

Usage:
    python scripts/evaluate.py \
        --model ./output/xeron \
        --dataset LocalLLaMA/typed-decisions \
        --split test \
        --device cuda
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from datasets import load_dataset

import laya
from laya.common import ece_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Fine-tuned model directory")
    ap.add_argument("--dataset", default="LocalLLaMA/typed-decisions")
    ap.add_argument("--config-name", default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="eval_results.json")
    args = ap.parse_args()

    print(f"Loading {args.split} split of {args.dataset}...")
    ds = load_dataset(args.dataset, args.config_name, split=args.split)

    agent_ft = laya.Agent(args.model, device=args.device)

    predictions = []
    latencies_ms = []

    print(f"Evaluating {len(ds)} cases on {args.device}...")
    t0_eval = time.time()

    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])

        t0 = time.perf_counter()
        res = agent_ft.predict(state, questions)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt_ms)

        predictions.append({
            "id": row.get("id"),
            "workflow": row.get("workflow"),
            "pred": res["answers"],
            "gold": gold,
            "questions": questions,
            "latency_ms": dt_ms,
        })

    print(f"Evaluated {len(ds)} cases in {time.time() - t0_eval:.1f}s")

    # ---- aggregate metrics: accuracy / soft accuracy / Brier / ECE / score MAE
    accs, soft_accs, briers, eces, score_maes, n = [], [], [], [], [], 0

    for p in predictions:
        for qid, q in p["questions"].items():
            if qid not in p["gold"]:
                continue
            t = q["type"]
            if qid not in (p["gold"] or {}):
                continue
            gold_dist = p["gold"][qid]["probabilities"]
            gold_label = max(gold_dist, key=gold_dist.get)
            _pe = (p["pred"] or {}).get(qid)
            if not isinstance(_pe, dict):
                continue

            if t == "choice":
                pred_dist = _pe.get("probabilities")
                if not isinstance(pred_dist, dict) or not pred_dist:
                    continue
                pred_label = max(pred_dist, key=pred_dist.get)
                n += 1
                accs.append(pred_label == gold_label)
                soft_accs.append(pred_dist.get(gold_label, 0.0))
                briers.append(
                    sum((pred_dist.get(k, 0.0) - 1.0 if k == gold_label else pred_dist.get(k, 0.0)) ** 2
                        for k in set(list(pred_dist) + list(gold_dist)))
                )
                _opts = sorted(set(list(pred_dist) + list(gold_dist)))
                eces.append(ece_score(
                    np.array([pred_dist.get(k, 0.0) for k in _opts]),
                    np.array([k == gold_label for k in _opts], dtype=bool),
                ))
            elif t == "score":
                pred_dist = _pe.get("distribution") or _pe.get("probabilities") or {}
                if not pred_dist:
                    continue
                gold_i = int(gold_label) if gold_label.isdigit() else gold_label
                try:
                    pred_label = (int(max(pred_dist.items(), key=lambda kv: kv[1])[0])
                                  if isinstance(pred_dist, dict) else pred_dist)
                except Exception:
                    pred_label = gold_i
                # score MAE against gold numeric level
                keys = sorted(pred_dist.keys(), key=lambda x: int(x)) if isinstance(pred_dist, dict) else range(len(pred_dist))
                score_maes.append(abs(float(gold_i) - float(pred_label)))
                n += 1

    summary = {
        "n_decisions": n,
        "accuracy": float(np.mean(accs)) if accs else None,
        "soft_accuracy": float(np.mean(soft_accs)) if soft_accs else None,
        "brier": float(np.mean(briers)) if briers else None,
        "ece": float(np.mean(eces)) if eces else None,
        "score_mae": float(np.mean(score_maes)) if score_maes else None,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_median": float(np.median(latencies_ms)),
        "cases": len(ds),
    }
    print(json.dumps(summary, indent=2))

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2)
    print(f"Saved results to {args.output}")

    # raw predictions sidecar (survives any downstream metric change)
    try:
        with open(args.output.replace(".json", ".preds.json"), "w") as f:
            json.dump(predictions, f)
    except Exception as e:
        print(f"sidecar note: {e}")


if __name__ == "__main__":
    main()