#!/usr/bin/env python3
"""Recompute score-MAE (and a type-wise breakdown) from saved evaluate.py predictions."""
import json, sys
import numpy as np
from laya.common import ece_score

def metrics(path):
    d = json.load(open(path))
    accs, softs, briers, eces, smaes = [], [], [], [], []
    per_type = {}
    for p in d["predictions"]:
        for qid, q in p["questions"].items():
            gold = (p["gold"] or {}).get(qid)
            pred = (p["pred"] or {}).get(qid)
            if not gold or not isinstance(pred, dict):
                continue
            gd = gold["probabilities"]
            gl = max(gd, key=gd.get)
            t = q["type"]
            if t == "choice":
                pd = pred.get("probabilities")
                if not isinstance(pd, dict) or not pd:
                    continue
                pl = max(pd, key=pd.get)
                accs.append(pl == gl)
                softs.append(pd.get(gl, 0.0))
                briers.append(sum((pd.get(k, 0.0) - (1.0 if k == gl else 0.0)) ** 2
                                  for k in set(list(pd) + list(gd))))
                opts = sorted(set(list(pd) + list(gd)))
                eces.append(ece_score(np.array([pd.get(k, 0.0) for k in opts]),
                                      np.array([k == gl for k in opts], dtype=bool)))
                per_type.setdefault("choice", []).append(pl == gl)
            elif t == "score":
                pd = pred.get("probabilities") or pred.get("distribution") or {}
                if not isinstance(pd, dict) or not pd:
                    continue
                gi = int(gl)
                pl = int(max(pd.items(), key=lambda kv: kv[1])[0])
                smaes.append(abs(gi - pl))
                per_type.setdefault("score", []).append(gi == pl)
            elif t == "noul":
                pd = pred.get("noul")
                if pd is None:
                    continue
                gv = gd.get("1", gd.get("true", 0.0))
                if not isinstance(gv, (int, float)):
                    gv = 1.0 if str(gl).lower() in ("1", "true") else 0.0
                per_type.setdefault("noul", []).append((pd >= 0.5) == (gv >= 0.5))
    return {
        "n_choice": len(accs),
        "accuracy": float(np.mean(accs)) if accs else None,
        "soft_accuracy": float(np.mean(softs)) if softs else None,
        "brier": float(np.mean(briers)) if briers else None,
        "ece": float(np.mean(eces)) if eces else None,
        "score_mae": float(np.mean(smaes)) if smaes else None,
        "per_type_acc": {k: round(float(np.mean(v)), 4) for k, v in per_type.items()},
    }

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        print(json.dumps(metrics(path), indent=2))
