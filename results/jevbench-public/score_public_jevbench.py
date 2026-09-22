#!/usr/bin/env python3
"""Score a local run against JevBench: matched public-subset comparison + JevBench axes."""
import json
import statistics
import sys

sys.path.insert(0, "/tmp/jevbench")
from jevbench.composite_v13 import (TIER_CHANCES, calibration, cost, intelligence,  # noqa: E402
                                    jevbench_score, speed)
from jevbench.metrics import ece_top_label  # noqa: E402
from jevbench.scoring import argmax_label  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

PUB = "/tmp/jevbench/datasets/public"
ART = "/tmp/jevbench/results/v1.2/jevbench-v1.2-per-task.json"
RES = "/tmp/jevbench/results/v1.2/jevbench-v1.2-results.json"
PRICE_IN_PER_M = 0.01          # same estimate basis the board used for Laya (size-class encoder, $0.01/M in)


def ours(prefix, label):
    recs = [json.loads(l) for l in open(f"{prefix}.results.jsonl")]
    tasks, tier_of = {}, {}
    for f, tier in (("original", "standard"), ("easy", "easy"), ("hard", "hard")):
        for t in load_jsonl(f"{PUB}/{f}.jsonl"):
            tasks[t.id] = t
            tier_of[t.id] = tier
    tiers = {"easy": [], "standard": [], "hard": []}
    pairs, tvds, lats, toks = [], [], [], []
    for r in recs:
        t = tasks[r["task_id"]]
        tiers[tier_of[r["task_id"]]].append(bool(r["correct"]))
        lats.append(r["latency_s"])
        toks.append((r.get("usage") or {}).get("input_tokens") or 0)
        if r.get("probs"):
            gold = str(t.expected)
            p = r["probs"]
            pairs.append((max(p.values()), argmax_label(p) == gold))
            gp = t.provenance.get("gold_probs")
            if gp:
                tvds.append(0.5 * sum(abs(p.get(k, 0.0) - v) for k, v in gp.items()))
    tier_acc = {k: (sum(v) / len(v) if v else None) for k, v in tiers.items()}
    ece = ece_top_label(pairs)["ece"]
    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    lat = sorted(lats)
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    usd_per_1000 = (sum(toks) / len(toks)) * PRICE_IN_PER_M / 1000.0
    axes = {
        "intelligence": intelligence(tier_acc),
        "calibration": calibration(ece, mean_tvd),
        "speed": speed(p50, p95, "cpu"),
        "cost": cost(usd_per_1000),
    }
    return {
        "label": label, "n": len(recs), "tier_accuracy": tier_acc,
        "overall_accuracy": sum(bool(r["correct"]) for r in recs) / len(recs),
        "ece_hard": ece, "n_calib_pairs": len(pairs),
        "mean_tvd": mean_tvd, "n_gold_dist": len(tvds),
        "latency_p50_raw_s": p50, "latency_p95_raw_s": p95,
        "mean_input_tokens": sum(toks) / len(toks),
        "usd_per_1000_est": usd_per_1000,
        "axes": axes, "jevbench_score_partial": jevbench_score(axes),
    }


def board_matched(key):
    art = json.load(open(ART))
    tmeta = art["tasks"]
    pub = {t["id"]: t["tier"] for t in tmeta}
    pt = art["systems"][key]["public_tasks"]
    tiers = {"easy": [], "standard": [], "judge": [], "hard": []}
    for tid, (code, _conf) in pt.items():
        tiers[pub[tid]].append(code == "c")
    tier_acc = {k: (sum(v) / len(v) if v else None) for k, v in tiers.items()}
    off = next(s for s in json.load(open(RES))["systems"] if s["key"] == key)
    return {
        "label": off["display"], "n": len(pt),
        "tier_accuracy_public_matched": tier_acc,
        "overall_accuracy_public_matched": sum(c == "c" for c, _ in pt.values()) / len(pt),
        "intelligence_public_matched": intelligence(tier_acc),
        "official_tiers_all_534": off["tiers"],
        "official_axes": off["axes"],
        "official_jevbench_score": off["jevbench_score"],
        "official_rank": off["rank"],
        "official_calibration": off["calibration"],
        "official_speed": off["speed"],
        "official_cost_usd_per_1000": off["cost"]["usd_per_1000"],
    }


if __name__ == "__main__":
    out = {"xeron": ours(sys.argv[1], sys.argv[2]),
           "laya": board_matched("laya"),
           "jev": board_matched("jev-1.13.0")}
    print(json.dumps(out, indent=1, default=str))
