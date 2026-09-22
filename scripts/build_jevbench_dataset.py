#!/usr/bin/env python3
"""Build a Laya-format JSONL from the Jevify `Praveenrajus/jev-bench` corpus (HF).

jev-bench = 22 public datasets reformatted into System One questions
(choice / score / noul) with human label distributions where they exist.
133,953 train rows total. https://huggingface.co/datasets/Praveenrajus/jev-bench

Output rows follow the Laya native format consumed by scripts/preprocess.py:
    {"id", "workflow", "state", "questions", "gold"}   (all JSON strings)

Option-count guard: the Laya head budget is `head_max_len` (256 in XERON), so
configs with a large option set (banking77 k=77, clinc150 k=151, ledgar k=100,
massive k=60) are skipped — their options would be truncated to ~4 tokens each.

Usage:
    python scripts/build_jevbench_dataset.py --output data/jevbench_typed.jsonl \
        [--max-options 32] [--limit 0] [--configs go_emotions,boolq,...]
"""
import argparse
import json
import os
from collections import Counter

from huggingface_hub import hf_hub_download

REPO = "Praveenrajus/jev-bench"

# counts/sizes come from the repo manifest; option counts from the manifest `k` field
SKIP_LARGE_K = {"banking77": 77, "clinc150": 151, "ledgar": 100, "massive": 60}


def load_manifest():
    p = hf_hub_download(REPO, "manifest.json", repo_type="dataset")
    with open(p) as f:
        return json.load(f)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_soft(soft, keys):
    """Normalize a soft label over `keys`; returns None if unusable.

    jev-bench emits soft labels in three shapes: a dict keyed by option, a list aligned to
    the option order (score levels / choice criteria), or — for noul — a scalar P(true).
    """
    if isinstance(soft, dict):
        vals = {k: float(soft.get(k, 0.0) or 0.0) for k in keys}
    elif isinstance(soft, (list, tuple)):
        if len(soft) != len(keys):
            return None
        vals = {k: float(v or 0.0) for k, v in zip(keys, soft)}
    else:
        return None
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in vals.items()}


def convert_row(row, max_options, max_levels):
    prim = row["primitive"]
    q = json.loads(row["question"])
    crit = q.get("criteria")
    soft = row.get("soft_label")
    if isinstance(soft, str):
        soft = None if soft.strip() in ("None", "") else json.loads(soft)

    if prim == "choice":
        if not isinstance(crit, dict) or not crit:
            return None
        keys = list(crit.keys())
        if len(keys) > max_options:
            return None
        probs = norm_soft(soft, keys)
        if probs is None:
            lab = row.get("label")
            if lab not in keys:
                return None
            probs = {k: (1.0 if k == lab else 0.0) for k in keys}
    elif prim == "score":
        n = len(crit) if isinstance(crit, list) else None
        if n is None or n > max_levels:
            return None
        keys = [str(i) for i in range(n)]
        probs = norm_soft(soft, keys)
        if probs is None:
            lab = str(row.get("label"))
            if lab not in keys:
                return None
            probs = {k: (1.0 if k == lab else 0.0) for k in keys}
    elif prim == "noul":
        lab = str(row.get("label")).strip().lower()
        if lab in ("1", "yes", "true"):
            p = 1.0
        elif lab in ("0", "no", "false"):
            p = 0.0
        else:
            return None
        # civil_comments ships a scalar: the fraction of annotators who said "true"
        if isinstance(soft, bool):
            p = float(soft)
        elif isinstance(soft, (int, float)) and 0.0 <= float(soft) <= 1.0:
            p = float(soft)
        elif isinstance(soft, dict) and soft:
            sv = {k.lower(): float(v or 0.0) for k, v in soft.items()}
            if "true" in sv or "yes" in sv:
                p = sv.get("true", sv.get("yes", p))
            elif sum(sv.values()) > 0:
                p = 1.0 - sv.get("false", sv.get("no", 1.0 - p))
        probs = {"false": 1.0 - p, "true": p}
    else:
        return None

    return {
        "id": row["id"],
        "workflow": f"jevbench-{row['source']}",
        "state": row["state"],                       # already a JSON string
        "questions": json.dumps({"decision": q}, ensure_ascii=False),
        "gold": json.dumps({"decision": {"probabilities": probs}}, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/jevbench_typed.jsonl")
    ap.add_argument("--max-options", type=int, default=32)
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap rows written (0 = all)")
    ap.add_argument("--configs", default=None, help="comma list; default = all supported")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    man = load_manifest()
    sources = man["sources"]
    want = set(args.configs.split(",")) if args.configs else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    written = 0
    per_source = Counter()
    skipped_k = []
    with open(args.output, "w") as out:
        for name, meta in sources.items():
            if want and name not in want:
                continue
            if name in SKIP_LARGE_K:
                skipped_k.append(f"{name}(k={SKIP_LARGE_K[name]})")
                continue
            rel = meta.get("files", {}).get(args.split)
            if not rel:
                continue
            path = hf_hub_download(REPO, rel, repo_type="dataset")
            n_ok = n_bad = 0
            for row in read_jsonl(path):
                conv = convert_row(row, args.max_options, args.max_levels)
                if conv is None:
                    n_bad += 1
                    continue
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n_ok += 1
                written += 1
                if args.limit and written >= args.limit:
                    break
            per_source[name] = n_ok
            print(f"  {name:26} {n_ok:7,} rows  ({n_bad} skipped)", flush=True)
            if args.limit and written >= args.limit:
                print("limit reached")
                break

    print(f"\nWrote {written:,} rows -> {args.output}")
    if skipped_k:
        print(f"Skipped large-option configs: {', '.join(skipped_k)}")
    print("By source:", dict(per_source))


if __name__ == "__main__":
    main()
