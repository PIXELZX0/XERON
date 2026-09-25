#!/usr/bin/env python3
"""Verify Laya-format bulk datasets (XERON-1.0): shape, gold, leakage, positional bias, duplicates.

Per file:
  1. row count + qtype distribution + option-count distribution
  2. every gold sums to 1.0 (full pass), choice arity 2..16, noul keys {false,true}
  3. no answer marker in `state` (regex) and no forbidden state keys
  4. gold-label position uniformity (max |observed - expected| share)
  5. empty/tiny state detection
  6. language distribution (if present in state)
  7. cross-file duplicate states (normalized) + duplicates vs --against globs

Usage:
    python scripts/verify_bulk.py "data/bulk/*.jsonl" [--against "data/*.jsonl"]
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

LEAK_PATTERNS = [
    re.compile(r"(?i)\banswer\s*[:=]\s*[A-P0-9]\b"),
    re.compile(r"(?i)\banswerkey\b"),
    re.compile(r"(?i)\banswer\s+key\b"),
    re.compile(r"(?i)\bcorrect\s+(answer|option|choice|label)\s*(is|:|=)"),
    re.compile(r"(?i)\bthe\s+answer\s+is\s*[A-P0-9]\b"),
    re.compile(r"(?i)\b(gold|correct|true)\s+(label|index|option|choice)\b"),
    re.compile(r"정답\s*[:：]?"),
    re.compile(r"(?i)\bground\s+truth\b"),
    re.compile(r'(?i)"(answer|answerkey|label|target|gold)"\s*:'),
]
FORBIDDEN_KEYS = re.compile(r"(?i)^(answer|answerkey|answer_key|label|gold|target|correct)$")


def norm_state(state_json):
    try:
        s = json.loads(state_json)
    except Exception:                                     # noqa: BLE001
        s = state_json
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def key_of(state_json):
    return hashlib.sha256(norm_state(state_json).encode()).hexdigest()[:16]


def check_file(path, seen_global):
    stats = {
        "file": os.path.basename(path), "rows": 0, "bad_json": 0, "bad_gold_sum": 0,
        "bad_arity": 0, "leak": 0, "forbidden_key": 0, "empty_state": 0,
        "dup_in_file": 0, "dup_cross_file": 0, "qtype": collections.Counter(),
        "arity": collections.Counter(), "langs": collections.Counter(),
        "gold_pos": collections.Counter(), "workflows": collections.Counter(),
        "gold_pos_by_arity": collections.defaultdict(collections.Counter),
    }
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                state = json.loads(r["state"])
                q = json.loads(r["questions"])["decision"]
                g = json.loads(r["gold"])["decision"]["probabilities"]
            except Exception:                             # noqa: BLE001
                stats["bad_json"] += 1
                continue
            stats["rows"] += 1
            stats["qtype"][q["type"]] += 1
            stats["workflows"][r.get("workflow", "?")] += 1
            if isinstance(state, dict):
                stats["langs"][str(state.get("language", "?"))] += 1
                if any(FORBIDDEN_KEYS.match(str(k)) for k in state):
                    stats["forbidden_key"] += 1
            if abs(sum(float(v) for v in g.values()) - 1.0) > 1e-6:
                stats["bad_gold_sum"] += 1
            if q["type"] == "choice":
                n = len(g)
                stats["arity"][n] += 1
                if not (2 <= n <= 16):
                    stats["bad_arity"] += 1
                top = max(g, key=lambda k: g[k])
                stats["gold_pos"][top] += 1
                stats["gold_pos_by_arity"][n][top] += 1
            elif q["type"] == "noul":
                if set(g) != {"false", "true"}:
                    stats["bad_arity"] += 1
            sjson = r["state"]
            if any(p.search(sjson) for p in LEAK_PATTERNS):
                stats["leak"] += 1
            txt = norm_state(sjson)
            if len(txt) < 10:
                stats["empty_state"] += 1
            k = key_of(sjson)
            if k in seen:
                stats["dup_in_file"] += 1
            seen.add(k)
            if k in seen_global:
                stats["dup_cross_file"] += 1
            else:
                seen_global.add(k)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob of files to verify")
    ap.add_argument("--against", default="", help="glob of extra files for cross-file duplicate check")
    ap.add_argument("--json-out", default="data/bulk/_verify.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"no files matched {args.pattern}")
        return 1
    seen_global = set()
    if args.against:
        for p in sorted(glob.glob(args.against)):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seen_global.add(key_of(json.loads(line)["state"]))
                    except Exception:                     # noqa: BLE001
                        continue
        print(f"against: {len(seen_global):,} existing state keys", flush=True)

    out, total = [], 0
    for p in files:
        st = check_file(p, seen_global)
        total += st["rows"]
        ok = (st["bad_json"] == 0 and st["bad_gold_sum"] == 0 and st["bad_arity"] == 0
              and st["leak"] == 0 and st["forbidden_key"] == 0 and st["dup_in_file"] == 0)
        flag = "OK " if ok else "WARN"
        pos = st["gold_pos"]
        tot = sum(pos.values()) or 1
        # deviation must be computed per arity: mixing 2-option and 3-option rows in one
        # histogram makes a perfectly uniform layout look skewed.
        dev = 0.0
        for n, c in (st.get("gold_pos_by_arity") or {}).items():
            t = sum(c.values()) or 1
            dev = max(dev, max(abs(v / t - 1.0 / max(1, len(c))) for v in c.values()))
        if not (st.get("gold_pos_by_arity") or {}):
            dev = max(abs(v / tot - 1.0 / max(1, len(pos))) for v in pos.values()) if pos else 0.0
        st["gold_pos_max_dev"] = round(dev, 3)
        print(f"[{flag}] {st['file']:28} rows={st['rows']:>8,} qtype={dict(st['qtype'])} "
              f"arity={dict(sorted(st['arity'].items()))} langs={len(st['langs'])} "
              f"leak={st['leak']} goldsum={st['bad_gold_sum']} dup_in={st['dup_in_file']} "
              f"dup_cross={st['dup_cross_file']} posdev={st['gold_pos_max_dev']} "
              f"empty={st['empty_state']} badjson={st['bad_json']}", flush=True)
        st["qtype"] = dict(st["qtype"])
        st["arity"] = {str(k): v for k, v in st["arity"].items()}
        st["langs"] = dict(st["langs"])
        st["gold_pos"] = dict(st["gold_pos"])
        st["gold_pos_by_arity"] = {str(k): dict(v) for k, v in st["gold_pos_by_arity"].items()}
        st["workflows"] = dict(st["workflows"])
        out.append(st)

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"files": out, "total_rows": total}, f, ensure_ascii=False, indent=1)
    print(f"\n=== {len(files)} files, {total:,} rows -> {args.json_out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
