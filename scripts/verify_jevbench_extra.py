#!/usr/bin/env python3
"""Full verification for the recovered JevBench configs (data/jevbench_extra_*.jsonl).

Checks per file:
  1. row count + qtype distribution
  2. label count in 2..12 (Laya head budget)
  3. gold probabilities sum to 1.0 (full pass); one-hot unless the source is soft-label
  4. answer-marker regex / forbidden state keys (no gold leak into `state`)
  5. lexical-shortcut metric: gold label name tokens appearing in the state vs
     distractor label name tokens (binary rate + mean token count)
  6. gold label-position histogram (positional bias)
  7. within-file and cross-file duplicate states
  8. state overlap against every existing data/*.jsonl

Usage:
    python scripts/verify_jevbench_extra.py data/jevbench_extra_*.jsonl
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_choice_utils import scan_leak

TOK_RE = re.compile(r"[a-z0-9]+")
FORBIDDEN_KEYS = re.compile(r"(?i)^(answer|answerkey|answer_key|label|gold|target|correct)$")
# sources whose gold is a human soft-label distribution, not one-hot
SOFT_SOURCES = {"chaosnli", "go_emotions", "measuring_hate_speech", "civil_comments"}


def toks(s):
    return {t for t in TOK_RE.findall(str(s).lower()) if len(t) > 2}


def render_opt(k, v):
    return k if (v is None or v == "") else f"{k}: {v}"


def norm_state(state):
    try:
        s = json.loads(state)
    except Exception:                                     # noqa: BLE001
        s = state
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def key_of(state):
    return hashlib.sha256(norm_state(state).encode()).hexdigest()[:16]


def iter_rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify(path, check_existing=True):
    r = {"path": path, "rows": 0, "qtypes": collections.Counter(),
         "n_labels": collections.Counter(), "bad_sum": 0, "bad_onehot": 0,
         "bad_labels": 0, "n_leak": 0, "leak_hits": [], "n_key_hits": 0,
         "gold_pos": collections.Counter(), "keys": set(),
         "gold_bin": 0, "gold_cnt": 0.0, "dist_bin": 0, "dist_cnt": 0.0,
         "dist_n": 0, "soft": False}
    for row in iter_rows(path):
        r["rows"] += 1
        state_raw = row["state"]
        try:
            state_obj = json.loads(state_raw)
        except Exception:                                 # noqa: BLE001
            state_obj = state_raw
        if isinstance(state_obj, dict):
            for k in state_obj:
                if FORBIDDEN_KEYS.match(str(k)):
                    r["n_key_hits"] += 1
        hits = scan_leak(state_raw)
        if hits:
            r["n_leak"] += 1
            if len(r["leak_hits"]) < 5:
                r["leak_hits"].append((row.get("id"), hits))

        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        stt = toks(state_obj)
        for qid, q in questions.items():
            r["qtypes"][q["type"]] += 1
            crit = q.get("criteria") or {}
            labs = list(crit.keys())
            r["n_labels"][len(labs)] += 1
            if not (2 <= len(labs) <= 12):
                r["bad_labels"] += 1
            probs = gold[qid]["probabilities"]
            s = sum(probs.values())
            if abs(s - 1.0) > 1e-6:
                r["bad_sum"] += 1
            top = max(probs, key=probs.get)
            r["gold_pos"][labs.index(top)] += 1
            is_soft = len({round(v, 6) for v in probs.values()}) > 2
            if is_soft:
                r["soft"] = True
            else:
                mx = max(probs.values())
                if not (abs(mx - 1.0) < 1e-9
                        and all(abs(v) < 1e-9 for v in probs.values() if v != mx)):
                    r["bad_onehot"] += 1
            # shortcut metric
            gtok = toks(render_opt(top, crit.get(top)))
            if gtok & stt:
                r["gold_bin"] += 1
            r["gold_cnt"] += len(gtok & stt)
            for lab in labs:
                if lab == top:
                    continue
                dtok = toks(render_opt(lab, crit.get(lab)))
                r["dist_n"] += 1
                if dtok & stt:
                    r["dist_bin"] += 1
                r["dist_cnt"] += len(dtok & stt)
        r["keys"].add(key_of(state_raw))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--no-cross", action="store_true")
    args = ap.parse_args()

    files = []
    for pat in args.files:
        files.extend(sorted(glob.glob(pat)))
    results = [verify(p) for p in files]

    print("=" * 78)
    for r in results:
        n = max(r["rows"], 1)
        gb = r["gold_bin"] / n
        db = r["dist_bin"] / max(r["dist_n"], 1)
        gc = r["gold_cnt"] / n
        dc = r["dist_cnt"] / max(r["dist_n"], 1)
        print(f"\n### {r['path']}")
        print(f"  rows            : {r['rows']:,}")
        print(f"  qtypes          : {dict(r['qtypes'])}")
        print(f"  labels/question : {dict(sorted(r['n_labels'].items()))}  (allowed 2..12)")
        print(f"  gold sum != 1.0 : {r['bad_sum']}   non-one-hot: {r['bad_onehot']}"
              f"  (soft-label source: {r['soft']})")
        print(f"  bad label count : {r['bad_labels']}")
        print(f"  leak regex      : {r['n_leak']} {r['leak_hits']}")
        print(f"  forbidden keys  : {r['n_key_hits']}")
        print(f"  unique states   : {len(r['keys']):,} / {r['rows']:,} "
              f"(within-file dups: {r['rows'] - len(r['keys'])})")
        print(f"  gold position   : {dict(sorted(r['gold_pos'].items()))}")
        print(f"  SHORTCUT  gold-name-in-state={gb:.3f}  distractor={db:.3f}  "
              f"ratio={gb / max(db, 1e-9):.2f}x   (count: {gc:.3f} vs {dc:.3f})")

    all_keys = {}
    for r in results:
        for k in r["keys"]:
            all_keys.setdefault(k, []).append(r["path"])
    cross = {k: v for k, v in all_keys.items() if len(v) > 1}
    print("\n" + "=" * 78)
    print(f"cross-file duplicate states among new files: {len(cross)}")

    if not args.no_cross:
        print("\noverlap with existing data/*.jsonl (complete-state duplicates):")
        new_keys = set(all_keys)
        for old in sorted(glob.glob("data/*.jsonl")):
            if os.path.basename(old).startswith("jevbench_extra_"):
                continue
            try:
                ok = {key_of(row["state"]) for row in iter_rows(old) if "state" in row}
            except FileNotFoundError:
                continue
            ov = len(new_keys & ok)
            flag = "" if ov == 0 else "  <-- NONZERO"
            print(f"  {os.path.basename(old):28} states={len(ok):8,}  overlap={ov}{flag}")

    ok = all(r["bad_sum"] == 0 and r["bad_labels"] == 0 and r["n_leak"] == 0
             and r["n_key_hits"] == 0 for r in results)
    print("\n" + "=" * 78)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
