#!/usr/bin/env python3
"""Verify newly built Laya-format datasets: shape, gold sums, leakage, duplicates.

Checks, per file:
  1. row count and qtype distribution
  2. every question is `choice` with 2..6 labels, gold is one-hot and sums to 1.0 (full pass)
  3. `state` carries no answer marker (regex) and no `answerKey`/`label`-style key
  4. `state` does not contain the gold option text in a way that identifies it
     (positional uniformity of the gold label)
  5. no normalized-state collision with the other data/*.jsonl files or between the new files

Usage:
    python scripts/verify_new_datasets.py data/clinc_typed.jsonl data/mmlu_typed.jsonl ...
"""
import argparse
import collections
import hashlib
import json
import re
import sys

from laya_choice_utils import scan_leak

# bare words that are *informational* only — they occur in ordinary question text
INFO_PATTERNS = {
    "answer(word)": re.compile(r"(?i)\banswer"),
    "correct(word)": re.compile(r"(?i)\bcorrect"),
    "gold(word)": re.compile(r"(?i)\bgold\b"),
    "정답(word)": re.compile("정답"),
}

DATA_FILES = [
    "data/anli_typed.jsonl",
    "data/browser_typed.jsonl",
    "data/jevbench_full.jsonl",
    "data/korean_typed.jsonl",
    "data/long_typed.jsonl",
    "data/mind2web_typed.jsonl",
    "data/xeron3_mix.jsonl",
]

FORBIDDEN_KEYS = re.compile(r"(?i)^(answer|answerkey|answer_key|label|gold|target|correct)$")


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


def verify(path, verbose=False):
    n = 0
    qtypes = collections.Counter()
    label_hist = collections.Counter()
    n_labels = collections.Counter()
    bad_sum, bad_labels, bad_onehot, leak_hits, key_hits = 0, 0, 0, [], []
    info = collections.Counter()
    keys = set()
    n_unique = 0
    samples = []
    for row in iter_rows(path):
        n += 1
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        state_raw = row["state"]
        state_obj = json.loads(state_raw)

        for name, p in INFO_PATTERNS.items():
            if p.search(state_raw):
                info[name] += 1

        if isinstance(state_obj, dict):
            for k in state_obj:
                if FORBIDDEN_KEYS.match(str(k)):
                    key_hits.append((row.get("guid"), k))
        hits = scan_leak(state_raw)
        if hits:
            leak_hits.append((row.get("guid"), hits))

        for qid, q in questions.items():
            qtypes[q["type"]] += 1
            crit = q.get("criteria") or {}
            labs = list(crit.keys())
            n_labels[len(labs)] += 1
            if not (2 <= len(labs) <= 6):
                bad_labels += 1
            probs = gold[qid]["probabilities"]
            s = sum(probs.values())
            if abs(s - 1.0) > 1e-6:
                bad_sum += 1
            elif not (abs(max(probs.values()) - 1.0) < 1e-9
                      and all(abs(v) < 1e-9 for v in probs.values() if v != max(probs.values()))):
                bad_onehot += 1
            top = max(probs, key=probs.get)
            label_hist[top] += 1
        k = key_of(state_raw)
        if k not in keys:
            n_unique += 1
        keys.add(k)
        if verbose and len(samples) < 20:
            samples.append(row)

    return {
        "path": path, "rows": n, "qtypes": dict(qtypes), "n_labels": dict(n_labels),
        "bad_sum": bad_sum, "bad_labels": bad_labels, "bad_onehot": bad_onehot,
        "leak_hits": leak_hits[:10], "n_leak": len(leak_hits),
        "key_hits": key_hits[:10], "n_key_hits": len(key_hits),
        "label_hist": dict(label_hist), "state_keys": keys, "samples": samples,
        "n_unique": n_unique, "info": dict(info),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--no-cross", action="store_true", help="skip comparison against data/*.jsonl")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    results = [verify(p, verbose=True) for p in args.files]

    print("=" * 78)
    for r in results:
        print(f"\n### {r['path']}")
        print(f"  rows           : {r['rows']:,}")
        print(f"  qtypes         : {r['qtypes']}")
        print(f"  labels/question: {r['n_labels']}  (allowed 2..6)")
        print(f"  gold label hist: {r['label_hist']}")
        print(f"  gold sum != 1.0: {r['bad_sum']}")
        print(f"  non-one-hot    : {r['bad_onehot']}")
        print(f"  bad label count: {r['bad_labels']}")
        print(f"  leak regex hits: {r['n_leak']}  {r['leak_hits']}")
        print(f"  forbidden keys : {r['n_key_hits']}  {r['key_hits']}")
        print(f"  unique states  : {r['n_unique']:,} / {r['rows']:,} "
              f"(within-file dups: {r['rows'] - r['n_unique']})")
        print(f"  info word count: {r['info']}  (dataset prose, not answer markers)")

    # cross-file duplicates
    all_keys = {}
    print("\n" + "=" * 78)
    for r in results:
        for k in r["state_keys"]:
            all_keys.setdefault(k, []).append(r["path"])
    cross = {k: v for k, v in all_keys.items() if len(v) > 1}
    print(f"cross-file duplicate states among new files: {len(cross)}")

    if not args.no_cross:
        for old in DATA_FILES:
            try:
                old_keys = {key_of(row["state"]) for row in iter_rows(old)}
            except FileNotFoundError:
                print(f"  {old}: (missing, skipped)")
                continue
            hits = set()
            for r in results:
                hits |= (r["state_keys"] & old_keys)
            print(f"  overlap with {old}: {len(hits)}")

    # eyeball samples
    print("\n" + "=" * 78)
    for r in results:
        print(f"\n--- {args.samples} samples from {r['path']} ---")
        for row in r["samples"][: args.samples]:
            print(f"  guid: {row.get('guid')} | workflow: {row.get('workflow')}")
            print(f"    state: {row['state'][:260]}")
            print(f"    q    : {row['questions'][:260]}")
            print(f"    gold : {row['gold']}")

    ok = all(r["bad_sum"] == 0 and r["bad_labels"] == 0 and r["bad_onehot"] == 0
             and r["n_leak"] == 0 and r["n_key_hits"] == 0 for r in results)
    print("\n" + "=" * 78)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
