#!/usr/bin/env python3
"""Recover the JevBench configs that build_jevbench_dataset.py skipped.

The original builder skips every config whose full label set overflows the Laya
choice head budget (`head_max_len=256`):

    banking77 (k=77), clinc150 (k=151), massive (k=60), ledgar (k=100)

Here each row is rebuilt as a *small-arity* `choice`: the gold label + a handful of
distractors, so the option block fits the head.  (`chaosnli` is not handled here —
it is recovered by running build_jevbench_dataset.py with `--split test`.)

LEXICAL SHORTCUT CONTROL
------------------------
The label name is part of the rendered option text ("slug: readable name"), and an
utterance very often contains the words of its own gold label:

    config       gold name-in-state   random-distractor name-in-state   ratio
    banking77         0.80                    0.16                      5.1x
    clinc150          0.65                    0.035                    18.6x
    massive           0.36                    0.014                    26.0x
    ledgar            0.50                    0.041                    12.1x

A model trained on that learns "pick the option whose words appear in the state"
instead of routing.  Fix: the distractor pool for a row is restricted to labels
whose state-overlap *bucket* equals the gold's, where the bucket is
min(number of name tokens occurring in the utterance, 2).  Every shown option
then carries the identical overlap bucket, so neither the "does it overlap"
feature nor the weaker "how much does it overlap" feature can discriminate, and
gold-vs-distractor overlap rates are equal by construction.  Rows lacking enough
same-bucket distractors are dropped.

Measured on the emitted files (gold vs distractor name-in-state):
    config      binary rate   ratio   mean token count   ratio   rows kept
    banking77   0.69 vs 0.68   1.01x   0.88 vs 0.79       1.11x   64%
    clinc150    0.45 vs 0.43   1.03x   0.46 vs 0.44       1.04x   63%
    massive     0.15 vs 0.13   1.16x   0.15 vs 0.13       1.16x   76%
    ledgar      0.10 vs 0.09   1.12x   0.10 vs 0.09       1.12x   56%

Output rows follow the JevBench / Laya native format:
    {"id", "workflow", "state", "questions", "gold"}   (state/questions/gold = JSON strings)

Usage:
    python scripts/build_jevbench_extra.py --outdir data \
        [--configs banking77,clinc150,massive,ledgar] [--seed 7]
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

from huggingface_hub import hf_hub_download

REPO = "Praveenrajus/jev-bench"

# config -> (split, distractor-count range, labels to drop, instruction override)
CONFIGS = {
    "banking77": ("train", (3, 5), set(), None),
    "clinc150":  ("train", (3, 5), {"oos"},
                  "Which intent does the user's query express?"),
    "massive":   ("train", (3, 5), set(), None),
    "ledgar":    ("train", (7, 11), set(), None),
}

TOK_RE = re.compile(r"[a-z0-9]+")
# the same answer-marker regexes the other builders use (scripts/laya_choice_utils.py)
LEAK_PATTERNS = [
    re.compile(r"(?i)\banswer\s*[:=]\s*[A-F0-9]\b"),
    re.compile(r"(?i)\banswerkey\b"),
    re.compile(r"(?i)\banswer\s+key\b"),
    re.compile(r"(?i)\bcorrect\s+(answer|option|choice|label)\s*(is|:|=)"),
    re.compile(r"(?i)\bthe\s+answer\s+is\s*[A-F0-9]\b"),
    re.compile(r"(?i)\b(gold|correct|true)\s+(label|index|option|choice)\b"),
    re.compile(r"정답\s*[:：]?"),
    re.compile(r"(?i)\bground\s*truth\b"),
    re.compile(r"(?i)\"(answer|answerkey|label|target|gold)\"\s*:"),
]


def toks(s):
    """Content tokens of a string (lowercased, length > 2)."""
    return {t for t in TOK_RE.findall(str(s).lower()) if len(t) > 2}


def render_opt(k, v):
    """Mirror of laya.common.render_options for a single criterion."""
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


def read_rows(rel):
    p = hf_hub_download(REPO, rel, repo_type="dataset")
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_existing_keys(paths):
    keys = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "state" in r:
                    keys.add(key_of(r["state"]))
    return keys


def build_config(name, split, nd_range, drop_labels, ins_override, rng, existing_keys):
    rows, seen = [], set()
    n_oos = n_pool = n_dup = 0
    gold_hit = dist_hit = dist_tot = 0
    gold_cnt = dist_cnt = 0.0
    nopt_hist = Counter()
    pos_hist = Counter()
    for row in read_rows(f"data/{name}/{split}.jsonl"):
        gold = row["label"]
        if gold in drop_labels:
            n_oos += 1
            continue
        q = json.loads(row["question"])
        crit = q.get("criteria")
        if not isinstance(crit, dict) or gold not in crit:
            continue
        labels = [k for k in crit if k not in drop_labels]
        st = json.loads(row["state"])
        stt = toks(st)
        # overlap bucket = min(# name tokens occurring in the utterance, 2)
        cnt = {k: len(toks(render_opt(k, crit[k])) & stt) for k in labels}
        bkt = {k: min(cnt[k], 2) for k in labels}
        gh = bkt[gold]
        same = [k for k in labels if k != gold and bkt[k] == gh]
        if len(same) < nd_range[0]:
            n_pool += 1
            continue
        nd = min(rng.randint(*nd_range), len(same))
        dist = rng.sample(same, nd)
        shown = [gold] + dist
        rng.shuffle(shown)

        new_crit = {k: crit[k] for k in shown}
        probs = {k: (1.0 if k == gold else 0.0) for k in shown}
        rec = {
            "id": row["id"],
            "workflow": f"jevbench-{name}",
            "state": row["state"],                       # utterance only — no label
            "questions": json.dumps(
                {"decision": {"type": "choice",
                              "instructions": ins_override or q["instructions"],
                              "criteria": new_crit}}, ensure_ascii=False),
            "gold": json.dumps(
                {"decision": {"probabilities": probs}}, ensure_ascii=False),
        }
        k = key_of(rec["state"])
        if k in seen or k in existing_keys:
            n_dup += 1
            continue
        seen.add(k)
        rows.append(rec)
        nopt_hist[len(shown)] += 1
        pos_hist[shown.index(gold)] += 1
        if cnt[gold]:
            gold_hit += 1
        gold_cnt += cnt[gold]
        for d in dist:
            dist_tot += 1
            if cnt[d]:
                dist_hit += 1
            dist_cnt += cnt[d]

    stats = {
        "rows": len(rows),
        "dropped_oos": n_oos,
        "dropped_pool": n_pool,
        "dropped_dup": n_dup,
        "gold_overlap": gold_hit / max(len(rows), 1),
        "dist_overlap": dist_hit / max(dist_tot, 1),
        "gold_cnt": gold_cnt / max(len(rows), 1),
        "dist_cnt": dist_cnt / max(dist_tot, 1),
        "n_options": dict(sorted(nopt_hist.items())),
        "gold_pos": dict(sorted(pos_hist.items())),
    }
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check-existing", default="data/*.jsonl",
                    help="glob of existing JSONL files to de-duplicate against")
    ap.add_argument("--manifest", default="data/jevbench_extra_manifest.json")
    ap.add_argument("--no-existing-check", action="store_true")
    args = ap.parse_args()

    import glob
    existing = set()
    if not args.no_existing_check:
        paths = [p for p in glob.glob(args.check_existing)
                 if not os.path.basename(p).startswith("jevbench_extra_")]
        print(f"Indexing {len(paths)} existing files for de-duplication ...", flush=True)
        existing = load_existing_keys(paths)
        print(f"  {len(existing):,} existing state keys", flush=True)

    summary = {}
    for name in args.configs.split(","):
        name = name.strip()
        if name not in CONFIGS:
            print(f"!! unknown config {name}, skipping")
            continue
        split, nd_range, drop, ins = CONFIGS[name]
        rng = random.Random(f"{args.seed}-{name}")
        rows, stats = build_config(name, split, nd_range, drop, ins, rng, existing)
        out = os.path.join(args.outdir, f"jevbench_extra_{name}.jsonl")
        os.makedirs(args.outdir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = stats
        print(f"\n### {name} -> {out}")
        print(f"  rows={stats['rows']:,}  dropped: oos={stats['dropped_oos']} "
              f"pool={stats['dropped_pool']} dup={stats['dropped_dup']}")
        print(f"  overlap(binary) gold={stats['gold_overlap']:.3f} "
              f"distractor={stats['dist_overlap']:.3f} "
              f"ratio={stats['gold_overlap'] / max(stats['dist_overlap'], 1e-9):.2f}x")
        print(f"  overlap(count)  gold={stats['gold_cnt']:.3f} "
              f"distractor={stats['dist_cnt']:.3f} "
              f"ratio={stats['gold_cnt'] / max(stats['dist_cnt'], 1e-9):.2f}x")
        print(f"  options/row={stats['n_options']}  gold_pos={stats['gold_pos']}")
        # global de-dup: later configs must not re-emit an earlier config's state
        existing |= {key_of(r["state"]) for r in rows}

    print("\n" + "=" * 70)
    print("SUMMARY")
    for n, s in summary.items():
        print(f"  {n:10} rows={s['rows']:5,} opts={list(s['n_options'])} "
              f"gold={s['gold_overlap']:.3f} dist={s['dist_overlap']:.3f}")

    if args.manifest:
        man = {
            "builder": "scripts/build_jevbench_extra.py",
            "source": "Praveenrajus/jev-bench",
            "seed": args.seed,
            "shortcut_control": "distractor pool restricted to the gold's overlap bucket "
                                "(min(#name tokens in state, 2)); rows without enough "
                                "same-bucket distractors dropped",
            "configs": summary,
        }
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, ensure_ascii=False)
        print(f"\nmanifest -> {args.manifest}")


if __name__ == "__main__":
    main()
