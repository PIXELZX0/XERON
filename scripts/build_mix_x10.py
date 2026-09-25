#!/usr/bin/env python3
"""Build `data/xeron10_mix.jsonl` — the XERON-1.0 final training mix (W4).

Concatenates **every** source (all `data/bulk/*.jsonl` W1/W2/W3 + the legacy typed sets)
and enforces exactly the invariants `scripts/verify_bulk.py` checks, so the mix verifies
clean (leak=0, goldsum=0, dup_in=0, bad_arity=0, empty=0, badjson=0):

  1. one question per row, qid normalized to `decision` — the legacy sets use `topic`,
     `relation`, `is_spam`, `element`, … ; `verify_bulk` reads `questions["decision"]` and
     `preprocess.build_training_item` indexes `gold` by the same qid, so the rename is
     lossless (no code path reads the qid string itself).
  2. choice: 2 <= arity <= MAX_K(16) and criteria keys == gold probability keys
     (the contract of `build_bulk_choice.py`, also `verify_bulk`'s 2..16 rule).
  3. score: criteria normalized to the **list** form. `build_bulk_choice.py` emits
     `{"0": "1 star", ...}` while `preprocess.build_training_item` does
     `n_levels = len(crit) if isinstance(crit, list) else 4` -> a 5-level dict silently
     truncated to 4 targets (5-star gold became a uniform 4-way target). Converting to the
     list form makes target length == marker count again.
  4. gold probabilities: all finite, sum > 0, rescaled to sum 1.0.
  5. `state` free of answer markers / forbidden keys, and non-empty (>=10 normalized chars).
  6. exact duplicates dropped on sha256 of the same normalized state that `mix_datasets` and
     `verify_bulk` use (normalized state is a superset of (state, questions) identity here,
     and verify counts state-only duplicates).

Excluded on purpose: `data/clinc_typed.jsonl`, `data/clinc_hard_typed.jsonl` (63 % of rows
carry the intent name inside the state) and the previous mix products
(`xeron3/5/9_mix.jsonl`, which would double-count their sources).

Sources are read-only; the output is a new file. Deterministic: no sampling, no RNG
(row order = source order, first occurrence wins a duplicate). seed=7 is the project-wide
seed and is recorded in the manifest for the downstream (sharded) preprocessing.

The output is written **atomically**: rows go to `<out>.tmp.<pid>` which is fsync'd and then
`os.replace`d onto `<out>`, so an interrupted run (kill / gateway restart) can never leave a
truncated mix behind — the previous file survives untouched until the new one is complete.
The stats file carries `rows_out` + `size_bytes` + `sha256`, and a re-run whose `<out>` already
matches the stats file is a no-op (idempotent; override with `--force`).

Usage:
    python scripts/build_mix_x10.py --out data/xeron10_mix.jsonl \
        --stats data/xeron10_mix_stats.json
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

from mix_datasets import _norm_state as norm_state          # noqa: E402
from verify_bulk import LEAK_PATTERNS, FORBIDDEN_KEYS       # noqa: E402

MIN_OPTS, MAX_K = 2, 16
EMPTY_CHARS = 10

LEGACY = [
    "data/jevbench_full.jsonl",
    "data/jevbench_extra_banking77.jsonl",
    "data/jevbench_extra_chaosnli.jsonl",
    "data/jevbench_extra_clinc150.jsonl",
    "data/jevbench_extra_ledgar.jsonl",
    "data/jevbench_extra_massive.jsonl",
    "data/korean_typed.jsonl",
    "data/anli_typed.jsonl",
    "data/browser_typed.jsonl",
    "data/long_typed.jsonl",
    "data/mind2web_typed.jsonl",
    "data/mmlu_typed.jsonl",
    "data/hellaswag_typed.jsonl",
    "data/arc_typed.jsonl",
]
EXCLUDE = {"clinc_typed.jsonl", "clinc_hard_typed.jsonl",
           "xeron3_mix.jsonl", "xeron5_mix.jsonl", "xeron9_mix.jsonl",
           "xeron5_base.jsonl"}


def sources():
    out = [p for p in sorted(glob.glob("data/bulk/*.jsonl"))
           if os.path.basename(p) not in EXCLUDE]
    out += LEGACY
    return out


def normalize_row(row):
    """Return (row, fix_notes) or (None, drop_reason)."""
    try:
        state_s = row["state"]
        qs = json.loads(row["questions"])
        gs = json.loads(row["gold"])
        state = json.loads(state_s)
    except Exception:                                        # noqa: BLE001
        return None, "bad_json"

    if not isinstance(qs, dict) or not isinstance(gs, dict) or len(qs) != 1:
        return None, "bad_shape"
    qid = next(iter(qs))
    fix_qid = qid != "decision"
    q = qs[qid]
    if qid not in gs:
        return None, "gold_qid_missing"
    gold_q = gs[qid]
    if fix_qid:
        qs = {"decision": q}
        gs = {"decision": gold_q}

    probs = gold_q.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None, "bad_gold"
    try:
        vals = {k: float(v) for k, v in probs.items()}
    except Exception:                                        # noqa: BLE001
        return None, "bad_gold"
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals.values()):
        return None, "bad_gold"
    if any(v < 0 for v in vals.values()):
        return None, "negative_gold"
    s = sum(vals.values())
    if s <= 0:
        return None, "gold_zero_sum"
    if abs(s - 1.0) > 1e-9:
        vals = {k: v / s for k, v in vals.items()}

    t = q.get("type")
    crit = q.get("criteria")
    notes = []
    if t == "choice":
        if not isinstance(crit, dict) or len(crit) < MIN_OPTS:
            return None, "bad_criteria"
        if set(crit) != set(vals):
            return None, "criteria_gold_key_mismatch"
        k = len(crit)
        if k > MAX_K:
            return None, f"choice_arity_gt_{MAX_K}"
    elif t == "score":
        if isinstance(crit, dict):
            keys = list(crit)
            try:
                order = sorted(keys, key=lambda x: int(x))
            except Exception:                                # noqa: BLE001
                return None, "bad_criteria"
            if [str(i) for i in range(len(order))] != order:
                return None, "score_keys_not_ordinal"
            crit = [crit[k_] for k_ in order]
            q = dict(q, criteria=crit)
            qs = {"decision": q}
            notes.append("score_dict_to_list")
        if not isinstance(crit, list) or not crit:
            return None, "bad_criteria"
        if set(vals) != {str(i) for i in range(len(crit))}:
            return None, "score_gold_key_mismatch"
    elif t == "noul":
        if set(vals) != {"false", "true"}:
            return None, "noul_gold_keys"
        if crit is not None and not isinstance(crit, dict):
            return None, "bad_criteria"
    else:
        return None, "unknown_qtype"

    if isinstance(state, dict) and any(FORBIDDEN_KEYS.match(str(k)) for k in state):
        return None, "forbidden_state_key"
    if any(p.search(state_s) for p in LEAK_PATTERNS):
        return None, "answer_marker_in_state"
    txt = norm_state(state_s)
    if len(txt) < EMPTY_CHARS:
        return None, "empty_state"

    row = dict(row)
    row["questions"] = json.dumps(qs, ensure_ascii=False)
    row["gold"] = json.dumps(gs, ensure_ascii=False)
    row.pop("id", None)
    return row, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/xeron10_mix.jsonl")
    ap.add_argument("--stats", default="data/xeron10_mix_stats.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: rows per source")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when --out already matches --stats")
    args = ap.parse_args()

    if not args.force and os.path.exists(args.out) and os.path.exists(args.stats):
        try:
            prev = json.load(open(args.stats))
            n = sum(1 for _ in open(args.out, "rb"))
            if prev.get("rows_out") == n and prev.get("size_bytes") == os.path.getsize(args.out):
                print(f"[skip] {args.out} already matches {args.stats} "
                      f"(rows_out={n:,}, size_bytes={prev['size_bytes']:,}) — nothing to do")
                return
            print(f"[rebuild] {args.out} disagrees with {args.stats}: "
                  f"rows {n:,} vs {prev.get('rows_out')}, "
                  f"size {os.path.getsize(args.out):,} vs {prev.get('size_bytes')}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[rebuild] cannot validate {args.out} against {args.stats}: "
                  f"{type(e).__name__}: {e}", flush=True)
    elif not os.path.exists(args.out):
        print(f"[build] {args.out} missing", flush=True)

    srcs = sources()
    print(f"sources: {len(srcs)} files", flush=True)
    seen = set()
    per_source = []
    drops = collections.Counter()
    fixes = collections.Counter()
    qtype = collections.Counter()
    arity = collections.Counter()
    langs = collections.Counter()
    workflows = collections.Counter()
    reason_by_source = collections.defaultdict(collections.Counter)
    total_in = total_out = 0

    tmp_out = f"{args.out}.tmp.{os.getpid()}"
    with open(tmp_out, "w") as fo:
        for path in srcs:
            n_in = n_out = 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    if args.limit and n_in > args.limit:
                        break
                    try:
                        row = json.loads(line)
                    except Exception:                        # noqa: BLE001
                        drops["bad_json"] += 1
                        reason_by_source[path]["bad_json"] += 1
                        continue
                    row, notes = normalize_row(row)
                    if row is None:
                        drops[notes] += 1
                        reason_by_source[path][notes] += 1
                        continue
                    for n in notes:
                        fixes[n] += 1
                    k = hashlib.sha256(norm_state(row["state"]).encode()).hexdigest()
                    if k in seen:
                        drops["duplicate_state"] += 1
                        reason_by_source[path]["duplicate_state"] += 1
                        continue
                    seen.add(k)
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_out += 1
                    q = json.loads(row["questions"])["decision"]
                    g = json.loads(row["gold"])["decision"]["probabilities"]
                    qtype[q["type"]] += 1
                    if q["type"] == "choice":
                        arity[len(g)] += 1
                    st = json.loads(row["state"])
                    if isinstance(st, dict):
                        langs[str(st.get("language", "?"))] += 1
                    workflows[row.get("workflow", "?")] += 1
            per_source.append({"file": path, "rows_in": n_in, "rows_out": n_out,
                               "dropped": dict(reason_by_source[path])})
            total_in += n_in
            total_out += n_out
            print(f"  {os.path.basename(path):34} in {n_in:>9,}  out {n_out:>9,}  "
                  f"drop {n_in - n_out:>6,}", flush=True)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp_out, args.out)          # atomic: old file survives until this point

    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)

    keep = total_out or 1
    top_wf = workflows.most_common(25)
    stats = {
        "out": args.out,
        "seed": args.seed,
        "size_bytes": os.path.getsize(args.out),
        "sha256": h.hexdigest(),
        "atomic_write": "tmp + fsync + os.replace",
        "repro": ("PYTHONPATH=scripts .venv/bin/python scripts/build_mix_x10.py "
                  "--out data/xeron10_mix.jsonl --stats data/xeron10_mix_stats.json"),
        "sources": per_source,
        "rows_in": total_in,
        "rows_out": total_out,
        "dropped_total": total_in - total_out,
        "dropped_by_reason": dict(drops.most_common()),
        "row_fixes": dict(fixes),
        "qtype": dict(qtype),
        "choice_arity": {str(k): v for k, v in sorted(arity.items())},
        "languages": {k: v for k, v in langs.most_common()},
        "n_language_tags": len(langs),
        "workflow_top25": top_wf,
        "max_workflow_share": round(top_wf[0][1] / keep, 4) if top_wf else 0.0,
        "non_choice_rows_without_language_tag": total_out - sum(langs.values()),
    }
    tmp_stats = f"{args.stats}.tmp.{os.getpid()}"
    with open(tmp_stats, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    os.replace(tmp_stats, args.stats)

    print(f"\nrows in {total_in:,} -> out {total_out:,} (dropped {total_in - total_out:,})")
    print(f"sha256 {stats['sha256']}  size {stats['size_bytes']:,} B")
    print(f"dropped by reason: {dict(drops.most_common())}")
    print(f"fixes: {dict(fixes)}")
    print(f"qtype: {dict(qtype)}   lang tags: {len(langs):,}   "
          f"max workflow share: {stats['max_workflow_share']:.3%} ({top_wf[0][0]})")
    print(f"-> {args.out}\n-> {args.stats}")


if __name__ == "__main__":
    main()
