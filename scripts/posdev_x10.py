#!/usr/bin/env python3
"""True per-arity positional deviation for the XERON-1.0 mix.

`verify_bulk.py` reports `posdev` as max over arity buckets of |share - 1/#distinct_keys| on the
**gold key string**. That is only meaningful when every row in a bucket names its options the
same way. The mix carries two conventions:

  * **letter-keyed** — `criteria = {"A": "<option text>", ...}` (everything `build_bulk_choice.py`
    emits, plus hellaswag/arc/mmlu typed sets). Positional bias is measurable here.
  * **name-keyed** — `criteria = {"entailment": null, ...}` / `{"track_order": "..."}`: the key
    *is* the label, so the "position" of a gold label is just its index in a fixed (usually
    alphabetical) order — the same convention the JevBench public-231 eval uses. Mixing both
    namespaces in one bucket makes `verify_bulk`'s `#distinct_keys` explode (arity 4 -> 297
    keys) and inflates `posdev` to ~0.24 without any row being positionally biased.

This script splits the two and reports the real number: the deviation of the gold's **display
index** (position in the `criteria` dict, which is the option order `render_options` emits).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/posdev_x10.py --mix data/xeron10_mix.jsonl \
        --json-out data/xeron10_posdev.json
"""
import argparse
import collections
import json
import re
import sys

LETTER = re.compile(r"^[A-P]$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/xeron10_mix.jsonl")
    ap.add_argument("--json-out", default="data/xeron10_posdev.json")
    args = ap.parse_args()

    letter = collections.defaultdict(collections.Counter)     # arity -> gold display index
    name = collections.defaultdict(collections.Counter)       # arity -> gold display index
    nletter = collections.Counter()
    nname = collections.Counter()
    ordered = collections.Counter()                           # name-keyed key-order check
    orders = collections.defaultdict(set)                     # workflow -> distinct key orders
    nname_rows = 0

    with open(args.mix) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = json.loads(r["questions"])["decision"]
            if q["type"] != "choice":
                continue
            g = json.loads(r["gold"])["decision"]["probabilities"]
            crit = q["criteria"]
            keys = list(crit)
            k = len(keys)
            top = max(g, key=lambda x: g[x])
            pos = keys.index(top) if top in crit else -1
            if all(LETTER.match(x) for x in keys) and keys == [chr(65 + i) for i in range(k)]:
                letter[k][pos] += 1
                nletter[k] += 1
            else:
                name[k][pos] += 1
                nname[k] += 1
                nname_rows += 1
                if keys == sorted(keys):
                    ordered["alphabetical"] += 1
                else:
                    ordered["other"] += 1
                if len(orders[r.get("workflow", "?")]) < 200:
                    orders[r.get("workflow", "?")].add(tuple(keys))

    def dev(counter, k):
        t = sum(counter.values()) or 1
        return max(abs(v / t - 1.0 / k) for v in counter.values())

    out = {"letter_keyed": {}, "name_keyed": {}, "max_letter_dev": 0.0,
           "max_name_dev": 0.0, "name_keyed_rows": nname_rows,
           "name_keyed_key_order": dict(ordered),
           "name_keyed_distinct_orders_per_workflow":
               {k: len(v) for k, v in sorted(orders.items(), key=lambda x: -len(x[1]))[:15]}}
    for k in sorted(set(letter) | set(name)):
        if letter[k]:
            out["letter_keyed"][str(k)] = {"rows": nletter[k], "distinct_gold": len(letter[k]),
                                           "dev": round(dev(letter[k], k), 4)}
            out["max_letter_dev"] = max(out["max_letter_dev"], dev(letter[k], k))
        if name[k]:
            out["name_keyed"][str(k)] = {"rows": nname[k], "distinct_gold": len(name[k]),
                                         "dev": round(dev(name[k], k), 4)}
            out["max_name_dev"] = max(out["max_name_dev"], dev(name[k], k))
    out["max_letter_dev"] = round(out["max_letter_dev"], 4)
    out["max_name_dev"] = round(out["max_name_dev"], 4)

    with open(args.json_out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("arity | letter rows / dev | name rows / dev")
    for k in sorted(set(letter) | set(name), key=int):
        print(f"  {k:>3} | {nletter[k]:>9,} / {out['letter_keyed'].get(str(k), {}).get('dev', '-')}"
              f" | {nname[k]:>9,} / {out['name_keyed'].get(str(k), {}).get('dev', '-')}")
    print(f"max letter-keyed dev = {out['max_letter_dev']} (target <= 0.15)")
    print(f"max name-keyed dev   = {out['max_name_dev']} (display index of the gold label)")
    print(f"name-keyed key order: {dict(ordered)}")
    print(f"distinct criteria orders per workflow (name-keyed): "
          f"{out['name_keyed_distinct_orders_per_workflow']}")
    print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()