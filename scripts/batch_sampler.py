#!/usr/bin/env python3
"""Length-bucketed, token-budget aware batch planning for the XERON trainers.

Why this exists
---------------
`collate_train_batch` pads every micro-batch to the **longest sequence inside it**.
Because the trainers shuffle items uniformly at random, a 128-token item frequently
shares a batch with an 8192-token item, so the whole batch is computed at 8192 tokens.
On the real `train_items_x10_8192.pt` distribution (3,389,832 items, mean 175 tokens,
p50 128, p90 268, p99 785, max 8192) uniform padding wastes **97.86 %** of the compute.

`MAX_TOKENS_BATCH` was previously only written into `rl_agent_config.json`; nothing
enforced it. This module makes it a real constraint.

Why it is safe for RLCD
-----------------------
The loss has no cross-item coupling: rewards are computed per item and normalised over
the GROUP_SIZE samples *within* that item (`adv = r - r.mean(0, keepdim=True)`), and the
batch reduction is a plain mean over independently-sampled items. Reordering which items
share an optimizer step therefore does not change the objective, only the composition of
each averaged gradient. Bucket shuffling (order of buckets + order inside each bucket is
reshuffled every epoch) keeps the length/step correlation from being fixed across epochs.

Usage (library)
---------------
    from batch_sampler import plan_batches
    batches, stats = plan_batches(lengths, micro_batch=8, max_tokens_batch=16384)

Usage (measure the win on a real items file / histogram)
--------------------------------------------------------
    python scripts/batch_sampler.py --items train_items_x10_8192.pt \
        --micro-batch 8 --max-tokens-batch 16384
    python scripts/batch_sampler.py --hist-json data/xeron10_len_stats.json \
        --micro-batch 8 --max-tokens-batch 16384
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_BUCKET_MULT = 64  # bucket width = micro_batch * bucket_mult items


def length_order(lengths: Sequence[int]) -> List[int]:
    """Ascending length order, stable on the original index."""
    return sorted(range(len(lengths)), key=lambda i: (lengths[i], i))


def _batch_max(lengths: Sequence[int], batch: Sequence[int]) -> int:
    return max(lengths[i] for i in batch) if batch else 0


def _summarise(lengths: Sequence[int], batches: List[List[int]],
               max_tokens_batch: Optional[int]) -> Dict:
    real = sum(lengths)
    padded = 0
    worst = 0
    over = 0
    single_over = 0
    for b in batches:
        m = _batch_max(lengths, b)
        pad = m * len(b)
        padded += pad
        worst = max(worst, pad)
        if max_tokens_batch and pad > max_tokens_batch:
            # a single item longer than the budget cannot be split, so it is unavoidable
            if len(b) == 1:
                single_over += 1
            else:
                over += 1
    return {
        "items": len(lengths),
        "batches": len(batches),
        "real_tokens": real,
        "padded_tokens": padded,
        "waste_pct": round(100 * (1 - real / padded), 2) if padded else 0.0,
        "mean_batch_size": round(len(lengths) / len(batches), 2) if batches else 0.0,
        "max_padded_per_batch": worst,
        "batches_over_token_budget": over,
        "single_item_over_budget": single_over,
    }


def plan_random(lengths: Sequence[int], micro_batch: int,
                max_tokens_batch: Optional[int] = None,
                rng: Optional[random.Random] = None) -> Tuple[List[List[int]], Dict]:
    """Legacy behaviour: uniform shuffle, fixed-size micro-batches."""
    rng = rng or random.Random(0)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]
    return batches, _summarise(lengths, batches, max_tokens_batch)


def plan_batches(lengths: Sequence[int], micro_batch: int,
                 max_tokens_batch: Optional[int] = None,
                 bucket_mult: int = DEFAULT_BUCKET_MULT,
                 rng: Optional[random.Random] = None) -> Tuple[List[List[int]], Dict]:
    """Length-bucketed batches under an optional padded-token budget.

    - items are ordered by length so each bucket spans a narrow length range
    - order inside each bucket is reshuffled (call once per epoch with a fresh rng)
    - a micro-batch is closed early when adding the next item would push
      ``max_len_in_batch * batch_size`` above ``max_tokens_batch``
    - batch order is shuffled so consecutive optimizer steps mix length ranges
    """
    rng = rng or random.Random(0)
    n = len(lengths)
    if n == 0:
        return [], _summarise(lengths, [], max_tokens_batch)

    order = length_order(lengths)
    width = max(int(micro_batch), int(micro_batch) * max(1, int(bucket_mult)))
    buckets = [order[i:i + width] for i in range(0, n, width)]
    for b in buckets:
        rng.shuffle(b)

    batches: List[List[int]] = []
    for b in buckets:
        cur: List[int] = []
        cur_max = 0
        for i in b:
            L = lengths[i]
            nxt_max = L if L > cur_max else cur_max
            over_size = len(cur) >= micro_batch
            over_budget = (max_tokens_batch is not None
                           and cur and nxt_max * (len(cur) + 1) > max_tokens_batch)
            if cur and (over_size or over_budget):
                batches.append(cur)
                cur, cur_max = [], 0
                nxt_max = L
            cur.append(i)
            cur_max = nxt_max
        if cur:
            batches.append(cur)

    rng.shuffle(batches)
    return batches, _summarise(lengths, batches, max_tokens_batch)


def plan_epoch(lengths: Sequence[int], micro_batch: int,
               max_tokens_batch: Optional[int] = None,
               mode: str = "length", bucket_mult: int = DEFAULT_BUCKET_MULT,
               seed: int = 0) -> Tuple[List[List[int]], Dict]:
    """One epoch of batches. `mode` is `length` (default) or `random` (legacy)."""
    rng = random.Random(seed)
    if mode == "random":
        return plan_random(lengths, micro_batch, max_tokens_batch, rng)
    return plan_batches(lengths, micro_batch, max_tokens_batch, bucket_mult, rng)


# ------------------------------------------------------------------ measurement
def lengths_from_items(path: str) -> List[int]:
    import torch  # imported lazily so the module stays lightweight
    items = torch.load(path, weights_only=False)
    out = [len(it["ids"]) for it in items]
    del items
    return out


def lengths_from_hist(path: str) -> List[int]:
    """Expand a length histogram (buckets with counts) into representative lengths.

    Within a bucket the lengths are spread uniformly, which reproduces the shape of the
    distribution closely enough to compare batch planning strategies.
    """
    h = json.load(open(path))
    out: List[int] = []
    if "buckets" not in h and isinstance(h.get("hist"), dict):
        # alternative format: {"hist": {"<upper_bin>": count}, "bin_width": N}
        prev = 0
        for upper_s, count in sorted(h["hist"].items(), key=lambda kv: int(kv[0])):
            upper, count = int(upper_s), int(count)
            if count <= 0:
                prev = upper
                continue
            lo = prev + 1
            span = max(1, upper - lo + 1)
            for k in range(count):
                out.append(min(upper, lo + (k * span) // count))
            prev = upper
        return out
    prev = 0
    for b in h.get("buckets", []):
        upper, count = int(b["upper"]), int(b["count"])
        if count <= 0:
            prev = upper
            continue
        lo = prev + 1
        span = max(1, upper - lo + 1)
        for k in range(count):
            out.append(min(upper, lo + (k * span) // count))
        prev = upper
    return out


def compare(lengths: Sequence[int], micro_batch: int,
            max_tokens_batch: Optional[int], bucket_mult: int = DEFAULT_BUCKET_MULT,
            seeds: int = 3) -> Dict:
    """Compare legacy random batching with length-bucketed batching."""
    res: Dict[str, Dict] = {}
    for mode in ("random", "length"):
        acc = None
        for s in range(seeds):
            _, st = plan_epoch(lengths, micro_batch, max_tokens_batch, mode, bucket_mult, 1234 + s)
            if acc is None:
                acc = {k: v / seeds for k, v in st.items() if isinstance(v, (int, float))}
            else:
                for k, v in st.items():
                    if isinstance(v, (int, float)):
                        acc[k] += v / seeds
        res[mode] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in acc.items()}
    r, l = res["random"], res["length"]
    if l["padded_tokens"]:
        res["speedup_padded_tokens"] = round(r["padded_tokens"] / l["padded_tokens"], 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="XERON batch planner / padding-waste report")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--items", help="train_items*.pt file")
    src.add_argument("--hist-json", help="length histogram JSON (buckets+counts)")
    src.add_argument("--lengths-json", help="JSON list of sequence lengths")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-tokens-batch", type=int, default=None)
    ap.add_argument("--bucket-mult", type=int, default=DEFAULT_BUCKET_MULT)
    ap.add_argument("--samples", type=int, default=0,
                    help="subsample this many lengths (hist expansion fills the file)")
    ap.add_argument("--out", default=None, help="write the comparison JSON here")
    args = ap.parse_args()

    if args.items:
        lengths = lengths_from_items(args.items)
    elif args.hist_json:
        lengths = lengths_from_hist(args.hist_json)
    else:
        lengths = [int(x) for x in json.load(open(args.lengths_json))]
    if args.samples and len(lengths) > args.samples:
        random.Random(0).shuffle(lengths)
        lengths = lengths[:args.samples]

    print(f"items={len(lengths):,} mean_len={sum(lengths)/max(1,len(lengths)):.1f} "
          f"max_len={max(lengths) if lengths else 0} "
          f"real_tokens={sum(lengths):,}")
    print(f"micro_batch={args.micro_batch} max_tokens_batch={args.max_tokens_batch} "
          f"bucket_mult={args.bucket_mult}")
    cmp = compare(lengths, args.micro_batch, args.max_tokens_batch, args.bucket_mult)
    for mode in ("random", "length"):
        s = cmp[mode]
        print(f"  {mode:>6}: batches={s['batches']:,.0f} padded={s['padded_tokens']:,.0f} "
              f"waste={s['waste_pct']:.2f}% mean_bs={s['mean_batch_size']} "
              f"max_batch_padded={s['max_padded_per_batch']:,.0f} "
              f"over_budget={s['batches_over_token_budget']:.0f} "
              f"single_item_over={s.get('single_item_over_budget', 0):.0f}")
    if "speedup_padded_tokens" in cmp:
        print(f"  => padded-token reduction: {cmp['speedup_padded_tokens']}x "
              f"(less wasted compute per epoch)")
    if args.out:
        json.dump(cmp, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
