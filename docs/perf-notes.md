# Training throughput notes

## 1. Padding waste from uniform shuffling (fixed)

`collate_train_batch` pads every micro-batch to the **longest sequence inside that batch**.
Both trainers used to shuffle items uniformly at random, so short items constantly shared a
batch with very long ones and the whole batch was computed at the long length.

Real length distribution (`train_items_x10_8192.pt`, 3,389,832 items):
mean 176 tokens, p50 128, p90 268, p99 785, max 8192 — i.e. the data is overwhelmingly short
while a handful of items are extremely long.

`MAX_TOKENS_BATCH` was documented as a memory ceiling but was only written into
`rl_agent_config.json`; nothing enforced it, so random batches regularly blew past it
(up to 4x in the measurements below), which is also a plausible source of intermittent OOMs.

### Measurements (simulated from the real length histograms — no GPU needed)

`scripts/batch_sampler.py --hist-json <histogram> --micro-batch N --max-tokens-batch T`

| dataset | micro batch | token budget | padded tokens (random) | padded tokens (length-bucketed) | waste random → length | reduction |
|---|---|---|---|---|---|---|
| 8192 (3.39 M items) | 8 | 16384 | 1,571,779,275 | 597,911,583 | 62.03 % → **0.19 %** | **2.63x** |
| 8192 (3.39 M items) | 2 | 16384 | 807,646,338 | 596,962,447 | 26.11 % → **0.03 %** | **1.35x** |
| 4096 (3.35 M items) | 8 | 8192 | 1,347,417,048 | 473,523,749 | 64.90 % → **0.12 %** | **2.85x** |

Batches exceeding the token budget: random 6,516 / 0 / 17,316 → length-bucketed **0**.
Worst observed padded batch under random batching: 32,768 tokens against an 8,192 budget.

Padded tokens are a good proxy for step time at these lengths (sequence lengths are short
relative to the compute per token, and ModernBERT uses sliding-window attention for two out
of three layers), but they are not the whole story — see the measured A/B below.

### Measured A/B on an A100 80GB (real training step, 20k-item sample of the 8192 data)

`scripts/a100/bench_batch_mode_ab.sh` — detached remote run, randomly initialised 1.329 B model,
`MAX_LEN=8192`, `GRAD_CKPT=1`, `DTYPE=bf16`, 12 timed steps after 3 warm-up steps, sample mean 173
tokens (p90 266, 98 items >1024).

| config | mode | step ms | seq/s | real tok/s | padded tok/seq | peak alloc |
|---|---|---|---|---|---|---|
| mb8, budget 16384 | random | 342.1 | 23.38 | 3,826 | 381.1 | 25.0 GB |
| mb8, budget 16384 | **length** | 286.7 | **27.91** | **4,954** | 187.4 | 25.1 GB |
| | | | **+19.4 %** | **+29.5 %** | **2.03x** | flat |
| mb2, budget 16384 | random | 241.1 | 8.30 | 1,094 | 160.9 | 25.1 GB |
| mb2, budget 16384 | length | 257.4 | 7.77 | 1,044 | 134.9 | 25.1 GB |
| | | | −6.4 % | −4.5 % | 1.19x | flat |

What this says:

- The change delivers a real win when micro-batches are large enough for kernel efficiency to
  dominate: **+19 % optimizer steps/s and +30 % real tokens/s at mb8**, with per-sequence padding
  cut 2.03x (matching the 2.63x simulated on the full distribution).
- At **mb2 it is a wash** (−6 % within a run-to-run spread of roughly ±40 % on step time), because
  fixed per-step overhead dominates and mb2 already pads relatively little.
- **Memory does not grow**: peak allocation stayed at 25.0-25.1 GB in both modes, since the token
  budget caps padded tokens per batch at 16,384 either way. That is the important operational
  point — a bigger `MICRO_BATCH` no longer costs extra activation memory, because the budget,
  not the item count, is the constraint.

Practical takeaway: keep `MAX_TOKENS_BATCH` set to the activation budget you can afford and raise
`MICRO_BATCH` to the largest value the budget still fills (mb8 with 16,384 here). That is the
configuration where the planner pays off, and it keeps the OOM ceiling unchanged.

## 2. What changed

- **`scripts/batch_sampler.py`** (new): length-bucketed, token-budget aware batch planning.
  - items sorted by length → buckets of `MICRO_BATCH * BATCH_BUCKET_MULT` items
  - order reshuffled **inside** each bucket, and bucket/batch order reshuffled per epoch,
    so the length↔step correlation is not fixed across epochs
  - a micro-batch closes early when adding the next item would exceed `MAX_TOKENS_BATCH`
    (padded-token accounting), and never exceeds `MICRO_BATCH` items
  - `plan_random` keeps the legacy behaviour for comparison
  - `--items/--hist-json/--lengths-json` CLI prints the padding-waste comparison
- **`scripts/train_ddp.py`**: epoch loop now consumes the plan instead of
  `random.shuffle` + fixed slicing. New env: `BATCH_MODE` (`length` default, `random` legacy),
  `BATCH_BUCKET_MULT` (default 64), `BATCH_SUMMARY` (default 1). The plan is logged once per run
  and recorded in `rl_agent_config.json` (`batch_mode`, `batch_bucket_mult`, `max_tokens_batch`).
- **`scripts/train_spmd_tpu.py`**: same planning over the shard, rebuilt every epoch; the
  DataLoader now serves pre-planned batches instead of fixed-size slices.
- **`scripts/test_batch_sampler.py`** (new): 9 self-tests, runnable without pytest:
  `.venv/bin/python scripts/test_batch_sampler.py`
  (covers coverage-once, budget compliance, unavoidable single-item overage reported
  separately, waste reduction vs random, per-epoch reshuffling, batch-size cap, legacy parity,
  histogram expansion, and an end-to-end plan→`collate_train_batch` shape check).

### Why reordering batches is safe for RLCD

The objective has no cross-item coupling: rewards are computed per item and normalised across
the `GROUP_SIZE` samples *within* that item (`adv = r - r.mean(0, keepdim=True)`), and the batch
term is a plain mean over independently sampled items. Changing which items share an optimizer
step changes the composition of that average, not the objective.

## 3. Reproduce

```bash
# unit + integration tests (CPU only)
.venv/bin/python scripts/test_batch_sampler.py

# padding-waste comparison on a real items file
.venv/bin/python scripts/batch_sampler.py --items train_items_x10_8192.pt \
    --micro-batch 8 --max-tokens-batch 16384 --out /tmp/batch_cmp.json

# same comparison from a histogram (no 2.5 GB load)
.venv/bin/python scripts/batch_sampler.py --hist-json data/xeron10_len_stats.json \
    --micro-batch 8 --max-tokens-batch 8192

# legacy behaviour for an A/B run
BATCH_MODE=random EPOCHS=1 MICRO_BATCH=8 MAX_TOKENS_BATCH=16384 torchrun ... scripts/train_ddp.py ...
```

## 4. Still open (not changed here)

- ~~`DDP(..., find_unused_parameters=True)`~~ — **done**: defaults to `False` now
  (`DDP_FIND_UNUSED=1` restores the old behaviour, `DDP_STATIC_GRAPH=1` opts into static graph).
  Removes one autograd-graph traversal per step. Needs a GPU A/B run to quantify.
- TPU: DCP checkpoint saving is synchronous (~75 s per save for the 1.33 B model); `CheckpointManager.save_async`
  or step-based cadence would hide it.
- TPU gradient clipping currently approximates the global norm as `local_norm * sqrt(ndev)`.
- Gradient checkpointing under `torch_xla` only works with `torch_xla.utils.checkpoint.checkpoint`
  and `use_reentrant=True`; enabling it correctly would allow larger batches at long context.
- Wall-clock A/B on GPU/TPU to confirm the projected speed-up. **Done for the A100/mb8 path**
  (+19 % steps/s, +30 % real tokens/s, memory flat); the long-context subset (mean 2,743 tokens)
  and the TPU path are still unmeasured.
