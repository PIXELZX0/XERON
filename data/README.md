# XERON Dataset Guide 📊

## Default (benchmark) dataset

- [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — 1,200 train / 400 test cases, 2,000 decisions
- 4 workflows: invoice processing, security incidents, customer service, agent-trace observability

## Custom dataset format (Laya native)

Each row must have the following 3 columns (`datasets.Dataset` or JSON Lines):

```json
{
  "id": "case-0001",
  "workflow": "customer-service",
  "state": "{\"subject\": \"Duplicate charge on invoice 4411\", \"body\": \"We were billed twice for March.\"}",
  "questions": "{\"department\": {\"type\": \"choice\", \"instructions\": \"Which team should handle this?\", \"criteria\": {\"billing\": \"invoices, payments, refunds\", \"technical\": \"bugs and outages\"}}, \"urgency\": {\"type\": \"score\", \"instructions\": \"How urgent is this?\", \"criteria\": [\"not urgent\", \"soon\", \"blocking\"]}, \"churn_risk\": {\"type\": \"noul\", \"instructions\": \"Does the user threaten to cancel?\"}}",
  "gold": "{\"department\": {\"probabilities\": {\"billing\": 0.9, \"technical\": 0.1}}, \"urgency\": {\"probabilities\": {\"0\": 0.1, \"1\": 0.3, \"2\": 0.6}}, \"churn_risk\": {\"probabilities\": {\"false\": 0.7, \"true\": 0.3}}}"
}
```

### Question types

| type | Description | criteria |
|---|---|---|
| `choice` | pick one of multiple options | dict: `{option_key: description}` |
| `score` | ordinal scale (0..n) | list or dict of levels |
| `noul` | calibrated probability P(true) | - |

### gold probability rules

- `choice`: probabilities over each option key sum to 1.0
- `score`: probabilities under string keys `"0"`, `"1"`, ...
- `noul`: `{"false": p, "true": 1-p}`

## Self-directed collection options
1. **Synthetic data generation**: generate state-question-gold distributions with an LLM and feed them straight into `scripts/preprocess.py`
2. **Real decision logs**: convert existing production routing/classification logs into the format above
3. **Human-in-the-loop**: annotators label probability distributions (most effective for calibration training)

> 💡 Once the dataset for XERON fine-tuning is finalized, put it under `data/` and update this README.

## Added 2026-09-24: 4 knowledge/reasoning/routing sources

To shore up JevBench's weak spots (routing/intent, knowledge/reasoning choice). All `choice` type, gold one-hot,
seed 7 fixed (`random.Random(7)`) so they are reproducible. Build scripts are `scripts/build_*_dataset.py`.

| File | Rows | workflow | Source |
|---|---|---|---|
| `data/clinc_typed.jsonl` | 20,000 | `clinc150-intent` | `clinc/clinc_oos` (`plus`, train+val+test) |
| `data/mmlu_typed.jsonl` | 15,000 | `mmlu-choice` | `cais/mmlu` (`all`, test+validation) |
| `data/hellaswag_typed.jsonl` | 10,000 | `hellaswag-continuation` | `Rowan/hellaswag` (validation) |
| `data/arc_typed.jsonl` | 3,000 | `arc-choice` | `allenai/ai2_arc` (Challenge val/test + Easy all) |
| `data/clinc_hard_typed.jsonl` | 20,000 | `clinc150-intent` | **optional** hard-negative variant of the CLINC above |

Build:

```bash
python scripts/build_clinc_dataset.py       --output data/clinc_typed.jsonl --target 20000
python scripts/build_mmlu_dataset.py        --output data/mmlu_typed.jsonl --target 15000
python scripts/build_hellaswag_dataset.py   --output data/hellaswag_typed.jsonl --target 10000
python scripts/build_arc_dataset.py         --output data/arc_typed.jsonl --target 3000
```

Verify (row count / exhaustive one-hot & probability-sum check / answer-leak regex / state duplicates vs existing data):

```bash
python scripts/verify_new_datasets.py data/clinc_typed.jsonl data/mmlu_typed.jsonl \
    data/hellaswag_typed.jsonl data/arc_typed.jsonl
```

Caveats:

- **Duplicate avoidance**: MMLU `dev` (jevbench-mmlu) and ARC-Challenge `train` (jevbench-arc_challenge) were
  deliberately excluded. Zero state overlap with existing `data/*.jsonl`.
- **CLINC150 vocabulary leak**: the share of rows where a token from the gold intent name appears verbatim in the utterance is 63%
  (2.1% for random distractors). This is intrinsic to CLINC150 and cannot be removed. Using `--distractor-mode hard` to draw
  distractors from intents with overlapping name tokens raises the distractor-side vocabulary overlap to 4.6% (partial mitigation).
- Labels (A..F) are randomly permuted per row, so there is no bias in the answer position.
## Added 2026-09-24 (2): Recovery of missing JevBench configs (5 kinds)

Recovered the 5 configs missing from the existing `jevbench_full.jsonl`
(17 configs / 101,953 rows) out of `Praveenrajus/jev-bench` (22 configs / 133,953 train rows). There were two reasons for the omission:

- **banking77 / clinc150 / massive / ledgar** — the original builder skipped them in `SKIP_LARGE_K` because the option count exceeded
  the limit (k=77/151/60/100) → exactly 4×8,000 = **32,000 rows** lost. Reconstructed as small-arity
  choice (gold + random distractors).
- **chaosnli** — only a `test` split exists (no train), so it was missed by the `--split train` default.
  Rerunning the existing builder with `--split test` recovers it as-is (soft labels preserved).

| File | Rows | workflow | Source | Options |
|---|---|---|---|---|
| `data/jevbench_extra_chaosnli.jsonl` | 1,599 | `jevbench-chaosnli` | `metaeval/chaos-mnli-ambiguity` (test) | 3 |
| `data/jevbench_extra_banking77.jsonl` | 5,092 | `jevbench-banking77` | `mteb/banking77` | 4–6 |
| `data/jevbench_extra_clinc150.jsonl` | 4,954 | `jevbench-clinc150` | `clinc/clinc_oos` (plus) | 4–6 |
| `data/jevbench_extra_massive.jsonl` | 6,022 | `jevbench-massive` | `mteb/amazon_massive_intent` (en) | 4–6 |
| `data/jevbench_extra_ledgar.jsonl` | 4,439 | `jevbench-ledgar` | `coastalcph/lex_glue` (ledgar) | 8–12 |

Build (seed 7 fixed, reproducible — identical sha256 on re-run):

```bash
python scripts/build_jevbench_dataset.py --configs chaosnli --split test \
    --output data/jevbench_extra_chaosnli.jsonl
python scripts/build_jevbench_extra.py --outdir data            # banking77/clinc150/massive/ledgar
python scripts/verify_jevbench_extra.py "data/jevbench_extra_*.jsonl"
```

**Vocabulary shortcut control (key)**: for intent/topic configs the label name appears in the option text, and
that word frequently appears verbatim in the utterance. Gold-vs-distractor name overlap ratio before control:

| config | gold | random distractor | ratio |
|---|---|---|---|
| banking77 | 0.80 | 0.16 | 5.1× |
| clinc150 | 0.65 | 0.035 | 18.6× |
| massive | 0.36 | 0.014 | 26.0× |
| ledgar | 0.50 | 0.041 | 12.1× |

→ Distractor candidates are drawn only from the **same overlap bucket as gold** (`min(number of label tokens appearing in state, 2)`).
Since every option is in the same bucket, the shortcut "pick the overlapping option" cannot work.
After control the ratios are all **1.01–1.16×** (both binary and count). Rows with fewer than 3 same-bucket distractors (7 for ledgar)
were discarded.

Verification (`scripts/verify_jevbench_extra.py`): all rows `choice`, probabilities sum to 1.0, 2–12 labels,
answer-leak regex 0, forbidden state keys 0, within-file/cross-file duplicates 0, exact duplicates vs existing `data/*.jsonl`
(including jevbench_full and xeron5_mix) 0. Laya head budget check: all 22,106 rows
pass `markers == k` in preprocess, max marker position 116 ≪ `head_max_len=256`.

> Note: utterances may overlap with the existing `clinc_typed.jsonl`/`clinc_hard_typed.jsonl`, but the state
> format differs, so exact duplicates are 0. This file has stronger shortcut control than the
> hard-negative variant (distractor overlap 4.6%, 13×).
