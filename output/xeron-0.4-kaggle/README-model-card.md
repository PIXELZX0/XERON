---
license: apache-2.0
language:
- en
- ko
- multilingual
base_model:
- PIXELZX/XERON-0.2
- jhu-clsp/mmBERT-base
library_name: laya
tags:
- decision-model
- system-1
- non-autoregressive
- typed-decisions
- calibration
- multilingual
- jevbench
pipeline_tag: text-classification
---

# XERON-0.4 🎯

**XERON-0.4** is the fourth release of the XERON family: a **typed-decision (System 1) model** forked from
[XERON-0.2](https://huggingface.co/PIXELZX/XERON-0.2) (itself a fine-tune of
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)'s multilingual checkpoint,
backbone `jhu-clsp/mmBERT-base`, 322M params).

It answers **typed questions over a state** — `choice` / `score` / `noul` (boolean) — in a **single forward pass**,
returning calibrated probabilities. It never generates text, so it cannot hallucinate and cannot emit malformed schemas.

**What changed vs 0.2:** 0.4 is a **short, stabilized refinement pass** on the 0.2 checkpoint, not a bigger-data run.
The family's 0.3 attempt (2 epochs on a 62k mixed corpus) came out **overconfident** — its fitted calibration
temperatures blew up to `[3.14, 5.41, 5.12]` and its soft-probability metrics collapsed. Diagnosis: the RLCD
policy-gradient term scales as `1/(2σ²)`, so annealing σ to 0.1 amplified the gradient ~50× and pushed the logits
scale upward; epoch-2 loss also rose (1.075 → 1.289 = overfitting).

0.4 therefore trains **1 epoch** with a tamed objective: `rl_weight 0.5`, `σ 0.3→0.2`, `lr_head 5e-5` (was 1e-4),
`weight_decay 0.02`. That recovers accuracy to a new family best while pulling calibration back most of the way.

Training: 1 epoch · 61,876 sequences · ~2.8 h · 2×T4 (fp16) · post-hoc temperature calibration `[2.254, 1.810, 3.188]`.

## 📈 Results

### JevBench v1.3 (public items only, matched subset)

JevBench's frozen set has 534 decisions; only **231 are public** (the judge tier is entirely held out), so these are
**not** directly comparable to the published board ranks. Every system below ran the *same 231 items* with the
official harness ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), `laya_local` adapter);
Jev/Laya rows come from the benchmark's own per-task artifact.

| System | easy (48) | standard (72) | hard (111) | Overall (231) | Intelligence |
|---|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe, API) | 1.000 | 0.986 | 0.730 | **0.866** | 82.2 |
| **XERON-0.4** (ours) | 0.958 | 0.611 | **0.351** | **0.558** | 36.0 |
| XERON-0.2 (ours) | 0.979 | 0.583 | 0.324 | 0.541 | 34.1 |
| laya-typed-decisions (Convai, 421M, tuned) | 0.979 | 0.653 | 0.270 | 0.537 | 38.0 |
| XERON-0.3 (ours, overconfident) | 0.938 | 0.569 | 0.324 | 0.528 | — |
| XERON-0.1 (ours) | 0.875 | 0.444 | 0.306 | 0.468 | 23.3 |
| laya-multilingual (base, untuned) | 0.896 | 0.403 | 0.324 | 0.468 | 21.5 |

### XERON-0.2 → XERON-0.4

| Metric | 0.2 | 0.3 | **0.4** |
|---|---|---|---|
| JevBench overall (231) | 0.541 | 0.528 | **0.558** ✅ |
| standard tier | 0.583 | 0.569 | **0.611** ✅ |
| hard tier | 0.324 | 0.324 | **0.351** ✅ |
| hard-tier ECE | 0.187 | 0.129 | **0.106** ✅ |
| typed-decisions acc | 0.7133 | 0.7117 | **0.7217** ✅ |
| typed-decisions soft acc | **0.5353** | 0.3504 | 0.4084 |
| typed-decisions Brier | **0.4242** | 0.5835 | 0.5152 |
| fitted temperature | **[0.96, 1.10, 0.57]** | [3.14, 5.41, 5.12] | [2.25, 1.81, 3.19] |

- **XERON-0.4 sets a new family best on overall JevBench accuracy** (0.558 vs 0.541) and improves the standard
  tier, hard tier, hard-tier ECE and typed-decisions accuracy.
- **Trade-off:** calibration did not fully return to 0.2 levels. If you need the softest, best-calibrated
  probability distributions, prefer **XERON-0.2**; if you want the highest decision accuracy, use **XERON-0.4**.

### typed-decisions benchmark

`LocalLLaMA/typed-decisions` test split, 400 cases / 1,400 decisions:

| Model | choice acc | soft acc | Brier | ECE |
|---|---|---|---|---|
| **XERON-0.4** | **0.7217** | 0.4084 | 0.5152 | 0.2525 |
| XERON-0.2 | 0.7133 | **0.5353** | **0.4242** | **0.2104** |
| XERON-0.1 | 0.7000 | 0.5171 | 0.4493 | 0.2143 |
| laya-typed-decisions | 0.7333 | 0.4460 | 0.4669 | 0.2380 |

## 🚀 Usage

```bash
pip install laya
```

```python
import laya

agent = laya.load("PIXELZX/XERON-0.4")

state = "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt."
questions = {
    "permitted": {"type": "noul", "instructions": "Under the stated policy, is the requested action permitted?"},
    "urgency":   {"type": "score",  "levels": ["0 — no pressure", "1 — routine", "2 — elevated", "3 — critical"]},
}

print(agent.predict(state, questions)["answers"])
```

## 📈 Reproduction

```bash
git clone https://github.com/PIXELZX0/XERON && cd XERON
# JevBench public 231 items (same harness and adapter)
git clone --depth 1 https://github.com/fstandhartinger/jevbench /tmp/jevbench
python results/jevbench-public/run_public_jevbench.py PIXELZX/XERON-0.4 XERON-0.4 /tmp/jb04
python results/jevbench-public/score_public_jevbench.py /tmp/jb04 XERON-0.4
# typed-decisions
python scripts/evaluate.py --model PIXELZX/XERON-0.4 --split test --device cuda --output eval.json
```

## ⚠️ Limitations

- **Calibration is not as good as 0.2** (temperature 2.25/1.81/3.19 vs ~1.0 for 0.2). Both 0.2 and 0.4 were run on
  non-A100 hardware (fp16), which may be a factor; worth verifying with a bf16 retrain.
- Trained at 4,096 tokens — ultra-long contexts need a `CTX_CAP` extension and a retrain.
- Ceiling on the number of options: Laya-family heads use `head_max_len=256`, so tasks with many options (e.g. 77/151 intents)
  have their option text truncated and performance collapses. Such configs were excluded from the training data as well.
- The gap between the JevBench hard tier (0.351) and Jev (0.730) is still large — the hard tier is dominated by long policy documents, ambiguous
  trade-offs and trap items, and our training data has almost no such authored rubric data.
- The training data is mostly single-label corpora, which hurts probability fidelity (TVD). Increasing the share of soft labels leaves room for improvement.

Built on [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations (Apache-2.0) and `jhu-clsp/mmBERT-base`.
Data: [Jevify jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) (mixed licenses, see its manifest).
