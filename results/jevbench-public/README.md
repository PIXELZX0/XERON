# XERON — JevBench v1.3-style evaluation (public item subset)

> 🆕 **For XERON-0.2 results see [section 7](#7-xeron-02--jevbench-style-data-fine-tuning-results)** — JevBench overall 0.468 → **0.541** (+7.3%p)

XERON-0.1 was measured **using JevBench v1.3.0 (Benchmark Heaven)'s exact scoring method, tasks and adapter**.
Harness: https://github.com/fstandhartinger/jevbench (MIT), as of 2026-09-22, using the `laya_local` adapter unchanged.

## ⚠️ Scope (must read first)

Of JevBench's 534 decisions, **only 231 are public** (the rest: the entire judge tier of 146 is private, plus standard 24, easy 24, hard 109 private).
This document therefore **cannot be directly compared to the official board ranks** — the same position as the board's "partial runs are shown without a rank" rule.

- Public: easy 48/72 · standard 72/96 · **judge 0/146** · hard 111/220
- Matched comparison: XERON, Laya and Jev were compared on the **same 231 items** (Jev/Laya use the public-item results from the official per-task artifacts)
- The three local systems (XERON, laya-typed-decisions, laya-multilingual) were run directly on the **same machine, same adapter, serial single request, threads=4, timed after model load**

## 1. Accuracy (same 231 public items, matched comparison)

| System | easy (48) | standard (72) | judge | hard (111) | Overall (231) | chance-corrected Intelligence |
|---|---|---|---|---|---|---|
| **Jev 1.13.0** (TypeSafe, API) | 1.000 | 0.986 | — | 0.730 | **0.866** | **82.2** |
| **laya-typed-decisions** (Convai, 421M, vendor-tuned) | 0.979 | 0.653 | — | 0.270 | **0.537** | 38.0 |
| **XERON-0.1** (PIXELZX, 322M, ours) | 0.875 | 0.444 | — | **0.306** | 0.468 | 23.3 |
| laya-multilingual (**untuned base**, 322M) | 0.896 | 0.403 | — | 0.324 | 0.468 | 21.5 |
| *Reference: Laya official board row (all 534)* | 0.944 | 0.729 | 0.692 | 0.341 | — | 45.8 |

**How to read this**

- **Jev dominates** (0.866 on the matched public set). This is expected, since JevBench is designed as a benchmark for zero-shot strong models.
- **XERON-0.1 vs its own backbone (untuned laya-multilingual)**: overall 0.468 = 0.468 (tie), but **Intelligence 21.5 → 23.3** and calibration is much better. In other words, the fine-tuning gain comes from **probability quality** rather than accuracy.
- **XERON-0.1 vs laya-typed-decisions, which the vendor tuned for this very benchmark**: −6.9%p down on overall. **However, on the hard tier XERON is ahead (0.306 vs 0.270).**
- In the easy tier, `fact`/`tool_selection` are near-perfect for all three; **the gap opens up on the standard tier (author-written rubrics: policy/adequacy/ordinal/routing).**

## 2. Calibration (hard tier, self-measured)

| System | ECE ↓ | probability fidelity (100×(1−TVD), 10 gold-distribution questions) | Calibration axis |
|---|---|---|---|
| laya-typed-decisions | **0.077** | **0.704** | **77.5** |
| XERON-0.1 | 0.211 | 0.611 | 59.5 |
| laya-multilingual (base) | 0.289 | 0.427 | 42.4 |
| Laya official board | 0.206 | 0.660 | 62.5 |
| Jev official board | 0.061 | 0.774 | 82.7 |

XERON is far better than the untuned base (ECE 0.289→0.211) but does not reach the vendor's English tuning.

## 3. Speed & cost (same machine, serial, measured after model load)

### 3a. Local CPU (yuchan-server, 12 vCPU, 4 threads)

| System | p50 (raw) | p95 (raw) | adjusted p50/p95 (×2+0.15s) | input tokens per decision | USD per 1,000 decisions (est.) | Speed axis | Cost axis |
|---|---|---|---|---|---|---|---|
| **XERON-0.1** | **0.129 s** | 5.683 s | 0.408 / 11.52 | 630 | $0.00630 | **73.3** | 76.0 |
| laya-typed-decisions | 0.394 s | 3.608 s | 0.937 / 7.37 | 338 | $0.00338 | 71.6 | 84.1 |
| laya-multilingual (base) | 0.129 s | 5.704 s | 0.408 / 11.56 | 630 | $0.00630 | 73.3 | 76.0 |
| Jev 1.13.0 (API) | 0.652 s | 0.722 s | 0.652 / 0.722 (no adjustment) | 950 | $0.0399 | 83.3 | 52.0 |

- Machine: yuchan-server (12 vCPU), CPU 4 threads, identical conditions.
- **The cost axis is sensitive to the tokenizer artifact** — JevBench estimates "system token count × $0.01/M". mmBERT (256k vocab) counts ~1.9× more tokens per decision than ModernBERT (50k vocab), so XERON's Cost axis reads lower than the actual GPU-cost difference. Since they are in the same size class (322M vs 421M), the reasonable view is that the **real cost is effectively identical**.

## 4. Colab T4 re-run (GPU) — identical accuracy, only faster

The exact same harness and adapter (`laya_local`) were re-run on a **Google Colab Tesla T4**.
The adapter was not modified — `laya.load()` auto-detects CUDA (`device=cuda` confirmed).
Execution path: `colab` CLI (google-colab-cli 0.6.0) → `colab new -s xeron-jb --gpu T4` → `colab install laya` → run script → `colab download` → `colab stop`.

| System | p50 (T4) | p95 (T4) | p50 (local CPU) | easy | standard | hard | Overall | ECE hard |
|---|---|---|---|---|---|---|---|---|
| XERON-0.1 | 0.0297 s | 0.136 s | 0.129 s | 0.875 | 0.444 | 0.306 | 0.468 | 0.211 |
| laya-typed-decisions | 0.0385 s | 0.085 s | 0.394 s | 0.979 | 0.653 | 0.270 | 0.537 | 0.072 |
| laya-multilingual (base) | 0.0279 s | 0.049 s | 0.129 s | 0.896 | 0.403 | 0.333 | 0.472 | 0.296 |

- **Full run of all 231 items: 14.1 s** (276 s on local CPU) → **~20× faster on GPU**. Even including the 34.5 s model load it is under a minute.
- Accuracy is effectively identical to the CPU run (XERON's easy/standard/hard match exactly). laya-multilingual's hard figure moving slightly from 0.324 to 0.333 is a marginal tie-breaking change caused by GPU/CPU floating-point differences.
- Raw: `*.colab-t4.results.jsonl`, aggregate: `summary-colab-t4.json`

## 5. JevBench Score (partial run — no rank)

Since the public items contain no judge tier, the missing tier's weight was renormalized as per the official rule. **Do not compare directly to official board scores.**

| System | Intelligence | Calibration | Speed | Cost | JevBench Score (partial) |
|---|---|---|---|---|---|
| laya-typed-decisions | 38.0 | 77.5 | 71.6 | 84.1 | **37.5** |
| XERON-0.1 | 23.3 | 59.5 | 73.3 | 76.0 | 11.5 |
| laya-multilingual (base) | 21.5 | 42.4 | 73.3 | 76.0 | 8.8 |

> When Intelligence < 50, the official rule multiplies in a `(Intelligence/50)²` penalty. For XERON that is 23.3 → ×0.217. This is the main reason the final score is dragged down so much.

## 6. Where it wins and where it loses (accuracy by family, 231 public items)

| family (tier) | XERON-0.1 | laya-td | laya-ml |
|---|---|---|---|
| tool_selection (easy) | **1.00** | 1.00 | 1.00 |
| fact (easy) | 0.917 | 0.917 | 0.833 |
| intent (easy) | 0.75 | 0.75 | 0.708 |
| adversarial (hard) | **0.50** | 0.333 | 0.667 |
| trap (hard) | **0.125** | 0.000 | 0.125 |
| judge_hard (hard) | **0.588** | 0.353 | 0.588 |
| temporal_numeric (hard) | **0.267** | 0.200 | 0.333 |
| routing_hard (hard) | 0.40 | 0.40 | 0.40 |
| probability (hard) | 0.40 | 0.40 | 0.30 |
| tradeoff (hard) | 0.50 | 0.50 | 0.167 |
| multi_hop (hard) | 0.167 | 0.167 | 0.056 |
| long_policy (hard) | 0.211 | 0.316 | 0.316 |
| extraction (std) | 0.583 | 0.792 | 0.667 |
| ordinal (std) | 0.50 | 0.75 | 0.25 |
| **policy (std)** | 0.50 | **0.917** | 0.25 |
| **adequacy (std)** | 0.25 | **0.583** | 0.167 |
| **routing (std)** | 0.333 | **0.583** | 0.75 |

**Interpretation**: XERON **beats the vendor model on the adversarial/trap/judge_hard families of the hard tier** and **loses on the author-written rubrics of the standard tier (policy, adequacy, routing, ordinal).**
The former overlaps with our training data (long context, web agent, multilingual judgment); the latter is JevBench's own "policy document + options" style — **a distribution entirely absent from our training data**.

Schema validity: all three are **231/231 strict valid, 0 renormalizations** — the outputs themselves are perfectly clean.

## 7. XERON-0.2 — JevBench-style data fine-tuning results

The second release, further trained from XERON-0.1 on **JevBench-style decision data**.

### Training
- **Data**: [Jevify `jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench) 22 public datasets → converted to System One format,
  with an option-count ceiling (32) guard applied, then 48,000 sampled rows + `LocalLLaMA/typed-decisions` EN 6,000 = **54,000 sequences**
- **Base**: `PIXELZX/XERON-0.1` (continued fine-tuning)
- **Config**: 2 epochs · MICRO_BATCH 8 / GRAD_ACCUM 8 (effective 64) · bf16 · MAX_LEN 4096 · 1×A100 40GB · ~41 min
- **Post-hoc calibration**: temperature `[0.96, 1.098, 0.569]`
- ⚠️ **The JevBench public 231 items are excluded from training** — kept as a clean held-out set

### Results (same 231 public items, official harness)

| System | easy (48) | standard (72) | hard (111) | Overall (231) | Intelligence | Calibration |
|---|---|---|---|---|---|---|
| Jev 1.13.0 | 1.000 | 0.986 | 0.730 | **0.866** | 82.2 | — |
| **XERON-0.2** | 0.979 | 0.583 | **0.324** | **0.541** | 34.1 | 58.0 |
| laya-typed-decisions | 0.979 | 0.653 | 0.270 | 0.537 | 38.0 | 77.5 |
| XERON-0.1 | 0.875 | 0.444 | 0.306 | 0.468 | 23.3 | 59.5 |
| laya-multilingual (base) | 0.896 | 0.403 | 0.324 | 0.468 | 21.5 | 42.4 |

| Metric | 0.1 | 0.2 | Δ |
|---|---|---|---|
| JevBench overall | 0.468 | **0.541** | **+7.3 %p** |
| standard tier | 0.444 | **0.583** | **+13.9 %p** |
| easy tier | 0.875 | **0.979** | **+10.4 %p** |
| hard tier | 0.306 | **0.324** | +1.8 %p |
| typed-decisions acc | 0.700 | **0.713** | +1.3 %p |
| Brier / score MAE | 0.449 / 0.421 | **0.424 / 0.373** | improved |
| hard ECE | 0.211 | **0.187** | −0.024 |
| hard prob. fidelity (TVD) | **0.389** | 0.465 | **+0.076 (worse)** |

**Interpretation**
- Changing the data made the standard tier (policy, rubric) jump from 0.444 to 0.583, and **overall accuracy moved ahead of the vendor-tuned version (0.537)**.
- The hard tier also improved to 0.324 versus the vendor (0.270) and the base (0.324) — though the gap to Jev (0.730) is still large.
- **Trade-off**: training on a mostly single-label corpus sharpens predictions, which improves ECE but **worsens fidelity to the gold distribution (TVD)**.
  Mixing in soft-label data leaves room for improvement.

### Artifacts
- HF: **https://huggingface.co/PIXELZX/XERON-0.2**
- `xeron-0.2.colab-a100.results.jsonl` (231 rows per-decision), `xeron-0.2.typed-decisions-eval.json`
- Training-script improvement: alongside each epoch checkpoint, an **inference-ready snapshot** is now saved too (commit `089be00`) — guarding against Colab VM loss

## 8. Conclusion and next steps

1. **XERON-0.1 is weaker than the vendor-tuned Laya on JevBench-type tasks.** The reason is clear — we trained on KLUE/browser/Mind2Web/SCOTUS, while JevBench measures "policy · rubric · option" judgment. The domains do not overlap.
2. Even so, **the fine-tuning effect is verified**: versus the same backbone, probability quality (ECE 0.289→0.211, TVD 0.573→0.389) and the hard tier improved.
3. **Suggested next step**: further fine-tuning on JevBench-style data.
   - Public items: `datasets/public/{original,easy,hard}.jsonl` (231 items, MIT)
   - At scale: HF `Praveenrajus/jev-bench` (22 configs · 166,054 rows, human-labeled System One questions)
   - Target: standard tier 0.44 → 0.65+, ECE hard 0.21 → 0.10 or below

## Reproduce

```bash
# Harness
git clone https://github.com/fstandhartinger/jevbench /tmp/jevbench
# public 231 items, laya_local adapter, serial single request, threads=4, timed after load
python run_public_jevbench.py <model_dir> <label> <out_prefix>
# Scoring (tier accuracy / ECE / TVD / axes / partial JevBench Score)
python score_public_jevbench.py <out_prefix> <label>
```

Raw per-decision results: `results/jevbench-public/*.results.jsonl` (231 rows each, including predictions, probability distributions and latency)
