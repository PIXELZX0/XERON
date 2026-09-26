# XERON-1.0 Data Plan (2026-09-25)

> Instruction: **expand the encoder from the 0.9 base** → data requirements grow substantially.
> **Language- and difficulty-agnostic, as much data as possible.**

## 0. Current state (starting point)

| Item | Content |
|---|---|
| Base | `PIXELZX/XERON-0.9` snapshot `output/xeron-0.9-snapshot/` (encoder + wide head, HEAD_SIZE supported) |
| 0.9 data | `data/xeron9_mix.jsonl` 123,737 rows = **JevBench single source** (17 configs + 5 kinds of extras) |
| 0.5 data | `data/xeron5_mix.jsonl` 153,582 rows = jevbench 55k / korean 25k / anli 16k / browser 12k / long 10k / mind2web 8k / mmlu 15k / hellaswag 10k / arc 3k |
| Preprocessing artifacts | `train_items_x3.pt`(61,876) / `x5.pt`(153,582) / `x9.pt` |
| Model code | `scripts/model_xeron.py` — `WideHeadDecisionModel` (encoder + independent head_size) |
| Training | Kaggle T4×2 (fp16) / Vast A100 (bf16, `scripts/a100/run_a100.sh`) |
| Evaluation | JevBench public-231 + `LocalLLaMA/typed-decisions` test |

## 1. 1.0 data principles

1. **Format unchanged** — Laya native 3 fields (`state`/`questions`/`gold`, all JSON strings).
   `preprocess.py` consumes them as-is. Do not change the schema.
2. **Unlimited languages** — en/ko/ja/zh/de/fr/es/ar/hi/th/vi/tr… (keep the `language` tag in `state`).
3. **Unlimited difficulty** — easy (sentiment/topic) to hard (multi-step reasoning/math/logic).
4. **Preserve the quality invariants** (this is the asset 0.2–0.9 accumulated):
   - Randomize the correct-answer position (no position shortcut)
   - No answer marker in `state` (`Answer: B`, `answerKey`, `정답`, …)
   - For datasets whose label names appear in the option text (k>6), draw distractors from the **same vocabulary-overlap bucket**
   - gold probabilities sum to 1.0, one-hot or soft (noul/score)
   - Remove exact duplicates (state,questions) + zero overlap with existing `data/*.jsonl` and with the **JevBench public-231 evaluation set**
5. **No evaluation leakage** — the JevBench public 231 (`/tmp/jevbench/datasets/public/*.jsonl`) and
   the typed-decisions test are never included in training under any circumstance.

## 2. Build tools

| File | Role |
|---|---|
| `scripts/build_bulk_choice.py` | Registry-driven bulk builder. Shapes: `nli / label_int / label_str / choices_list / dict_choices / abc_columns / ordinal / soft_binary / copa / story_cloze / pair_binary / auto` |
| `scripts/probe_candidates.py` | Inspect candidate HF dataset schemas/splits **without downloading** (`data/candidate_probe.json`) |
| `scripts/verify_bulk.py` | Verify artifacts: gold sum/arity/leak/position bias/duplicates/language distribution |

Run:

```bash
cd ~/XERON
PYTHONPATH=scripts .venv/bin/python scripts/build_bulk_choice.py \
    --outdir data/bulk --only massive,xnli            # by name
PYTHONPATH=scripts .venv/bin/python scripts/build_bulk_choice.py \
    --outdir data/bulk --specs-file scripts/specs_multilingual.py   # separate registry
PYTHONPATH=scripts .venv/bin/python scripts/verify_bulk.py "data/bulk/*.jsonl" \
    --against "data/*.jsonl"
```

* In `--outdir data/bulk`: `{name}.jsonl` + `{name}.manifest.json` + `_build_summary.json`.
* Exclusion-key cache: `data/bulk/_exclude_keys.json` (existing `data/*.jsonl` + public-231).
  If existing data changes, use `--rebuild-exclude-cache`.
* For large volumes (millions of rows), use `stream="shuffle"` (buffer shuffle then take) or `stream=True` (A-Res reservoir).
* seed=7 fixed → reproducible.

## 3. Target scale

| Wave | Content | Target rows |
|---|---|---|
| W1 (in progress) | multilingual routing/NLI (MASSIVE 51 languages, XNLI 15 languages) + English volume (SNLI/MNLI/ANLI/RACE/Amazon/Yelp/DBpedia/AGNews/CivilComments…) + Korean (KLUE/KoBEST) | ~1.0M |
| W2 (done) | multilingual expansion (Global-MMLU 42 languages, Belebele 122 languages, SIB-200 205 languages, XCOPA, XStoryCloze, PAWS-X, MMLU-ProX, INCLUDE-44, mNLI-26lang, multilingual reviews) | **881,570** (44 specs) |
| W3 (done) | other languages (Japanese JGLUE, Chinese CLUE/C3, Korean KMMLU/KorNLI/NSMC) + reasoning/math (LogiQA2, MathQA, SocialIQA, CosmosQA, FEVER) + 8 GLUE sets + MTOP routing | **493,935** (32 specs) |
| Final | `data/xeron10_mix.jsonl` → `train_items_x10.pt` (preprocessing, multiprocess) | **3,352,791 rows → 3,352,791 sequences** |

> **W4 measured (2026-09-25)**: final mix + preprocessing complete. See `data/bulk/W4_report.md`.
> All 112 sources (98 bulk + 14 legacy), no cap → from 3,388,667 input rows, 35,876 dropped (1.06%:
> arity>16 26,675 · duplicate state 8,813 · empty state 312 · state answer marker 76) → **3,352,791 rows**.
> Format normalization: score criteria dict→list **80,255 rows** (fixing a bug where preprocess misread
> the dict format from the W1/W2 builders as 4-level), qid→`decision` 142,327 rows (legacy).
> Verification: `verify_bulk` leak=0 / goldsum=0 / dup_in=0 / bad_arity=0 / empty=0 / badjson=0,
> measured position deviation ≤ 0.1374 (the tool's posdev 0.24 is an artifact of mixing letter/name key namespaces — `scripts/posdev_x10.py`),
> eval leakage 0 (JevBench public-231 · typed-decisions test 400).
> Preprocessing: `scripts/preprocess_shard.py` (reusing `preprocess.build_training_item`) + line-offset index,
> 10 shards × 10 processes, MAX_LEN 4096 / HEAD_MAX_LEN 256 → **6 min 30 s**, merge 3 min 20 s,
> `train_items_x10.pt` 2.43 GB / 3,352,791 items. Smoke PASS (202 items including the longest 4096 and shortest 35, NaN 0).
> Kaggle: `pistonx/xeron-1-0-train-items` (private, 2.43 GB) — `kaggle/dataset-x10/`.
> Languages 353 tags→257 normalized (en 30.8%), qtype choice 89.9%/noul 6.2%/score 3.9%,
> largest workflow massive-intent 16.7% (xnli 11.2%).

> **W3 measured (2026-09-25)**: `scripts/specs_extra.py` (new registry, 32 specs) → `data/bulk/*.jsonl`
> **493,935 rows** (en 240,210 / ko 93,063 / zh 84,091 / ja 46,961 / de·es·fr·hi·th 37,463).
> Language coverage: ko KMMLU·KorNLI·NSMC, ja JNLI·MARC-ja·JSTS, zh CLUE (TNEWS/OCNLI/CMNLI/AFQMC/WSC)·C3.
> Reasoning·math: LogiQA2, MathQA, SocialIQA, CosmosQA, FEVER-NLI, 8 GLUE sets (mrpc/qqp/sst2/cola/qnli/rte/wnli/stsb).
> probe ERR recovery: 12 of 13 (parquet mirrors), failures are LogiQA v1·CLUE-iflytek·CLUE-csl·mteb/amazon_reviews_multi.
> Verification: `verify_bulk` → leak=0 / goldsum=0 / dup_in=0 / dup_cross=0 / empty=0. Details in `data/bulk/W3_report.md`.
> Run: `--specs-file` is resolved relative to scripts, so specify an absolute path.

> **W2 measured (2026-09-25)**: `scripts/specs_multilingual.py` (new registry, 44 specs + 4 custom adapters) → `data/bulk/*.jsonl`
> **881,570 rows**, `state.language` tags 351 kinds → **206 languages** after normalization. Largest single-language share 4.27% (zh), meeting the 20% ceiling.
> Composition: belebele 109,484 (122 languages) · sib200 205,820 (205 languages) · global_mmlu 146,403 (42) · massive_scenario 91,800 (51) ·
> mmlu_prox 69,620 (28) · paws_x 55,565 (7) · amazon_reviews_multi 71,721 (6, ordinal) · afrisenti 34,178 (15) ·
> mlnli 32,489 (26) · masakhanews 22,154 (16) · xstory_cloze 20,578 (11) · include_lite_44 10,716 (44) · xcopa 6,600 (11) · xwinograd 4,442 (6).
> probe recovery: `google/paws-x` deleted → `google-research-datasets/paws-x`, `mteb/amazon_reviews_multi` script removed → `SetFit/amazon_reviews_multi_*`.
> Excluded: `m3exam` (unclear answer encoding), `language-identification` (answer = language, leaks into state), `mlqa`/`tydiqa`/`mtop_*` (script-only), xcopa `translation-*`, MMLU-ProX `en` (duplicate of W1).
> Verification: `verify_bulk` → all files leak=0 / goldsum=0 / dup_in=0 / dup_cross=0 / empty=0, posdev ≤ 0.111. Details in `data/bulk/W2_report.md`.

Preprocessing bottleneck: `preprocess.py` is single-process → 2M rows takes hours.
→ shard-parallel (`scripts/preprocess_shard.py`, new) then merge.

### W4 tools (new)

| File | Role |
|---|---|
| `scripts/build_mix_x10.py` | Combine all sources + enforce invariants (state/qid/arity/gold normalization) + drop/distribution stats |
| `scripts/preprocess_shard.py` | `--shard i --num-shards N` + line-offset index (`--build-index-only`), reuses `preprocess.build_training_item` by import |
| `scripts/merge_shards_x10.py` | Merge shards → `train_items_x10.pt` |
| `scripts/smoke_x10.py` | 200 random entries from the merged file + longest/shortest → 0.9 snapshot CPU forward check |
| `scripts/check_eval_leak.py` | Hard gate for zero leakage into public-231 + typed-decisions test |
| `scripts/posdev_x10.py` | True position-deviation measurement separating letter/name namespaces |
| `scripts/make_x10_manifest.py` | Assemble `data/xeron10_mix_manifest.json` |

## 4. Open decisions (owner)

- 1.0 encoder expansion dimensions (hidden/layers/vocab) → after expansion, a partial-load strategy for the 0.9 weights is needed
  (reuse the encoder-only load pattern from `69aea9f`).
- Context length: currently 4096/head 256. Whether to increase it for 1.0.
- Training compute: Kaggle free (T4×2, fp16) vs Vast A100 (bf16, $0.8/h) — 2M rows × 1 epoch is
  a scale that T4×2 cannot handle.
