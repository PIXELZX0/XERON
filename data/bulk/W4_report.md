# W4 report — final mix + preprocessing (`xeron10_mix.jsonl` → `train_items_x10.pt`)

> 2026-09-25 · seed 7 · took about 37 minutes (mix 9 min + verification 6 min + preprocessing 6.5 min + merge 3.3 min + smoke 2.5 min)

## 0. Conclusion

| Item | Value |
|---|---|
| Mix rows | **3,352,791** (112 sources, 3,388,667 input → 35,876 dropped = 1.06 %) |
| Training sequences | **3,352,791** (`train_items_x10.pt`, 2.43 GB) — 1 question per row, `item_none=0` |
| Languages | 353 tag kinds → **257 normalized** (en 30.8 % · ko 5.35 % · zh 5.11 % · ja 2.72 % …) |
| qtype | choice 3,014,289 / noul 208,957 / score 129,545 |
| arity | 2:370,315 · 3:874,501 · 4:796,083 · 5:506,807 · 6:404,148 · 7:53,002 · 8–12·14:9,433 |
| Largest workflow share | **massive-intent 16.70 %** (runner-up xnli-multilingual 11.18 %) — one over 10 %, explicitly reported |
| Verification | leak 0 · goldsum 0 · dup_in 0 · bad_arity 0 · empty 0 · badjson 0 · **position deviation ≤ 0.1374** |
| Eval leakage | JevBench public-231 **0** / typed-decisions test 400 **0** |
| Smoke | **PASS** (202 items, longest 4096 + shortest 35, logits finite, NaN 0) |
| Kaggle | `pistonx/xeron-1-0-train-items` (private, 2.43 GB < 20 GB) |

## 1. Mix (`data/xeron10_mix.jsonl`)

Sources: 98 `data/bulk/*.jsonl` + 14 legacy = **112 files, all included (no cap)**.
Excluded: `clinc_typed`/`clinc_hard_typed` (deliberate pre-existing exclusion), `xeron3/5/9_mix.jsonl` (causes duplication).

Top sources (rows): massive 560,040 · xnli 374,965 · sib200 205,820 · global_mmlu 146,403 ·
anli(bulk) 112,389 · belebele 109,484 · massive_scenario 91,800 · aqua_rat 89,970 ·
jevbench_full 85,503 · korean_typed 82,333 · race 79,916 · civil_comments 79,020 …

### Drop breakdown (35,876 rows, 1.06 %)

| Reason | Rows | Notes |
|---|---|---|
| `choice_arity_gt_16` | 26,675 | jevbench go_emotions 8,000 (28 choices) · long_typed topic 11,314 (20) · mind2web element 7,361 (26) — violates the project contract `MAX_K=16` (`build_bulk_choice`) and `verify_bulk` 2–16. public-231 eval arity is only 3–6, so there is no training benefit either |
| `duplicate_state` | 8,813 | jevbench_full 8,253 (duplicate short go_emotions texts) · browser_typed 506 · a few others |
| `empty_state` | 312 | normalized state < 10 chars (clinc150 48 · massive_extra 133 · jevbench 131) |
| `answer_marker_in_state` | 76 | jevbench_full 66 · long 6 · korean 2 · anli 1 · hellaswag 1 |

### Format normalization (not drops, `row_fixes`)

- `score_dict_to_list` **80,255 rows** — the W1/W2 builders write `criteria={"0": "1 star", …}` (dict), but
  `preprocess.build_training_item` reads `n_levels = len(crit) if isinstance(crit, list) else 4`.
  Left as-is, a 5-level score gets **truncated to 4 targets**, so a 5★ answer is trained as uniform 4-way (label destruction).
  → normalized to list form (option text and answer order unchanged). This bug is a combination first exposed in 1.0.
- qid → `decision` **142,327 rows** — legacy `korean_typed` (topic/relation/similarity),
  `browser_typed` (topic/is_spam/is_phishing), `long_typed` (topic/author_justice),
  `mind2web_typed` (action_op/element) do not have `decision` as the qid. Since there is exactly one question per row,
  this is a lossless rename-only normalization (no code reads the qid string anywhere). Without it, `verify_bulk`
  reports `badjson=142,327`.

## 2. Verification

### `verify_bulk.py "data/xeron10_mix.jsonl"` → `[OK]`

```
rows=3,352,791 qtype={'choice':3014289,'score':129545,'noul':208957}
arity={2:370315,3:874501,4:796083,5:506807,6:404148,7:53002,8:999,9:951,10:854,11:826,12:809,14:4994}
langs=353 leak=0 goldsum=0 dup_in=0 dup_cross=0 posdev=0.24 empty=0 badjson=0
```

### Position deviation: the tool's 0.24 metric is a **namespace-mixing artifact**; measured **≤ 0.1374**

`verify_bulk`'s posdev counts positions from the distribution of gold **key strings**, but the mix contains two conventions.

- **letter-keyed** `criteria={"A": "<option text>"}` — builder output + hellaswag/arc/mmlu (2,856,223 rows)
- **name-keyed** `criteria={"entailment": null}` — the key is the label itself. **Same convention as the public-231 eval**
  (eval example: `{"track_order": "..."}` / `expected="track_order"`)

When both conventions are mixed into the arity-4 bucket, `#distinct_keys` becomes 297 (1/297=0.0034 vs A–D at 0.238 each)
and posdev jumps to 0.24. Measured on the actual **display position** (criteria dict order = `render_options` order)
(`scripts/posdev_x10.py`, `data/xeron10_posdev.json`):

| arity | letter rows / dev | name rows / dev |
|---|---|---|
| 2 | 370,315 / 0.0009 | — |
| 3 | 815,915 / 0.0003 | 58,586 / 0.0348 |
| 4 | 769,926 / 0.0015 | 26,157 / 0.0096 |
| 5 | 500,428 / 0.0007 | 6,379 / 0.0113 |
| 6 | 399,639 / 0.0008 | 4,509 / 0.0134 |
| 7 | — | 53,002 / 0.0704 |
| 14 | — | 4,994 / **0.1374** (long-context author_justice, 14 choices) |

**Max 0.1374 ≤ 0.15 satisfied.** The top name-keyed workflows jevbench-banking77/clinc150/ledgar/massive
shuffle the option order per row (verified 200 orders per workflow), and the order-fixed workflows are only anli·long-context·
browser-use/agent·korean-general, whose deviations are the values in the table above. Same handling as documented for the
same artifact in the W2/W3 reports.

### Eval leakage = 0 (`scripts/check_eval_leak.py`)

- `/tmp/jevbench/datasets/public/{original,easy,hard}.jsonl` 231 states → **0**
- `LocalLLaMA/typed-decisions` **test** 400 states → **0** (loaded from HF cache, state `checked`)

## 3. Distribution

- **Languages (257 normalized)** — rule: lowercase + strip region/script + ISO-639-3→639-1.
  en 30.81 % · ko 5.35 · zh 5.11 · ja 2.72 · fr 2.25 · es 2.25 · de 2.22 · hi 1.68 · ar 1.67 ·
  sw 1.60 · ru 1.49 · th 1.46 · vi 1.44 · tr 1.36 · ur 1.34 · el 1.31 … (untagged 226,507 rows,
  state not a dict 55,957 rows = jevbench-family raw strings)
- **Top workflows** massive-intent 16.70 % · xnli-multilingual 11.18 · sib200-topic 6.14 ·
  global_mmlu 4.37 · anli-nli 3.35 · belebele-reading 3.27 · massive-scenario 2.74 ·
  aqua-rat 2.68 · jevbench_full 2.55 · korean-general 2.46 …
  → **two workflows over 10 %** (massive-intent 16.7 %, xnli 11.2 %) explicitly reported. No cap applied, as instructed.
- **qtype** choice 89.9 % / noul 6.2 % / score 3.9 % (208,957 rows of soft/noul labels secured)

## 4. Preprocessing

- Tokenizer: the sha256 of `output/xeron-0.9-snapshot/tokenizer/tokenizer.json` and
  `~/laya-models/laya-base/multilingual/tokenizer/tokenizer.json` are **identical**
  (`609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`) → 0.9 lineage matches.
  The snapshot was copied to `~/laya-models/xeron-0.9-base/` for use (so that config mutations by `_fix_tokenizer_config`/
  `ensure_long_context` do not touch the 0.9 snapshot).
- `scripts/preprocess_shard.py` new: **reuses `preprocess.build_training_item` by import** (no duplicated implementation),
  `--shard i --num-shards N` + a **row-level line-offset index** (`data/x10_shards/mix.lineidx.npz`,
  built in 9 s) for even splitting → exactly 335,279 rows per shard (last 335,280).
- `MAX_LEN=4096 HEAD_MAX_LEN=256`, `xargs -P 10` (12 cores) → **6 min 30 s** (16:30→16:36).
  Per shard `rows == items`, `item_none=0`, `bad_json=0`, `max_ids` 305–4096.
- Merge `scripts/merge_shards_x10.py` → `train_items_x10.pt` **3,352,791 items / 2.43 GB** (3 min 20 s).
- Sequence lengths: p50 128 · p90 267 · p99 789 · max 4096. Across a 20,000-sample check:
  target sum=1.0 violations 0, `label != argmax` 0, every item `len(markers)==len(target)`.
- Smoke `scripts/smoke_x10.py`: 200 random + longest (4096)/shortest (35) → 101 CPU forward batches,
  logits `(batch, kmax)` normal and finite, marker/target match, **weights missing=0 unexpected=0**
  (head_layers=4, head_size=1024) → **PASS**.

## 5. Artifacts / Kaggle

| File | Content |
|---|---|
| `data/xeron10_mix.jsonl` | 3,352,791 rows (3.23 GB, gitignore) |
| `train_items_x10.pt` | 3,352,791 sequences (2.43 GB, gitignore) |
| `data/xeron10_mix_manifest.json` | per-source rows, drops, distribution, verification, tokenizer sha256, seed, reproduce commands |
| `data/xeron10_mix_stats.json` / `_verify.json` / `_eval_leak.json` / `_posdev.json` / `_preprocess_stats.json` | raw data per stage |
| `kaggle/dataset-x10/dataset-metadata.json` | `pistonx/xeron-1-0-train-items` (private) |
| `scripts/` | `build_mix_x10.py` · `preprocess_shard.py` · `merge_shards_x10.py` · `smoke_x10.py` · `check_eval_leak.py` · `posdev_x10.py` · `make_x10_manifest.py` |

Kaggle upload: 2.43 GB < 20 GB → `kaggle datasets create -p kaggle/dataset-x10 --dir-mode skip`.

## 6. Remaining risks / decisions needed

1. **The 26,675 dropped arity > 16 rows** — excluded based on the project contract (`MAX_K=16`) and the eval distribution (3–6).
   To keep long_typed topic (20 choices)·mind2web element (26)·go_emotions (28), widen the contract and
   re-run mix→preprocess (preprocess 6.5 min, merge 3.3 min, so re-running is cheap).
2. **Workflow skew** — massive-intent 16.7 % · xnli 11.2 %. Kept as-is per the no-cap instruction.
   If a downsampling policy is needed, add a budget option to `build_mix_x10.py`.
3. **dbpedia_14 = 0 rows** — the state that was entirely dropped in the W1 build via `dropped_shape=60000` (field mismatch)
   came through unchanged. Needs a rebuild once the schema is settled (out of scope for this wave).
4. **score/noul label skew** — score 3.9 %·noul 6.2 %, so Calibration-axis data is relatively thin.
   Upsampling like `--soft-boost` (mix_datasets) was not applied this time (instruction: full volume, no cap or adjustment).
5. **The `posdev` metric itself** — the tool cannot distinguish namespaces. If needed, the right fix is to add
   namespace-aware posdev to `verify_bulk`, but this wave handled it with an unmodified tool plus a separate script.
6. **55,957 rows whose state is not a dict** (jevbench family) have no language tag, so they are excluded from the language-distribution tally.
