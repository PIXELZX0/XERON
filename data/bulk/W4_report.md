# W4 리포트 — 최종 믹스 + 전처리 (`xeron10_mix.jsonl` → `train_items_x10.pt`)

> 2026-09-25 · seed 7 · 소요 약 37분(믹스 9분 + 검증 6분 + 전처리 6.5분 + 병합 3.3분 + 스모크 2.5분)

## 0. 결론

| 항목 | 값 |
|---|---|
| 믹스 행수 | **3,352,791** (소스 112개, 입력 3,388,667 → 드롭 35,876 = 1.06 %) |
| 학습 시퀀스 | **3,352,791** (`train_items_x10.pt`, 2.43 GB) — 행당 질문 1개, `item_none=0` |
| 언어 | 태그 353종 → **정규화 257개** (en 30.8 % · ko 5.35 % · zh 5.11 % · ja 2.72 % …) |
| qtype | choice 3,014,289 / noul 208,957 / score 129,545 |
| arity | 2:370,315 · 3:874,501 · 4:796,083 · 5:506,807 · 6:404,148 · 7:53,002 · 8–12·14:9,433 |
| 최대 workflow 비중 | **massive-intent 16.70 %** (2위 xnli-multilingual 11.18 %) — 단일 10 % 초과, 명시 보고 |
| 검증 | leak 0 · goldsum 0 · dup_in 0 · bad_arity 0 · empty 0 · badjson 0 · **위치편차 ≤ 0.1374** |
| 평가 누수 | JevBench public-231 **0** / typed-decisions test 400 **0** |
| 스모크 | **PASS** (202 items, longest 4096 + shortest 35, logits 유한, NaN 0) |
| Kaggle | `pistonx/xeron-1-0-train-items` (private, 2.43 GB < 20 GB) |

## 1. 믹스 (`data/xeron10_mix.jsonl`)

소스: `data/bulk/*.jsonl` 98개 + 레거시 14개 = **112개, 전량 포함(캡 없음)**.
제외: `clinc_typed`/`clinc_hard_typed`(의도적 기존 제외), `xeron3/5/9_mix.jsonl`(중복 원인).

상위 소스(행수): massive 560,040 · xnli 374,965 · sib200 205,820 · global_mmlu 146,403 ·
anli(bulk) 112,389 · belebele 109,484 · massive_scenario 91,800 · aqua_rat 89,970 ·
jevbench_full 85,503 · korean_typed 82,333 · race 79,916 · civil_comments 79,020 …

### 드롭 내역 (35,876행, 1.06 %)

| 사유 | 행수 | 비고 |
|---|---|---|
| `choice_arity_gt_16` | 26,675 | jevbench go_emotions 8,000(28지선) · long_typed topic 11,314(20) · mind2web element 7,361(26) — 프로젝트 계약 `MAX_K=16`(`build_bulk_choice`)·`verify_bulk` 2–16 위반. public-231 평가 arity는 3–6뿐이라 학습 이득도 없음 |
| `duplicate_state` | 8,813 | jevbench_full 8,253(짧은 go_emotions 텍스트 중복) · browser_typed 506 · 나머지 소수 |
| `empty_state` | 312 | 정규화 state < 10자 (clinc150 48 · massive_extra 133 · jevbench 131) |
| `answer_marker_in_state` | 76 | jevbench_full 66 · long 6 · korean 2 · anli 1 · hellaswag 1 |

### 포맷 정규화 (드롭 아님, `row_fixes`)

- `score_dict_to_list` **80,255행** — W1/W2 빌더는 `criteria={"0": "1 star", …}`(dict)로 쓰지만
  `preprocess.build_training_item`은 `n_levels = len(crit) if isinstance(crit, list) else 4` 로 읽는다.
  그대로 두면 5단계 score가 **target 4개로 잘려** 5★ 정답이 uniform 4-way로 학습된다(라벨 파괴).
  → list 형으로 정규화(옵션 텍스트·정답 순서 불변). 이 버그는 1.0에서 처음 노출된 조합이다.
- qid → `decision` **142,327행** — 레거시 `korean_typed`(topic/relation/similarity),
  `browser_typed`(topic/is_spam/is_phishing), `long_typed`(topic/author_justice),
  `mind2web_typed`(action_op/element)는 qid가 `decision`이 아니다. 행당 질문이 정확히 1개라
  이름만 바꾸는 무손실 정규화(코드 어디서도 qid 문자열을 읽지 않음). 이게 없으면 `verify_bulk`가
  `badjson=142,327`을 보고한다.

## 2. 검증

### `verify_bulk.py "data/xeron10_mix.jsonl"` → `[OK]`

```
rows=3,352,791 qtype={'choice':3014289,'score':129545,'noul':208957}
arity={2:370315,3:874501,4:796083,5:506807,6:404148,7:53002,8:999,9:951,10:854,11:826,12:809,14:4994}
langs=353 leak=0 goldsum=0 dup_in=0 dup_cross=0 posdev=0.24 empty=0 badjson=0
```

### 위치편차: 툴 지표 0.24는 **namespace 혼합 아티팩트**, 실측 **≤ 0.1374**

`verify_bulk`의 posdev는 gold **키 문자열**의 분포로 위치를 세는데, 믹스에는 두 관례가 섞여 있다.

- **letter-keyed** `criteria={"A": "<option text>"}` — 빌더 산출물 + hellaswag/arc/mmlu (2,856,223행)
- **name-keyed** `criteria={"entailment": null}` — 키가 곧 라벨. **public-231 평가와 같은 관례**
  (평가 예: `{"track_order": "..."}` / `expected="track_order"`)

arity 4 버킷에 두 관례가 섞이면 `#distinct_keys`가 297이 되어(1/297=0.0034 vs A–D 각 0.238)
posdev가 0.24로 튄다. 실제 **표시 위치**(criteria dict 순서 = `render_options` 순서) 기준 실측
(`scripts/posdev_x10.py`, `data/xeron10_posdev.json`):

| arity | letter rows / dev | name rows / dev |
|---|---|---|
| 2 | 370,315 / 0.0009 | — |
| 3 | 815,915 / 0.0003 | 58,586 / 0.0348 |
| 4 | 769,926 / 0.0015 | 26,157 / 0.0096 |
| 5 | 500,428 / 0.0007 | 6,379 / 0.0113 |
| 6 | 399,639 / 0.0008 | 4,509 / 0.0134 |
| 7 | — | 53,002 / 0.0704 |
| 14 | — | 4,994 / **0.1374** (long-context author_justice, 14지선) |

**최대 0.1374 ≤ 0.15 충족.** name-keyed 상위 workflow는 jevbench-banking77/clinc150/ledgar/massive가
행마다 옵션 순서를 셔플(워크플로당 200개 순서 확인)했고, 순서 고정 workflow는 anli·long-context·
browser-use/agent·korean-general뿐이며 편차는 위 표 값이다. W2/W3 리포트가 같은 아티팩트를
문서화한 것과 동일한 처리다.

### 평가 누수 = 0 (`scripts/check_eval_leak.py`)

- `/tmp/jevbench/datasets/public/{original,easy,hard}.jsonl` 231 state → **0**
- `LocalLLaMA/typed-decisions` **test** 400 state → **0** (HF 캐시 로드, 상태 `checked`)

## 3. 분포

- **언어(정규화 257개)** — 규칙: 소문자화 + 지역/스크립트 제거 + ISO-639-3→639-1.
  en 30.81 % · ko 5.35 · zh 5.11 · ja 2.72 · fr 2.25 · es 2.25 · de 2.22 · hi 1.68 · ar 1.67 ·
  sw 1.60 · ru 1.49 · th 1.46 · vi 1.44 · tr 1.36 · ur 1.34 · el 1.31 … (태그 없음 226,507행,
  state가 dict 아님 55,957행 = jevbench 계열 원문 문자열)
- **workflow 상위** massive-intent 16.70 % · xnli-multilingual 11.18 · sib200-topic 6.14 ·
  global_mmlu 4.37 · anli-nli 3.35 · belebele-reading 3.27 · massive-scenario 2.74 ·
  aqua-rat 2.68 · jevbench_full 2.55 · korean-general 2.46 …
  → **단일 workflow 10 % 초과 2건**(massive-intent 16.7 %, xnli 11.2 %)을 명시 보고. 캡은 지시대로 미적용.
- **qtype** choice 89.9 % / noul 6.2 % / score 3.9 % (soft/noul 라벨 208,957행 확보)

## 4. 전처리

- 토크나이저: `output/xeron-0.9-snapshot/tokenizer/tokenizer.json` 과
  `~/laya-models/laya-base/multilingual/tokenizer/tokenizer.json` 의 sha256이 **동일**
  (`609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`) → 0.9 계보 일치.
  스냅샷을 `~/laya-models/xeron-0.9-base/` 로 복사해 사용(`_fix_tokenizer_config`/
  `ensure_long_context` 의 config 변조가 0.9 스냅샷을 건드리지 않도록).
- `scripts/preprocess_shard.py` 신규: `preprocess.build_training_item` **import 재사용**(중복 구현 없음),
  `--shard i --num-shards N` + **행 단위 line-offset 인덱스**(`data/x10_shards/mix.lineidx.npz`,
  생성 9초)로 균등 분할 → 샤드당 정확히 335,279행(마지막 335,280).
- `MAX_LEN=4096 HEAD_MAX_LEN=256`, `xargs -P 10` (12코어) → **6분 30초** (16:30→16:36).
  샤드별 `rows == items`, `item_none=0`, `bad_json=0`, `max_ids` 305–4096.
- 병합 `scripts/merge_shards_x10.py` → `train_items_x10.pt` **3,352,791 items / 2.43 GB** (3분 20초).
- 시퀀스 길이: p50 128 · p90 267 · p99 789 · max 4096. 20,000개 샘플 검사에서
  target 합=1.0 위반 0, `label != argmax` 0, 전 아이템 `len(markers)==len(target)`.
- 스모크 `scripts/smoke_x10.py`: 무작위 200 + 최장(4096)/최단(35) → 101 배치 CPU forward,
  logits `(batch, kmax)` 정상·유한, marker/target 일치, **weights missing=0 unexpected=0**
  (head_layers=4, head_size=1024) → **PASS**.

## 5. 산출물 / Kaggle

| 파일 | 내용 |
|---|---|
| `data/xeron10_mix.jsonl` | 3,352,791행 (3.23 GB, gitignore) |
| `train_items_x10.pt` | 3,352,791 시퀀스 (2.43 GB, gitignore) |
| `data/xeron10_mix_manifest.json` | 소스별 행수·드롭·분포·검증·토크나이저 sha256·시드·재현 명령 |
| `data/xeron10_mix_stats.json` / `_verify.json` / `_eval_leak.json` / `_posdev.json` / `_preprocess_stats.json` | 단계별 원자료 |
| `kaggle/dataset-x10/dataset-metadata.json` | `pistonx/xeron-1-0-train-items` (private) |
| `scripts/` | `build_mix_x10.py` · `preprocess_shard.py` · `merge_shards_x10.py` · `smoke_x10.py` · `check_eval_leak.py` · `posdev_x10.py` · `make_x10_manifest.py` |

Kaggle 업로드: 2.43 GB < 20 GB → `kaggle datasets create -p kaggle/dataset-x10 --dir-mode skip`.

## 6. 남은 리스크 / 결정 필요

1. **arity > 16 제외 26,675행** — 프로젝트 계약(`MAX_K=16`)·평가 분포(3–6)에 근거해 제외했다.
   long_typed topic(20지선)·mind2web element(26)·go_emotions(28)을 살리려면 계약을 넓히고
   믹스→전처리를 재실행해야 한다(전처리 6.5분, 병합 3.3분이라 재실행 비용은 낮다).
2. **workflow 편중** — massive-intent 16.7 % · xnli 11.2 %. 캡 미적용 지시대로 유지.
   다운샘플 정책이 필요하면 `build_mix_x10.py`에 예산 옵션을 추가하면 된다.
3. **dbpedia_14 = 0행** — W1 빌드에서 `dropped_shape=60000`(필드 불일치)으로 전량 드롭된 상태가
   그대로 들어왔다. 스키마 확정 후 재빌드 필요(이번 웨이브 범위 아님).
4. **score/noul 라벨 편중** — score 3.9 %·noul 6.2 %로 Calibration 축 데이터가 상대적으로 얇다.
   `--soft-boost`(mix_datasets) 같은 업샘플링은 이번엔 미적용(지시: 캡·조정 없이 전량).
5. **`posdev` 지표 자체** — 툴이 namespace를 구분하지 못한다. 필요하면 `verify_bulk`에
   namespace-aware posdev를 추가하는 것이 옳지만, 이번 웨이브는 툴 미수정 + 별도 스크립트로 처리.
6. **state가 dict 아닌 55,957행**(jevbench 계열)은 언어 태그가 없어 언어 분포 집계에서 제외된다.
