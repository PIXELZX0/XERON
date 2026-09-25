# XERON-1.0 데이터 계획 (2026-09-25)

> 지시: **0.9 기반에서 인코더를 확장**한다 → 데이터 요구량이 크게 늘어난다.
> **언어·난이도 무관, 최대한 많은 데이터**를 넣는다.

## 0. 현재 상태 (출발점)

| 구분 | 내용 |
|---|---|
| 베이스 | `PIXELZX/XERON-0.9` 스냅샷 `output/xeron-0.9-snapshot/` (encoder + wide head, HEAD_SIZE 지원) |
| 0.9 데이터 | `data/xeron9_mix.jsonl` 123,737행 = **JevBench 단일 소스**(17 config + extras 5종) |
| 0.5 데이터 | `data/xeron5_mix.jsonl` 153,582행 = jevbench 55k / korean 25k / anli 16k / browser 12k / long 10k / mind2web 8k / mmlu 15k / hellaswag 10k / arc 3k |
| 전처리 산출 | `train_items_x3.pt`(61,876) / `x5.pt`(153,582) / `x9.pt` |
| 모델 코드 | `scripts/model_xeron.py` — `WideHeadDecisionModel` (encoder + 독립 head_size) |
| 학습 | Kaggle T4×2(fp16) / Vast A100(bf16, `scripts/a100/run_a100.sh`) |
| 평가 | JevBench public-231 + `LocalLLaMA/typed-decisions` test |

## 1. 1.0 데이터 원칙

1. **포맷 불변** — Laya 네이티브 3필드(`state`/`questions`/`gold`, 전부 JSON 문자열).
   `preprocess.py`가 그대로 소비한다. 스키마를 바꾸지 않는다.
2. **언어 무제한** — en/ko/ja/zh/de/fr/es/ar/hi/th/vi/tr… (state에 `language` 태그 유지).
3. **난이도 무제한** — easy(감정/토픽) ~ hard(다단 추론/수학/논리).
4. **품질 불변식 유지** (이게 0.2~0.9가 벌어둔 자산):
   - 정답 위치 무작위화 (위치 shortcut 금지)
   - `state`에 정답 마커 금지 (`Answer: B`, `answerKey`, `정답` …)
   - 라벨 이름이 옵션 텍스트에 들어가는 데이터셋(k>6)은 **동일 어휘겹침 버킷**에서 오답 추출
   - gold 확률합 = 1.0, one-hot 또는 soft(noul/score)
   - 정확 중복(state,questions) 제거 + 기존 `data/*.jsonl` 및 **JevBench public-231 평가셋과 중복 0**
5. **평가 누수 금지** — JevBench public 231(`/tmp/jevbench/datasets/public/*.jsonl`)과
   typed-decisions test는 어떤 경우에도 학습에 넣지 않는다.

## 2. 빌드 도구

| 파일 | 역할 |
|---|---|
| `scripts/build_bulk_choice.py` | 레지스트리 기반 대량 빌더. shape: `nli / label_int / label_str / choices_list / dict_choices / abc_columns / ordinal / soft_binary / copa / story_cloze / pair_binary / auto` |
| `scripts/probe_candidates.py` | 후보 HF 데이터셋 스키마/스플릿을 **다운로드 없이** 조사 (`data/candidate_probe.json`) |
| `scripts/verify_bulk.py` | 산출물 검증: gold 합/arity/누수/위치편향/중복/언어분포 |

실행:

```bash
cd ~/XERON
PYTHONPATH=scripts .venv/bin/python scripts/build_bulk_choice.py \
    --outdir data/bulk --only massive,xnli            # 이름 지정
PYTHONPATH=scripts .venv/bin/python scripts/build_bulk_choice.py \
    --outdir data/bulk --specs-file scripts/specs_multilingual.py   # 별도 레지스트리
PYTHONPATH=scripts .venv/bin/python scripts/verify_bulk.py "data/bulk/*.jsonl" \
    --against "data/*.jsonl"
```

* `--outdir data/bulk` 에 `{name}.jsonl` + `{name}.manifest.json` + `_build_summary.json`.
* 제외키 캐시: `data/bulk/_exclude_keys.json` (기존 `data/*.jsonl` + public-231).
  기존 데이터가 바뀌면 `--rebuild-exclude-cache`.
* 대용량(수백만 행)은 `stream="shuffle"`(버퍼 셔플 후 take) 또는 `stream=True`(A-Res reservoir).
* seed=7 고정 → 재현 가능.

## 3. 목표 규모

| 웨이브 | 내용 | 목표 행수 |
|---|---|---|
| W1 (진행 중) | 다국어 라우팅/NLI(MASSIVE 51개 언어, XNLI 15개 언어) + 영어 볼륨(SNLI/MNLI/ANLI/RACE/Amazon/Yelp/DBpedia/AGNews/CivilComments…) + 한국어(KLUE/KoBEST) | ~1.0M |
| W2 (완료) | 다국어 확장(Global-MMLU 42개 언어, Belebele 122개 언어, SIB-200 205개 언어, XCOPA, XStoryCloze, PAWS-X, MMLU-ProX, INCLUDE-44, mNLI-26lang, 리뷰 다국어) | **881,570** (44 스펙) |
| W3 (완료) | 타언어(일본어 JGLUE, 중국어 CLUE/C3, 한국어 KMMLU/KorNLI/NSMC) + 추론/수학(LogiQA2, MathQA, SocialIQA, CosmosQA, FEVER) + GLUE 세트 8종 + MTOP 라우팅 | **493,935** (32 스펙) |
| 최종 | `data/xeron10_mix.jsonl` → `train_items_x10.pt` (전처리, 멀티프로세스) | **3,352,791행 → 3,352,791 시퀀스** |

> **W4 실측(2026-09-25)**: 최종 믹스 + 전처리 완료. `data/bulk/W4_report.md` 참조.
> 소스 112개(98 bulk + 14 레거시) 전량, 캡 없음 → 입력 3,388,667행에서 드롭 35,876(1.06 %:
> arity>16 26,675 · 중복 state 8,813 · 빈 state 312 · state 정답마커 76) → **3,352,791행**.
> 포맷 정규화: score criteria dict→list **80,255행**(W1/W2 빌더의 dict 형식을 preprocess가
> 4-level로 오독하던 버그 수정), qid→`decision` 142,327행(레거시).
> 검증: `verify_bulk` leak=0 / goldsum=0 / dup_in=0 / bad_arity=0 / empty=0 / badjson=0,
> 위치편차 실측 ≤ 0.1374(툴 posdev 0.24는 letter/name 키 namespace 혼합 아티팩트 — `scripts/posdev_x10.py`),
> 평가 누수 0(JevBench public-231 · typed-decisions test 400).
> 전처리: `scripts/preprocess_shard.py`(preprocess.build_training_item 재사용) + line-offset 인덱스로
> 10샤드 × 10프로세스, MAX_LEN 4096 / HEAD_MAX_LEN 256 → **6분 30초**, 병합 3분 20초,
> `train_items_x10.pt` 2.43 GB / 3,352,791 items. 스모크 PASS(최장 4096·최단 35 포함 202개, NaN 0).
> Kaggle: `pistonx/xeron-1-0-train-items` (private, 2.43 GB) — `kaggle/dataset-x10/`.
> 언어 353태그→정규화 257개(en 30.8 %), qtype choice 89.9 %/noul 6.2 %/score 3.9 %,
> 최대 workflow massive-intent 16.7 %(xnli 11.2 %).

> **W3 실측(2026-09-25)**: `scripts/specs_extra.py` (신규 레지스트리, 32 스펙) → `data/bulk/*.jsonl`
> **493,935행** (en 240,210 / ko 93,063 / zh 84,091 / ja 46,961 / de·es·fr·hi·th 37,463).
> 언어권: ko KMMLU·KorNLI·NSMC, ja JNLI·MARC-ja·JSTS, zh CLUE(TNEWS/OCNLI/CMNLI/AFQMC/WSC)·C3.
> 추론·수학: LogiQA2, MathQA, SocialIQA, CosmosQA, FEVER-NLI, GLUE 8종(mrpc/qqp/sst2/cola/qnli/rte/wnli/stsb).
> probe ERR 복구: 13종 중 12종(parquet 미러), 실패는 LogiQA v1·CLUE-iflytek·CLUE-csl·mteb/amazon_reviews_multi.
> 검증: `verify_bulk` → leak=0 / goldsum=0 / dup_in=0 / dup_cross=0 / empty=0. 상세 `data/bulk/W3_report.md`.
> 실행: `--specs-file`는 scripts 기준 상대경로로 해석되므로 절대경로로 지정할 것.

> **W2 실측(2026-09-25)**: `scripts/specs_multilingual.py` (신규 레지스트리, 44 스펙 + 커스텀 어댑터 4종) → `data/bulk/*.jsonl`
> **881,570행**, `state.language` 태그 351종 → 정규화 기준 **206개 언어**. 최대 단일 언어 비중 4.27 %(zh)로 상한 20 % 충족.
> 구성: belebele 109,484(122언어) · sib200 205,820(205언어) · global_mmlu 146,403(42) · massive_scenario 91,800(51) ·
> mmlu_prox 69,620(28) · paws_x 55,565(7) · amazon_reviews_multi 71,721(6, ordinal) · afrisenti 34,178(15) ·
> mlnli 32,489(26) · masakhanews 22,154(16) · xstory_cloze 20,578(11) · include_lite_44 10,716(44) · xcopa 6,600(11) · xwinograd 4,442(6).
> probe 복구: `google/paws-x` 삭제 → `google-research-datasets/paws-x`, `mteb/amazon_reviews_multi` script 제거 → `SetFit/amazon_reviews_multi_*`.
> 제외: `m3exam`(answer 인코딩 불명확), `language-identification`(정답=언어라 state에 누출), `mlqa`/`tydiqa`/`mtop_*`(script-only), xcopa `translation-*`, MMLU-ProX `en`(W1 중복).
> 검증: `verify_bulk` → 전 파일 leak=0 / goldsum=0 / dup_in=0 / dup_cross=0 / empty=0, posdev ≤ 0.111. 상세 `data/bulk/W2_report.md`.

전처리 병목: `preprocess.py`는 단일 프로세스 → 2M행이면 수 시간.
→ 샤드 병렬(`scripts/preprocess_shard.py`, 신규) 후 병합.

### W4 도구 (신규)

| 파일 | 역할 |
|---|---|
| `scripts/build_mix_x10.py` | 소스 전량 결합 + 불변식 강제(state/qid/arity/gold 정규화) + 드롭·분포 통계 |
| `scripts/preprocess_shard.py` | `--shard i --num-shards N` + line-offset 인덱스(`--build-index-only`), `preprocess.build_training_item` import 재사용 |
| `scripts/merge_shards_x10.py` | 샤드 → `train_items_x10.pt` 병합 |
| `scripts/smoke_x10.py` | 병합 파일 무작위 200 + 최장/최단 → 0.9 스냅샷 CPU forward 검사 |
| `scripts/check_eval_leak.py` | public-231 + typed-decisions test 누수 0 하드 게이트 |
| `scripts/posdev_x10.py` | letter/name namespace를 분리한 진짜 위치편차 측정 |
| `scripts/make_x10_manifest.py` | `data/xeron10_mix_manifest.json` 조립 |

## 4. 남은 결정 (주인님)

- 1.0 인코더 확장 폭(hidden/layers/vocab) → 확장 후에는 0.9 가중치의 부분 로드 전략 필요
  (`69aea9f`의 encoder-only load 패턴 재사용).
- 컨텍스트 길이: 현재 4096/헤드 256. 1.0에서 늘릴지 여부.
- 학습 컴퓨트: Kaggle 무료(T4×2, fp16) vs Vast A100(bf16, $0.8/h) — 2M행 × 1 epoch는
  T4×2로 감당이 안 되는 규모.
