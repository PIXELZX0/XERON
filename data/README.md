# XERON 데이터셋 가이드 📊

## 기본 (벤치마크) 데이터셋

- [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — 1,200 train / 400 test 케이스, 2,000 decisions
- 4개 workflow: invoice processing, security incidents, customer service, agent-trace observability

## 커스텀 데이터셋 포맷 (Laya 네이티브)

각 행(row)은 다음 3개 컬럼을 가져야 합니다 (`datasets.Dataset` 또는 JSON Lines):

```json
{
  "id": "case-0001",
  "workflow": "customer-service",
  "state": "{\"subject\": \"Duplicate charge on invoice 4411\", \"body\": \"We were billed twice for March.\"}",
  "questions": "{\"department\": {\"type\": \"choice\", \"instructions\": \"Which team should handle this?\", \"criteria\": {\"billing\": \"invoices, payments, refunds\", \"technical\": \"bugs and outages\"}}, \"urgency\": {\"type\": \"score\", \"instructions\": \"How urgent is this?\", \"criteria\": [\"not urgent\", \"soon\", \"blocking\"]}, \"churn_risk\": {\"type\": \"noul\", \"instructions\": \"Does the user threaten to cancel?\"}}",
  "gold": "{\"department\": {\"probabilities\": {\"billing\": 0.9, \"technical\": 0.1}}, \"urgency\": {\"probabilities\": {\"0\": 0.1, \"1\": 0.3, \"2\": 0.6}}, \"churn_risk\": {\"probabilities\": {\"false\": 0.7, \"true\": 0.3}}}"
}
```

### 질문 타입

| type | 설명 | criteria |
|---|---|---|
| `choice` | 다중 옵션 중 선택 | dict: `{option_key: description}` |
| `score` | 서수 척도 (0..n) | list 또는 dict of levels |
| `noul` | 보정된 확률 P(true) | - |

### gold 확률 규칙

- `choice`: 각 옵션 키에 확률 합계 1.0
- `score`: `"0"`, `"1"`, ... 문자열 키로 확률
- `noul`: `{"false": p, "true": 1-p}`

## 자기 주도 수집 옵션
1. **합성 데이터 생성**: LLM으로 상태-질문-골드 분포 생성 후 `scripts/preprocess.py`에 바로 투입
2. **실제 결정 로그**: 기존 운영 라우팅/분류 로그를 위 포맷으로 변환
3. **Human-in-the-loop**: 라벨러가 확률 분포로 주석 (보정 학습에 가장 효과적)

> 💡 XERON 파인튜닝용 데이터셋 확정되면 `data/` 아래에 넣고 README를 갱신하세요.

## 2026-09-24 추가: 지식·추론·라우팅 소스 4종

JevBench 약점(routing/intent, 지식·추론 choice) 보강용. 전부 `choice` 타입, 정답 one-hot,
seed 7 고정(`random.Random(7)`)으로 재현 가능. 빌드 스크립트는 `scripts/build_*_dataset.py`.

| 파일 | 행 수 | workflow | 원본 |
|---|---|---|---|
| `data/clinc_typed.jsonl` | 20,000 | `clinc150-intent` | `clinc/clinc_oos` (`plus`, train+val+test) |
| `data/mmlu_typed.jsonl` | 15,000 | `mmlu-choice` | `cais/mmlu` (`all`, test+validation) |
| `data/hellaswag_typed.jsonl` | 10,000 | `hellaswag-continuation` | `Rowan/hellaswag` (validation) |
| `data/arc_typed.jsonl` | 3,000 | `arc-choice` | `allenai/ai2_arc` (Challenge val/test + Easy all) |
| `data/clinc_hard_typed.jsonl` | 20,000 | `clinc150-intent` | 위 CLINC의 **선택적** hard-negative 변형 |

빌드:

```bash
python scripts/build_clinc_dataset.py       --output data/clinc_typed.jsonl --target 20000
python scripts/build_mmlu_dataset.py        --output data/mmlu_typed.jsonl --target 15000
python scripts/build_hellaswag_dataset.py   --output data/hellaswag_typed.jsonl --target 10000
python scripts/build_arc_dataset.py         --output data/arc_typed.jsonl --target 3000
```

검증(행 수 / one-hot·확률합 전수 검사 / 정답 누수 정규식 / 기존 데이터와 state 중복):

```bash
python scripts/verify_new_datasets.py data/clinc_typed.jsonl data/mmlu_typed.jsonl \
    data/hellaswag_typed.jsonl data/arc_typed.jsonl
```

주의사항:

- **중복 회피**: MMLU `dev`(jevbench-mmlu)와 ARC-Challenge `train`(jevbench-arc_challenge)은
  의도적으로 제외했다. 기존 `data/*.jsonl`과 state 중복 0.
- **CLINC150 어휘 누수**: gold intent 이름의 토큰이 발화문에 그대로 등장하는 비율이 63%다
  (무작위 오답 기준 2.1%). CLINC150 고유의 성질이라 제거 불가. `--distractor-mode hard`로
  오답을 이름 토큰이 겹치는 intent로 뽑으면 오답 쪽 어휘 겹침이 4.6%로 올라간다(부분 완화).
- 라벨(A..F)은 행마다 무작위 순열이라 정답 위치에 편향이 없다.
## 2026-09-24 추가 2: JevBench 누락 config 복구 (5종)

`Praveenrajus/jev-bench`(22 config / 133,953 train행) 중 기존 `jevbench_full.jsonl`
(17 config / 101,953행)에서 빠진 5종을 복구. 누락 사유는 두 갈래였다:

- **banking77 / clinc150 / massive / ledgar** — 원본 빌더가 옵션 수 초과(k=77/151/60/100)로
  `SKIP_LARGE_K`에서 스킵 → 정확히 4×8,000 = **32,000행**이 사라진 원인. 소규모 arity
  choice(정답 + 무작위 오답)로 재구성.
- **chaosnli** — `test` split만 존재(train 없음)라 `--split train` 기본값에 걸려 누락.
  기존 빌더를 `--split test`로 실행해 그대로 복구(soft label 유지).

| 파일 | 행 수 | workflow | 원본 | 옵션 수 |
|---|---|---|---|---|
| `data/jevbench_extra_chaosnli.jsonl` | 1,599 | `jevbench-chaosnli` | `metaeval/chaos-mnli-ambiguity` (test) | 3 |
| `data/jevbench_extra_banking77.jsonl` | 5,092 | `jevbench-banking77` | `mteb/banking77` | 4–6 |
| `data/jevbench_extra_clinc150.jsonl` | 4,954 | `jevbench-clinc150` | `clinc/clinc_oos` (plus) | 4–6 |
| `data/jevbench_extra_massive.jsonl` | 6,022 | `jevbench-massive` | `mteb/amazon_massive_intent` (en) | 4–6 |
| `data/jevbench_extra_ledgar.jsonl` | 4,439 | `jevbench-ledgar` | `coastalcph/lex_glue` (ledgar) | 8–12 |

빌드 (seed 7 고정, 재현 가능 — 재실행 시 sha256 동일):

```bash
python scripts/build_jevbench_dataset.py --configs chaosnli --split test \
    --output data/jevbench_extra_chaosnli.jsonl
python scripts/build_jevbench_extra.py --outdir data            # banking77/clinc150/massive/ledgar
python scripts/verify_jevbench_extra.py "data/jevbench_extra_*.jsonl"
```

**어휘 shortcut 통제 (핵심)**: intent/topic config는 라벨 이름이 옵션 텍스트에 들어가고
그 단어가 발화문에 자주 그대로 등장한다. 통제 전 gold-vs-오답 이름 겹침 비율:

| config | gold | 무작위 오답 | 비율 |
|---|---|---|---|
| banking77 | 0.80 | 0.16 | 5.1× |
| clinc150 | 0.65 | 0.035 | 18.6× |
| massive | 0.36 | 0.014 | 26.0× |
| ledgar | 0.50 | 0.041 | 12.1× |

→ 오답 후보를 **gold와 동일한 겹침 버킷**(`min(state에 등장하는 라벨 토큰 수, 2)`)에서만
추출. 모든 옵션이 같은 버킷이라 "겹치는 옵션을 고른다"는 shortcut이 성립하지 않는다.
통제 후 비율은 전부 **1.01–1.16×** (binary·count 모두). 같은 버킷 오답이 3개(ledgar 7개)
미만인 행은 버렸다.

검증(`scripts/verify_jevbench_extra.py`): 전 행 `choice`, 확률합 1.0, 라벨 2–12개,
정답 누수 정규식 0, 금지 state 키 0, 파일 내/파일 간 중복 0, 기존 `data/*.jsonl`
(jevbench_full·xeron5_mix 포함)과 완전 중복 0. Laya head 예산 검증: 22,106행 전부
preprocess에서 `markers == k` 통과, 최대 marker 위치 116 ≪ `head_max_len=256`.

> 참고: 기존 `clinc_typed.jsonl`/`clinc_hard_typed.jsonl`과 발화문은 겹칠 수 있으나 state
> 포맷이 달라 완전 중복은 0. hard-negative 변형(오답 겹침 4.6%, 13×)보다 본 파일이
> shortcut 통제가 강하다.
