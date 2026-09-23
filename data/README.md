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