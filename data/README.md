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