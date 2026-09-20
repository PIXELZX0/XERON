# XERON 🎯

**XERON** — [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 기반 파인튜닝 결정(decision) 모델 프로젝트.

Laya는 Multilingual · Non-autoregressive **System 1 decision model**로, 상태(text/email/ticket/JSON)와 타입이 지정된 질문(choice / score / noul)을 받아 **단일 forward pass**로 타입화된 답변과 보정된 확률을 반환합니다. 텍스트를 생성하지 않으므로 파싱이 필요 없고 할루시네이션이 없습니다.

| 항목 | 값 |
|---|---|
| Base 모델 | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large backbone, 421M params) |
| 학습 방식 | RLCD (Reinforcement Learning for Calibrated Decisions) — strictly proper scoring rule + GRPO-style baseline |
| 라이선스 | Apache-2.0 |
| 체크포인트 | English (root) / multilingual / typed-decisions |

## 🗂 체크포인트

| 체크포인트 | Backbone | Params | Context | 용도 |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large | 421M | 512 | English |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions workflows |

## 🏗 파인튜닝 파이프라인

```
데이터셋 (HF dataset / JSON Lines)
        │
        ▼
scripts/preprocess.py ──► train_items.pt (토큰화 + 옵션 마커 + target 분포)
        │
        ▼
scripts/train_ddp.py  ──► torchrun DDP (RLCD policy gradient + soft CE guidance)
        │                       + 사후 temperature calibration
        ▼
   output/ (model.safetensors, encoder/, tokenizer/, rl_agent_config.json)
        │
        ├──► scripts/evaluate.py   (벤치마크: accuracy, Brier, ECE, score MAE)
        └──► scripts/upload_hf.py  (HuggingFace Hub 업로드)
```

### 요구 환경

- **GPU 2x 이상** (참고: T4 x2 기준 전체 학습 ~4–6분)
- Python ≥ 3.10, PyTorch ≥ 2.x, CUDA 환경

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

### 2) 데이터 준비

기본 벤치마크 데이터셋: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

커스텀 데이터셋을 쓰려면 `data/README.md`의 포맷을 따르세요 (Laya 네이티브 format: `state` + `questions` + `gold`).

```bash
python scripts/preprocess.py \
  --dataset LocalLLaMA/typed-decisions \
  --config-name all \
  --split train \
  --output train_items.pt
```

### 3) 학습 (DDP)

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  convaiinnovations/laya \
  ./output/xeron \
  ./train_items.pt
```

하이퍼파라미터는 `configs/finetune.json`에서 조정하거나 환경변수(`EPOCHS`, `MICRO_BATCH`, `GRAD_ACCUM`, `LR_ENCODER` ...)로 오버라이드할 수 있습니다.

### 4) 평가

```bash
python scripts/evaluate.py \
  --model ./output/xeron \
  --dataset LocalLLaMA/typed-decisions \
  --split test
```

### 5) HuggingFace 업로드

```bash
HF_TOKEN=... python scripts/upload_hf.py \
  --model-dir ./output/xeron \
  --repo-id PIXELZX/XERON
```

## 📄 라이선스

- Base 모델: Apache-2.0 (상업 이용 가능)
- 본 레포 코드: Apache-2.0