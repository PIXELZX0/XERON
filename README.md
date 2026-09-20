# XERON 🎯

**XERON** — [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 기반 파인튜닝 결정(decision) 모델 프로젝트.

> 🎯 **목표: 범용(general-purpose) 파인튜닝** — 특정 도메인에 한정하지 않고 다양한 결정 태스크(분류/라우팅/스코어링/위험 판단)를 커버하는 범용 XERON 모델을 만든다. **영어 + 한국어 입력 + 브라우저 사용(웹 결정)까지 지원**.
> - 영어 트랙: English 체크포인트 + [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (4개 workflow: invoice / security incidents / customer service / agent-trace)
> - 한국어 트랙: 멀티링궈얼 체크포인트 + 한국어 데이터 (KLUE 기반 82,344 시퀀스)
> - 브라우저 트랙: 웹 페이지 주제 분류(BBC/AG News) + 스팸/피싱 탐지 (28,899 시퀀스)

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
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages (**한국어 포함**) |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions workflows |

## 🌐 학습 트랙 (Language Tracks)

| 트랙 | 베이스 | 데이터 | 시퀀스 | items 파일 |
|---|---|---|---|---|
| **XERON-EN** | `laya` (English) | typed-decisions train | 6,000 | `train_items.pt` |
| **XERON-KR** | `laya-multilingual` | 한국어 KLUE (ynat 45,678 + nli 24,998 + sts 11,668) | 82,344 | `train_items_kr.pt` |
| **XERON-MIX** 🎯 | `laya-multilingual` | EN typed-decisions + KR KLUE | 88,344 | `train_items_mix.pt` |
| **XERON-BROWSE** | `laya-multilingual` | 웹 페이지 분류(BBC 1,225 + AG 20,000) + 스팸 5,574 + 피싱 2,100 | 28,899 | `train_items_browser.pt` |
| **XERON-ALL** 🏆 | `laya-multilingual` | EN + KR + Browser 전체 혼합 | 117,243 | `train_items_all.pt` |

> 🎯 **기본 학습은 XERON-MIX 또는 XERON-ALL**: 멀티링궈얼 베이스에 영어+한국어(+브라우저) 혼합 데이터로 파인튜닝하면 모든 입력을 처리하는 단일 범용 모델이 됩니다.
> Laya `Router`는 언어를 자동 감지하므로, English/멀티링궈얼 체크포인트를 함께 배포하면 언어별 라우팅도 가능합니다.

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

### 현재 준비 상태 (2026-09-20)

| 항목 | 상태 |
|---|---|
| GitHub 레포 | ✅ PIXELZX0/XERON (public) |
| 로컬 venv + 의존성 | ✅ `.venv` (laya 0.3.3 / transformers 5.17 / torch 2.14) |
| 베이스 모델 (EN + Multilingual) | ✅ `/home/yuchan/laya-models/laya-base` (2.3GB) |
| 영어 데이터 전처리 | ✅ `train_items.pt` — 6,000 seq (typed-decisions) |
| 한국어 데이터 구축 | ✅ `data/korean_typed.jsonl` — 82,344행 (KLUE ynat/nli/sts) |
| 한국어 전처리 | ✅ `train_items_kr.pt` — 82,344 seq |
| MIX 전처리 (EN+KR) | ✅ `train_items_mix.pt` — 88,344 seq |
| 브라우저 데이터 구축 | ✅ `data/browser_typed.jsonl` — 28,899행 (웹 분류 BBC/AG + 스팸/피싱) |
| 브라우저 전처리 | ✅ `train_items_browser.pt` — 28,899 seq |
| ALL 전처리 (EN+KR+Browser) | ✅ `train_items_all.pt` — 117,243 seq |
| 실제 학습 | ⏳ GPU 필요 (T4 x2 기준 EN만 ~4–6분, MIX/ALL은 수십 분~시간 내외) |

### XERON-MIX (권장 — 영어 + 한국어 범용)

```bash
# 1) 한국어 데이터셋 빌드 (KLUE) — 82,344행 JSONL 생성
python scripts/build_korean_dataset.py --output data/korean_typed.jsonl

# 2) 영어 + 한국어를 multilingual 토크나이저로 전처리
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --dataset LocalLLaMA/typed-decisions --config-name all --split train \
  --output train_items_en_multi.pt
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --data-files data/korean_typed.jsonl --output train_items_kr.pt

# 3) 병합 (선택)
python -c "import torch; torch.save(torch.load('train_items_en_multi.pt')+torch.load('train_items_kr.pt'),'train_items_mix.pt')"

# 4) MIX 학습 (GPU 2x, 데이터 규모상 EPOCHS=2 권장)
EPOCHS=2 torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  /home/yuchan/laya-models/laya-base/multilingual \
  ./output/xeron-mix \
  train_items_mix.pt
```

### XERON-ALL (권장 — 영어 + 한국어 + 브라우저)**

```bash
# 1) 브라우저 사용 데이터셋 빌드 — 28,899행 JSONL (웹 페이지 분류 + 스팸/피싱 탐지)
python scripts/build_browser_dataset.py --output data/browser_typed.jsonl

# 2) browser 데이터 전처리 (multilingual 토크나이저)
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --data-files data/browser_typed.jsonl --output train_items_browser.pt

# 3) 전체 혼합: EN + KR + Browser
python -c "import torch; a=torch.load('train_items_en_multi.pt'); b=torch.load('train_items_kr.pt'); c=torch.load('train_items_browser.pt'); torch.save(a+b+c,'train_items_all.pt')"

# 4) ALL 학습 (EPOCHS=1~2 권장 — 117K 시퀀스)
EPOCHS=2 torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  /home/yuchan/laya-models/laya-base/multilingual \
  ./output/xeron-all \
  train_items_all.pt
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