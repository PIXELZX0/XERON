# XERON 🎯

**XERON** — [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 기반 파인튜닝 결정(decision) 모델 프로젝트.

> 🎯 **목표: 범용(general-purpose) 파인튜닝** — 특정 도메인에 한정하지 않고 다양한 결정 태스크(분류/라우팅/스코어링/위험 판단)를 커버하는 범용 XERON 모델을 만든다. **영어 + 한국어 입력 + 브라우저 사용(웹 결정 + 웹 에이전트 액션)까지 지원**.
> - 영어 트랙: English 체크포인트 + [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (4개 workflow: invoice / security incidents / customer service / agent-trace)
> - 한국어 트랙: 멀티링궈얼 체크포인트 + 한국어 데이터 (KLUE 기반 82,344 시퀀스)
> - 브라우저 트랙: 웹 페이지 주제 분류(BBC/AG News) + 스팸/피싱 탐지 (28,899 시퀀스)
> - 웹 에이전트 트랙: [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) 기반 다음 액션/요소 결정 (14,724 시퀀스)

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
| **XERON-WEBAGENT** | `laya-multilingual` | Mind2Web 다음 액션(op) + 요소 선택 (7,362 스텝) | 14,724 | `train_items_webagent.pt` |
| **XERON-ALL** 🏆 | `laya-multilingual` | EN + KR + Browser 혼합 | 117,243 | `train_items_all.pt` |
| **XERON-ALL-v2** 🚀 | `laya-multilingual` | EN + KR + Browser + WebAgent 전체 | 131,967 | `train_items_all_v2.pt` |

> 🎯 **기본 학습은 XERON-ALL-v2**: 멀티링궈얼 베이스에 영어+한국어+브라우저+웹 에이전트 데이터로 파인튜닝하면 일반 결정부터 웹 브라우저 액션 결정까지 처리하는 단일 범용 모델이 됩니다.
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
| Mind2Web 웹 에이전트 | ✅ `data/mind2web_typed.jsonl` — 14,724행 (다음 액션/요소 선택) |
| 웹 에이전트 전처리 | ✅ `train_items_webagent.pt` — 14,724 seq |
| ALL 전처리 (EN+KR+Browser) | ✅ `train_items_all.pt` — 117,243 seq |
| ALL-v2 (EN+KR+Browser+WebAgent) 🚀 | ✅ `train_items_all_v2.pt` — 131,967 seq |
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

### 학습 환경변수 (파라미터 확장)

| 변수 | 기본 | 설명 |
|---|---|---|
| `EPOCHS` | 4 | 학습 에폭 |
| `MICRO_BATCH` | 8 | GPU당 1회 forward 배치 |
| `GRAD_ACCUM` | 4 | 그래디언트 누적 — 유효 배치 = MICRO_BATCH × GPU 수 × GRAD_ACCUM |
| `GROUP_SIZE` | 4 | GRPO baseline 샘플 수 |
| `LR_ENCODER` / `LR_HEAD` | 2.5e-5 / 1e-4 | 인코더/헤드 학습률 |
| `SIGMA_START` / `SIGMA_END` | 0.4 / 0.1 | 탐색 노이즈 |
| `CHECKPOINT_EVERY` | 0 | N 에폭마다 체크포인트 저장 (Colab 등 세션 끊김 대비: 1) |
| `RESUME` | (없음) | 체크포인트 경로 또는 `auto`(output_dir 최신 체크포인트)로 이어서 학습 |
| `MAX_LEN` | 1024 | **컨텍스트 길이** — 인코더 한도 8192 (ModernBERT RoPE). English 베이스 기본 512 → 2048/4096 확장 가능 |
| `HEAD_MAX_LEN` | 256 | 결정 헤드 마커 윈도우 |
| `MAX_TOKENS_BATCH` | 4096 | 마이크로배치당 최대 토큰 (메모리 상한 — 컨텍스트↑면 MICRO_BATCH↓) |

### 📏 컨텍스트 확장 방법

1. **전처리와 학습 양쪽에서 동일한 `MAX_LEN` 사용** (토크나이즈 길이가 달라지면 안 됨):
   ```bash
   MAX_LEN=2048 python scripts/preprocess.py --model-id <영어베이스> --data-files data.jsonl --output items.pt
   MAX_LEN=2048 torchrun --standalone --nproc_per_node=1 scripts/train_ddp.py <영어베이스> ./output/xeron items.pt
   ```
2. **English 베이스 권장** (`layaroot`, 512→무제한 확장): ModernBERT는 RoPE 기반이라 2048/4096/8192 모두 동작
3. **메모리 트레이드오프**: 시퀀스 2배 = attention 메모리 약 4배 → `MICRO_BATCH`를 반으로 (예: 8→4), `MAX_TOKENS_BATCH` 조정
4. Colab 노트북 ⚙️ 설정의 `MAX_LEN` 항목만 바꾸면 전 과정 자동 반영

```bash
# 예: 세션 끊김 대비 1 에폭마다 저장 + 재개 가능한 학습
CHECKPOINT_EVERY=1 RESUME=auto torchrun ...
```

## 🚀 Google Colab 파인튜닝 (T4 1-GPU)

무료 T4에서 돌릴 수 있도록 준비된 노트북: **`notebooks/XERON_finetune_colab.ipynb`**
([Colab에서 열기](https://colab.research.google.com/github/PIXELZX0/XERON/blob/main/notebooks/XERON_finetune_colab.ipynb))

1. **데이터 준비 (3경로 중 1)**:
   - ☁️ **S3 호환 (권장)**: `scripts/upload_s3.py`로 데이터 업로드 → Colab Secrets에 `S3_ENDPOINT/S3_BUCKET/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY` 등록
   - 📁 Drive: `train_items_all_v2.pt`(117MB)를 `xeron/` 폴더에 업로드
   - 🔧 Colab에서 JSONL 재구성 (백업 경로)
2. 노트북 셀 1~6 실행 (GPU 확인 → 설치 → 모델 다운로드 → 데이터 확보 → 설정)
3. 셀 7: 학습 실행 (1 GPU 전용 GRAD_ACCUM 확장, 에폭마다 체크포인트)
4. 셀 8: 평가 + Drive 복사 · 셀 9: HF 업로드

**1-GPU 파라미터 확장 실험 매트릭스** (원본 2×T4 유효배치 64와 동일 기준):

| 실험 | EPOCHS | GRAD_ACCUM | 유효배치 | 예상 시간 (T4) |
|---|---|---|---|---|
| A (빠른 검증) | 1 | 4 | 32 | ~30–45분 |
| **B (권장)** | **2** | **8** | **64** | ~1.5–3시간 |
| C (정밀) | 3 | 8 | 64 | Pro 권장 |

> 💡 Colab 세션이 끊기면: 상단 셀 재실행 후 학습 셀에서 `RESUME="auto"`로 바꿔 실행하면 마지막 에폭 체크포인트부터 이어집니다.

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