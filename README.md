# XERON 🎯

**XERON** — a decision-model fine-tuning project built on [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya).

> 🎯 **Goal: general-purpose fine-tuning** — build a general-purpose XERON model that covers a wide range of decision tasks (classification / routing / scoring / risk judgment) instead of being limited to a single domain. **English + Korean input + browser use (web decisions + web-agent actions) are all supported.**
> - English track: English checkpoint + [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (4 workflows: invoice / security incidents / customer service / agent-trace)
> - Korean track: multilingual checkpoint + Korean data (KLUE-based 82,344 sequences)
> - Browser track: web page topic classification (BBC/AG News) + spam/phishing detection (28,899 sequences)
> - Web-agent track: next action/element decisions based on [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) (14,724 sequences)

Laya is a Multilingual · Non-autoregressive **System 1 decision model** that takes a state (text/email/ticket/JSON) and typed questions (choice / score / noul) and returns typed answers with calibrated probabilities in a **single forward pass**. It does not generate text, so no parsing is required and there are no hallucinations.

| Item | Value |
|---|---|
| Base model | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large backbone, 421M params) |
| Training method | RLCD (Reinforcement Learning for Calibrated Decisions) — strictly proper scoring rule + GRPO-style baseline |
| License | Apache-2.0 |
| Checkpoints | English (root) / multilingual / typed-decisions |

## 🗂 Checkpoints

| Checkpoint | Backbone | Params | Context | Purpose |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large | 421M | 512 | English |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages (**including Korean**) |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions workflows |

## 🌐 Language Tracks

| Track | Base | Data | Sequences | items file |
|---|---|---|---|---|
| **XERON-EN** | `laya` (English) | typed-decisions train | 6,000 | `train_items.pt` |
| **XERON-KR** | `laya-multilingual` | Korean KLUE (ynat 45,678 + nli 24,998 + sts 11,668) | 82,344 | `train_items_kr.pt` |
| **XERON-MIX** 🎯 | `laya-multilingual` | EN typed-decisions + KR KLUE | 88,344 | `train_items_mix.pt` |
| **XERON-BROWSE** | `laya-multilingual` | web page classification (BBC 1,225 + AG 20,000) + spam 5,574 + phishing 2,100 | 28,899 | `train_items_browser.pt` |
| **XERON-WEBAGENT** | `laya-multilingual` | Mind2Web next action (op) + element selection (7,362 steps) | 14,724 | `train_items_webagent.pt` |
| **XERON-LONG** 🧾 | `laya-multilingual` | **long-context**: SCOTUS Supreme Court opinion justice classification (5,000) + 20 Newsgroups (11,314) | 16,314 | `train_items_long4096.pt` |
| **XERON-ALL** 🏆 | `laya-multilingual` | EN + KR + Browser mixture | 117,243 | `train_items_all.pt` |
| **XERON-ALL-v2** 🚀 | `laya-multilingual` | EN + KR + Browser + WebAgent, all combined | 131,967 | `train_items_all_v2.pt` |
| **XERON-ALL-4096+LONG** 🏆 | `laya-multilingual` | 4096-preprocessed ALL + long-context (SCOTUS/Newsgroups) — **22% uses the 4096 trunk** | 148,281 | `train_items_all_multi4096_long.pt` |

> 🎯 **The default training run is XERON-ALL-v2**: fine-tuning the multilingual base on English + Korean + browser + web-agent data yields a single general-purpose model that handles everything from general decisions to web-browser action decisions.
> Laya's `Router` detects the language automatically, so shipping the English and multilingual checkpoints together also enables per-language routing.

## 🏗 Fine-tuning pipeline

```
Dataset (HF dataset / JSON Lines)
        │
        ▼
scripts/preprocess.py ──► train_items.pt (tokenize + option markers + target distribution)
        │
        ▼
scripts/train_ddp.py  ──► torchrun DDP (RLCD policy gradient + soft CE guidance)
        │                       + post-hoc temperature calibration
        ▼
   output/ (model.safetensors, encoder/, tokenizer/, rl_agent_config.json)
        │
        ├──► scripts/evaluate.py   (benchmarks: accuracy, Brier, ECE, score MAE)
        └──► scripts/upload_hf.py  (upload to HuggingFace Hub)
```

### Current readiness (2026-09-20)

| Item | Status |
|---|---|
| GitHub repo | ✅ PIXELZX0/XERON (public) |
| Local venv + dependencies | ✅ `.venv` (laya 0.3.3 / transformers 5.17 / torch 2.14) |
| Base models (EN + Multilingual) | ✅ `/home/yuchan/laya-models/laya-base` (2.3GB) |
| English data preprocessing | ✅ `train_items.pt` — 6,000 seq (typed-decisions) |
| Korean data build | ✅ `data/korean_typed.jsonl` — 82,344 rows (KLUE ynat/nli/sts) |
| Korean preprocessing | ✅ `train_items_kr.pt` — 82,344 seq |
| MIX preprocessing (EN+KR) | ✅ `train_items_mix.pt` — 88,344 seq |
| Browser data build | ✅ `data/browser_typed.jsonl` — 28,899 rows (web classification BBC/AG + spam/phishing) |
| Browser preprocessing | ✅ `train_items_browser.pt` — 28,899 seq |
| Mind2Web web agent | ✅ `data/mind2web_typed.jsonl` — 14,724 rows (next action/element selection) |
| Web-agent preprocessing | ✅ `train_items_webagent.pt` — 14,724 seq |
| ALL preprocessing (EN+KR+Browser) | ✅ `train_items_all.pt` — 117,243 seq |
| ALL-v2 (EN+KR+Browser+WebAgent) 🚀 | ✅ `train_items_all_v2.pt` — 131,967 seq |
| Actual training | ⏳ GPU required (EN alone ~4–6 min on 2×T4; MIX/ALL take tens of minutes to a few hours) |

### XERON-MIX (recommended — general-purpose English + Korean)

```bash
# 1) Build the Korean dataset (KLUE) — produces an 82,344-row JSONL
python scripts/build_korean_dataset.py --output data/korean_typed.jsonl

# 2) Preprocess English + Korean with the multilingual tokenizer
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --dataset LocalLLaMA/typed-decisions --config-name all --split train \
  --output train_items_en_multi.pt
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --data-files data/korean_typed.jsonl --output train_items_kr.pt

# 3) Merge (optional)
python -c "import torch; torch.save(torch.load('train_items_en_multi.pt')+torch.load('train_items_kr.pt'),'train_items_mix.pt')"

# 4) Train MIX (2× GPU; EPOCHS=2 recommended given the data size)
EPOCHS=2 torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  /home/yuchan/laya-models/laya-base/multilingual \
  ./output/xeron-mix \
  train_items_mix.pt
```

### XERON-ALL (recommended — English + Korean + browser)**

```bash
# 1) Build the browser-use dataset — 28,899-row JSONL (web page classification + spam/phishing detection)
python scripts/build_browser_dataset.py --output data/browser_typed.jsonl

# 2) Preprocess the browser data (multilingual tokenizer)
python scripts/preprocess.py \
  --model-id /home/yuchan/laya-models/laya-base/multilingual \
  --data-files data/browser_typed.jsonl --output train_items_browser.pt

# 3) Full mixture: EN + KR + Browser
python -c "import torch; a=torch.load('train_items_en_multi.pt'); b=torch.load('train_items_kr.pt'); c=torch.load('train_items_browser.pt'); torch.save(a+b+c,'train_items_all.pt')"

# 4) Train ALL (EPOCHS=1~2 recommended — 117K sequences)
EPOCHS=2 torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  /home/yuchan/laya-models/laya-base/multilingual \
  ./output/xeron-all \
  train_items_all.pt
```

### Requirements

- **2× GPU or more** (reference: full training ~4–6 min on 2×T4)
- Python ≥ 3.10, PyTorch ≥ 2.x, CUDA environment

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare data

Default benchmark dataset: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

To use a custom dataset, follow the format in `data/README.md` (Laya native format: `state` + `questions` + `gold`).

```bash
python scripts/preprocess.py \
  --dataset LocalLLaMA/typed-decisions \
  --config-name all \
  --split train \
  --output train_items.pt
```

### 3) Training (DDP)

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/train_ddp.py \
  convaiinnovations/laya \
  ./output/xeron \
  ./train_items.pt
```

Hyperparameters can be tuned in `configs/finetune.json` or overridden via environment variables (`EPOCHS`, `MICRO_BATCH`, `GRAD_ACCUM`, `LR_ENCODER`, ...).

### Training environment variables (parameter scaling)

| Variable | Default | Description |
|---|---|---|
| `EPOCHS` | 4 | training epochs |
| `MICRO_BATCH` | 8 | forward batch per GPU |
| `GRAD_ACCUM` | 4 | gradient accumulation — effective batch = MICRO_BATCH × #GPUs × GRAD_ACCUM |
| `GROUP_SIZE` | 4 | number of GRPO baseline samples |
| `LR_ENCODER` / `LR_HEAD` | 2.5e-5 / 1e-4 | encoder/head learning rate |
| `SIGMA_START` / `SIGMA_END` | 0.4 / 0.1 | exploration noise |
| `CHECKPOINT_EVERY` | 0 | save a checkpoint every N epochs (use 1 for Colab etc. to survive session drops) |
| `RESUME` | (none) | checkpoint path or `auto` (latest checkpoint in output_dir) to resume training |
| `MAX_LEN` | 2048 | **actual training context** — the default limit is automatically raised by preprocess/train up to **32,768 (4096×8)** |
| `HEAD_MAX_LEN` | 256 | decision-head marker window |
| `MAX_TOKENS_BATCH` | 4096 | max tokens per micro-batch (memory ceiling — lower MICRO_BATCH if context ↑) |
| `CTX_CAP` | 32768 | encoder `max_position_embeddings` cap (RoPE, 4096×8) |
| `DTYPE` | fp16 | fp16 (T4/V100) · **bf16 (A100/H100 — native, no GradScaler needed)** |

### 🚀 Recommended settings per GPU (MAX_LEN=4096, 148K long-context rows, effective batch 64)

| GPU | Memory | DTYPE | MICRO_BATCH | GRAD_ACCUM | MAX_TOKENS_BATCH | Est. per epoch |
|---|---|---|---|---|---|---|
| T4 (Colab free) | 15GB | fp16 | 2 | 32 | 8192 | ~3–4 h |
| **A100 (Colab Pro)** | **40GB** | **bf16** | **8** | **8** | **16384** | **~30–60 min** |
| A100 80GB | 80GB | bf16 | 16 | 4 | 32768 | ~25–45 min |
| H100 | 80GB | bf16 | 16–24 | 2–4 | 32768 | ~15–30 min |

> A100 is **3–5× faster than T4** thanks to native bf16 and more memory. For MAX_LEN=8192 ultra-long-context
> experiments, use MICRO_BATCH=4, MAX_TOKENS_BATCH=16384 on A100.

### 📏 How to extend context

0. **The default limit is raised automatically**: on startup, `preprocess.py` and `train_ddp.py` call `ctx_extend.ensure_long_context()` to raise the encoder `max_position_embeddings` up to **32,768** and set `rl_agent_config.max_len` to the default (idempotent).
   Use the `CTX_CAP` env var to adjust the cap.
1. **Use the same `MAX_LEN` in both preprocessing and training** (the tokenized length must not change):
   ```bash
   MAX_LEN=8192 python scripts/preprocess.py --model-id <base> --data-files data.jsonl --output items.pt
   MAX_LEN=8192 torchrun --standalone --nproc_per_node=1 scripts/train_ddp.py <base> ./output/xeron items.pt
   ```
2. **RoPE-based bases recommended** (both multilingual/english are ModernBERT): there is no position-embedding
   table, so 2048/4096/8192/32768 all work without any architecture change (4,000-token forward verified)
3. **Memory trade-off**: 2× sequence length ≈ 4× attention memory → halve `MICRO_BATCH` (e.g. 8→4) and adjust `MAX_TOKENS_BATCH`
4. Changing only the `MAX_LEN` entry in the Colab notebook's ⚙️ settings propagates through the whole pipeline automatically

```bash
# e.g. save every epoch (session-drop resilient) and resume-capable training
CHECKPOINT_EVERY=1 RESUME=auto torchrun ...
```

## 🚀 Google Colab fine-tuning (T4 1-GPU)

Notebook ready to run on a free T4: **`notebooks/XERON_finetune_colab.ipynb`**
([Open in Colab](https://colab.research.google.com/github/PIXELZX0/XERON/blob/main/notebooks/XERON_finetune_colab.ipynb))

1. **Prepare data (one of 3 paths)**:
   - ☁️ **S3-compatible (recommended)**: upload data with `scripts/upload_s3.py` → register `S3_ENDPOINT/S3_BUCKET/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY` in Colab Secrets
   - 📁 Drive: upload `train_items_all_v2.pt` (117MB) into the `xeron/` folder
   - 🔧 Rebuild from JSONL in Colab (fallback path)
2. Run notebook cells 1–6 (GPU check → install → download model → obtain data → configure)
3. Cell 7: run training (single-GPU GRAD_ACCUM scaling, checkpoint every epoch)
4. Cell 8: evaluate + copy to Drive · Cell 9: upload to HF

**1-GPU parameter-scaling experiment matrix** (same basis as the original 2×T4 effective batch 64):

| Experiment | EPOCHS | GRAD_ACCUM | Effective batch | Est. time (T4) |
|---|---|---|---|---|
| A (quick validation) | 1 | 4 | 32 | ~30–45 min |
| **B (recommended)** | **2** | **8** | **64** | ~1.5–3 h |
| C (thorough) | 3 | 8 | 64 | Pro recommended |

> 💡 If the Colab session drops: rerun the top cells, then in the training cell set `RESUME="auto"` and run again to resume from the last epoch checkpoint.

### 4) Evaluation

```bash
python scripts/evaluate.py \
  --model ./output/xeron \
  --dataset LocalLLaMA/typed-decisions \
  --split test
```

### 5) HuggingFace upload

```bash
HF_TOKEN=... python scripts/upload_hf.py \
  --model-dir ./output/xeron \
  --repo-id PIXELZX/XERON
```

## 📄 License

- Base model: Apache-2.0 (commercial use allowed)
- This repo's code: Apache-2.0
