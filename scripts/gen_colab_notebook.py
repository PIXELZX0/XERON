#!/usr/bin/env python3
"""Generate notebooks/XERON_finetune_colab.ipynb (Colab T4 1-GPU fine-tuning)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks", "XERON_finetune_colab.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(
"""# 🎯 XERON — Colab 파인튜닝 (T4 1-GPU, 하이퍼파라미터 확장)

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) multilingual 체크포인트를
**XERON-ALL-v2 데이터 (131,967 시퀀스: EN 6,000 + KR 82,344 + Browser 28,899 + WebAgent 14,724)**로 파인튜닝합니다.

- 런타임: **GPU (T4 권장, A100/T4 Pro도 동작)** — 메뉴 → 런타임 → 런타임 유형 변경
- 1 GPU용으로 GRAD_ACCUM을 확장해 2xT4(64 유효배치)와 동일한 유효 배치를 유지합니다
- 체크포인트를 에폭마다 저장하므로 **세션이 끊겨도 재개**할 수 있습니다

> ⚠️ 무료 T4 세션은 런타임이 끊길 수 있습니다. 완료 후 반드시 결과를 Drive로 복사하세요.
"""))

cells.append(code(
"""# 1) GPU 확인
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"({p.total_memory/1e9:.1f} GB)")
"""))

cells.append(code(
"""# 2) 의존성 설치
!pip install -q "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate
import laya, transformers, torch
print("laya", laya.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
"""))

cells.append(code(
"""# 3) XERON 레포 (학습/평가 스크립트) 가져오기
!git clone --depth 1 https://github.com/PIXELZX0/XERON.git /content/XERON 2>/dev/null || (cd /content/XERON && git pull)
import sys; sys.path.insert(0, "/content/XERON")
%cd /content/XERON
!ls scripts/
"""))

cells.append(md(
"""## 4) 학습 데이터 준비 — 둘 중 하나 선택

**경로 A (권장):** 로컬 컴퓨터에서 생성한 `train_items_all_v2.pt` (~136MB)를
[Google Drive](https://drive.google.com)의 `xeron/` 폴더에 업로드합니다.

**경로 B:** 소스 JSONL에서 Colab에서 직접 전처리 (데이터 빌드 스크립트 실행, 몇 분 소요).
"""))

cells.append(code(
"""# 4A) Google Drive 마운트 + items 파일 경로
from google.colab import drive
drive.mount("/content/drive")

import os
BASE = "/content/drive/MyDrive/xeron"
os.makedirs(BASE, exist_ok=True)
ITEMS = os.path.join(BASE, "train_items_all_v2.pt")

# 데이터 빌드 스크립트 경로 (경로 B용)
KR_JSONL  = os.path.join(BASE, "korean_typed.jsonl")
BR_JSONL  = os.path.join(BASE, "browser_typed.jsonl")
M2W_JSONL = os.path.join(BASE, "mind2web_typed.jsonl")

print("items exists:", os.path.exists(ITEMS))
print("korean jsonl:", os.path.exists(KR_JSONL))
print("브라우저 jsonl:", os.path.exists(BR_JSONL))
print("mind2web jsonl:", os.path.exists(M2W_JSONL))
"""))

cells.append(code(
"""# 4B) 경로 B 전용: JSONL/HF에서 items 재구성 (경로 A를 쓰면 그냥 통과)
import os, torch, subprocess

def run_preprocess(model_dir, out_path, data_files=None):
    cmd = ["python", "scripts/preprocess.py",
           "--model-id", model_dir, "--output", out_path]
    if data_files:
        cmd += ["--data-files", data_files]
    else:
        cmd += ["--dataset", "LocalLLaMA/typed-decisions",
                "--config-name", "all", "--split", "train"]
    subprocess.run(cmd, check=True, cwd="/content/XERON")
    return torch.load(out_path, weights_only=False)

if not os.path.exists(ITEMS):
    os.makedirs("/content/items", exist_ok=True)
    print("items 없음 → JSONL/HF에서 재구성 (몇 분 소요)")
    en = run_preprocess(BASE_MODEL, "/content/items/en.pt")
    kr = run_preprocess(BASE_MODEL, "/content/items/kr.pt", KR_JSONL) if os.path.exists(KR_JSONL) else []
    br = run_preprocess(BASE_MODEL, "/content/items/br.pt", BR_JSONL) if os.path.exists(BR_JSONL) else []
    mw = run_preprocess(BASE_MODEL, "/content/items/mw.pt", M2W_JSONL) if os.path.exists(M2W_JSONL) else []
    all_items = en + kr + br + mw
    torch.save(all_items, ITEMS)
    print(f"재구성 완료: {len(all_items)} 시퀀스 -> {ITEMS}")
else:
    print("items 존재 — 경로 A 사용:", ITEMS)
"""))

cells.append(md(
"""## 5) 베이스 모델 (multilingual 체크포인트) 다운로드

`convaiinnovations/laya`의 `multilingual/` 서브폴더(mmBERT-base, 322M)만 받습니다 (약 650MB).
"""))

cells.append(code(
"""# 5) multilingual 베이스 모델 다운로드
import os
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config

MODEL_ROOT = "/content/laya-multilingual"
if not os.path.exists(os.path.join(MODEL_ROOT, "multilingual", "model.safetensors")):
    snapshot_download("convaiinnovations/laya", local_dir=MODEL_ROOT,
                      allow_patterns=["multilingual/*"])

BASE_MODEL = os.path.join(MODEL_ROOT, "multilingual")
_fix_tokenizer_config(os.path.join(BASE_MODEL, "tokenizer"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(BASE_MODEL, "tokenizer"))
print("Base model ready:", BASE_MODEL)
print("KR smoke test:", tok.encode("안녕하세요, 주문한 상품이 아직 안 왔어요.")[:8])
"""))

cells.append(md(
"""## 6) 하이퍼파라미터 설정 (파라미터 확장 🚀)

1 GPU(T4) 기준. **2xT4(유효배치 64)와 동일한 유효 배치를 1 GPU로 확장**합니다:
- `GRAD_ACCUM=8` (2 GPU 기준 4 → 8로 확장) → 유효 배치 = 8 × 1 × 8 = **64**
- `EPOCHS=2` (131K 규모, T4 1장에서 ~1~1.5시간/epoch 예상)
- `CHECKPOINT_EVERY=1` → 매 에폭 체크포인트 (런타임 끊김 대비)

| 실험 | EPOCHS | GRAD_ACCUM | 유효배치 | 비고 |
|---|---|---|---|---|
| A (빠른 검증) | 1 | 4 | 32 | ~30~45분 |
| **B (권장)** | **2** | **8** | **64** | ~1.5~3시간 |
| C (정밀) | 3 | 8 | 64 | Pro 권장 |
"""))

cells.append(code(
"""# 6) 하이퍼파라미터 확장 설정 (env로 전달)
import os
os.environ["EPOCHS"]           = "2"    # 2-3 권장
os.environ["MICRO_BATCH"]      = "8"    # T4 fp16 + grad checkpoint 안전값
os.environ["GRAD_ACCUM"]       = "8"    # 1 GPU 확장: 유효배치 64 (= 2xT4 원본)
os.environ["GROUP_SIZE"]       = "4"
os.environ["LR_ENCODER"]       = "2.5e-5"
os.environ["LR_HEAD"]          = "1e-4"
os.environ["CHECKPOINT_EVERY"] = "1"    # 에폭마다 체크포인트
os.environ["RESUME"]           = ""     # 재개 시 "auto"로 변경

OUT_DIR = "/content/output/xeron-all-v2"
print("유효 배치:", int(os.environ["MICRO_BATCH"]) * int(os.environ["GRAD_ACCUM"]),
      "| EPOCHS:", os.environ["EPOCHS"])
"""))

cells.append(md(
"""## 7) 학습 실행

- 처음 실행: 그대로 실행
- **런타임이 끊겼다면**: 상단 셀들을 다시 실행한 뒤 아래에서 `RESUME="auto"`로 바꿔 재개
"""))

cells.append(code(
"""# 7) DDP 학습 실행 (1 GPU)
%cd /content/XERON
!torchrun --standalone --nproc_per_node=1 --max_restarts=0 \\
  scripts/train_ddp.py "$BASE_MODEL" "$OUT_DIR" "$ITEMS"
"""))

cells.append(md(
"""## 8) 평가 + 결과 저장

eval은 test split 400 케이스(2,000 결정) 기준입니다. 타 데이터셋은 `--dataset`으로 지정 가능.
"""))

cells.append(code(
"""# 8A) 벤치마크 평가
%cd /content/XERON
!python scripts/evaluate.py --model "$OUT_DIR" --device cuda --output /content/eval_results.json
import json
print(json.dumps(json.load(open("/content/eval_results.json"))["summary"], indent=2))
"""))

cells.append(code(
"""# 8B) 결과를 Drive로 복사 (필수 — 세션 종료 대비)
!mkdir -p "$BASE/output"
!cp -r "$OUT_DIR" "$BASE/output/"
!cp /content/eval_results.json "$BASE/output/eval_results.json"
print("Saved to Drive:", "$BASE/output/")
"""))

cells.append(code(
"""# 9) [선택] HuggingFace Hub 업로드 (HF_TOKEN 필요)
# !HF_TOKEN=hf_xxx python scripts/upload_hf.py --model-dir "$OUT_DIR" --repo-id PIXELZX/XERON
print("skip or uncomment with your HF_TOKEN")
"""))

cells.append(md(
"""## 📋 빠른 체크리스트

1. 런타임 → 런타임 유형 변경 → **T4 GPU** ✅
2. Drive에 `xeron/train_items_all_v2.pt` 업로드 ✅
3. 셀 1~6 순서대로 실행 ✅
4. 학습 실행 (셀 7) — 에폭마다 체크포인트 자동 저장 ✅
5. 평가 + Drive 복사 (셀 8) ✅
6. 끊기면 RESUME="auto"로 재개 ✅

> 데이터 파일이 없으면: 로컬 `/home/yuchan/laya-models/` 에서
> `train_items_all_v2.pt`(136MB) / `korean_typed.jsonl` / `browser_typed.jsonl` / `mind2web_typed.jsonl` 을 Drive에 업로드하세요.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT} with {len(cells)} cells")