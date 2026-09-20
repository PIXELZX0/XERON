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

> ⚠️ 무료 T4 세션은 런타임이 끊길 수 있습니다. 완료 후 반드시 결과를 저장하세요.
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
!pip install -q "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate boto3
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
"""## ⚙️ 설정 — 여기서 값을 넣어주세요

아래 셀의 `SETTINGS` 딕셔너리를 수정하면 나머지 과정이 자동 반영됩니다.
- **빈 값으로 두면** Colab Secrets(🔑 아이콘 → + New secret)에서 같은 이름으로 읽습니다
- ⚠️ **S3 자격 증명을 이 셀에 직접 넣어도 동작하지만, GitHub에 노트북을 커밋하지 마세요**
  (이 레포는 public이라 키가 노출됩니다 — Colab Secrets 사용을 권장)
"""))

cells.append(code(
"""# ⚙️ 설정 (Settings) — 값을 직접 넣거나 Secrets에서 읽습니다
SETTINGS = {
    # ── 데이터: S3 호환(MinIO 등) ──────────────────────────────
    "S3_ENDPOINT": "",            # 예: http://192.168.0.100:9000 (빈 값 = Secrets)
    "S3_BUCKET": "xeron",
    "S3_KEY": "train_items_all_v2.pt",
    "AWS_ACCESS_KEY_ID": "",      # 빈 값 = Secrets
    "AWS_SECRET_ACCESS_KEY": "",  # 빈 값 = Secrets

    # ── 베이스 모델 ────────────────────────────────────────────
    "BASE_MODEL": "multilingual", # multilingual(322M, 한/영/웹) | english(421M)

    # ── 학습 하이퍼파라미터 (파라미터 확장) ────────────────────
    "EPOCHS": "2",                # 1~3 권장 (131K 규모)
    "MICRO_BATCH": "8",           # T4 fp16 + grad checkpoint 안전값
    "GRAD_ACCUM": "8",            # 1 GPU 확장 → 유효배치 = 8×1×8 = 64
    "GROUP_SIZE": "4",
    "LR_ENCODER": "2.5e-5",
    "LR_HEAD": "1e-4",
    "CHECKPOINT_EVERY": "1",      # N 에폭마다 체크포인트 (세션 끊김 대비)
    "RESUME": "",                 # 재개 시 "auto"

    # ── 결과 ───────────────────────────────────────────────────
    "OUTPUT_NAME": "xeron-all-v2",
}

# 설정창(Secrets) 폴백 헬퍼
try:
    from google.colab import userdata
    def _sec(name, default=""):
        try:
            return userdata.get(name, default) or default
        except Exception:
            return default
except ImportError:
    userdata = None
    def _sec(name, default=""):
        return default

for k in ["S3_ENDPOINT", "S3_BUCKET", "S3_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
    if not SETTINGS.get(k):
        SETTINGS[k] = _sec(k, SETTINGS.get(k, ""))

import os
os.environ.update({k: str(v) for k, v in SETTINGS.items() if v})
print("✅ 설정 적용됨")
print("   S3:", SETTINGS["S3_ENDPOINT"] or "(미설정 — Drive 모드)", "| bucket:", SETTINGS["S3_BUCKET"])
print("   BASE_MODEL:", SETTINGS["BASE_MODEL"], "| EPOCHS:", SETTINGS["EPOCHS"],
      "| GRAD_ACCUM:", SETTINGS["GRAD_ACCUM"])
"""))

cells.append(md(
"""## 4) 학습 데이터 준비 — 3가지 경로 (자동 우선순위)

**① S3 호환(MinIO 등)** → **② Google Drive** → **③ JSONL 재구성**

- S3 설정을 넣었거나 Secrets에 있으면 자동으로 S3에서 다운로드합니다
- S3 미설정이면 Drive 마운트를 시도합니다
- 둘 다 없으면 마지막 셀이 JSONL에서 직접 재구성합니다
"""))

cells.append(code(
"""# 4A) 학습 데이터 확보 (S3 → Drive → 재구성 자동 시도)
import os
ITEMS = "/content/train_items_all_v2.pt"

# ① S3 호환 (MinIO 등)
if not os.path.exists(ITEMS) and SETTINGS.get("S3_ENDPOINT"):
    import boto3
    ep = SETTINGS["S3_ENDPOINT"]
    ck = dict(
        aws_access_key_id=SETTINGS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=SETTINGS["AWS_SECRET_ACCESS_KEY"],
    )
    if "amazonaws.com" not in ep:  # MinIO/R2/B2 등 S3 호환
        ck["endpoint_url"] = ep
        ck["region_name"] = "us-east-1"
    s3 = boto3.client("s3", **{k: v for k, v in ck.items() if v})
    try:
        s3.download_file(SETTINGS["S3_BUCKET"], SETTINGS["S3_KEY"], ITEMS)
        print("☁️ S3에서 다운로드 완료:", ITEMS)
    except Exception as e:
        print("⚠️ S3 다운로드 실패:", str(e)[:200], "→ Drive 모드로 전환")

# ② Google Drive
if not os.path.exists(ITEMS):
    from google.colab import drive
    drive.mount("/content/drive")
    GDRIVE = "/content/drive/MyDrive/xeron/" + SETTINGS["S3_KEY"]
    if os.path.exists(GDRIVE):
        !cp "$GDRIVE" "$ITEMS"
        print("📁 Drive에서 복사 완료")

print("ITEMS exists:", os.path.exists(ITEMS),
      "| size:", (os.path.getsize(ITEMS) // 1048576) if os.path.exists(ITEMS) else 0, "MB")
"""))

cells.append(code(
"""# 4B) 경로 ③ 전용: JSONL/HF에서 items 재구성 (items가 없을 때만 실행)
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
    GDRIVE = "/content/drive/MyDrive/xeron"
    KR_JSONL  = os.path.join(GDRIVE, "korean_typed.jsonl")
    BR_JSONL  = os.path.join(GDRIVE, "browser_typed.jsonl")
    M2W_JSONL = os.path.join(GDRIVE, "mind2web_typed.jsonl")
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
    print("items 존재 — 재구성 불필요:", ITEMS)
"""))

cells.append(md(
"""## 5) 베이스 모델 다운로드

`SETTINGS["BASE_MODEL"]`에 따라 받습니다:
- `multilingual` → `multilingual/` 서브폴더 (mmBERT-base, 322M, 한국어 포함, 권장)
- `english` → repo root (ModernBERT-large, 421M)
"""))

cells.append(code(
"""# 5) 베이스 모델 다운로드
import os
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config

MODEL_ROOT = "/content/laya-model"
if SETTINGS["BASE_MODEL"] == "english":
    pattern = ["*.safetensors", "encoder/*", "tokenizer/*", "rl_agent_config.json"]
else:
    pattern = ["multilingual/*"]

if not os.path.exists(os.path.join(MODEL_ROOT, SETTINGS["BASE_MODEL"] == "english" and "model.safetensors" or "multilingual", "model.safetensors")):
    snapshot_download("convaiinnovations/laya", local_dir=MODEL_ROOT, allow_patterns=pattern)

BASE_MODEL = MODEL_ROOT if SETTINGS["BASE_MODEL"] == "english" else os.path.join(MODEL_ROOT, "multilingual")
_fix_tokenizer_config(os.path.join(BASE_MODEL, "tokenizer"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(BASE_MODEL, "tokenizer"))
print("Base model ready:", BASE_MODEL)
print("KR smoke test:", tok.encode("안녕하세요, 주문한 상품이 아직 안 왔어요.")[:8])
"""))

cells.append(md(
"""## 6) 하이퍼파라미터 — 파라미터 확장 🚀

1 GPU(T4) 기준. **2xT4(유효배치 64)와 동일한 유효 배치를 1 GPU로 확장**합니다:
- `GRAD_ACCUM=8` (2 GPU 기준 4 → 8로 확장) → 유효 배치 = 8 × 1 × 8 = **64**
- `EPOCHS=2` (131K 규모, T4 1장에서 ~1~1.5시간/epoch 예상)
- `CHECKPOINT_EVERY=1` → 매 에폭 체크포인트 (런타임 끊김 대비)

| 실험 | EPOCHS | GRAD_ACCUM | 유효배치 | 비고 |
|---|---|---|---|---|
| A (빠른 검증) | 1 | 4 | 32 | ~30~45분 |
| **B (권장)** | **2** | **8** | **64** | ~1.5~3시간 |
| C (정밀) | 3 | 8 | 64 | Pro 권장 |

> 값은 위 ⚙️ 설정 셀에서 변경하면 됩니다 (여기서는 그 값을 env로 전달).
"""))

cells.append(code(
"""# 6) ⚙️ 설정 값을 학습 환경변수로 전달
import os
os.environ["EPOCHS"]           = SETTINGS["EPOCHS"]
os.environ["MICRO_BATCH"]      = SETTINGS["MICRO_BATCH"]
os.environ["GRAD_ACCUM"]       = SETTINGS["GRAD_ACCUM"]
os.environ["GROUP_SIZE"]       = SETTINGS["GROUP_SIZE"]
os.environ["LR_ENCODER"]       = SETTINGS["LR_ENCODER"]
os.environ["LR_HEAD"]          = SETTINGS["LR_HEAD"]
os.environ["CHECKPOINT_EVERY"] = SETTINGS["CHECKPOINT_EVERY"]
os.environ["RESUME"]           = SETTINGS["RESUME"]

OUT_DIR = f"/content/output/{SETTINGS['OUTPUT_NAME']}"
print("유효 배치:", int(SETTINGS["MICRO_BATCH"]) * int(SETTINGS["GRAD_ACCUM"]),
      "| EPOCHS:", SETTINGS["EPOCHS"], "| OUT:", OUT_DIR)
"""))

cells.append(md(
"""## 7) 학습 실행

- 처음 실행: 그대로 실행
- **런타임이 끊겼다면**: ⚙️ 설정 셀에서 `RESUME`을 `"auto"`로 바꾸고 상단 셀들을 재실행
"""))

cells.append(code(
"""# 7) DDP 학습 실행 (1 GPU) — Python subprocess 방식
import subprocess, os

env = dict(os.environ)
cmd = [
    "torchrun", "--standalone", "--nproc_per_node=1", "--max_restarts=0",
    "scripts/train_ddp.py", BASE_MODEL, OUT_DIR, ITEMS,
]
print(" ".join(cmd))
subprocess.run(cmd, env=env, check=True, cwd="/content/XERON")
print("✅ 학습 완료:", OUT_DIR)
"""))

cells.append(md(
"""## 8) 평가 + 결과 저장"""))

cells.append(code(
"""# 8A) 벤치마크 평가
%cd /content/XERON
!python scripts/evaluate.py --model "$OUT_DIR" --device cuda --output /content/eval_results.json
import json
try:
    print(json.dumps(json.load(open("/content/eval_results.json"))["summary"], indent=2))
except Exception as e:
    print("평가 결과 파싱 실패:", e)
"""))

cells.append(code(
"""# 8B) 결과 저장 — S3 (설정 시) 또는 Drive
import os
if SETTINGS.get("S3_ENDPOINT"):
    import boto3
    ck = dict(aws_access_key_id=SETTINGS["AWS_ACCESS_KEY_ID"],
              aws_secret_access_key=SETTINGS["AWS_SECRET_ACCESS_KEY"])
    if "amazonaws.com" not in SETTINGS["S3_ENDPOINT"]:
        ck["endpoint_url"] = SETTINGS["S3_ENDPOINT"]
        ck["region_name"] = "us-east-1"
    s3 = boto3.client("s3", **{k: v for k, v in ck.items() if v})
    # 모델 디렉토리 통째로 업로드
    for root, _dirs, files in os.walk(OUT_DIR):
        for f in files:
            local = os.path.join(root, f)
            key = f"output/{os.path.basename(OUT_DIR)}/{os.path.relpath(local, OUT_DIR)}"
            s3.upload_file(local, SETTINGS["S3_BUCKET"], key)
    s3.upload_file("/content/eval_results.json", SETTINGS["S3_BUCKET"], "output/eval_results.json")
    print("☁️ 결과를 S3에 업로드 완료")
else:
    !mkdir -p "/content/drive/MyDrive/xeron/output"
    !cp -r "$OUT_DIR" "/content/drive/MyDrive/xeron/output/"
    !cp /content/eval_results.json "/content/drive/MyDrive/xeron/output/"
    print("📁 결과를 Drive에 복사 완료")
"""))

cells.append(code(
"""# 9) [선택] HuggingFace Hub 업로드 (HF_TOKEN 필요)
# !HF_TOKEN=hf_xxx python scripts/upload_hf.py --model-dir "$OUT_DIR" --repo-id PIXELZX/XERON
print("skip or uncomment with your HF_TOKEN")
"""))

cells.append(md(
"""## 📋 빠른 체크리스트

1. 런타임 → 런타임 유형 변경 → **T4 GPU** ✅
2. **⚙️ 설정 셀**에서 값 확인 (S3 미설정 시 Drive에 `xeron/train_items_all_v2.pt` 업로드) ✅
3. 셀 1~6 순서대로 실행 ✅
4. 학습 실행 (셀 7) — 에폭마다 체크포인트 자동 저장 ✅
5. 평가 + 결과 저장 (셀 8) — S3/Drive 자동 ✅
6. 끊기면 `RESUME="auto"`로 재개 ✅

### S3(MinIO) 사용 시 참고
- 노트북 왼쪽 🔑 **Secrets**에 등록: `S3_ENDPOINT`, `S3_BUCKET`, `S3_KEY`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (설정 셀에 직접 넣어도 동작)
- MinIO 접근은 Colab 외부망에서 가능해야 합니다 (퍼블릭 IP/포트포워딩/터널)
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