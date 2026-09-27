#!/bin/bash
set -uo pipefail
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q --upgrade "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || true
pip install -q laya transformers datasets safetensors scipy pandas tabulate 2>&1 | tail -2 || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
rm -rf /root/XERON && git clone -q --depth 1 -b perf/train-throughput https://github.com/PIXELZX0/XERON /root/XERON && echo "[remote] repo ok"
cd /root/XERON && git log --oneline -1
python - <<'PY'
import torch
items = torch.load('/root/data/train_long.pt', weights_only=False)
lens = [len(it['ids']) for it in items]
print('[data]', len(items), 'items tokens', sum(lens), 'max', max(lens), flush=True)
PY
echo "[train] start $(date -Is)"
EPOCHS=1 MICRO_BATCH=8 GRAD_ACCUM=4 GROUP_SIZE=4 \
BATCH_MODE=length BATCH_BUCKET_MULT=64 MAX_TOKENS_BATCH=16384 \
LR_ENCODER=2e-5 LR_HEAD=5e-5 SIGMA_START=0.3 SIGMA_END=0.2 RL_WEIGHT=0.5 WEIGHT_DECAY=0.02 \
HEAD_LAYERS=4 HEAD_SIZE=1024 HEAD_DROPOUT=0.0 DTYPE=bf16 MAX_LEN=8192 HEAD_MAX_LEN=256 \
CHECKPOINT_EVERY=1 PYTHONUNBUFFERED=1 \
timeout 36000 torchrun --standalone --nproc_per_node=1 scripts/train_ddp.py \
  /root/base ./output/xeron-1.0-long-e2 /root/data/train_long.pt
echo "TRAIN_EXIT=$?"
echo "[train] done $(date -Is)"
ls -la ./output/xeron-1.0-long-e2 2>/dev/null | head -6
