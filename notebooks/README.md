# XERON 🎯

## Fine-tuning notebook

Original Kaggle notebook (2x T4, typed-decisions fine-tuning):
👉 [NandhaKishorM/laya · notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)

This repo's `scripts/` is a port of that pipeline for **local/cloud GPUs alike**.

## Execution order (GPU server or Colab/Kaggle)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download the model (to a local directory)
python - <<'EOF'
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
d = snapshot_download("convaiinnovations/laya")
_fix_tokenizer_config(d)
print(d)
EOF

# 3. Preprocess
python scripts/preprocess.py --dataset LocalLLaMA/typed-decisions --output /tmp/train_items.pt

# 4. Train (when you have 2 GPUs)
torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py <model_directory> ./output/xeron /tmp/train_items.pt

# 5. Evaluate
python scripts/evaluate.py --model ./output/xeron

# 6. Upload
HF_TOKEN=... python scripts/upload_hf.py --model-dir ./output/xeron --repo-id PIXELZX/XERON
```

> ⚠️ With a single GPU, run `torchrun --nproc_per_node=1` (correct the effective batch size with GRAD_ACCUM).
