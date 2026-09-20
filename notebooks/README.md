# XERON 🎯

## 파인튜닝 노트북

원본 Kaggle 노트북 (2x T4, typed-decisions 파인튜닝):
👉 [NandhaKishorM/laya · notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)

이 레포의 `scripts/`는 위 파이프라인을 **로컬/클라우드 GPU 공용**으로 포팅한 버전입니다.

## 실행 순서 (GPU 서버 또는 Colab/Kaggle)

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 모델 다운로드 (로컬 디렉토리로)
python - <<'EOF'
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
d = snapshot_download("convaiinnovations/laya")
_fix_tokenizer_config(d)
print(d)
EOF

# 3. 전처리
python scripts/preprocess.py --dataset LocalLLaMA/typed-decisions --output /tmp/train_items.pt

# 4. 학습 (GPU 2개일 때)
torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py <모델_디렉토리> ./output/xeron /tmp/train_items.pt

# 5. 평가
python scripts/evaluate.py --model ./output/xeron

# 6. 업로드
HF_TOKEN=... python scripts/upload_hf.py --model-dir ./output/xeron --repo-id PIXELZX/XERON
```

> ⚠️ GPU 1장이면 `torchrun --nproc_per_node=1`로 실행 (효과 배치 크기는 GRAD_ACCUM으로 보정).