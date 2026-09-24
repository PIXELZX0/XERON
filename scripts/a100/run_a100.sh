#!/bin/bash
# XERON-0.6: bf16 re-run on A100 to test whether fp16 training caused the overconfidence.
#
# Evidence: same-condition fp32 inference gives logit spread 15.4 for 0.4 (fp16-trained)
# vs 7.1 for 0.2 (bf16-trained), and temperatures [2.27,2.19,3.31] vs [1.26,1.71,2.08].
# So the logits scale is baked into the weights, not an inference artifact. 0.6 keeps every
# hyperparameter identical to 0.4 and changes ONLY the precision (fp16 -> bf16).
#
# Env: HF_TOKEN (required), DATA_FILE (default train_items_x3.pt), RUN_NAME (default xeron-0.6)
set -euo pipefail

RUN_NAME="${RUN_NAME:-xeron-0.6}"
DATA_FILE="${DATA_FILE:-train_items_x3.pt}"
HEAD_LAYERS="${HEAD_LAYERS:-0}"          # 0 = keep 0.2's 2-layer head; >0 = rebuild the head
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -V

echo "=== deps ==="
# The Vast.ai base image ships torch 2.4.1, but transformers>=5 requires torch>=2.5
# (without this, AutoModel silently disables PyTorch and build_model raises ImportError).
pip install -q --upgrade "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124
# torchvision/torchaudio in the image are pinned to torch 2.4.1 and break the torch 2.6
# ABI ("operator torchvision::nms does not exist" while transformers imports image_utils).
# Our local env has neither installed; drop them here too.
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q laya tabulate scipy
python -c "import torch, transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"
python -c "from transformers import ModernBertModel; print('ModernBertModel import OK')"

echo "=== repo ==="
# already inside the repo (scripts/ present) -> use it; otherwise clone
if [ ! -f scripts/train_ddp.py ]; then
  [ -d XERON ] || git clone --depth 1 https://github.com/PIXELZX0/XERON.git XERON
  cd XERON
fi
git log --oneline -1 2>/dev/null || true

echo "=== data + base ==="
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
tok = os.environ["HF_TOKEN"]
data = os.environ.get("DATA_FILE", "train_items_x3.pt")
p = hf_hub_download("PIXELZX/xeron-train-items", data, repo_type="dataset", token=tok, local_dir=".")
b = snapshot_download("PIXELZX/XERON-0.2", allow_patterns=["*.json", "*.safetensors", "tokenizer/*"], token=tok)
print("DATA=" + p, flush=True)
print("BASE=" + b, flush=True)
with open("/tmp/dirs.txt", "w") as fh:
    fh.write(p + "\n" + b + "\n")
PY
DATA=$(sed -n 1p /tmp/dirs.txt)
BASE=$(sed -n 2p /tmp/dirs.txt)
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
mkdir -p outputs

echo "=== train (bf16, 1 epoch) head_layers=$HEAD_LAYERS ==="
# head is freshly initialized when HEAD_LAYERS differs from the base's 2 -> train it harder (1e-4)
EPOCHS=1 MICRO_BATCH=4 GRAD_ACCUM=8 GROUP_SIZE=4 \
LR_ENCODER=2e-5 LR_HEAD=1e-4 SIGMA_START=0.3 SIGMA_END=0.2 RL_WEIGHT=0.5 WEIGHT_DECAY=0.02 \
HEAD_LAYERS="$HEAD_LAYERS" HEAD_DROPOUT="$HEAD_DROPOUT" \
DTYPE=bf16 MAX_LEN=4096 HEAD_MAX_LEN=256 MAX_TOKENS_BATCH=8192 CHECKPOINT_EVERY=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node="$NGPU" scripts/train_ddp.py "$BASE" "./output/$RUN_NAME" "$DATA"

echo "=== fitted temperature (the key number) ==="
python -c "import json;d=json.load(open('./output/$RUN_NAME/rl_agent_config.json'));print('TEMP',d.get('temperature'));print('TRAINING',json.dumps(d.get('training')))"

echo "=== snapshot -> HF (644MB, faster than workspace sync) ==="
python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "PIXELZX/" + os.environ.get("RUN_NAME", "xeron-0.6")
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path="./output/$RUN_NAME",
                  commit_message="XERON-0.6 bf16 re-run (A100)")
print("UPLOADED " + repo, flush=True)
PY

echo "=== eval: jevbench public-231 ==="
[ -d /tmp/jevbench ] || git clone --depth 1 https://github.com/fstandhartinger/jevbench.git /tmp/jevbench
python - <<PY
import sys, time
sys.path.insert(0, "/tmp/jevbench")
from jevbench.adapters import LayaLocalAdapter
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.tasks import load_jsonl
PUB = "/tmp/jevbench/datasets/public"
tasks = []
for f in ("original", "easy", "hard"):
    tasks.extend(load_jsonl(f"{PUB}/{f}.jsonl"))
print("public tasks:", len(tasks), flush=True)
ad = LayaLocalAdapter(endpoint="./output/$RUN_NAME", model="$RUN_NAME", threads=4)
ad.load()
for t in tasks[:2]:
    ad.run(t)
r = Runner(ad, Ledger("outputs/jevbench.ledger.jsonl"), "outputs/jevbench.raw")
recs = r.run_all(tasks, progress_every=50, results_path="outputs/$RUN_NAME.results.jsonl")
print("jevbench done %d/%d" % (sum(1 for x in recs if x["ok"]), len(recs)), flush=True)
PY

echo "=== eval: typed-decisions ==="
python scripts/evaluate.py --model "./output/$RUN_NAME" --split test --device cuda \
  --output "outputs/$RUN_NAME.typed-decisions-eval.json"

echo "=== diag: fp32 logit scale (compare with 0.4's 15.417) ==="
python -u scripts/diag_logit_scale.py "./output/$RUN_NAME" --dtype fp32 --n 200 --items "$DATA" 2>&1 | grep -aE "^\[diag\]" || true

echo "PIPELINE COMPLETE"
