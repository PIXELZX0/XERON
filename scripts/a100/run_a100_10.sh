#!/bin/bash
# XERON-1.0 A100 80GB 학습 하네스 + 실측 벤치마크 (hidden 1536 / layers 44 / vocab 256000).
#
# scripts/a100/run_a100.sh (XERON-0.6 파이프라인)는 그대로 보존되어 있고, 이 스크립트는
# 그 환경 구성(deps/repo)만 계승해 벤치마크 전용으로 확장한 것이다. 본학습은 하지 않는다.
#
# 대상 아키텍처(주인님 확정): ModernBERT enc hidden 1536 / 44 layers / 24 heads / vocab 256000,
#   MAX_LEN 8192, HEAD_SIZE 1024, HEAD_LAYERS 4, bf16. 가중치는 불필요(랜덤 초기화).
#   INTER 는 기본 2304(명세 그대로, mmBERT 스케일링) 이며 INTER=4096 변형(≈1.69B)도 측정한다.
#
# 사용 (로컬):
#   gpu run --gpu-type "NVIDIA A100-SXM4-80GB" bash -c 'cd XERON && bash scripts/a100/run_a100_10.sh'
# 필요한 시크릿(선택): KAGGLE_API_TOKEN (사설 Kaggle 데이터셋 다운로드용)
#
# 주요 env:
#   BUDGET_SECS(2700)  STEPS_BASE(100)  STEPS_OTHER(40)  SAMPLE_ITEMS(40000)
#   DATA_SOURCE(auto|file|kaggle|hf|synth)  DATA_FILE  ONLY  SKIP  CFG_TIMEOUT(600)
set -uo pipefail

RUN_NAME="${RUN_NAME:-xeron-1.0}"
STEPS_BASE="${STEPS_BASE:-100}"
STEPS_OTHER="${STEPS_OTHER:-40}"
WARMUP="${WARMUP:-3}"
SAMPLE_ITEMS="${SAMPLE_ITEMS:-40000}"
BUDGET_SECS="${BUDGET_SECS:-2700}"
CFG_TIMEOUT="${CFG_TIMEOUT:-900}"
DATA_SOURCE="${DATA_SOURCE:-auto}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
KAGGLE_DATASET="${KAGGLE_DATASET:-pistonx/xeron-1-0-train-items}"

LOGFILE="${LOGFILE:-logs/bulk/W6_a100_bench.log}"
OUT_DIR="${OUT_DIR:-outputs/bench}"
OUT_JSONL="${OUT_JSONL:-$OUT_DIR/xeron10_a100_bench.jsonl}"
mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1

T0=$(date +%s)
echo "########## XERON-1.0 A100 bench harness ##########"
echo "host=$(hostname) date=$(date -Is)"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -V

echo "=== deps ==="
# Vast/기본 이미지가 torch<2.5면 transformers 5가 PyTorch를 꺼버린다(run_a100.sh와 동일한 이유).
TORCH_OK=$(python -c "import torch,sys;v=torch.__version__.split('+')[0].split('.');print(1 if (int(v[0]),int(v[1]))>=(2,5) else 0)" 2>/dev/null || echo 0)
if [ "$TORCH_OK" != "1" ]; then
  echo "[deps] upgrading torch (found <2.5)"
  pip install -q --upgrade "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124
fi
pip install -q laya tabulate scipy pandas bitsandbytes 2>&1 | tail -2 || true

# transformers 는 import 시점에 torchaudio/torchvision 을 끌어온다(processing_utils -> audio_utils).
# base 이미지(/opt/conda)의 사본이 venv torch 와 ABI 가 어긋나면 ModernBert import 자체가 죽는다.
# 이 벤치에는 둘 다 불필요 -> (1) venv 제거 (2) torchaudio 같은 버전 설치 (3) 그래도 import 되면
# base 사본을 *.disabled 로 옮겨 '존재하지 않게' 만든다.
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
if ! python -c "import torchaudio" >/dev/null 2>&1; then
  TV=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  if [ -n "$TV" ]; then
    echo "[deps] pip install torchaudio==$TV"
    pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || true
  fi
fi
python scripts/_disable_audio_deps.py || true
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"
python -c "from transformers import ModernBertModel;print('ModernBertModel import OK')" || echo "[deps] WARN: ModernBert import failed"

echo "=== repo ==="
if [ ! -f scripts/bench_xeron10.py ]; then
  if [ -d XERON ]; then cd XERON; else git clone --depth 1 https://github.com/PIXELZX0/XERON.git XERON && cd XERON; fi
fi
git log --oneline -1 2>/dev/null || true

echo
# preflight: 작은 아키텍처로 import/스텝/loss 경로를 먼저 검증한다(환경 깨짐으로
# pod 사이클을 통째로 날리는 사고를 막는다).
echo "=== preflight (tiny arch, 2 steps) ==="
rm -f /tmp/preflight.jsonl
if HIDDEN=64 LAYERS=2 INTER=96 HEADS=4 VOCAB=512 HEAD_SIZE=64 HEAD_LAYERS=2 MAX_LEN=64 MAXPOS=256 \
   DATA_MODE=synth SYNTH_LEN=64 SAMPLE_ITEMS=8 MICRO_BATCH=2 GRAD_ACCUM=1 GRAD_CKPT=0 \
   STEPS=2 WARMUP=1 OUT_JSONL=/tmp/preflight.jsonl TAG=preflight \
   python scripts/bench_xeron10.py && grep -q '"ok": true' /tmp/preflight.jsonl; then
  echo "[preflight] OK"
else
  echo "[preflight] FAILED -> sweep 중단 (env 문제)"; exit 1
fi


echo "=== data ==="
# 1) 이미 있으면 그대로 사용 (8192판이 있으면 그걸 우선)
DATA_FILE_LOCAL=""
for cand in train_items_x10_8192.pt train_items_x10.pt; do
  if [ -f "$cand" ]; then DATA_FILE_LOCAL="$cand"; break; fi
done
if [ -z "$DATA_FILE_LOCAL" ] && { [ "$DATA_SOURCE" = "kaggle" ] || { [ "$DATA_SOURCE" = "auto" ] && [ -n "${KAGGLE_API_TOKEN:-}" ]; }; }; then
  echo "[data] kaggle download: $KAGGLE_DATASET"
  pip install -q kaggle 2>&1 | tail -2 || true
  KB="$(dirname "$(python -c 'import sys;print(sys.executable)')")/kaggle"
  mkdir -p /tmp/x10
  if [ -x "$KB" ]; then
    "$KB" datasets download -d "$KAGGLE_DATASET" -p /tmp/x10 --unzip -q
  else
    python -m kaggle.cli datasets download -d "$KAGGLE_DATASET" -p /tmp/x10 --unzip -q
  fi || echo "[data] kaggle download 실패"
  PT=$(find /tmp/x10 -maxdepth 2 -name "train_items_x10*.pt" | head -1)
  if [ -n "$PT" ]; then cp -f "$PT" ./train_items_x10.pt && DATA_FILE_LOCAL=train_items_x10.pt; fi
fi
if [ -z "$DATA_FILE_LOCAL" ] && [ -n "${DATA_FILE:-}" ] && [ -f "${DATA_FILE:-}" ]; then
  DATA_FILE_LOCAL="$DATA_FILE"
fi
if [ -z "$DATA_FILE_LOCAL" ]; then
  echo "[data] 실데이터 없음 -> 합성(synth) 모드로만 측정한다 (길이분포 기반)"
  DATA_SOURCE=synth
else
  echo "[data] DATA_FILE=$DATA_FILE_LOCAL ($(du -h "$DATA_FILE_LOCAL" | cut -f1))"
  [ "$DATA_SOURCE" = "auto" ] && DATA_SOURCE=file
fi
ls -la "$DATA_FILE_LOCAL" 2>/dev/null || true

echo "=== length stats ==="
if [ ! -f configs/xeron10_len_stats.json ] && [ -d data/x10_shards ]; then
  python scripts/len_stats_x10.py || true
  [ -f data/xeron10_len_stats.json ] && cp -f data/xeron10_len_stats.json configs/xeron10_len_stats.json
fi
[ -f configs/xeron10_len_stats.json ] && head -c 400 configs/xeron10_len_stats.json; echo

echo "=== bench sample (작은 서브셋 -> 설정마다 재로딩 비용 제거) ==="
REALFILE=""
if [ "$DATA_SOURCE" = "file" ] || [ "$DATA_SOURCE" = "kaggle" ] || [ "$DATA_SOURCE" = "hf" ]; then
  free -g | sed -n 1,2p || true
  if DATA_FILE="$DATA_FILE_LOCAL" SAMPLE_ITEMS="$SAMPLE_ITEMS" DATA_MODE=file ONLY_PREP=1 \
       python scripts/bench_xeron10.py; then
    REALFILE=data/xeron10_bench_sample.pt
  else
    echo "[data] sample prep 실패 -> 실데이터 설정은 건너뛴다"
  fi
fi
SKIP_REAL=""
[ -z "$REALFILE" ] && SKIP_REAL="1"

echo
echo "########## sweep start (budget ${BUDGET_SECS}s) ##########"
SKIPPED=""; RAN=""
T0=$(date +%s)

run_cfg() {
  local tag="$1"; shift
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$tag,"; then return 0; fi
  if [ -n "$SKIP" ] && echo ",$SKIP," | grep -q ",$tag,"; then echo "[skip] $tag (SKIP)"; return 0; fi
  case " $* " in *" DATA_MODE=file "*) [ -n "$SKIP_REAL" ] && { echo "[skip] $tag (실데이터 없음 -> synth만)"; SKIPPED="$SKIPPED $tag"; return 0; };; esac
  local now=$(( $(date +%s) - T0 ))
  if [ "$now" -gt "$BUDGET_SECS" ]; then echo "[budget] t+${now}s > ${BUDGET_SECS}s -> skip $tag"; SKIPPED="$SKIPPED $tag"; return 0; fi
  echo; echo "=== CONFIG $tag (t+${now}s) :: $* ==="
  [ -n "$REALFILE" ] && export DATA_FILE="$REALFILE"
  timeout "$CFG_TIMEOUT" env "$@" OUT_JSONL="$OUT_JSONL" TAG="$tag" \
    torchrun --standalone --nproc_per_node=1 scripts/bench_xeron10.py
  local rc=$?
  echo "=== CONFIG $tag rc=$rc (t+$(( $(date +%s) - T0 ))s) ==="
  RAN="$RAN $tag"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
  sleep 2
}

while IFS='|' read -r tag envs; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  run_cfg "$tag" $envs
done <<EOF
# ---- 실데이터(train_items_x10.pt, 평균 길이 실측 기반) ----
R1_mb4_ga8_ckpt_L4096|DATA_MODE=file MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=1 STEPS=$STEPS_BASE WARMUP=$WARMUP
R2_mb1_ga32_ckpt_L4096|DATA_MODE=file MICRO_BATCH=1 GRAD_ACCUM=32 GRAD_CKPT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R3_mb2_ga16_ckpt_L4096|DATA_MODE=file MICRO_BATCH=2 GRAD_ACCUM=16 GRAD_CKPT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R4_mb8_ga4_ckpt_L4096|DATA_MODE=file MICRO_BATCH=8 GRAD_ACCUM=4 GRAD_CKPT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R5_mb16_ga2_ckpt_L4096|DATA_MODE=file MICRO_BATCH=16 GRAD_ACCUM=2 GRAD_CKPT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R6_mb4_ga8_nockpt_L4096|DATA_MODE=file MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=0 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R7_mb4_ga8_ckpt_8bit_L4096|DATA_MODE=file MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=1 OPT8BIT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
R8_mb4_ga8_ckpt_L8192|DATA_MODE=file MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=1 MAX_LEN=8192 STEPS=$STEPS_OTHER WARMUP=$WARMUP
# ---- 합성 최악 케이스: 모든 시퀀스가 정확히 8192 토큰 (OOM 경계 탐색) ----
S1_8192_mb1|DATA_MODE=synth SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=1 GRAD_ACCUM=32 GRAD_CKPT=1 MAX_LEN=8192 STEPS=12 WARMUP=2
S2_8192_mb2|DATA_MODE=synth SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=2 GRAD_ACCUM=16 GRAD_CKPT=1 MAX_LEN=8192 STEPS=12 WARMUP=2
S3_8192_mb4|DATA_MODE=synth SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=1 MAX_LEN=8192 STEPS=12 WARMUP=2
S4_8192_mb8|DATA_MODE=synth SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=8 GRAD_ACCUM=4 GRAD_CKPT=1 MAX_LEN=8192 STEPS=8 WARMUP=2
S5_8192_mb4_nockpt|DATA_MODE=synth SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=0 MAX_LEN=8192 STEPS=8 WARMUP=2
# ---- 대체 아키텍처: INTER=4096 (총 ~1.69B, 보고된 "약 1.7B"와 일치) ----
A1_inter4096_mb4_ga8_ckpt_L4096|DATA_MODE=file INTER=4096 MICRO_BATCH=4 GRAD_ACCUM=8 GRAD_CKPT=1 STEPS=$STEPS_OTHER WARMUP=$WARMUP
A2_inter4096_8192_mb2|DATA_MODE=synth INTER=4096 SYNTH_LEN=8192 SAMPLE_ITEMS=64 MICRO_BATCH=2 GRAD_ACCUM=16 GRAD_CKPT=1 MAX_LEN=8192 STEPS=8 WARMUP=2
EOF

echo
echo "########## sweep done (t+$(( $(date +%s) - T0 ))s) ##########"
echo "RAN:$RAN"
echo "SKIPPED:$SKIPPED"
echo "=== results ==="
[ -f "$OUT_JSONL" ] && python - <<PY
import json
rows=[json.loads(l) for l in open("$OUT_JSONL")]
for r in rows:
    t=r.get("throughput") or {}
    m=r.get("mem") or {}
    print("%-28s ok=%-5s oom=%-5s seq/s=%-7s tok/s=%-9s peak=%-8s %s" % (
        r.get("tag"), r.get("ok"), r.get("oom", False), t.get("seq_s"), t.get("tok_s_padded"),
        m.get("peak_alloc_mb"), (r.get("error") or "")[:60]))
PY
python - <<PY
import json, os, time
s={"run_name":"$RUN_NAME","device":None,"start":$T0,"end":int(time.time()),
   "ran":"$RAN".split(),"skipped":"$SKIPPED".split(),"data_file":"$DATA_FILE_LOCAL",
   "sample_file":"$REALFILE","jsonl":"$OUT_JSONL","data_source":"$DATA_SOURCE"}
try:
    import torch
    s["device"]=torch.cuda.get_device_name(0); s["torch"]=torch.__version__
    import transformers; s["transformers"]=transformers.__version__
except Exception as e: s["err"]=str(e)
os.makedirs("$OUT_DIR",exist_ok=True)
p="$OUT_DIR/xeron10_a100_bench_summary.json"
open(p+".tmp","w").write(json.dumps(s,indent=1)); os.replace(p+".tmp",p)
print("wrote",p)
PY
echo "BENCH COMPLETE"
