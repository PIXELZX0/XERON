#!/bin/bash
# XERON-1.0 롱컨텍스트(>1024 토큰) 파인튜닝 — A100 80GB, Vast API 직접 임대.
#
# Kaggle TPU v5e-8(칩당 16GiB)은 seq>=2048에서 OOM이므로, 1024 토큰 초과 18,802개를
# **A100에서 먼저** 학습시킨다. 베이스는 현재 XERON-1.0인 output/xeron-1.0-a1 에서 이어받는다.
# 데이터는 길이 오름차순 정렬(train_items_x10_long1024.pt) — collate가 배치 내 최대 길이로
# 패딩하므로 낭비가 최소화된다.
#
# env: VAST_OFFER(21050981 = A100 SXM4 80GB $1.056/h) EPOCHS(2) MAX_TRAIN_SECS(36000)
#      MICRO_BATCH(2) GRAD_ACCUM(16) LR_ENCODER(2e-5) LR_HEAD(5e-5) RUN_NAME(xeron-1.0-long)
set -uo pipefail

: "${VAST_API_KEY:?VAST_API_KEY 필요 (secretEnv)}"
OFFER="${VAST_OFFER:-}"
DISK="${INSTANCE_DISK:-120}"
MAX_WAIT="${MAX_WAIT:-1200}"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_SECS="${MAX_TRAIN_SECS:-36000}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_ENCODER="${LR_ENCODER:-2e-5}"
LR_HEAD="${LR_HEAD:-5e-5}"
SIGMA_START="${SIGMA_START:-0.3}"
SIGMA_END="${SIGMA_END:-0.2}"
RUN_NAME="${RUN_NAME:-xeron-1.0-long-e2}"
BASE_DIR="${BASE_DIR:-$HOME/XERON/output/xeron-1.0-long}"
BRANCH="${BRANCH:-perf/train-throughput}"
BATCH_MODE="${BATCH_MODE:-length}"
BATCH_BUCKET_MULT="${BATCH_BUCKET_MULT:-64}"
DATA_FILE="${DATA_FILE:-$HOME/XERON/train_items_x10_long1024.pt}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
API="https://console.vast.ai/api/v0"
KEY="$HOME/.ssh/xeron_vast_ed25519"
LOGDIR=/tmp/xeron-bench
mkdir -p "$LOGDIR"
LOG="$LOGDIR/vast_long_train.log"
exec > >(tee -a "$LOG") 2>&1

ID=""
cleanup() {
  if [ -n "$ID" ]; then
    echo "[cleanup] destroying instance $ID"
    curl -s -m 30 -X DELETE "$API/instances/$ID/" -H "Authorization: Bearer $VAST_API_KEY" | head -c 200; echo
  fi
}
trap cleanup EXIT

echo "########## XERON long-context TRAIN start $(date -Is) offer=$OFFER epochs=$EPOCHS ##########"
[ -f "$DATA_FILE" ] || { echo "데이터 없음: $DATA_FILE"; exit 1; }
[ -f "$BASE_DIR/model.safetensors" ] || { echo "베이스 없음: $BASE_DIR"; exit 1; }
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "xeron-vast-direct" -f "$KEY"
curl -s -m 30 -X POST "$API/ssh/" -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"ssh_key":sys.argv[1]}))' "$(cat "$KEY.pub")")" >/dev/null
echo "[ssh] key registered"

# Vast inventory is volatile: pick the cheapest rentable A100 80GB right now unless pinned.
if [ -z "$OFFER" ]; then
  OFFER=$(curl -s -m 60 "$API/bundles/?q=%7B%22gpu_name%22%3A%7B%22eq%22%3A%22A100%20SXM4%22%7D%2C%22num_gpus%22%3A%7B%22eq%22%3A1%7D%2C%22gpu_ram%22%3A%7B%22gte%22%3A80000%7D%2C%22rentable%22%3A%7B%22eq%22%3Atrue%7D%2C%22disk_space%22%3A%7B%22gte%22%3A$DISK%7D%2C%22inet_down%22%3A%7B%22gte%22%3A200%7D%2C%22order%22%3A%5B%5B%22dph_total%22%2C%22asc%22%5D%5D%2C%22limit%22%3A10%7D" \
    | python3 -c '
import sys, json
try:
    offs = json.load(sys.stdin).get("offers", [])
except Exception:
    offs = []
offs = [o for o in offs if o.get("verification") in ("verified", "deverified")] or offs
if offs:
    o = offs[0]
    print(o["id"])
    print("[offer] id=%s %s %.0fGB $%.3f/h %s" % (o["id"], o.get("gpu_name"), (o.get("gpu_ram") or 0)/1024,
          o.get("dph_total", 0), o.get("geolocation")), file=sys.stderr)
' 2>/tmp/xeron-bench/offer.txt)
  cat /tmp/xeron-bench/offer.txt 2>/dev/null || true
fi
[ -z "$OFFER" ] && { echo "[offer] 사용 가능한 A100 80GB 없음"; exit 1; }
echo "[offer] using $OFFER"

BODY=$(python3 - "$DISK" "$IMAGE" "$RUN_NAME" <<'PY'
import json, sys, time
disk, image, name = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({"client_id": "me", "image": image, "disk": float(disk), "runtype": "ssh",
                  "label": f"{name}-{time.strftime('%m%d-%H%M')}"}))
PY
)
CREATE=$(curl -s -m 60 -X PUT "$API/asks/$OFFER/" -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' -d "$BODY")
ID=$(echo "$CREATE" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("new_contract") or "")')
[ -z "$ID" ] && { echo "[create] 실패: $CREATE"; exit 1; }
echo "[create] instance=$ID"

HOST=""; PORT=""
for i in $(seq 1 $((MAX_WAIT/10))); do
  INFO=$(curl -s -m 30 "$API/instances/$ID/" -H "Authorization: Bearer $VAST_API_KEY")
  ST=$(echo "$INFO" | python3 -c 'import sys,json;i=(json.load(sys.stdin).get("instances") or {});print(i.get("actual_status") or "?")' 2>/dev/null)
  HOST=$(echo "$INFO" | python3 -c 'import sys,json;i=(json.load(sys.stdin).get("instances") or {});print(i.get("ssh_host") or "")' 2>/dev/null)
  PORT=$(echo "$INFO" | python3 -c 'import sys,json;i=(json.load(sys.stdin).get("instances") or {});print(i.get("ssh_port") or "")' 2>/dev/null)
  echo "[wait#$i] $ST $HOST:$PORT"
  [ "$ST" = "running" ] && [ -n "$HOST" ] && [ -n "$PORT" ] && break
  sleep 10
done
[ -z "$HOST" ] && { echo "[wait] SSH 정보 없음"; exit 1; }

SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
SSH="ssh -p $PORT $SSHOPT root@$HOST"
for i in $(seq 1 30); do $SSH 'echo ok' >/dev/null 2>&1 && { echo "[ssh] connected"; break; }; sleep 10; done

echo "[upload] base snapshot $(du -sh "$BASE_DIR" | cut -f1) + data $(du -sh "$DATA_FILE" | cut -f1) ..."
$SSH 'mkdir -p /root/base /root/data'
scp -q -r $SSHOPT -P "$PORT" "$BASE_DIR/model.safetensors" root@$HOST:/root/base/ && echo "[upload] weights done"
scp -q -r $SSHOPT -P "$PORT" "$BASE_DIR/encoder" "$BASE_DIR/tokenizer" "$BASE_DIR/rl_agent_config.json" root@$HOST:/root/base/ && echo "[upload] aux done"
scp -q $SSHOPT -P "$PORT" "$DATA_FILE" root@$HOST:/root/data/train_long.pt && echo "[upload] data done"

REMOTE=$(cat <<REMOTE_EOF
#!/bin/bash
set -uo pipefail
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q --upgrade "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || true
pip install -q laya transformers datasets safetensors scipy pandas tabulate 2>&1 | tail -2 || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
rm -rf /root/XERON && git clone -q --depth 1 -b $BRANCH https://github.com/PIXELZX0/XERON /root/XERON && echo "[remote] repo $BRANCH ok"
cd /root/XERON && git log --oneline -1
python - <<'PY'
import torch
items = torch.load('/root/data/train_long.pt', weights_only=False)
lens = [len(it['ids']) for it in items]
print('[data] items', len(items), 'tokens', sum(lens), 'min', min(lens), 'max', max(lens), flush=True)
PY
echo "[train] start \$(date -Is)  base=/root/base  epochs=$EPOCHS"
EPOCHS=$EPOCHS MICRO_BATCH=$MICRO_BATCH GRAD_ACCUM=$GRAD_ACCUM GROUP_SIZE=4 \
BATCH_MODE=$BATCH_MODE BATCH_BUCKET_MULT=$BATCH_BUCKET_MULT MAX_TOKENS_BATCH=16384 \
LR_ENCODER=$LR_ENCODER LR_HEAD=$LR_HEAD SIGMA_START=$SIGMA_START SIGMA_END=$SIGMA_END \
RL_WEIGHT=0.5 WEIGHT_DECAY=0.02 HEAD_LAYERS=4 HEAD_SIZE=1024 HEAD_DROPOUT=0.0 \
DTYPE=bf16 MAX_LEN=8192 HEAD_MAX_LEN=256 CHECKPOINT_EVERY=1 \
PYTHONUNBUFFERED=1 timeout $MAX_TRAIN_SECS torchrun --standalone --nproc_per_node=1 \
  scripts/train_ddp.py /root/base ./output/$RUN_NAME /root/data/train_long.pt
echo "TRAIN_EXIT=\$?"
echo "[train] done \$(date -Is)"
ls -la ./output/$RUN_NAME 2>/dev/null | head
REMOTE_EOF
)
# 원격 학습은 **detach** 해서 돌린다 — 로컬 SSH/런처가 죽어도 SIGHUP 으로 학습이 죽지 않게.
# (2026-09-27 사고: 로컬 런처가 SIGKILL 되자 원격 train_ddp 가 SIGHUP 으로 종료됐고,
#  인스턴스만 살아남아 계속 과금됨. epoch1 스냅샷은 회수했지만 epoch2 46% 지점에서 유실)
printf '%s' "$REMOTE" | $SSH 'cat > /root/run_train.sh && chmod +x /root/run_train.sh' \
  || { echo "[remote] 스크립트 업로드 실패"; exit 1; }
$SSH 'setsid nohup bash /root/run_train.sh > /root/train.log 2>&1 < /dev/null & echo "[remote] detached"'
echo "[remote] detached; 폴링 시작 ($(date -Is))"
start=$(date +%s)
while true; do
  state=$($SSH 'pgrep -f "train_dd[p]" >/dev/null 2>&1 && echo RUNNING || echo DONE' 2>/dev/null | tail -1)
  # 원격 로그의 마지막 진행률을 주기적으로 남긴다
  $SSH 'tr "\r" "\n" < /root/train.log | grep -aE "^Epoch |TRAIN_EXIT|=== Epoch" | tail -1' 2>/dev/null \
    | tee -a "$LOGDIR/vast_long_train_remote.log" >/dev/null || true
  if [ "$state" = "DONE" ]; then
    echo "[remote] 학습 프로세스 종료 ($(( $(date +%s) - start ))s 폴링)"
    break
  fi
  if [ $(( $(date +%s) - start )) -gt 86400 ]; then
    echo "[remote] 24h 초과 — 중단하고 회수 단계로"; break
  fi
  sleep 60
done
$SSH 'tr "\r" "\n" < /root/train.log | tail -25' 2>&1 | tee -a "$LOGDIR/vast_long_train_remote.log" || true

echo "[pull] snapshot back ..."
mkdir -p "$HOME/XERON/output/$RUN_NAME"
for rel in model.safetensors rl_agent_config.json; do
  scp -q $SSHOPT -P "$PORT" "root@$HOST:/root/XERON/output/$RUN_NAME/$rel" "$HOME/XERON/output/$RUN_NAME/" || echo "[pull] miss $rel"
done
scp -q -r $SSHOPT -P "$PORT" "root@$HOST:/root/XERON/output/$RUN_NAME/encoder" "root@$HOST:/root/XERON/output/$RUN_NAME/tokenizer" "$HOME/XERON/output/$RUN_NAME/" 2>/dev/null || true
echo "[pull] done: $(du -sh "$HOME/XERON/output/$RUN_NAME" 2>/dev/null | cut -f1)"
echo "########## long train finished $(date -Is) ##########"
