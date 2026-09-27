#!/bin/bash
# A/B bench: BATCH_MODE=random (legacy) vs length (length-bucketed + token budget).
#
# Runs the real XERON-1.0 training step on a rented A100 80GB with a randomly initialised
# model (no weights needed), on a 20k-item sample that preserves the true length distribution.
# Remote work is detached (setsid+noop) so a local SSH drop cannot kill it.
#
# env: VAST_OFFER(자동 선택) DATA_FILE(bench_items_20k.pt) STEPS(12) WARMUP(3) BRANCH(perf/train-throughput)
set -uo pipefail

: "${VAST_API_KEY:?VAST_API_KEY 필요 (secretEnv)}"
OFFER="${VAST_OFFER:-}"
DISK="${INSTANCE_DISK:-60}"
MAX_WAIT="${MAX_WAIT:-900}"
STEPS="${STEPS:-12}"
WARMUP="${WARMUP:-3}"
BRANCH="${BRANCH:-perf/train-throughput}"
DATA_FILE="${DATA_FILE:-$HOME/XERON/bench_items_20k.pt}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
API="https://console.vast.ai/api/v0"
KEY="$HOME/.ssh/xeron_vast_ed25519"
LOGDIR=/tmp/xeron-bench
mkdir -p "$LOGDIR"
LOG="$LOGDIR/bench_ab.log"
exec > >(tee -a "$LOG") 2>&1

ID=""
cleanup() {
  if [ -n "$ID" ]; then
    echo "[cleanup] destroying instance $ID"
    curl -s -m 30 -X DELETE "$API/instances/$ID/" -H "Authorization: Bearer $VAST_API_KEY" | head -c 120; echo
  fi
}
trap cleanup EXIT

echo "########## A/B bench start $(date -Is) ##########"
[ -f "$DATA_FILE" ] || { echo "데이터 없음: $DATA_FILE"; exit 1; }
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "xeron-vast-direct" -f "$KEY"
curl -s -m 30 -X POST "$API/ssh/" -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"ssh_key":sys.argv[1]}))' "$(cat "$KEY.pub")")" >/dev/null

if [ -z "$OFFER" ]; then
  OFFER=$(curl -s -m 60 "$API/bundles/?q=%7B%22gpu_name%22%3A%7B%22eq%22%3A%22A100%20SXM4%22%7D%2C%22num_gpus%22%3A%7B%22eq%22%3A1%7D%2C%22gpu_ram%22%3A%7B%22gte%22%3A80000%7D%2C%22rentable%22%3A%7B%22eq%22%3Atrue%7D%2C%22disk_space%22%3A%7B%22gte%22%3A$DISK%7D%2C%22inet_down%22%3A%7B%22gte%22%3A200%7D%2C%22order%22%3A%5B%5B%22dph_total%22%2C%22asc%22%5D%5D%2C%22limit%22%3A10%7D" \
    | python3 -c '
import sys, json
offs = json.load(sys.stdin).get("offers", [])
offs = [o for o in offs if o.get("verification") in ("verified", "deverified")] or offs
if offs:
    o = offs[0]; print(o["id"])
    print("[offer] %s %s %.0fGB $%.3f/h %s" % (o["id"], o.get("gpu_name"), (o.get("gpu_ram") or 0)/1024,
          o.get("dph_total", 0), o.get("geolocation")), file=sys.stderr)
' 2>"$LOGDIR/offer_ab.txt")
  cat "$LOGDIR/offer_ab.txt" 2>/dev/null || true
fi
[ -z "$OFFER" ] && { echo "[offer] 사용 가능한 A100 80GB 없음"; exit 1; }
echo "[offer] using $OFFER"

BODY=$(python3 - "$DISK" "$IMAGE" <<'PY'
import json, sys, time
disk, image = sys.argv[1], sys.argv[2]
print(json.dumps({"client_id": "me", "image": image, "disk": float(disk), "runtype": "ssh",
                  "label": f"xeron-ab-{time.strftime('%m%d-%H%M')}"}))
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

SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30"
SSH="ssh -p $PORT $SSHOPT root@$HOST"
for i in $(seq 1 30); do $SSH 'echo ok' >/dev/null 2>&1 && { echo "[ssh] connected"; break; }; sleep 10; done

echo "[upload] data $(du -h "$DATA_FILE" | cut -f1)"
$SSH 'mkdir -p /root/data' && scp -q $SSHOPT -P "$PORT" "$DATA_FILE" root@$HOST:/root/data/bench_items.pt && echo "[upload] ok"

REMOTE=$(cat <<REMOTE_EOF
set -uo pipefail
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q --upgrade "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || true
pip install -q laya transformers safetensors scipy pandas tabulate 2>&1 | tail -2 || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
rm -rf /root/XERON && git clone -q --depth 1 -b $BRANCH https://github.com/PIXELZX0/XERON /root/XERON && echo "[remote] repo $BRANCH ok"
cd /root/XERON && git log --oneline -1
python - <<'PY'
import torch
items = torch.load('/root/data/bench_items.pt', weights_only=False)
lens = [len(it['ids']) for it in items]
print('[data]', len(items), 'items mean', round(sum(lens)/len(lens),1), 'max', max(lens),
      'p90', sorted(lens)[int(len(lens)*0.9)], '>1024', sum(1 for L in lens if L>1024), flush=True)
PY
echo "==== A/B: mb8 budget16384 ===="
for MODE in random length; do
  echo "---- BATCH_MODE=\$MODE ----"
  OUT_JSONL=/root/ab_mb8.jsonl TAG=ab_mb8_\$MODE BATCH_MODE=\$MODE MICRO_BATCH=8 GRAD_ACCUM=4 \\
  MAX_TOKENS_BATCH=16384 MAX_LEN=8192 GRAD_CKPT=1 DTYPE=bf16 DATA_MODE=file \\
  DATA_FILE=/root/data/bench_items.pt SAMPLE_ITEMS=0 STEPS=$STEPS WARMUP=$WARMUP \\
  python scripts/bench_xeron10.py 2>&1 | grep -aE "step|tok_s|BENCH|\[bench\]|peak|OOM|Traceback|Error" | tail -12
done
echo "==== A/B: mb2 budget16384 ===="
for MODE in random length; do
  echo "---- BATCH_MODE=\$MODE ----"
  OUT_JSONL=/root/ab_mb2.jsonl TAG=ab_mb2_\$MODE BATCH_MODE=\$MODE MICRO_BATCH=2 GRAD_ACCUM=16 \\
  MAX_TOKENS_BATCH=16384 MAX_LEN=8192 GRAD_CKPT=1 DTYPE=bf16 DATA_MODE=file \\
  DATA_FILE=/root/data/bench_items.pt SAMPLE_ITEMS=0 STEPS=$STEPS WARMUP=$WARMUP \\
  python scripts/bench_xeron10.py 2>&1 | grep -aE "step|tok_s|BENCH|\[bench\]|peak|OOM|Traceback|Error" | tail -12
done
echo "AB_DONE"
REMOTE_EOF
)
printf '%s' "$REMOTE" | $SSH 'cat > /root/run_ab.sh && chmod +x /root/run_ab.sh' || { echo "[remote] 업로드 실패"; exit 1; }
$SSH 'setsid nohup /root/run_ab.sh > /root/ab.log 2>&1 < /dev/null & echo "[remote] detached"'
start=$(date +%s)
while true; do
  state=$($SSH 'pgrep -f "bench_xeron10|run_ab.sh" >/dev/null 2>&1 && echo RUNNING || echo DONE' 2>/dev/null | tail -1)
  $SSH 'tail -c 1200 /root/ab.log' 2>/dev/null | tail -3 | sed 's/^/[ab] /'
  [ "$state" = "DONE" ] && { echo "[remote] 완료"; break; }
  [ $(( $(date +%s) - start )) -gt 5400 ] && { echo "[remote] 90분 초과 중단"; break; }
  sleep 45
done
$SSH 'grep -a AB_DONE /root/ab.log || tail -30 /root/ab.log' | tail -5
echo "==== 결과 회수 ===="
for f in ab_mb8 ab_mb2; do scp -q $SSHOPT -P "$PORT" "root@$HOST:/root/$f.jsonl" "$LOGDIR/" 2>/dev/null && echo "pulled $f.jsonl"; done
echo "########## A/B bench finished $(date -Is) ##########"
