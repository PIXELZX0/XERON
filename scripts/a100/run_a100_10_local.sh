#!/bin/bash
# XERON-1.0 A100 벤치 로컬 오케스트레이터.
#   pod 프로비저닝 -> 원격 하네스(scripts/a100/run_a100_10.sh) 실행 -> 즉시 pod 삭제 -> 잔존 확인
# 지출 상한 보호: gpu.jsonc 의 keep_alive_minutes/max_idle_spend + 아래 WAIT_TIMEOUT + 무조건 stop.
#
# 사용: nohup bash scripts/a100/run_a100_10_local.sh > /dev/null 2>&1 &
#   env: GPU_TYPE(기본 A100 SXM4)  WAIT_TIMEOUT(2400)  RUN_TAG
set -uo pipefail
export PATH="$HOME/.gpu-cli/bin:$PATH"
cd "$(dirname "$0")/../.." || exit 1

GPU_TYPE="${GPU_TYPE:-A100 SXM4}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2700}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
LOGDIR="outputs/bench"
mkdir -p "$LOGDIR" logs/bulk
LOG="logs/bulk/W6_a100_bench_runner.log"
exec > >(tee -a "$LOG") 2>&1

echo "########## runner start $(date -Is) tag=$RUN_TAG ##########"

# Kaggle 토큰(사설 데이터셋). 값은 절대 출력하지 않는다.
KENV=()
if [ -f "$HOME/.kaggle/access_token" ]; then
  KW=$(tr -d '\r\n' < "$HOME/.kaggle/access_token")
  if [ -n "$KW" ]; then KENV=(-e "KAGGLE_API_TOKEN=$KW"); echo "[runner] kaggle token loaded (len=${#KW})"; fi
fi

REMOTE_CMD='cd XERON 2>/dev/null || true; pwd; git fetch -q --depth 1 origin main 2>/dev/null; git reset -q --hard origin/main 2>/dev/null; git log --oneline -1; export PYTHONUNBUFFERED=1 BUDGET_SECS=${BUDGET_SECS:-1800} STEPS_BASE=${STEPS_BASE:-100} STEPS_OTHER=${STEPS_OTHER:-40} SAMPLE_ITEMS=${SAMPLE_ITEMS:-40000}; timeout 2700 bash scripts/a100/run_a100_10.sh; echo "HARNESS_EXIT=$?"'

echo "[runner] submitting..."
SUBMIT=$(gpu run -d --json --progress-style minimal --gpu-type "$GPU_TYPE" "${KENV[@]}" bash -c "$REMOTE_CMD" 2>&1)
echo "$SUBMIT" | tee "$LOGDIR/submit_$RUN_TAG.json"
JOB=$(echo "$SUBMIT" | sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\(job_[^"]*\)".*/\1/p' | head -1)
echo "[runner] job_id=$JOB"
if [ -z "$JOB" ]; then echo "[runner] submit 실패"; gpu status 2>&1 | head -20; exit 1; fi

echo "[runner] waiting (timeout ${WAIT_TIMEOUT}s)..."
set +e
gpu wait "$JOB" --timeout "$WAIT_TIMEOUT" --json | tee "$LOGDIR/wait_$RUN_TAG.json"
rc=${PIPESTATUS[0]}
set -e
echo "[runner] wait rc=$rc"

echo "[runner] pulling job logs..."
gpu logs -j "$JOB" > "logs/bulk/W6_a100_bench.gpu.log" 2>&1
echo "[runner] log lines: $(wc -l < logs/bulk/W6_a100_bench.gpu.log)"

echo "[runner] stopping pod (무조건)..."
gpu stop -y 2>&1 | tail -5
sleep 3
echo "[runner] pod status after stop:"
gpu status 2>&1 | tee "$LOGDIR/pod_status_$RUN_TAG.txt" | head -30
gpu status --all --json > "$LOGDIR/pod_status_$RUN_TAG.json" 2>&1 || true

echo "[runner] artifacts:"
ls -la outputs/bench/ logs/bulk/ 2>/dev/null | head -40
echo "########## runner done $(date -Is) ##########"
