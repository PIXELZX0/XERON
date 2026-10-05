#!/bin/bash
# XERON-1.0 숏컨텍스트 — Kaggle TPU v5e-8 4세션 파이프라인 러너.
#
#   ./run_spmd_shard.sh 0            # 0번 샤드 본런 (DCP 를 HF 로 올려 이어받기)
#   ./run_spmd_shard.sh 0 --smoke    # 스모크: 2만 seq, 20분 예산, DCP 는 로컬만
#
# 커널은 매 세션마다 자기 샤드의 DCP 체크포인트를 HF(CKPT_HF)에서 이어받고,
# 끝나면 다시 올린다. Kaggle 제약: 주 20h / 1세션 최대 9h / 동시 1세션.
#   → 4샤드 × ~9h = 1에폭, 약 1.8주 (20h/주 한도)
#
# 본런 시간 예산(MAX_TRAIN_MIN): 9h 세션 한도 안에서 DCP 저장(~12분)+HF 업로드(7.9GB)를
# 끝내야 하므로 기본 420분(7h).  예: MAX_TRAIN_MIN=480 ./run_spmd_shard.sh 0
set -euo pipefail

SHARD="${1:?usage: $0 <shard_index 0..3> [--smoke]}"
SMOKE="${2:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
KDIR="$REPO/kaggle/kernel-spmd-short"
REF="pistonx/xeron-1-0-short-spmd"
LOG_DIR="$REPO/kaggle/logs"
mkdir -p "$LOG_DIR"

# 커널 파일의 SHARD_INDEX 기본값을 이번 세션 샤드로 바꿔 push (push 시 env 주입이 불가하므로)
sed -i -E "s/^SHARD_INDEX = _i\(\"SHARD_INDEX\", [0-9]+\)/SHARD_INDEX = _i(\"SHARD_INDEX\", ${SHARD})/" "$KDIR/kernel.py"

# BASE_HF 를 지정하면 그 값으로 베이스를 바꾼다 (미지정 = 커널 기본값 유지)
if [ -n "${BASE_HF:-}" ]; then
  sed -i -E "s|^BASE_HF = _s\(\"BASE_HF\", \"[^\"]*\"\)|BASE_HF = _s(\"BASE_HF\", \"${BASE_HF}\")|" "$KDIR/kernel.py"
fi
grep -n '^SHARD_INDEX\|^BASE_HF' "$KDIR/kernel.py"

if [ "$SMOKE" = "--smoke" ]; then
  ITEMS="${SMOKE_ITEMS:-20000}"
  MINS="${SMOKE_MIN:-20}"
  MODE="${SMOKE_CKPT:-local}"
else
  ITEMS="${MAIN_ITEMS:-0}"
  MINS="${MAX_TRAIN_MIN:-420}"
  MODE="${MAIN_CKPT:-hf}"
fi

sed -i -E "s/^MAX_ITEMS = _i\(\"MAX_ITEMS\", [0-9]+\)/MAX_ITEMS = _i(\"MAX_ITEMS\", ${ITEMS})/" "$KDIR/kernel.py"
sed -i -E "s/^MINUTES = _i\(\"MAX_TRAIN_MIN\", [0-9]+\)/MINUTES = _i(\"MAX_TRAIN_MIN\", ${MINS})/" "$KDIR/kernel.py"
sed -i -E "s/^CKPT_MODE = _s\(\"CKPT_MODE\", \"[a-z0-9]*\"\)/CKPT_MODE = _s(\"CKPT_MODE\", \"${MODE}\")/" "$KDIR/kernel.py"
# 프로브 잔여 설정 제거: 본런은 PROBE_N=0 이어야 한다(프로브 커널이 남긴 값이 그대로 따라오면
# 24 micro 에서 조기 종료해 DCP 저장 경로를 타지 않는다).
PN="${PROBE_N:-0}"
sed -i -E "s/^_probe = _s\(\"PROBE_N\", \"[0-9]*\"\)/_probe = _s(\"PROBE_N\", \"${PN}\")/" "$KDIR/kernel.py"
grep -n '^SHARD_INDEX\|^MAX_ITEMS\|^MINUTES\|^CKPT_MODE\|^_probe' "$KDIR/kernel.py"
echo "[push] $REF (shard $SHARD, MAX_ITEMS=$ITEMS, MAX_TRAIN_MIN=$MINS, CKPT_MODE=$MODE)"
kaggle kernels push -p "$KDIR" 2>&1 | tee -a "$LOG_DIR/shard${SHARD}.log"

# Kaggle CLI 는 push 후 상태 폴링만 제공: 상태/로그 확인용 안내
sleep 20
kaggle kernels status "$REF" 2>&1 | tee -a "$LOG_DIR/shard${SHARD}.log" || true
echo
echo "진행 확인:  kaggle kernels logs -f $REF"
echo "결과 회수:  kaggle kernels output $REF -p $LOG_DIR/out-shard${SHARD}"
