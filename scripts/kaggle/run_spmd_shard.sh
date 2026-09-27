#!/bin/bash
# XERON-1.0 숏컨텍스트 — Kaggle TPU v5e-8 4세션 파이프라인 러너.
#
#   ./run_spmd_shard.sh 0        # 0번 샤드 세션 push (9h)
#   ./run_spmd_shard.sh 0 --smoke  # 스모크: 2만 seq만
#
# 커널은 매 세션마다 자기 샤드의 DCP 체크포인트를 HF(CKPT_HF)에서 이어받고,
# 끝나면 다시 올린다. Kaggle 제약: 주 20h / 1세션 최대 9h / 동시 1세션.
#   → 4샤드 × ~9h = 1에폭, 약 1.8주 (20h/주 한도)
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
  sed -i -E "s/^MAX_ITEMS = _i\(\"MAX_ITEMS\", [0-9]+\)/MAX_ITEMS = _i(\"MAX_ITEMS\", 20000)/" "$KDIR/kernel.py"
  sed -i -E "s/^MINUTES = _i\(\"MAX_TRAIN_MIN\", [0-9]+\)/MINUTES = _i(\"MAX_TRAIN_MIN\", 40)/" "$KDIR/kernel.py"
  echo "[smoke] MAX_ITEMS=20000 MAX_TRAIN_MIN=40"
else
  sed -i -E "s/^MAX_ITEMS = _i\(\"MAX_ITEMS\", [0-9]+\)/MAX_ITEMS = _i(\"MAX_ITEMS\", 0)/" "$KDIR/kernel.py"
  sed -i -E "s/^MINUTES = _i\(\"MAX_TRAIN_MIN\", [0-9]+\)/MINUTES = _i(\"MAX_TRAIN_MIN\", 0)/" "$KDIR/kernel.py"
fi

echo "[push] $REF (shard $SHARD)"
kaggle kernels push -p "$KDIR" 2>&1 | tee -a "$LOG_DIR/shard${SHARD}.log"

# Kaggle CLI 는 push 후 상태 폴링만 제공: 상태/로그 확인용 안내
sleep 20
kaggle kernels status "$REF" 2>&1 | tee -a "$LOG_DIR/shard${SHARD}.log" || true
echo
echo "진행 확인:  kaggle kernels logs -f $REF"
echo "결과 회수:  kaggle kernels output $REF -p $LOG_DIR/out-shard${SHARD}"
