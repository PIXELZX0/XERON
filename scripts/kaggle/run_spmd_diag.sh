#!/bin/bash
# XERON-1.0 SPMD 처리량 진단 (DIAG=1) — 같은 커널 슬롯을 쓴다.
#
#   ./run_spmd_diag.sh          # 진단: 변형별 micro 4개 (full / ce / full_fixed_eps / fwd)
#
# 학습은 하지 않는다(DIAG=1 이면 트레이너가 진단 후 즉시 return). CKPT 는 로컬(업로드 없음).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
KDIR="$REPO/kaggle/kernel-spmd-short"
REF="pistonx/xeron-1-0-short-spmd"
LOG_DIR="$REPO/kaggle/logs"
mkdir -p "$LOG_DIR"

sed -i -E "s/^DIAG = _i\(\"DIAG\", [0-9]+\)/DIAG = _i(\"DIAG\", 1)/" "$KDIR/kernel.py"
sed -i -E "s/^SHARD_INDEX = _i\(\"SHARD_INDEX\", [0-9]+\)/SHARD_INDEX = _i(\"SHARD_INDEX\", 0)/" "$KDIR/kernel.py"
sed -i -E "s/^MAX_ITEMS = _i\(\"MAX_ITEMS\", [0-9]+\)/MAX_ITEMS = _i(\"MAX_ITEMS\", 2000)/" "$KDIR/kernel.py"
sed -i -E "s/^MINUTES = _i\(\"MAX_TRAIN_MIN\", [0-9]+\)/MINUTES = _i(\"MAX_TRAIN_MIN\", 1)/" "$KDIR/kernel.py"
sed -i -E "s/^CKPT_MODE = _s\(\"CKPT_MODE\", \"[a-z0-9]*\"\)/CKPT_MODE = _s(\"CKPT_MODE\", \"local\")/" "$KDIR/kernel.py"
grep -n '^SHARD_INDEX\|^MAX_ITEMS\|^MINUTES\|^CKPT_MODE\|^DIAG' "$KDIR/kernel.py"
echo "[push] $REF (DIAG=1, shard 0, MAX_ITEMS=2000, MAX_TRAIN_MIN=1, CKPT_MODE=local)"
kaggle kernels push -p "$KDIR" 2>&1 | tee -a "$LOG_DIR/diag.log"

sleep 20
kaggle kernels status "$REF" 2>&1 | tee -a "$LOG_DIR/diag.log" || true
echo
echo "진행:  kaggle kernels logs -f $REF"
echo "회수:  kaggle kernels logs $REF > $LOG_DIR/diag-raw.log"
