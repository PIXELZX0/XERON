#!/bin/bash
# XERON-1.0 SPMD 처리량 프로브 — 학습 루프 자체를 micro 단위로 계측한다.
#
#   ./run_spmd_probe.sh              # 기본: PROBE_N=24, PROBE_METRICS=1, MAX_ITEMS=2000
#   PROBE_N=48 ./run_spmd_probe.sh   # 더 길게
#
# 왜 필요한가: v8 스모크는 20분 예산에 micro 30개(=960 seq)만 돌았는데 로그의
#   `19.3 seq/s` 는 **계획된 아이템 수(20,000)/경과시간** 이라 실제(0.93 seq/s)의 20배였다.
#   같은 런의 스텝 로그는 255/221/276 s/step(8 micro) = ~30 s/micro 로 일정했고,
#   DIAG 는 캐시가 데워진 같은 shape 를 **0.4 s/micro** 로 쟀다(컴파일 1회성).
#   즉 29.6 s/micro 는 입력 경로(MpDeviceLoader/H2D)이거나 매 micro 재컴파일이다.
#   이 프로브가 data 대기 시간과 XLA 카운터를 직접 찍어 둘 중 하나를 확정한다.
#
# 세션을 아끼려고 DIAG=0(학습 경로) + PROBE_N 으로 한 번에 잰다. DCP 저장은 하지 않는다.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
KDIR="$REPO/kaggle/kernel-spmd-short"
REF="pistonx/xeron-1-0-short-spmd"
LOG_DIR="$REPO/kaggle/logs"
mkdir -p "$LOG_DIR"

PN="${PROBE_N:-24}"
PM="${PROBE_METRICS:-1}"
RE="${RATE_EVERY:-8}"
ITEMS="${PROBE_ITEMS:-2000}"
MINS="${PROBE_MIN:-30}"

sed -i -E "s/^DIAG = _i\(\"DIAG\", [0-9]+\)/DIAG = _i(\"DIAG\", 0)/" "$KDIR/kernel.py"
sed -i -E "s/^SHARD_INDEX = _i\(\"SHARD_INDEX\", [0-9]+\)/SHARD_INDEX = _i(\"SHARD_INDEX\", 0)/" "$KDIR/kernel.py"
sed -i -E "s/^MAX_ITEMS = _i\(\"MAX_ITEMS\", [0-9]+\)/MAX_ITEMS = _i(\"MAX_ITEMS\", ${ITEMS})/" "$KDIR/kernel.py"
sed -i -E "s/^MINUTES = _i\(\"MAX_TRAIN_MIN\", [0-9]+\)/MINUTES = _i(\"MAX_TRAIN_MIN\", ${MINS})/" "$KDIR/kernel.py"
sed -i -E "s/^CKPT_MODE = _s\(\"CKPT_MODE\", \"[a-z0-9]*\"\)/CKPT_MODE = _s(\"CKPT_MODE\", \"local\")/" "$KDIR/kernel.py"
grep -n '^SHARD_INDEX\|^MAX_ITEMS\|^MINUTES\|^CKPT_MODE\|^DIAG' "$KDIR/kernel.py"

# PROBE_* 는 커널이 _s() 로 읽으므로 kernel.py 의 기본값을 직접 바꿔 넣는다.
python3 - "$KDIR/kernel.py" "$PN" "$PM" "$RE" <<'PY'
import re, sys
p, pn, pm, re_ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
src = open(p, encoding="utf-8").read()
src = re.sub(r'_s\("PROBE_N", "[^"]*"\)', f'_s("PROBE_N", "{pn}")', src)
src = re.sub(r'_s\("PROBE_METRICS", "[^"]*"\)', f'_s("PROBE_METRICS", "{pm}")', src)
src = re.sub(r'_s\("RATE_EVERY", "[^"]*"\)', f'_s("RATE_EVERY", "{re_}")', src)
open(p, "w", encoding="utf-8").write(src)
PY
grep -n 'PROBE_N", \|PROBE_METRICS", \|RATE_EVERY", ' "$KDIR/kernel.py"

echo "[push] $REF (DIAG=0 PROBE_N=$PN PROBE_METRICS=$PM RATE_EVERY=$RE, shard 0, MAX_ITEMS=$ITEMS, MAX_TRAIN_MIN=$MINS, CKPT_MODE=local)"
kaggle kernels push -p "$KDIR" 2>&1 | tee -a "$LOG_DIR/probe.log"

sleep 20
kaggle kernels status "$REF" 2>&1 | tee -a "$LOG_DIR/probe.log" || true
echo
echo "진행:  kaggle kernels logs -f $REF"
echo "회수:  kaggle kernels logs $REF > $LOG_DIR/probe-raw.log"
