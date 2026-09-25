#!/bin/bash
# XERON-1.0 W5 — step B: 8192 preprocessing of the (verified) mix.
#   index -> 10 shards x 10 procs -> merge -> length stats -> smoke
# Every stage writes to a temp file and renames (atomic), and skips when its output is
# already valid (idempotent), so a kill / gateway restart can be resumed by re-running this.
set -euo pipefail
cd /home/yuchan/XERON
export PYTHONPATH=scripts
MIX=data/xeron10_mix.jsonl
SHDIR=data/x10_8192_shards
MODEL=/home/yuchan/laya-models/xeron-0.9-base-8192
ITEMS=train_items_x10_8192.pt
mkdir -p "$SHDIR"

echo "=== [B1] line-offset index ==="
if [ -f "$SHDIR/mix.lineidx.npz" ]; then
    echo "[skip] index exists: $SHDIR/mix.lineidx.npz"
else
    .venv/bin/python scripts/preprocess_shard.py --data-files "$MIX" \
        --index-file "$SHDIR/mix.lineidx.npz" --build-index-only
fi

echo
echo "=== [B2] 10 shards x 10 procs, MAX_LEN=8192 HEAD_MAX_LEN=256 ==="
missing=0
for i in $(seq 0 9); do [ -f "$SHDIR/shard_$i.pt" ] || missing=1; done
if [ "$missing" = "0" ]; then
    echo "[skip] all 10 shards exist"
else
    seq 0 9 | xargs -P 10 -I{} env PYTHONPATH=scripts MAX_LEN=8192 HEAD_MAX_LEN=256 \
        .venv/bin/python scripts/preprocess_shard.py \
        --data-files "$MIX" \
        --index-file "$SHDIR/mix.lineidx.npz" \
        --shard {} --num-shards 10 \
        --output "$SHDIR/shard_{}.pt" \
        --model-id "$MODEL"
fi

echo
echo "=== [B3] merge shards -> $ITEMS ==="
.venv/bin/python scripts/merge_shards_x10.py \
    --shards "$SHDIR/shard_*.pt" --output "$ITEMS" \
    --stats data/xeron10_preprocess_stats_8192.json

echo
echo "=== [B4] length stats (4096 vs 8192) ==="
.venv/bin/python scripts/lenstats_x10.py \
    --items 4096=train_items_x10.pt \
    --items 8192="$ITEMS" \
    --json-out data/xeron10_lenstats_8192.json

echo
echo "=== [B5] smoke (8192 model dir) ==="
.venv/bin/python scripts/smoke_x10.py --items "$ITEMS" \
    --model-id "$MODEL" --n 200 --max-len 8192

echo
echo "=== [B] DONE ==="
