#!/bin/bash
# XERON-1.0 preprocessing: 10 shards, 10 procs (12 cores)
cd /home/yuchan/XERON
seq 0 9 | xargs -P 10 -I{} env PYTHONPATH=scripts MAX_LEN=4096 HEAD_MAX_LEN=256 \
    .venv/bin/python scripts/preprocess_shard.py \
    --data-files data/xeron10_mix.jsonl \
    --index-file data/x10_shards/mix.lineidx.npz \
    --shard {} --num-shards 10 \
    --output data/x10_shards/shard_{}.pt \
    --model-id /home/yuchan/laya-models/xeron-0.9-base
