#!/bin/bash
# XERON-1.0 W5 rerun — step A: verify the rebuilt mix (dbpedia included).
# Run AFTER scripts/build_mix_x10.py finished (data/xeron10_mix.jsonl final).
set -euo pipefail
cd /home/yuchan/XERON
export PYTHONPATH=scripts

echo "=== [A1] verify_bulk.py (full mix) ==="
.venv/bin/python scripts/verify_bulk.py "data/xeron10_mix.jsonl" \
    --json-out data/xeron10_mix_verify.json

echo
echo "=== [A2] check_eval_leak.py (JevBench public-231 + typed-decisions test) ==="
.venv/bin/python scripts/check_eval_leak.py \
    --mix data/xeron10_mix.jsonl --json-out data/xeron10_eval_leak.json

echo
echo "=== [A3] posdev_x10.py (true per-namespace positional deviation) ==="
.venv/bin/python scripts/posdev_x10.py \
    --mix data/xeron10_mix.jsonl --json-out data/xeron10_posdev.json

echo
echo "=== [A] DONE ==="
