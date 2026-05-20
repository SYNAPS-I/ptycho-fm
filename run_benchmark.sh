#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python benchmark_dummy_training.py \
  --iterations 1000 \
  --batch-size 64 \
  --num-workers 4 \
  "$@"
