#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src:.:scripts "$PYTHON_BIN" scripts/diagnose_mask_word_overlap.py \
  --wds-root data/tokenized_wds/train-clean-100-aligned \
  --output-dir exp/mask_word_overlap_word1_check \
  --num-samples 200 \
  --mode word-aligned \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --word-mask-max-words 1 \
  --codec-frame-rate 50
