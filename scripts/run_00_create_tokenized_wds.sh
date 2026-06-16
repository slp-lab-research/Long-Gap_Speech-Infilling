#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src "$PYTHON_BIN" scripts/create_tokenized_wds.py \
  --root data/LibriTTS_R \
  --split train-clean-100 \
  --output-dir data/tokenized_wds/train-clean-100 \
  --prefix librittsr-train-clean-100 \
  --shard-size 1000 \
  --device cuda \
  --overwrite
