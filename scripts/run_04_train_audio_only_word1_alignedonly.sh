#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src:.:scripts "$PYTHON_BIN" scripts/train_inpainting_audio_only_roles.py \
  --wds-root data/tokenized_wds/train-clean-100-aligned \
  --output-dir exp/inpainting_audio_only_roles_word1_alignedonly_bestonly_3000 \
  --limit 120 \
  --eval-split-size 16 \
  --eval-on-train \
  --max-steps 3000 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-grad-norm 0.5 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --lr-scheduler-type cosine \
  --log-every 100 \
  --eval-every 500 \
  --eval-max-batches 16 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --eval-mask-len 20 \
  --mask-mode word-aligned \
  --word-mask-max-words 1 \
  --require-aligned-mask \
  --trainable-scope speech-head-last-n-llm \
  --unfreeze-llm-layers 8 \
  --save-every 0 \
  --save-best \
  --no-save-last
