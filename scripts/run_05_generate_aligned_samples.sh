#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-exp/inpainting_audio_only_roles_word1_alignedonly_bestonly_3000/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/inpainting_audio_only_roles_word1_alignedonly_best_samples}"

CUDA_VISIBLE_DEVICES="$GPU" NUMBA_CACHE_DIR=/tmp/voicecraftx_numba_cache MPLCONFIGDIR=/tmp/voicecraftx_mpl \
  PYTHONPATH=src:.:scripts "$PYTHON_BIN" scripts/generate_inpainting_samples_audio_only_roles.py \
  --wds-root data/tokenized_wds/train-clean-100-aligned \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --eval-split-size 100 \
  --num-samples 8 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --eval-mask-len 20 \
  --mask-mode word-aligned \
  --word-mask-max-words 1 \
  --require-aligned-mask \
  --top-k 1 \
  --device cuda
