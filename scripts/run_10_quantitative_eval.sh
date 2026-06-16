#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/quantitative_eval}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-exp/eval_model_cache}"
WHISPER_MODEL="${WHISPER_MODEL:-base.en}"

if [ "$#" -eq 0 ]; then
  set -- --samples-root "exp/audio_only_generation_samples" --method-name "audio_only"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/evaluate.py \
  "$@" \
  --output-dir "$OUTPUT_DIR" \
  --model-cache-dir "$MODEL_CACHE_DIR" \
  --device "$DEVICE" \
  --whisper-model "$WHISPER_MODEL"
