#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_ROOT="${SAMPLES_ROOT:-exp/inpainting_audio_only_roles_word1_alignedonly_best_samples/best}"

PYTHONPATH=src:.:scripts "$PYTHON_BIN" scripts/evaluate_generated_audio_agent.py \
  --samples-root "$SAMPLES_ROOT" \
  --top-k 8
