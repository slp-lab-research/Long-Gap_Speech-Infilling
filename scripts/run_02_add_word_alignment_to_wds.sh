#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" scripts/add_word_alignment_to_wds.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --alignment-root data/mfa_alignments/train-clean-100 \
  --output-dir data/tokenized_wds/train-clean-100-aligned \
  --alignment-format auto \
  --missing skip
