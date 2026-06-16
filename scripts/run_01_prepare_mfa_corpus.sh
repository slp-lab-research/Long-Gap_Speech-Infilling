#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" scripts/prepare_mfa_corpus.py \
  --libritts-root data/LibriTTS_R \
  --split train-clean-100 \
  --output-dir data/mfa_corpus/train-clean-100

cat <<'MSG'

MFA corpus is ready under data/mfa_corpus/train-clean-100.
Run Montreal Forced Aligner externally, then place alignments under:
  data/mfa_alignments/train-clean-100

After that, run:
  bash scripts/run_02_add_word_alignment_to_wds.sh
MSG
