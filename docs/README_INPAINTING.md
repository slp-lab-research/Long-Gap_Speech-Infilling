# VoiceCraft-X Audio-Only Speech Gap Infilling

This repository state is for the current team handoff.

The active project direction is **audio-only speech gap infilling**:

- No transcript/text prompt is used for training.
- Input is speech-token context only: prefix speech tokens + suffix speech tokens.
- Target is the missing middle speech-token span.
- Current best setup masks one word-aligned gap and filters out fallback frame masks.

This is not a final high-quality model yet. The current code is a reproducible experimental baseline for continuing the work from the same state.

## Current Best Result

Best checkpoint from the latest local run:

```text
exp/inpainting_audio_only_roles_word1_alignedonly_bestonly_3000/best.pt
```

This checkpoint is provided separately when needed. It can also be regenerated
with the training command below.

Current best run summary:

```text
script: scripts/train_inpainting_audio_only_roles.py
dataset: data/tokenized_wds/train-clean-100-aligned
mask_mode: word-aligned
word_mask_max_words: 1
require_aligned_mask: enabled
limit: 120
train/eval after filtering: 81 / 16
max_steps: 3000
trainable_scope: speech-head-last-n-llm
unfreeze_llm_layers: 8
eval_masked_token_accuracy: 0.5694
generated_audio_triage_avg_score: 40.08
generated_audio_triage: pass 0 / partial 2 / fail 6
```

The model has learned to place non-silent speech-like audio in the gap, but it does **not** reliably reconstruct the correct content or natural boundaries yet.

## Required Local Files

Large files are ignored and must exist locally:

```text
VoiceCraft-X/
  pretrained_models/
    voicecraftx.ckpt
    voicecraftx-en.ckpt
    voicecraftx-zh.ckpt
    encodec.th
    speech_campplus.onnx
  data/
    LibriTTS_R/
```

The local experiment used Python 3.10. The shell wrappers use the active
`python` by default. Override it when needed:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_04_train_audio_only_word1_alignedonly.sh
```

## Environment

Start from the official VoiceCraft-X environment, then install/pin the extra experiment dependencies in:

```text
requirements-inpainting.txt
```

Locally tested major versions:

```text
torch==2.1.0
torchaudio==2.1.0
torchvision==0.16.0
transformers==4.51.3
flash-attn==2.5.6
setuptools==69.5.1
```

## Reproducible Workflow

All current commands are wrapped as shell scripts under `scripts/run_*.sh`.

### 1. Tokenize LibriTTS-R Into WDS

```bash
GPU=0 bash scripts/run_00_create_tokenized_wds.sh
```

Output:

```text
data/tokenized_wds/train-clean-100
```

### 2. Prepare MFA Corpus

```bash
bash scripts/run_01_prepare_mfa_corpus.sh
```

Output:

```text
data/mfa_corpus/train-clean-100
```

Then run Montreal Forced Aligner externally and place alignments at:

```text
data/mfa_alignments/train-clean-100
```

### 3. Add Word Alignment Metadata To WDS

```bash
bash scripts/run_02_add_word_alignment_to_wds.sh
```

Output:

```text
data/tokenized_wds/train-clean-100-aligned
```

### 4. Check Mask Quality

```bash
bash scripts/run_03_diagnose_word_masks.sh
```

Expected direction:

```text
partial_or_mixed_rate should be 0.0 for word-aligned masks.
```

### 5. Train Current Best Audio-Only Recipe

```bash
GPU=0 bash scripts/run_04_train_audio_only_word1_alignedonly.sh
```

Important parameters:

```text
--mask-mode word-aligned
--word-mask-max-words 1
--require-aligned-mask
--trainable-scope speech-head-last-n-llm
--unfreeze-llm-layers 8
--gradient-accumulation-steps 4
--max-grad-norm 0.5
--lr 3e-4
--warmup-steps 50
--lr-scheduler-type cosine
--save-best
--no-save-last
```

Output:

```text
exp/inpainting_audio_only_roles_word1_alignedonly_bestonly_3000/best.pt
```

Only the best checkpoint is saved to avoid filling the disk.

### 6. Generate Listening Samples

```bash
GPU=0 bash scripts/run_05_generate_aligned_samples.sh
```

Output:

```text
exp/inpainting_audio_only_roles_word1_alignedonly_best_samples/best
```

Each sample directory contains:

```text
original.wav
masked_input.wav
generated_matched.wav
generated_gap_matched.wav
metadata.json
```

Listen in this order:

```text
original.wav -> masked_input.wav -> generated_matched.wav -> generated_gap_matched.wav
```

The generation script also supports `--require-aligned-mask`; the wrapper enables it so fallback frame masks are skipped.

### 7. Run Signal-Level Audio Triage

```bash
bash scripts/run_06_evaluate_generated_samples.sh
```

Output:

```text
audio_agent_report.md
audio_agent_report.json
```

This report is only a quick signal-level triage. Human listening is still required.

## Important Code Files

Core current path:

```text
src/dataset/tokenized_wds.py
scripts/create_tokenized_wds.py
scripts/prepare_mfa_corpus.py
scripts/add_word_alignment_to_wds.py
scripts/diagnose_mask_word_overlap.py
scripts/train_inpainting_audio_only_roles.py
scripts/generate_inpainting_samples_audio_only_roles.py
scripts/evaluate_generated_audio_agent.py
```

Support modules still used by the current script:

```text
scripts/train_inpainting_audio_only.py
scripts/train_inpainting_generation_context.py
scripts/train_inpainting_aligned_fim.py
scripts/train_inpainting_energy.py
scripts/generate_inpainting_samples_audio_only.py
```

These support modules contain helper functions imported by the active audio-only role scripts.

## Experiment History

Detailed experiment history is in:

```text
EXPERIMENTS_INPAINTING.md
audio_only_gap_infilling_experiment_summary.md
```

Key conclusion so far:

```text
The model is learning part of the task, but current audio quality is not good enough.
Word-aligned one-word masks helped substantially.
Fallback frame masks should be avoided for listening/eval.
Next work should focus on generation quality, boundary continuity, and content-aware gap reconstruction.
```

## Git Hygiene

Ignored by `.gitignore`:

```text
pretrained_models/
data/LibriTTS_R/
data/tokenized_wds/
data/mfa_corpus/
data/mfa_alignments/
exp/
tmp/
*.pt
*.wav
*.log
```

Only code, docs, shell wrappers, and lightweight tests should be committed.
