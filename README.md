# Audio-Only Speech Gap Infilling with VoiceCraft-X

This repository contains the code for our GenAI final project:
**audio-only long-gap speech infilling** with
[VoiceCraft-X](https://github.com/zszheng147/VoiceCraft-X).

The goal is to test whether a pretrained LLM-based speech foundation model can
fill a missing speech segment using only surrounding speech-token context. In
contrast to text-guided speech editing, this project removes transcript
conditioning and trains the model to predict the missing codec tokens from the
prefix and suffix speech tokens.

```text
prefix speech tokens + gap marker + suffix speech tokens
    -> missing middle speech tokens
```

The repository is organized to reproduce the training setup, generate samples,
and inspect experiment outputs.

## Project Structure

```text
src/
  dataset/                 Dataset, masking, and tokenized WDS utilities
  models/                  VoiceCraft-X model code
  utils/                   Audio tokenization helpers

scripts/
  create_tokenized_wds.py                  Build tokenized LibriTTS-R WDS
  train_inpainting_audio_only_roles.py     Train audio-only role-FIM infilling
  generate_inpainting_samples_audio_only_roles.py
                                           Generate wav samples from a checkpoint
  evaluate.py                              PESQ/STOI/WER/speaker-sim evaluation

tests/
  Unit tests for masking, WDS loading, and training tensor shapes

docs/
  Experiment notes and failure analysis

figures/
  Report figures generated from the experiments
```

## Environment

The project was run with Python 3.10 on RTX 4090 GPUs.

```bash
conda create -n voicecraftx python=3.10
conda activate voicecraftx
pip install -r requirements.txt
pip install -r requirements-inpainting.txt
```

Locally tested major packages:

```text
torch==2.1.0
torchaudio==2.1.0
torchvision==0.16.0
transformers==4.51.3
flash-attn==2.5.6
```

## Data and Checkpoints

The dataset and model weights are not included in this repository. Prepare the
VoiceCraft-X pretrained files under:

```text
pretrained_models/
  voicecraftx.ckpt
  encodec.th
  speech_campplus.onnx
  config.json
  tokenizer.json
  vocab.json
```

The upstream pretrained files are available from the official
[VoiceCraft-X Hugging Face repository](https://huggingface.co/zhisheng01/VoiceCraft-X).
Project checkpoints used only for sample generation can be placed under `exp/`
when available.

Prepare LibriTTS-R locally as:

```text
data/LibriTTS_R/
  train-clean-100/
  dev-clean/
  test-clean/
```

## Usage

Run commands from the repository root.

### 1. Build Tokenized WebDataset

The raw wav files are converted once into VoiceCraft-X codec tokens. This avoids
running the codec encoder during every training step.

```bash
GPU=0 bash scripts/run_00_create_tokenized_wds.sh
```

Output:

```text
data/tokenized_wds/train-clean-100
```

### 2. Train

Train the audio-only infilling model on tokenized speech data:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:.:scripts python scripts/train_inpainting_audio_only_roles.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --output-dir exp/audio_only_gap_infilling \
  --limit 1000 \
  --eval-split-size 100 \
  --max-steps 3000 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-grad-norm 0.5 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --lr-scheduler-type cosine \
  --log-every 100 \
  --eval-every 500 \
  --eval-max-batches 100 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --eval-mask-len 20 \
  --mask-mode frame \
  --trainable-scope speech-head-last-n-llm \
  --unfreeze-llm-layers 8 \
  --save-best \
  --no-save-last
```

For a quick smoke test, reduce `--limit` and `--max-steps`.

### 3. Generate Samples

If a checkpoint has been downloaded separately, generate listening samples:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:.:scripts python scripts/generate_inpainting_samples_audio_only_roles.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --checkpoint exp/audio_only_gap_infilling/best.pt \
  --output-dir exp/audio_only_generation_samples \
  --limit 1000 \
  --eval-split-size 100 \
  --num-samples 8 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --eval-mask-len 20 \
  --mask-mode frame \
  --temperature 1.0 \
  --top-k 1 \
  --device cuda
```

Each generated sample directory contains:

```text
original.wav
masked_input.wav
generated_matched.wav
generated_gap_matched.wav
metadata.json
```

Listen in this order:

```text
original.wav -> masked_input.wav -> generated_matched.wav
```

### 4. Evaluate Generated Samples

For generated sample directories, compute PESQ/STOI, ASR WER, and speaker
similarity:

```bash
pip install -r requirements-eval.txt
```

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --samples-root exp/audio_only_generation_samples \
  --method-name audio_only \
  --output-dir exp/quantitative_eval
```

This command may download Whisper and SpeechBrain models into the evaluation
output directory unless `--model-cache-dir` is provided.

## Tests

Run the lightweight unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

These tests do not require the full pretrained checkpoints. Full training and
generation require the local VoiceCraft-X pretrained files.

## Experiment Notes

Detailed experiment notes and failure analysis are available in:

```text
docs/audio_only_failure_analysis.md
docs/audio_only_gap_infilling_experiment_summary.md
docs/EXPERIMENTS_INPAINTING.md
```

## Attribution

This repository is based on the official VoiceCraft-X codebase:

```bibtex
@inproceedings{zheng2025voicecraft,
  title={VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing},
  author={Zheng, Zhisheng and Peng, Puyuan and Diwan, Anuj and Huynh, Cong Phuoc and Sun, Xiaohang and Liu, Zhu and Bhat, Vimal and Harwath, David},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={2737--2756},
  year={2025}
}
```

Voice cloning and speech editing should only be performed with speaker consent.
