# Audio-Only Gap Infilling Failure Analysis

## Current Goal

The project direction is audio-only speech gap infilling:

```text
prefix speech tokens + gap marker + suffix speech tokens -> middle gap speech tokens
```

No transcript or text prompt is used. The model should infer a plausible missing speech-token span from surrounding acoustic-token context.

At this point, the goal is no longer to search for a stronger recipe. The goal is to explain why the current VoiceCraft-X-based audio-only setup failed to produce reliable gap infilling.

## Main Failure Question

The key question is:

```text
Did the code/objective fail globally, or does it only fail once the gap becomes a realistic random speech gap?
```

To test this, two diagnostic experiments were run.

## Probe A: Tiny Word-Aligned Overfit

Purpose:

```text
Check whether the current forward/loss path can learn anything at all
when the task is made extremely easy.
```

Setting:

```text
GPU: 1
script: scripts/train_inpainting_audio_only_roles.py
wds: data/tokenized_wds/train-clean-100-aligned
limit: 16
filtered aligned candidates: 8
eval_on_train: true
mask_mode: word-aligned
word_mask_max_words: 1
require_aligned_mask: true
mask_len: 1-5 frames
eval_mask_len: 3 frames
trainable_scope: speech-head-last-n-llm
unfreeze_llm_layers: 8
lr: 3e-4
scheduler: cosine with 50 warmup steps
gradient_accumulation_steps: 4
max_grad_norm: 0.5
checkpoint saving: disabled
```

Key result:

```text
step 250:
  eval_loss = 0.4106
  eval_masked_token_accuracy = 0.8438

step 500:
  eval_loss = 0.0128
  eval_masked_token_accuracy = 1.0000
  eval_codebook_0_accuracy = 1.0000
  eval_codebook_1_accuracy = 1.0000
  eval_codebook_2_accuracy = 1.0000
  eval_codebook_3_accuracy = 1.0000
```

Interpretation:

```text
The training path is not completely broken.
The model can overfit a very small, very easy aligned-gap setting.
```

This weakens the hypothesis that the implementation is globally invalid. The forward pass, target construction, CE loss, and trainable parameter path are at least capable of producing a strong learning signal under easy conditions.

Important limitation:

```text
This does not prove real gap infilling works.
The experiment has only 8 aligned candidates and very short gaps.
It is an overfit sanity check, not a generalization result.
```

## Probe B: Random Frame Gap Under Similar Training Recipe

Purpose:

```text
Check whether the same training setup collapses when the mask becomes
a realistic random frame gap instead of an aligned short word-level gap.
```

Setting:

```text
GPU: 2
script: scripts/train_inpainting_audio_only_roles.py
wds: data/tokenized_wds/train-clean-100-aligned
limit: 120
eval_split_size: 16
eval_on_train: true
mask_mode: frame
mask_len: 10-30 frames
eval_mask_len: 20 frames
trainable_scope: speech-head-last-n-llm
unfreeze_llm_layers: 8
lr: 3e-4
scheduler: cosine with 50 warmup steps
gradient_accumulation_steps: 4
max_grad_norm: 0.5
checkpoint saving: disabled
```

Key result:

```text
step 250:
  train loss = 7.5383
  train masked_token_accuracy = 0.0185

step 250 eval:
  eval_loss = 6.6461
  eval_masked_token_accuracy = 0.0352
  eval_target_tokens = 1280
  eval_codebook_0_accuracy = 0.0656
  eval_codebook_1_accuracy = 0.0406
  eval_codebook_2_accuracy = 0.0250
  eval_codebook_3_accuracy = 0.0094
```

Interpretation:

```text
The same code path that overfits the easy aligned case fails badly
on random 10-30 frame gaps.
```

This supports the hypothesis that realistic random frame gaps are much harder than the word-aligned sanity setting.

## What These Probes Prove

The two probes together support this conclusion:

```text
The current implementation can learn when the task is made very small and clean,
but it does not learn useful masked-token reconstruction for random frame gaps.
```

Therefore the failure is unlikely to be caused only by a trivial bug such as:

```text
wrong optimizer wiring
all trainable parameters frozen
loss disconnected from model outputs
target labels never used
```

Instead, the stronger failure explanation is:

```text
The current audio-only objective and prompt format are not sufficient
for realistic random gap infilling with VoiceCraft-X.
```

## Main Failure Causes

### 1. Random Frame Gaps Often Cut Inside Speech Units

Random frame masks can remove:

```text
partial phonemes
coarticulation regions
word interiors
boundary transitions
short silence/breath regions
```

This makes exact codec-token reconstruction extremely ambiguous.

In contrast, word-aligned masks create a cleaner missing unit. That makes the task easier, but it is not the same as realistic random dropout.

### 2. Exact Codec Token CE Is Too Strict

The model is trained with CE against one original codec-token sequence.

But audio-only gap infilling is one-to-many:

```text
many possible middle speech segments may sound natural between the same prefix and suffix
```

If the generated segment is plausible but differs from the original codec tokens, CE and token accuracy still treat it as wrong.

This is especially severe for random gaps and longer gaps.

### 3. VoiceCraft-X Was Not Pretrained for This Exact Input Distribution

The current audio-only setup removes:

```text
transcript/text condition
explicit linguistic target
the original text-guided editing structure
```

So the model must adapt from a text-conditioned generation/editing distribution to a speech-token-only infilling distribution.

The tiny overfit result shows adaptation is possible in principle. The random-frame result shows that this adaptation does not scale under the current objective.

### 4. Token Accuracy Does Not Equal Perceptual Quality

Previous generated wav listening checks showed that even when token metrics improve in controlled settings, audible gap quality can remain weak.

So final evaluation should not rely only on:

```text
CE loss
masked token accuracy
codebook accuracy
```

It should also include:

```text
listening tests
boundary continuity
RMS/envelope continuity
log-mel distance
speaker similarity
PESQ/STOI if applicable
ASR WER only as an auxiliary diagnostic
```

## Final Failure Statement

The current result should be reported as:

```text
VoiceCraft-X can be adapted to memorize very small and clean audio-only aligned gaps,
which suggests the implemented training path is functional.
However, under realistic random frame masking, the same setup fails to learn useful
middle-token reconstruction. This indicates that the main limitation is not a simple
training bug, but a mismatch between the audio-only random-gap task and the current
codec-token CE / prompt adaptation strategy.
```

## Recommended Paper/Presentation Framing

Use this framing:

```text
This project explored whether a text-conditioned speech foundation model can be
repurposed for transcript-free audio-only speech gap infilling.
Although the model can overfit a tiny aligned-gap setting, it fails to generalize
to realistic random frame gaps. The failure analysis suggests that future work needs
an objective and evaluation protocol that better match perceptual gap continuity,
rather than relying only on exact codec-token reconstruction.
```

Avoid overclaiming:

```text
Do not claim that VoiceCraft-X successfully performs audio-only gap infilling.
Do not claim that word-aligned masking solves the real random-gap problem.
Do not use token accuracy alone as evidence of perceptual success.
```

