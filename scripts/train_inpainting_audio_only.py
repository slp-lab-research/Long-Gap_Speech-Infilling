from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from transformers import DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, VoiceCraftXTokenizedWDSCollator  # noqa: E402
from helper import load_voicecraftx  # noqa: E402
from train_inpainting_aligned_fim import (  # noqa: E402
    apply_trainable_scope,
    parse_loss_codebooks,
    print_trainable_summary,
    restore_frozen_module_eval_mode,
    select_loss_codebooks,
    _format_metric,
    _progress_metrics,
    _to_float,
)
from train_inpainting_generation_context import (  # noqa: E402
    accuracy_counts,
    delayed_labels,
    detach_cache,
    first_trainable_llm_layer,
    load_generation_context_checkpoint,
    save_generation_context_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VoiceCraft-X on audio-only speech-token gap infilling.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--output-dir", default="exp/inpainting_audio_only")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-split-size", type=int, default=0)
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Evaluate on the beginning of the train subset for overfit sanity checks.",
    )
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=1)
    parser.add_argument("--mask-len-max", type=int, default=5)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--deterministic-mask", action="store_true")
    parser.add_argument("--eval-mask-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--loss-codebooks", default="all")
    parser.add_argument(
        "--trainable-scope",
        choices=["default", "speech-head", "speech-head-last-n-llm"],
        default="speech-head",
    )
    parser.add_argument("--unfreeze-llm-layers", type=int, default=0)
    parser.add_argument("--force-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trainable-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("audio-only generation-context training currently supports --batch-size 1")
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_voicecraftx(config).to(device)
    model.train()
    apply_trainable_scope(model, args)
    if args.init_checkpoint:
        load_generation_context_checkpoint(model, Path(args.init_checkpoint))
    loss_codebooks = parse_loss_codebooks(args.loss_codebooks, int(config.model.num_codebooks))
    print_trainable_summary(model, args)
    print(
        {
            "task": "audio_only_speech_token_gap_infilling",
            "loss_codebooks": args.loss_codebooks,
            "loss_context": "speech_only_prefix_suffix_prefill_teacher_forcing",
            "force_length": str(args.force_length),
        },
        flush=True,
    )

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size < 0:
        raise ValueError("--eval-split-size must be non-negative")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than the selected dataset size")

    train_indices = list(range(selected_size - args.eval_split_size))
    train_dataset = Subset(dataset, train_indices)
    if args.eval_split_size and args.eval_on_train:
        eval_dataset = Subset(dataset, train_indices[: args.eval_split_size])
    elif args.eval_split_size:
        eval_dataset = Subset(dataset, list(range(selected_size - args.eval_split_size, selected_size)))
    else:
        eval_dataset = None
    collator = VoiceCraftXTokenizedWDSCollator(
        speech_mask_idx=int(config.model.speech_mask_idx),
        pad_token=int(config.model.speech_empty_idx),
        mask_len_min=args.mask_len_min,
        mask_len_max=args.mask_len_max,
        max_n_spans=args.max_n_spans,
        mask_sample_dist=args.mask_sample_dist,
        min_gap=args.min_gap,
        eval_mask_len=args.eval_mask_len,
        deterministic_eval=args.deterministic_mask,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_collator = VoiceCraftXTokenizedWDSCollator(
            speech_mask_idx=int(config.model.speech_mask_idx),
            pad_token=int(config.model.speech_empty_idx),
            eval_mask_len=args.eval_mask_len,
            deterministic_eval=True,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collator,
            drop_last=False,
        )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    step = 0
    examples_seen = 0
    loss_sum = 0.0
    loss_window_sum = 0.0
    loss_window_count = 0
    train_start = time.perf_counter()
    dataset_size = len(train_dataset)
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in dataloader:
            speaker_embeddings = batch["speaker_embeddings"]
            if not isinstance(speaker_embeddings, torch.Tensor):
                raise RuntimeError("speaker embeddings are required for VoiceCraft-X audio-only training")
            speaker_embeddings = speaker_embeddings.to(device)
            inference_prompt, middle = build_generation_training_tensors(batch, config, device)
            labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = audio_only_generation_context_forward(
                    model=model,
                    prompt_speech_token=inference_prompt,
                    speaker_emb=speaker_embeddings,
                    labels=labels,
                    force_length=args.force_length,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            grad_norm = None
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            examples_seen += 1
            loss_value = float(loss.detach().cpu().item() * args.gradient_accumulation_steps)
            loss_sum += loss_value
            loss_window_sum += loss_value
            loss_window_count += 1
            if step % args.log_every == 0:
                correct, total, codebook_correct, codebook_total = accuracy_counts(outputs["logits"].detach(), labels.detach())
                metrics = {
                    "loss": _format_metric(loss_value),
                    "loss_moving_avg": _format_metric(loss_window_sum / max(loss_window_count, 1)),
                    "masked_token_accuracy": _format_metric(correct / max(total, 1)),
                    "grad_norm": _format_metric(_to_float(grad_norm)),
                    "lr": _format_metric(optimizer.param_groups[0]["lr"]),
                    "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
                    "target_tokens": str(total),
                    "step": str(step),
                }
                for codebook_idx, (cb_correct, cb_total) in enumerate(zip(codebook_correct, codebook_total)):
                    metrics[f"codebook_{codebook_idx}_accuracy"] = _format_metric(int(cb_correct.item()) / max(int(cb_total.item()), 1))
                metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(metrics, flush=True)
                loss_window_sum = 0.0
                loss_window_count = 0

            if eval_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(model, eval_dataloader, config, args, device, loss_codebooks)
                eval_metrics["epoch"] = _format_metric(examples_seen / max(dataset_size, 1))
                eval_metrics["step"] = str(step)
                eval_metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(eval_metrics, flush=True)

            if not args.no_save and args.save_every > 0 and step % args.save_every == 0:
                save_generation_context_checkpoint(output_dir / f"step-{step:06d}.pt", model, optimizer, step, args)
            if step >= args.max_steps:
                break

    if not args.no_save:
        save_generation_context_checkpoint(output_dir / "last.pt", model, optimizer, step, args)
    train_runtime = time.perf_counter() - train_start
    print(
        {
            "train_runtime": _format_metric(train_runtime),
            "train_samples_per_second": _format_metric(examples_seen / train_runtime if train_runtime > 0 else 0.0),
            "train_steps_per_second": _format_metric(step / train_runtime if train_runtime > 0 else 0.0),
            "train_loss": _format_metric(loss_sum / max(step, 1)),
            "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
        },
        flush=True,
    )


@torch.no_grad()
def evaluate(model, dataloader, config, args, device, loss_codebooks):
    was_training = model.training
    model.eval()
    eval_start = time.perf_counter()
    loss_sum = 0.0
    samples = 0
    steps = 0
    correct_tokens = 0
    total_tokens = 0
    codebook_correct = None
    codebook_total = None
    for batch in dataloader:
        if args.eval_max_batches is not None and steps >= args.eval_max_batches:
            break
        speaker_embeddings = batch["speaker_embeddings"]
        if not isinstance(speaker_embeddings, torch.Tensor):
            raise RuntimeError("speaker embeddings are required for VoiceCraft-X audio-only eval")
        inference_prompt, middle = build_generation_training_tensors(batch, config, device)
        labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)
        outputs = audio_only_generation_context_forward(
            model=model,
            prompt_speech_token=inference_prompt,
            speaker_emb=speaker_embeddings.to(device),
            labels=labels,
            force_length=args.force_length,
        )
        correct, total, batch_codebook_correct, batch_codebook_total = accuracy_counts(outputs["logits"], labels)
        loss_sum += float(outputs["loss"].detach().cpu().item())
        correct_tokens += correct
        total_tokens += total
        if codebook_correct is None:
            codebook_correct = batch_codebook_correct
            codebook_total = batch_codebook_total
        else:
            codebook_correct += batch_codebook_correct
            codebook_total += batch_codebook_total
        samples += 1
        steps += 1
    runtime = time.perf_counter() - eval_start
    if was_training:
        model.train()
        restore_frozen_module_eval_mode(model, args)
    metrics = {
        "eval_loss": _format_metric(loss_sum / max(samples, 1)),
        "eval_masked_token_accuracy": _format_metric(correct_tokens / max(total_tokens, 1)),
        "eval_target_tokens": str(total_tokens),
        "eval_runtime": _format_metric(runtime),
        "eval_samples_per_second": _format_metric(samples / runtime if runtime > 0 else 0.0),
        "eval_steps_per_second": _format_metric(steps / runtime if runtime > 0 else 0.0),
    }
    if codebook_correct is not None and codebook_total is not None:
        for codebook_idx, (correct, total) in enumerate(zip(codebook_correct, codebook_total)):
            metrics[f"eval_codebook_{codebook_idx}_accuracy"] = _format_metric(int(correct.item()) / max(int(total.item()), 1))
    return metrics


def build_generation_training_tensors(batch, config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    if input_ids.shape[0] != 1:
        raise ValueError("audio-only generation-context training currently supports batch size 1")
    active_positions = labels[0].ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if active_positions.numel() == 0:
        raise RuntimeError("batch has no masked middle labels")
    middle_start = int(active_positions.min().item())
    middle_end = int(active_positions.max().item()) + 1
    prompt_without_middle = input_ids[:, :, :middle_start]
    middle = labels[:, :, middle_start:middle_end].long()
    cushion = torch.tensor(
        [int(x) for x in config.model.silence_tokens],
        dtype=input_ids.dtype,
        device=device,
    ).view(1, -1, 1).expand(1, -1, int(config.model.silence_nums))
    inference_prompt = torch.cat([prompt_without_middle, cushion], dim=-1)
    return inference_prompt, middle


def audio_only_generation_context_forward(model, prompt_speech_token, speaker_emb, labels, force_length: bool):
    hidden_states = build_audio_only_prefill_hidden(model, prompt_speech_token, speaker_emb)
    logits_by_step = []
    cache = DynamicCache()
    llm_is_frozen = not any(param.requires_grad for param in model.llm.parameters())
    for step_idx in range(labels.shape[-1]):
        attention_mask = torch.tril(
            torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]),
                device=hidden_states.device,
            )
        ).to(torch.bool)
        if llm_is_frozen:
            with torch.no_grad():
                outputs = model.llm.model(
                    inputs_embeds=hidden_states.detach(),
                    attention_mask=attention_mask[:, -1, :],
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=cache,
                )
            cache = detach_cache(outputs.past_key_values)
            step_hidden = outputs.last_hidden_state[:, -1:]
        else:
            step_hidden, cache = llm_generation_step_with_frozen_prefix(
                model=model,
                hidden_states=hidden_states,
                attention_mask=attention_mask[:, -1, :],
                cache=cache,
            )
        step_logits = torch.stack([model.llm_decoder[k](step_hidden) for k in range(model.num_codebooks)], dim=1).squeeze(2)
        step_logits[:, :, model.speech_empty_idx] = -1e9
        step_logits[:, :, model.speech_mask_idx] = -1e9
        if force_length:
            step_logits[:, :, model.speech_eos_idx] = -1e9
        logits_by_step.append(step_logits)
        if step_idx + 1 < labels.shape[-1]:
            next_tokens = labels[:, :, step_idx].clamp_min(int(model.speech_empty_idx))
            next_speech_token = torch.stack(
                [model.speech_embedding[k](next_tokens[:, k]) for k in range(model.num_codebooks)],
                dim=1,
            ).sum(dim=1, keepdim=True)
            hidden_states = next_speech_token
    logits = torch.stack(logits_by_step, dim=2)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    return {"loss": loss, "logits": logits}


def llm_generation_step_with_frozen_prefix(model, hidden_states: torch.Tensor, attention_mask: torch.Tensor, cache: DynamicCache):
    llm_model = model.llm.model
    past_seen_tokens = cache.get_seq_length() if cache is not None else 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + hidden_states.shape[1],
        device=hidden_states.device,
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = llm_model._update_causal_mask(
        attention_mask,
        hidden_states,
        cache_position,
        cache,
        False,
    )
    position_embeddings = llm_model.rotary_emb(hidden_states, position_ids)

    first_trainable_layer = first_trainable_llm_layer(model)
    for layer_idx, decoder_layer in enumerate(llm_model.layers[: llm_model.config.num_hidden_layers]):
        if layer_idx < first_trainable_layer:
            with torch.no_grad():
                layer_outputs = decoder_layer(
                    hidden_states.detach(),
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0].detach()
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
    hidden_states = llm_model.norm(hidden_states)
    return hidden_states[:, -1:], detach_cache(cache)


def build_audio_only_prefill_hidden(model, prompt_speech_token, speaker_emb):
    shifted = model.shift_batch(prompt_speech_token)[:, :, :-int(model.config.silence_nums)]
    speech_emb = model.embed_speech(shifted)
    speaker_emb = model.spk_embed_affine_layer(F.normalize(speaker_emb, dim=-1)).unsqueeze(1)
    return torch.cat([speaker_emb, speech_emb], dim=1)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
