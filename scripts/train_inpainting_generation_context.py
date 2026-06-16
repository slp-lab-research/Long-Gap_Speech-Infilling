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

from dataset.text_gap_alignment import encode_aligned_fim_texts  # noqa: E402
from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, VoiceCraftXTokenizedWDSCollator  # noqa: E402
from helper import load_tokenizer, load_voicecraftx  # noqa: E402
from utils.simple_lora import (  # noqa: E402
    apply_lora_to_last_llm_layers,
    lora_config_from_args,
    trainable_state_dict,
)
from train_inpainting_aligned_fim import (  # noqa: E402
    apply_trainable_scope,
    parse_loss_codebooks,
    print_trainable_summary,
    restore_frozen_module_eval_mode,
    select_loss_codebooks,
    _format_metric,
    _progress_metrics,
    _source_summary,
    _to_float,
)
from train_inpainting_energy import load_or_build_token_energy_table, token_rms_energy_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VoiceCraft-X inpainting directly on generation-context CE.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--output-dir", default="exp/inpainting_generation_context")
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
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=10)
    parser.add_argument("--mask-len-max", type=int, default=30)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--deterministic-mask", action="store_true")
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--language", default="english")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--loss-codebooks", default="all")
    parser.add_argument("--energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-table-samples", type=int, default=5000)
    parser.add_argument("--energy-table-path", default=None)
    parser.add_argument("--energy-log-scale", type=float, default=100.0)
    parser.add_argument("--text-gap-mode", choices=["auto", "aligned", "proportional", "whole"], default="auto")
    parser.add_argument("--trainable-scope", choices=["default", "speech-head", "speech-head-last-n-llm"], default="speech-head")
    parser.add_argument("--unfreeze-llm-layers", type=int, default=0)
    parser.add_argument("--force-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--save-trainable-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only trainable parameters. This keeps LoRA/speech-head checkpoints small.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("generation-context training currently supports --batch-size 1")
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_tokenizer, _ = load_tokenizer(config)
    model = load_voicecraftx(config).to(device)
    model.train()
    apply_trainable_scope(model, args)
    lora_injected = 0
    if args.lora:
        lora_config = lora_config_from_args(args)
        lora_injected = apply_lora_to_last_llm_layers(
            model,
            num_layers=lora_config.num_layers,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
        restore_frozen_module_eval_mode(model, args)
    if args.init_checkpoint:
        load_generation_context_checkpoint(model, Path(args.init_checkpoint))
    loss_codebooks = parse_loss_codebooks(args.loss_codebooks, int(config.model.num_codebooks))
    print_trainable_summary(model, args)
    print(
        {
            "text_gap_mode": args.text_gap_mode,
            "loss_codebooks": args.loss_codebooks,
            "loss_context": "generation_prefill_teacher_forcing",
            "force_length": str(args.force_length),
            "lora": str(args.lora),
            "lora_injected": str(lora_injected),
        },
        flush=True,
    )

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size < 0:
        raise ValueError("--eval-split-size must be non-negative")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than the selected dataset size")

    energy_table = None
    if args.energy_loss_weight > 0:
        energy_table_path = Path(args.energy_table_path) if args.energy_table_path else output_dir / "token_energy_table.pt"
        energy_table = load_or_build_token_energy_table(
            dataset=dataset,
            config=config,
            path=energy_table_path,
            max_samples=min(args.energy_table_samples, selected_size),
        ).to(device)

    train_dataset = Subset(dataset, list(range(selected_size - args.eval_split_size)))
    eval_dataset = Subset(dataset, list(range(selected_size - args.eval_split_size, selected_size))) if args.eval_split_size else None

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
        batch_size=args.batch_size,
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
    ce_sum = 0.0
    energy_sum = 0.0
    loss_window_sum = 0.0
    loss_window_count = 0
    train_start = time.perf_counter()
    dataset_size = len(train_dataset)
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in dataloader:
            text, text_attention_mask, text_splits = encode_batch_text(batch, text_tokenizer, config, args)
            speaker_embeddings = batch["speaker_embeddings"]
            if not isinstance(speaker_embeddings, torch.Tensor):
                raise RuntimeError("speaker embeddings are required for VoiceCraft-X training")

            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            speaker_embeddings = speaker_embeddings.to(device)
            inference_prompt, middle = build_generation_training_tensors(batch, config, device)
            labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = generation_context_forward(
                    model=model,
                    text=text,
                    text_attention_mask=text_attention_mask,
                    prompt_speech_token=inference_prompt,
                    speaker_emb=speaker_embeddings,
                    labels=labels,
                    force_length=args.force_length,
                )
                ce_loss = outputs["loss"]
                energy_loss = outputs["logits"].new_tensor(0.0)
                if energy_table is not None:
                    energy_loss = token_rms_energy_loss(
                        logits=outputs["logits"],
                        shifted_labels=labels,
                        energy_table=energy_table,
                        log_scale=args.energy_log_scale,
                    )
                total_loss = ce_loss + args.energy_loss_weight * energy_loss
                loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            grad_norm = None
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            examples_seen += int(inference_prompt.shape[0])
            loss_value = float(total_loss.detach().cpu().item())
            ce_value = float(ce_loss.detach().cpu().item())
            energy_value = float(energy_loss.detach().cpu().item())
            loss_sum += loss_value
            ce_sum += ce_value
            energy_sum += energy_value
            loss_window_sum += loss_value
            loss_window_count += 1
            if step % args.log_every == 0:
                correct, total, codebook_correct, codebook_total = accuracy_counts(outputs["logits"].detach(), labels.detach())
                metrics = {
                    "loss": _format_metric(loss_value),
                    "ce_loss": _format_metric(ce_value),
                    "energy_loss": _format_metric(energy_value),
                    "loss_moving_avg": _format_metric(loss_window_sum / max(loss_window_count, 1)),
                    "masked_token_accuracy": _format_metric(correct / max(total, 1)),
                    "grad_norm": _format_metric(_to_float(grad_norm)),
                    "lr": _format_metric(optimizer.param_groups[0]["lr"]),
                    "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
                    "target_tokens": str(total),
                    "text_gap_source": _source_summary(text_splits),
                    "step": str(step),
                }
                for codebook_idx, (cb_correct, cb_total) in enumerate(zip(codebook_correct, codebook_total)):
                    metrics[f"codebook_{codebook_idx}_accuracy"] = _format_metric(int(cb_correct.item()) / max(int(cb_total.item()), 1))
                metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(metrics, flush=True)
                loss_window_sum = 0.0
                loss_window_count = 0

            if eval_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(model, eval_dataloader, text_tokenizer, config, args, device, loss_codebooks)
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
            "train_ce_loss": _format_metric(ce_sum / max(step, 1)),
            "train_energy_loss": _format_metric(energy_sum / max(step, 1)),
            "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
        },
        flush=True,
    )


@torch.no_grad()
def evaluate(model, dataloader, text_tokenizer, config, args, device, loss_codebooks):
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
        text, text_attention_mask, _ = encode_batch_text(batch, text_tokenizer, config, args)
        speaker_embeddings = batch["speaker_embeddings"]
        if not isinstance(speaker_embeddings, torch.Tensor):
            raise RuntimeError("speaker embeddings are required for VoiceCraft-X eval")
        inference_prompt, middle = build_generation_training_tensors(batch, config, device)
        labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)
        outputs = generation_context_forward(
            model=model,
            text=text.to(device),
            text_attention_mask=text_attention_mask.to(device),
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


def encode_batch_text(batch, text_tokenizer, config, args):
    return encode_aligned_fim_texts(
        text_tokenizer=text_tokenizer,
        texts=batch["texts"],
        mask_intervals=batch["mask_intervals"],
        y_lens=batch["y_lens"],
        language=args.language,
        pad_token_id=int(config.model.llm_padding_idx),
        metadata=batch["metadata"],
        codec_frame_rate=int(config.CODEC_FRAME_RATE),
        mode=args.text_gap_mode,
    )


def build_generation_training_tensors(batch, config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    if input_ids.shape[0] != 1:
        raise ValueError("generation-context training currently supports batch size 1")
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


def generation_context_forward(model, text, text_attention_mask, prompt_speech_token, speaker_emb, labels, force_length: bool):
    hidden_states = build_generation_prefill_hidden(model, text, prompt_speech_token, speaker_emb)
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


def detach_cache(cache):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache = [tensor.detach() for tensor in cache.key_cache]
        cache.value_cache = [tensor.detach() for tensor in cache.value_cache]
    return cache


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


def first_trainable_llm_layer(model) -> int:
    for layer_idx, layer in enumerate(model.llm.model.layers):
        if any(param.requires_grad for param in layer.parameters()):
            return layer_idx
    return len(model.llm.model.layers)


def build_generation_prefill_hidden(model, text, prompt_speech_token, speaker_emb):
    text_emb = model.llm.model.embed_tokens(text)
    shifted = model.shift_batch(prompt_speech_token)[:, :, :-int(model.config.silence_nums)]
    speech_emb = model.embed_speech(shifted)
    speaker_emb = model.spk_embed_affine_layer(F.normalize(speaker_emb, dim=-1)).unsqueeze(1)
    return torch.cat([text_emb, speaker_emb, speech_emb], dim=1)


def delayed_labels(middle: torch.Tensor, num_codebooks: int, ignore_index: int = -100) -> torch.Tensor:
    batch_size, _, middle_len = middle.shape
    labels = torch.full(
        (batch_size, num_codebooks, middle_len + num_codebooks),
        ignore_index,
        dtype=middle.dtype,
        device=middle.device,
    )
    for codebook_idx in range(num_codebooks):
        labels[:, codebook_idx, codebook_idx : codebook_idx + middle_len] = middle[:, codebook_idx]
    return labels


def accuracy_counts(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    correct = predictions.eq(labels) & mask
    codebook_correct = correct.sum(dim=(0, 2)).detach().cpu()
    codebook_total = mask.sum(dim=(0, 2)).detach().cpu()
    return int(correct.sum().item()), int(mask.sum().item()), codebook_correct, codebook_total


def save_generation_context_checkpoint(path: Path, model, optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = trainable_state_dict(model) if args.save_trainable_only else model.state_dict()
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "trainable_only": bool(args.save_trainable_only),
        },
        path,
    )


def load_generation_context_checkpoint(model, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        {
            "init_checkpoint": str(path),
            "missing_keys": str(len(missing)),
            "unexpected_keys": str(len(unexpected)),
        },
        flush=True,
    )


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
