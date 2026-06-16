from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from transformers import DynamicCache
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from dataset.tokenized_wds import (  # noqa: E402
    TokenizedLibriTTSRWDSDataset,
    VoiceCraftXTokenizedWDSCollator,
    word_aligned_mask_candidates,
)
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
from train_inpainting_audio_only import (  # noqa: E402
    build_generation_training_tensors,
    llm_generation_step_with_frozen_prefix,
)
from train_inpainting_generation_context import (  # noqa: E402
    accuracy_counts,
    delayed_labels,
    detach_cache,
    load_generation_context_checkpoint,
)
from train_inpainting_energy import load_or_build_token_energy_table, token_rms_energy_loss  # noqa: E402


PREFIX_ROLE = 0
SUFFIX_ROLE = 1
MIDDLE_ROLE = 2
GAP_ROLE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VoiceCraft-X audio-only infilling with learnable speech FIM role embeddings.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--output-dir", default="exp/inpainting_audio_only_roles")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-split-size", type=int, default=0)
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=10)
    parser.add_argument("--mask-len-max", type=int, default=30)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--mask-mode", choices=["frame", "word-aligned"], default="frame")
    parser.add_argument("--word-mask-max-words", type=int, default=4)
    parser.add_argument(
        "--require-aligned-mask",
        action="store_true",
        help="When using --mask-mode word-aligned, filter out samples without a valid aligned mask instead of falling back to frame masks.",
    )
    parser.add_argument("--codec-frame-rate", type=float, default=None)
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--deterministic-mask", action="store_true")
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--loss-codebooks", default="all")
    parser.add_argument("--trainable-scope", choices=["default", "speech-head", "speech-head-last-n-llm"], default="speech-head")
    parser.add_argument("--unfreeze-llm-layers", type=int, default=0)
    parser.add_argument("--force-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--role-init-std", type=float, default=0.02)
    parser.add_argument(
        "--prompt-format",
        choices=["roles-fim", "original-mask-roles"],
        default="roles-fim",
        help=(
            "roles-fim keeps the existing speech-only FIM prompt. "
            "original-mask-roles keeps an explicit speech mask marker between prefix and suffix, "
            "then adds learnable role embeddings."
        ),
    )
    parser.add_argument("--save-trainable-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-table-samples", type=int, default=1000)
    parser.add_argument("--energy-table-path", default=None)
    parser.add_argument("--energy-log-scale", type=float, default=100.0)
    parser.add_argument(
        "--context-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary speech-only CE weight on prefix/suffix tokens. The masked middle remains supervised by the main loss.",
    )
    return parser.parse_args()


def filter_word_aligned_indices(
    dataset: TokenizedLibriTTSRWDSDataset,
    indices: list[int],
    mask_len_min: int,
    mask_len_max: int,
    codec_frame_rate: float,
    word_mask_max_words: int,
) -> list[int]:
    filtered: list[int] = []
    for index in indices:
        item = dataset[index]
        tokens = item["tokens"]
        candidates = word_aligned_mask_candidates(
            metadata=item.get("metadata", {}),
            y_len=int(tokens.shape[-1]),
            mask_len_min=mask_len_min,
            mask_len_max=mask_len_max,
            codec_frame_rate=codec_frame_rate,
            max_words=word_mask_max_words,
        )
        if candidates:
            filtered.append(index)
    return filtered


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("audio-only roles training currently supports --batch-size 1")
    if args.require_aligned_mask and args.mask_mode != "word-aligned":
        raise ValueError("--require-aligned-mask requires --mask-mode word-aligned")
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    codec_frame_rate = float(args.codec_frame_rate or config.CODEC_FRAME_RATE)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_voicecraftx(config).to(device)
    model.train()
    apply_trainable_scope(model, args)
    if args.init_checkpoint:
        load_generation_context_checkpoint(model, Path(args.init_checkpoint))

    hidden_size = int(model.llm.model.embed_tokens.embedding_dim)
    role_count = 4 if args.prompt_format == "original-mask-roles" else 3
    role_embeddings = torch.nn.Embedding(role_count, hidden_size).to(device)
    role_embeddings.prompt_format = args.prompt_format
    torch.nn.init.normal_(role_embeddings.weight, mean=0.0, std=args.role_init_std)

    loss_codebooks = parse_loss_codebooks(args.loss_codebooks, int(config.model.num_codebooks))
    print_trainable_summary(model, args)
    print(
        {
            "task": "audio_only_speech_token_gap_infilling_roles",
            "role_format": (
                "speaker + PREFIX_ROLE + prefix + GAP_ROLE + speech_mask + SUFFIX_ROLE + suffix + MIDDLE_ROLE"
                if args.prompt_format == "original-mask-roles"
                else "speaker + PREFIX_ROLE + prefix + SUFFIX_ROLE + suffix + MIDDLE_ROLE"
            ),
            "prompt_format": args.prompt_format,
            "role_trainable_params": str(sum(param.numel() for param in role_embeddings.parameters())),
            "loss_codebooks": args.loss_codebooks,
            "force_length": str(args.force_length),
            "mask_mode": args.mask_mode,
            "codec_frame_rate": _format_metric(codec_frame_rate),
            "energy_loss_weight": _format_metric(args.energy_loss_weight),
            "context_loss_weight": _format_metric(args.context_loss_weight),
        },
        flush=True,
    )

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_indices = list(range(min(args.limit, len(dataset)) if args.limit is not None else len(dataset)))
    if args.require_aligned_mask:
        selected_indices = filter_word_aligned_indices(
            dataset=dataset,
            indices=selected_indices,
            mask_len_min=args.mask_len_min,
            mask_len_max=args.mask_len_max,
            codec_frame_rate=codec_frame_rate,
            word_mask_max_words=args.word_mask_max_words,
        )
        if not selected_indices:
            raise RuntimeError("no samples have a valid word-aligned mask candidate")
        print({"aligned_candidate_filtered_size": str(len(selected_indices))}, flush=True)
    selected_size = len(selected_indices)
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

    train_indices = selected_indices[: selected_size - args.eval_split_size]
    train_dataset = Subset(dataset, train_indices)
    if args.eval_split_size and args.eval_on_train:
        eval_dataset = Subset(dataset, train_indices[: args.eval_split_size])
    elif args.eval_split_size:
        eval_dataset = Subset(dataset, selected_indices[selected_size - args.eval_split_size :])
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
        mask_mode=args.mask_mode,
        codec_frame_rate=codec_frame_rate,
        word_mask_max_words=args.word_mask_max_words,
        require_word_aligned_mask=args.require_aligned_mask,
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
            mask_len_min=args.mask_len_min,
            mask_len_max=args.mask_len_max,
            max_n_spans=args.max_n_spans,
            mask_sample_dist=args.mask_sample_dist,
            min_gap=args.min_gap,
            eval_mask_len=args.eval_mask_len,
            deterministic_eval=True,
            mask_mode=args.mask_mode,
            codec_frame_rate=codec_frame_rate,
            word_mask_max_words=args.word_mask_max_words,
            require_word_aligned_mask=args.require_aligned_mask,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collator,
            drop_last=False,
        )

    trainable_params = [param for param in model.parameters() if param.requires_grad] + list(role_embeddings.parameters())
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_optimizer_steps = max(1, (args.max_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    scheduler = build_scheduler(optimizer, args, total_optimizer_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    step = 0
    examples_seen = 0
    loss_sum = 0.0
    ce_sum = 0.0
    energy_sum = 0.0
    context_sum = 0.0
    loss_window_sum = 0.0
    loss_window_count = 0
    train_start = time.perf_counter()
    dataset_size = len(train_dataset)
    optimizer_step = 0
    best_eval_accuracy = -1.0
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in dataloader:
            speaker_embeddings = batch["speaker_embeddings"]
            if not isinstance(speaker_embeddings, torch.Tensor):
                raise RuntimeError("speaker embeddings are required for VoiceCraft-X audio-only roles training")
            speaker_embeddings = speaker_embeddings.to(device)
            inference_prompt, middle = build_generation_training_tensors(batch, config, device)
            labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = audio_only_roles_forward(
                    model=model,
                    role_embeddings=role_embeddings,
                    prompt_speech_token=inference_prompt,
                    speaker_emb=speaker_embeddings,
                    labels=labels,
                    force_length=args.force_length,
                )
                ce_loss = outputs["loss"]
                energy_loss = outputs["logits"].new_tensor(0.0)
                context_loss = outputs["logits"].new_tensor(0.0)
                if energy_table is not None:
                    energy_loss = token_rms_energy_loss(
                        logits=outputs["logits"],
                        shifted_labels=labels,
                        energy_table=energy_table,
                        log_scale=args.energy_log_scale,
                    )
                if args.context_loss_weight > 0:
                    context_tokens, context_labels = build_context_training_tensors(
                        batch,
                        device,
                        speech_mask_idx=int(model.speech_mask_idx),
                    )
                    context_labels = select_loss_codebooks(context_labels, loss_codebooks)
                    context_loss = audio_only_context_forward(
                        model=model,
                        speech_tokens=context_tokens,
                        speaker_emb=speaker_embeddings,
                        labels=context_labels,
                    )
                total_loss = ce_loss + args.energy_loss_weight * energy_loss + args.context_loss_weight * context_loss
                loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            grad_norm = None
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer_step += 1
                optimizer.zero_grad(set_to_none=True)

            step += 1
            examples_seen += 1
            loss_value = float(total_loss.detach().cpu().item())
            ce_value = float(ce_loss.detach().cpu().item())
            energy_value = float(energy_loss.detach().cpu().item())
            context_value = float(context_loss.detach().cpu().item())
            loss_sum += loss_value
            ce_sum += ce_value
            energy_sum += energy_value
            context_sum += context_value
            loss_window_sum += loss_value
            loss_window_count += 1
            if step % args.log_every == 0:
                correct, total, codebook_correct, codebook_total = accuracy_counts(outputs["logits"].detach(), labels.detach())
                metrics = {
                    "loss": _format_metric(loss_value),
                    "ce_loss": _format_metric(ce_value),
                    "energy_loss": _format_metric(energy_value),
                    "context_loss": _format_metric(context_value),
                    "loss_moving_avg": _format_metric(loss_window_sum / max(loss_window_count, 1)),
                    "masked_token_accuracy": _format_metric(correct / max(total, 1)),
                    "grad_norm": _format_metric(_to_float(grad_norm)),
                    "lr": _format_metric(optimizer.param_groups[0]["lr"]),
                    "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
                    "target_tokens": str(total),
                    "step": str(step),
                    "optimizer_step": str(optimizer_step),
                }
                for source, count in Counter(batch.get("mask_sources", [])).items():
                    metrics[f"mask_source_{source}"] = str(count)
                for codebook_idx, (cb_correct, cb_total) in enumerate(zip(codebook_correct, codebook_total)):
                    metrics[f"codebook_{codebook_idx}_accuracy"] = _format_metric(int(cb_correct.item()) / max(int(cb_total.item()), 1))
                metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(metrics, flush=True)
                loss_window_sum = 0.0
                loss_window_count = 0

            if eval_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(model, role_embeddings, eval_dataloader, config, args, device, loss_codebooks)
                eval_metrics["epoch"] = _format_metric(examples_seen / max(dataset_size, 1))
                eval_metrics["step"] = str(step)
                eval_metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(eval_metrics, flush=True)
                eval_accuracy = float(eval_metrics["eval_masked_token_accuracy"])
                if not args.no_save and args.save_best and eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    save_roles_checkpoint(output_dir / "best.pt", model, role_embeddings, optimizer, step, args)
                    print(
                        {
                            "saved_best_checkpoint": str(output_dir / "best.pt"),
                            "best_eval_masked_token_accuracy": _format_metric(best_eval_accuracy),
                            "step": str(step),
                        },
                        flush=True,
                    )

            if not args.no_save and args.save_every > 0 and step % args.save_every == 0:
                save_roles_checkpoint(output_dir / f"step-{step:06d}.pt", model, role_embeddings, optimizer, step, args)
            if step >= args.max_steps:
                break

    if not args.no_save and args.save_last:
        save_roles_checkpoint(output_dir / "last.pt", model, role_embeddings, optimizer, step, args)
    train_runtime = time.perf_counter() - train_start
    print(
        {
            "train_runtime": _format_metric(train_runtime),
            "train_samples_per_second": _format_metric(examples_seen / train_runtime if train_runtime > 0 else 0.0),
            "train_steps_per_second": _format_metric(step / train_runtime if train_runtime > 0 else 0.0),
            "train_loss": _format_metric(loss_sum / max(step, 1)),
            "train_ce_loss": _format_metric(ce_sum / max(step, 1)),
            "train_energy_loss": _format_metric(energy_sum / max(step, 1)),
            "train_context_loss": _format_metric(context_sum / max(step, 1)),
            "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
        },
        flush=True,
    )


@torch.no_grad()
def evaluate(model, role_embeddings, dataloader, config, args, device, loss_codebooks):
    was_training = model.training
    model.eval()
    role_embeddings.eval()
    eval_start = time.perf_counter()
    loss_sum = 0.0
    samples = 0
    steps = 0
    correct_tokens = 0
    total_tokens = 0
    codebook_correct = None
    codebook_total = None
    mask_source_counts: Counter[str] = Counter()
    for batch in dataloader:
        if args.eval_max_batches is not None and steps >= args.eval_max_batches:
            break
        mask_source_counts.update(batch.get("mask_sources", []))
        speaker_embeddings = batch["speaker_embeddings"]
        if not isinstance(speaker_embeddings, torch.Tensor):
            raise RuntimeError("speaker embeddings are required for VoiceCraft-X audio-only roles eval")
        inference_prompt, middle = build_generation_training_tensors(batch, config, device)
        labels = select_loss_codebooks(delayed_labels(middle, model.num_codebooks), loss_codebooks)
        outputs = audio_only_roles_forward(
            model=model,
            role_embeddings=role_embeddings,
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
        role_embeddings.train()
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
    for source, count in sorted(mask_source_counts.items()):
        metrics[f"eval_mask_source_{source}"] = str(count)
    return metrics


def audio_only_roles_forward(model, role_embeddings, prompt_speech_token, speaker_emb, labels, force_length: bool):
    hidden_states = build_roles_prefill_hidden(
        model,
        role_embeddings,
        prompt_speech_token,
        speaker_emb,
        prompt_format=getattr(role_embeddings, "prompt_format", "roles-fim"),
    )
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


def build_context_training_tensors(batch, device: torch.device, speech_mask_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    middle_labels = batch["labels"].to(device)
    if input_ids.shape[0] != 1:
        raise ValueError("context auxiliary loss currently supports batch size 1")

    active_positions = middle_labels[0].ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if active_positions.numel() == 0:
        raise RuntimeError("batch has no masked middle labels")
    middle_start_in_input = int(active_positions.min().item())
    prompt = input_ids[:, :, :middle_start_in_input]
    middle = middle_labels[:, :, middle_start_in_input : int(active_positions.max().item()) + 1].long()

    mask_columns = prompt[0].eq(speech_mask_idx).all(dim=0).nonzero(as_tuple=False).flatten()
    if mask_columns.numel() < 2:
        raise RuntimeError("expected prompt to contain two speech mask columns")
    first_mask = int(mask_columns[0].item())
    second_mask = int(mask_columns[1].item())
    prefix = prompt[:, :, :first_mask]
    suffix = prompt[:, :, first_mask + 1 : second_mask]
    original = torch.cat([prefix, middle, suffix], dim=-1)
    labels = original.clone().long()
    labels[:, :, prefix.shape[-1] : prefix.shape[-1] + middle.shape[-1]] = -100
    return original, labels


def audio_only_context_forward(model, speech_tokens: torch.Tensor, speaker_emb: torch.Tensor, labels: torch.Tensor):
    empty_text = torch.empty((speech_tokens.shape[0], 0), dtype=torch.long, device=speech_tokens.device)
    empty_text_attention = torch.empty((speech_tokens.shape[0], 0), dtype=torch.bool, device=speech_tokens.device)
    outputs = model(
        text=empty_text,
        prompt_speech_token=speech_tokens,
        speaker_emb=speaker_emb,
        labels=labels,
        text_attention_mask=empty_text_attention,
        return_logits=False,
    )
    return outputs["loss"]


def build_roles_prefill_hidden(model, role_embeddings, prompt_speech_token, speaker_emb, prompt_format: str = "roles-fim"):
    raw_prompt = prompt_speech_token[:, :, : -int(model.config.silence_nums)]
    if raw_prompt.shape[0] != 1:
        raise ValueError("audio-only roles training currently supports batch size 1")
    mask_columns = raw_prompt[0].eq(int(model.speech_mask_idx)).all(dim=0).nonzero(as_tuple=False).flatten()
    if mask_columns.numel() < 2:
        raise RuntimeError("expected prompt to contain two speech mask columns")
    first_mask = int(mask_columns[0].item())
    second_mask = int(mask_columns[1].item())
    prefix = raw_prompt[:, :, :first_mask]
    gap_marker = raw_prompt[:, :, first_mask : first_mask + 1]
    suffix = raw_prompt[:, :, first_mask + 1 : second_mask]

    speaker_hidden = model.spk_embed_affine_layer(F.normalize(speaker_emb, dim=-1)).unsqueeze(1)
    prefix_hidden = embed_segment(model, prefix)
    suffix_hidden = embed_segment(model, suffix)
    batch_size = prompt_speech_token.shape[0]
    if prompt_format == "original-mask-roles":
        role_ids = torch.tensor(
            [PREFIX_ROLE, GAP_ROLE, SUFFIX_ROLE, MIDDLE_ROLE],
            dtype=torch.long,
            device=prompt_speech_token.device,
        )
        prefix_role, gap_role, suffix_role, middle_role = role_embeddings(role_ids).view(4, 1, 1, -1).expand(
            -1, batch_size, -1, -1
        )
        return torch.cat(
            [
                speaker_hidden,
                prefix_role,
                prefix_hidden,
                gap_role,
                embed_segment(model, gap_marker),
                suffix_role,
                suffix_hidden,
                middle_role,
            ],
            dim=1,
        )

    role_ids = torch.tensor([PREFIX_ROLE, SUFFIX_ROLE, MIDDLE_ROLE], dtype=torch.long, device=prompt_speech_token.device)
    prefix_role, suffix_role, middle_role = role_embeddings(role_ids).view(3, 1, 1, -1).expand(-1, batch_size, -1, -1)

    return torch.cat(
        [
            speaker_hidden,
            prefix_role,
            prefix_hidden,
            suffix_role,
            suffix_hidden,
            middle_role,
        ],
        dim=1,
    )


def embed_segment(model, speech_tokens: torch.Tensor) -> torch.Tensor:
    if speech_tokens.shape[-1] == 0:
        hidden_size = int(model.llm.model.embed_tokens.embedding_dim)
        return torch.empty(speech_tokens.shape[0], 0, hidden_size, device=speech_tokens.device)
    shifted = model.shift_batch(speech_tokens)
    return model.embed_speech(shifted)


def save_roles_checkpoint(path: Path, model, role_embeddings, optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    torch.save(
        {
            "step": step,
            "model": model_state if args.save_trainable_only else model.state_dict(),
            "role_embeddings": role_embeddings.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "trainable_only": bool(args.save_trainable_only),
        },
        path,
    )


def build_scheduler(optimizer, args: argparse.Namespace, total_optimizer_steps: int):
    warmup_steps = min(max(args.warmup_steps, 0), total_optimizer_steps)
    if args.lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    if args.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
