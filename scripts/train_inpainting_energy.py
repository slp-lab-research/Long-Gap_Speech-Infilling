from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, VoiceCraftXTokenizedWDSCollator  # noqa: E402
from helper import load_tokenizer, load_voicecraftx  # noqa: E402
from train_inpainting import (  # noqa: E402
    apply_trainable_scope,
    encode_texts,
    evaluate,
    print_trainable_summary,
    restore_frozen_module_eval_mode,
    save_checkpoint,
    _format_metric,
    _progress_metrics,
    _to_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VoiceCraft-X inpainting with CE + token-RMS energy proxy loss.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100")
    parser.add_argument("--output-dir", default="exp/inpainting_energy")
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
    parser.add_argument("--mask-len-max", type=int, default=150)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--deterministic-mask", action="store_true")
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--language", default="english")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument(
        "--trainable-scope",
        choices=["default", "speech-head", "speech-head-last-n-llm"],
        default="default",
    )
    parser.add_argument("--unfreeze-llm-layers", type=int, default=0)
    parser.add_argument("--energy-loss-weight", type=float, default=0.1)
    parser.add_argument("--energy-table-samples", type=int, default=5000)
    parser.add_argument("--energy-table-path", default=None)
    parser.add_argument("--energy-log-scale", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_tokenizer, _ = load_tokenizer(config)
    model = load_voicecraftx(config).to(device)
    model.train()
    apply_trainable_scope(model, args)
    print_trainable_summary(model, args)

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size < 0:
        raise ValueError("--eval-split-size must be non-negative")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than the selected dataset size")

    energy_table_path = Path(args.energy_table_path) if args.energy_table_path else output_dir / "token_energy_table.pt"
    energy_table = load_or_build_token_energy_table(
        dataset=dataset,
        config=config,
        path=energy_table_path,
        max_samples=min(args.energy_table_samples, selected_size),
    ).to(device)

    train_indices = list(range(selected_size - args.eval_split_size))
    eval_indices = list(range(selected_size - args.eval_split_size, selected_size))
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices) if eval_indices else None

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
            batch_size=args.batch_size,
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
            text, text_attention_mask = encode_texts(
                text_tokenizer=text_tokenizer,
                texts=batch["texts"],
                language=args.language,
                pad_token_id=int(config.model.llm_padding_idx),
            )
            speaker_embeddings = batch["speaker_embeddings"]
            if not isinstance(speaker_embeddings, torch.Tensor):
                raise RuntimeError("speaker embeddings are required for VoiceCraft-X training")

            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            speaker_embeddings = speaker_embeddings.to(device)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(
                    text=text,
                    text_attention_mask=text_attention_mask,
                    prompt_speech_token=input_ids,
                    speaker_emb=speaker_embeddings,
                    labels=labels,
                    return_logits=True,
                )
                ce_loss = outputs["loss"]
                energy_loss = token_rms_energy_loss(
                    logits=outputs["logits"],
                    shifted_labels=outputs["shifted_labels"],
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
            batch_size = int(input_ids.shape[0])
            examples_seen += batch_size
            loss_value = float(total_loss.detach().cpu().item())
            ce_value = float(ce_loss.detach().cpu().item())
            energy_value = float(energy_loss.detach().cpu().item())
            loss_sum += loss_value
            ce_sum += ce_value
            energy_sum += energy_value
            loss_window_sum += loss_value
            loss_window_count += 1
            if step % args.log_every == 0:
                target_tokens = int(batch["loss_mask"].sum().item())
                metrics = {
                    "loss": _format_metric(loss_value),
                    "ce_loss": _format_metric(ce_value),
                    "energy_loss": _format_metric(energy_value),
                    "loss_moving_avg": _format_metric(loss_window_sum / max(loss_window_count, 1)),
                    "grad_norm": _format_metric(_to_float(grad_norm)),
                    "lr": _format_metric(optimizer.param_groups[0]["lr"]),
                    "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
                    "target_frames": str(target_tokens),
                    "step": str(step),
                }
                metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(metrics, flush=True)
                loss_window_sum = 0.0
                loss_window_count = 0

            if eval_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(model, eval_dataloader, text_tokenizer, config, args, device)
                eval_metrics["epoch"] = _format_metric(examples_seen / max(dataset_size, 1))
                eval_metrics["step"] = str(step)
                eval_metrics.update(_progress_metrics(train_start, step, args.max_steps))
                print(eval_metrics, flush=True)

            if not args.no_save and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"step-{step:06d}.pt", model, optimizer, step, args)

            if step >= args.max_steps:
                break

    if not args.no_save:
        save_checkpoint(output_dir / "last.pt", model, optimizer, step, args)
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


def load_or_build_token_energy_table(dataset, config, path: Path, max_samples: int) -> torch.Tensor:
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        table = payload["energy_table"] if isinstance(payload, dict) else payload
        print({"energy_table": str(path), "status": "loaded"}, flush=True)
        return table.float()

    sample_rate = int(config.SAMPLE_RATE)
    frame_rate = int(config.CODEC_FRAME_RATE)
    num_codebooks = int(config.model.num_codebooks)
    vocab_size = int(config.model.speech_token_size)
    sums = torch.zeros(num_codebooks, vocab_size, dtype=torch.float64)
    counts = torch.zeros(num_codebooks, vocab_size, dtype=torch.float64)
    global_sum = 0.0
    global_count = 0

    for index in range(max_samples):
        item = dataset[index]
        tokens = item["tokens"].long()
        wav_path = Path(item["wav_path"])
        if not wav_path.is_absolute():
            wav_path = REPO_ROOT / wav_path
        wav, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        frame_rms = wav_frame_rms(wav.squeeze(0), tokens.shape[-1], sample_rate, frame_rate)
        global_sum += float(frame_rms.sum().item())
        global_count += int(frame_rms.numel())
        for codebook_idx in range(num_codebooks):
            token_ids = tokens[codebook_idx].clamp(0, vocab_size - 1)
            sums[codebook_idx].scatter_add_(0, token_ids, frame_rms.double())
            counts[codebook_idx].scatter_add_(0, token_ids, torch.ones_like(frame_rms, dtype=torch.float64))
        if (index + 1) == 1 or (index + 1) % 500 == 0:
            print({"energy_table_samples": str(index + 1), "max_samples": str(max_samples)}, flush=True)

    global_mean = global_sum / max(global_count, 1)
    table = torch.where(counts > 0, sums / counts.clamp_min(1), torch.full_like(sums, global_mean)).float()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "energy_table": table,
            "max_samples": max_samples,
            "sample_rate": sample_rate,
            "frame_rate": frame_rate,
            "global_mean": global_mean,
        },
        path,
    )
    print({"energy_table": str(path), "status": "built", "global_mean": _format_metric(global_mean)}, flush=True)
    return table


def wav_frame_rms(wav: torch.Tensor, num_frames: int, sample_rate: int, frame_rate: int) -> torch.Tensor:
    values = []
    for frame_idx in range(num_frames):
        start = round(frame_idx * sample_rate / frame_rate)
        end = round((frame_idx + 1) * sample_rate / frame_rate)
        segment = wav[start:end]
        if segment.numel() == 0:
            values.append(torch.tensor(0.0, dtype=wav.dtype))
        else:
            values.append(torch.sqrt(torch.mean(segment.float().pow(2)) + 1e-12))
    return torch.stack(values).float()


def token_rms_energy_loss(
    logits: torch.Tensor,
    shifted_labels: torch.Tensor,
    energy_table: torch.Tensor,
    log_scale: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    mask = shifted_labels.ne(ignore_index)
    frame_mask = mask.any(dim=1)
    if not frame_mask.any():
        return logits.new_tensor(0.0)

    energy_table = pad_energy_table(energy_table, logits.shape[-1])
    probs = torch.softmax(logits.float(), dim=-1)
    pred_codebook_energy = torch.einsum("bktv,kv->bkt", probs, energy_table.float())
    safe_labels = shifted_labels.clamp_min(0)
    target_codebook_energy = energy_table.gather(1, safe_labels.transpose(0, 1).reshape(energy_table.shape[0], -1))
    target_codebook_energy = target_codebook_energy.reshape(energy_table.shape[0], shifted_labels.shape[0], shifted_labels.shape[2])
    target_codebook_energy = target_codebook_energy.permute(1, 0, 2)
    target_codebook_energy = torch.where(mask, target_codebook_energy, torch.zeros_like(target_codebook_energy))

    codebook_counts = mask.sum(dim=1).clamp_min(1)
    pred_frame_energy = (pred_codebook_energy * mask).sum(dim=1) / codebook_counts
    target_frame_energy = target_codebook_energy.sum(dim=1) / codebook_counts
    pred_log = torch.log1p(pred_frame_energy * log_scale)
    target_log = torch.log1p(target_frame_energy * log_scale)
    return F.smooth_l1_loss(pred_log[frame_mask], target_log[frame_mask])


def pad_energy_table(energy_table: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if energy_table.shape[-1] == vocab_size:
        return energy_table
    if energy_table.shape[-1] > vocab_size:
        return energy_table[:, :vocab_size]
    pad_value = energy_table.mean(dim=1, keepdim=True)
    pad = pad_value.expand(-1, vocab_size - energy_table.shape[-1])
    return torch.cat([energy_table, pad], dim=1)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
