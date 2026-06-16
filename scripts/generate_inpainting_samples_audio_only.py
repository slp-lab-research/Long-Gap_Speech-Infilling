from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from transformers import DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from dataset.online_masking import build_voicecraftx_inpainting_input, sample_eval_mask_intervals  # noqa: E402
from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset  # noqa: E402
from helper import load_tokenizer, load_voicecraftx  # noqa: E402
from train_inpainting_audio_only import build_audio_only_prefill_hidden  # noqa: E402
from train_inpainting_generation_context import detach_cache, load_generation_context_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate audio-only VoiceCraft-X speech-token infilling samples.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="exp/inpainting_audio_only_generation_check")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-split-size", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--eval-mask-len", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _, audio_tokenizer = load_tokenizer(config)
    audio_tokenizer = audio_tokenizer.to(device)
    model = load_voicecraftx(config).to(device)
    load_generation_context_checkpoint(model, Path(args.checkpoint))
    model.eval()

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size <= 0:
        raise ValueError("--eval-split-size must be positive")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than selected dataset size")

    eval_start = selected_size - args.eval_split_size
    sample_indices = list(range(eval_start + args.sample_offset, selected_size))[: args.num_samples]
    sample_items = [dataset[index] for index in sample_indices]
    y_lens = [int(item["tokens"].shape[-1]) for item in sample_items]
    mask_intervals, _ = sample_eval_mask_intervals(y_lens, mask_len=args.eval_mask_len)

    run_dir = output_root / Path(args.checkpoint).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, (item, intervals) in enumerate(zip(sample_items, mask_intervals)):
        sample_dir = run_dir / f"{local_idx:02d}_{item['utterance_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        tokens = item["tokens"].long()
        example = build_voicecraftx_inpainting_input(
            tokens,
            intervals,
            speech_mask_idx=int(config.model.speech_mask_idx),
        )
        prompt = _build_audio_only_prompt(
            example,
            speech_mask_idx=int(config.model.speech_mask_idx),
            silence_tokens=tuple(int(x) for x in config.model.silence_tokens),
            silence_nums=int(config.model.silence_nums),
        ).unsqueeze(0).to(device)
        speaker_embedding = item["speaker_embedding"]
        if not isinstance(speaker_embedding, torch.Tensor):
            raise RuntimeError(f"speaker embedding missing for {item['utterance_id']}")
        speaker_embedding = speaker_embedding.to(device)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        gap_len = intervals[0][1] - intervals[0][0]
        with torch.inference_mode():
            generated_span = audio_only_generate(
                model=model,
                prompt_speech_token=prompt,
                speaker_emb=speaker_embedding,
                target_frames=gap_len,
                temperature=args.temperature,
                top_k=args.top_k,
            ).cpu()

        generated_full = torch.cat([example.prefix.cpu(), generated_span, example.suffix.cpu()], dim=-1)
        original_wav = _load_original_wav(item["wav_path"], int(config.SAMPLE_RATE))
        masked_wav = _make_masked_wav(
            original_wav,
            intervals[0],
            sample_rate=int(config.SAMPLE_RATE),
            codec_frame_rate=int(config.CODEC_FRAME_RATE),
        )
        generated_wav = _decode_tokens(audio_tokenizer, generated_full.to(device))
        generated_gap_wav = _try_decode_tokens(audio_tokenizer, generated_span.to(device))

        _save_wav(sample_dir / "original.wav", original_wav, int(config.SAMPLE_RATE))
        _save_wav(sample_dir / "masked_input.wav", masked_wav, int(config.SAMPLE_RATE))
        _save_wav(sample_dir / "generated_matched.wav", generated_wav, int(config.SAMPLE_RATE))
        if generated_gap_wav is not None:
            _save_wav(sample_dir / "generated_gap_matched.wav", generated_gap_wav, int(config.SAMPLE_RATE))

        metadata = {
            "checkpoint": str(args.checkpoint),
            "task": "audio_only_speech_token_gap_infilling",
            "utterance_id": item["utterance_id"],
            "speaker_id": item.get("speaker_id"),
            "chapter_id": item.get("chapter_id"),
            "text": item.get("text"),
            "wav_path": item["wav_path"],
            "original_frames": int(tokens.shape[-1]),
            "mask_interval": list(intervals[0]),
            "mask_start_sec": intervals[0][0] / float(config.CODEC_FRAME_RATE),
            "mask_end_sec": intervals[0][1] / float(config.CODEC_FRAME_RATE),
            "generated_frames": int(generated_span.shape[-1]),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {sample_dir}", flush=True)


def audio_only_generate(
    model,
    prompt_speech_token: torch.Tensor,
    speaker_emb: torch.Tensor,
    target_frames: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    num_codebooks = int(model.num_codebooks)
    total_steps = target_frames + num_codebooks
    hidden_states = build_audio_only_prefill_hidden(model, prompt_speech_token, speaker_emb)
    cache = DynamicCache()
    delayed_predictions = torch.full(
        (prompt_speech_token.shape[0], num_codebooks, total_steps),
        int(model.speech_empty_idx),
        dtype=torch.long,
        device=prompt_speech_token.device,
    )

    for step_idx in range(total_steps):
        attention_mask = torch.tril(
            torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]),
                device=hidden_states.device,
            )
        ).to(torch.bool)
        outputs = model.llm.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask[:, -1, :],
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        cache = detach_cache(outputs.past_key_values)
        step_hidden = outputs.last_hidden_state[:, -1:]
        step_logits = torch.stack([model.llm_decoder[k](step_hidden) for k in range(num_codebooks)], dim=1).squeeze(2)
        step_logits[:, :, model.speech_empty_idx] = -1e9
        step_logits[:, :, model.speech_mask_idx] = -1e9
        step_logits[:, :, model.speech_eos_idx] = -1e9
        next_tokens = _sample_tokens(step_logits, temperature=temperature, top_k=top_k)
        delayed_predictions[:, :, step_idx] = next_tokens
        if step_idx + 1 < total_steps:
            next_speech_token = torch.stack(
                [model.speech_embedding[k](next_tokens[:, k]) for k in range(num_codebooks)],
                dim=1,
            ).sum(dim=1, keepdim=True)
            hidden_states = next_speech_token

    frames = []
    for frame_idx in range(target_frames):
        codebook_tokens = []
        for codebook_idx in range(num_codebooks):
            codebook_tokens.append(delayed_predictions[:, codebook_idx, frame_idx + codebook_idx])
        frames.append(torch.stack(codebook_tokens, dim=1))
    return torch.stack(frames, dim=-1).squeeze(0)


def _sample_tokens(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0 or top_k == 1:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        probs = F.softmax(values, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).view(*probs.shape[:-1])
        return indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).view(*probs.shape[:-1])


def _build_audio_only_prompt(example, speech_mask_idx: int, silence_tokens: tuple[int, ...], silence_nums: int) -> torch.Tensor:
    speech_mask = torch.full(
        (example.prefix.shape[0], 1),
        speech_mask_idx,
        dtype=example.prefix.dtype,
        device=example.prefix.device,
    )
    cushion = torch.tensor(
        silence_tokens,
        dtype=example.prefix.dtype,
        device=example.prefix.device,
    ).view(-1, 1).expand(-1, silence_nums)
    return torch.cat([example.prefix, speech_mask, example.suffix, speech_mask, cushion], dim=-1)


def _decode_tokens(audio_tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    wav = audio_tokenizer.decode(tokens.unsqueeze(0))
    return wav.detach().cpu()


def _try_decode_tokens(audio_tokenizer, tokens: torch.Tensor) -> torch.Tensor | None:
    try:
        return _decode_tokens(audio_tokenizer, tokens)
    except RuntimeError as exc:
        print(f"warning: skipped short gap decode: {exc}", flush=True)
        return None


def _load_original_wav(wav_path: str | None, sample_rate: int) -> torch.Tensor:
    if wav_path is None:
        raise RuntimeError("wav_path is missing from metadata")
    path = Path(wav_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.unsqueeze(0)


def _make_masked_wav(wav: torch.Tensor, interval: tuple[int, int], sample_rate: int, codec_frame_rate: int) -> torch.Tensor:
    masked = wav.clone()
    start, end = interval
    start_sample = round(start * sample_rate / codec_frame_rate)
    end_sample = round(end * sample_rate / codec_frame_rate)
    masked[..., start_sample:end_sample] = 0.0
    return masked


def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu()
    while wav.dim() > 2:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sample_rate)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
