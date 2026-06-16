from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
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
from dataset.tokenized_wds import (  # noqa: E402
    TokenizedLibriTTSRWDSDataset,
    sample_word_aligned_mask_interval,
    word_aligned_mask_candidates,
)
from generate_inpainting_samples_audio_only import (  # noqa: E402
    _build_audio_only_prompt,
    _decode_tokens,
    _load_original_wav,
    _make_masked_wav,
    _resolve_device,
    _sample_tokens,
    _save_wav,
    _try_decode_tokens,
)
from helper import load_tokenizer, load_voicecraftx  # noqa: E402
from train_inpainting_audio_only_roles import build_roles_prefill_hidden  # noqa: E402
from train_inpainting_generation_context import detach_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate audio-only role-embedding VoiceCraft-X infilling samples.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--pretrained-only",
        action="store_true",
        help="Generate with the pretrained VoiceCraft-X weights only, without loading a fine-tuned role checkpoint.",
    )
    parser.add_argument(
        "--role-init",
        choices=["random", "zero"],
        default="random",
        help="Role embedding initialization used with --pretrained-only.",
    )
    parser.add_argument("--output-dir", default="exp/inpainting_audio_only_roles_generation_check")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-split-size", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--mask-len-min", type=int, default=10)
    parser.add_argument("--mask-len-max", type=int, default=30)
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--mask-mode", choices=["frame", "word-aligned"], default="frame")
    parser.add_argument("--word-mask-max-words", type=int, default=4)
    parser.add_argument(
        "--require-aligned-mask",
        action="store_true",
        help="When using --mask-mode word-aligned, skip eval samples without a valid aligned mask instead of falling back to frame masks.",
    )
    parser.add_argument("--codec-frame-rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    codec_frame_rate = float(args.codec_frame_rate or config.CODEC_FRAME_RATE)
    if args.require_aligned_mask and args.mask_mode != "word-aligned":
        raise ValueError("--require-aligned-mask requires --mask-mode word-aligned")
    device = _resolve_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _, audio_tokenizer = load_tokenizer(config)
    audio_tokenizer = audio_tokenizer.to(device)
    model = load_voicecraftx(config).to(device)
    role_embeddings = torch.nn.Embedding(3, int(model.llm.model.embed_tokens.embedding_dim)).to(device)
    if args.pretrained_only:
        if args.checkpoint:
            raise ValueError("--pretrained-only cannot be combined with --checkpoint")
        if args.role_init == "zero":
            role_embeddings.weight.data.zero_()
        print({"checkpoint": "pretrained_only", "role_embeddings": args.role_init}, flush=True)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --pretrained-only is set")
        load_roles_checkpoint(model, role_embeddings, Path(args.checkpoint))
    model.eval()
    role_embeddings.eval()

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size <= 0:
        raise ValueError("--eval-split-size must be positive")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than selected dataset size")

    eval_start = selected_size - args.eval_split_size
    candidate_indices = list(range(eval_start + args.sample_offset, selected_size))
    if args.require_aligned_mask:
        candidate_indices = filter_generation_indices_with_aligned_masks(
            dataset=dataset,
            indices=candidate_indices,
            mask_len_min=args.mask_len_min,
            mask_len_max=args.mask_len_max,
            codec_frame_rate=codec_frame_rate,
            word_mask_max_words=args.word_mask_max_words,
        )
        if not candidate_indices:
            raise RuntimeError("no eval samples have a valid word-aligned mask candidate")
        print({"aligned_generation_candidates": str(len(candidate_indices))}, flush=True)
    sample_indices = candidate_indices[: args.num_samples]
    sample_items = [dataset[index] for index in sample_indices]
    y_lens = [int(item["tokens"].shape[-1]) for item in sample_items]
    mask_intervals, mask_sources = build_generation_mask_intervals(
        sample_items=sample_items,
        y_lens=y_lens,
        args=args,
        codec_frame_rate=codec_frame_rate,
    )

    run_name = Path(args.checkpoint).stem if args.checkpoint else f"pretrained_only_{args.role_init}_roles"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, (item, intervals, mask_source) in enumerate(zip(sample_items, mask_intervals, mask_sources)):
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
            generated_span = audio_only_roles_generate(
                model=model,
                role_embeddings=role_embeddings,
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
            "checkpoint": str(args.checkpoint) if args.checkpoint else "pretrained_only",
            "role_init": args.role_init if args.pretrained_only else "checkpoint",
            "task": "audio_only_speech_token_gap_infilling_roles",
            "utterance_id": item["utterance_id"],
            "speaker_id": item.get("speaker_id"),
            "chapter_id": item.get("chapter_id"),
            "text": item.get("text"),
            "wav_path": item["wav_path"],
            "original_frames": int(tokens.shape[-1]),
            "mask_interval": list(intervals[0]),
            "mask_start_sec": intervals[0][0] / float(config.CODEC_FRAME_RATE),
            "mask_end_sec": intervals[0][1] / float(config.CODEC_FRAME_RATE),
            "mask_source": mask_source,
            "mask_mode": args.mask_mode,
            "generated_frames": int(generated_span.shape[-1]),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {sample_dir}", flush=True)


def build_generation_mask_intervals(sample_items, y_lens, args, codec_frame_rate: float):
    if args.mask_mode == "frame":
        mask_intervals, _ = sample_eval_mask_intervals(y_lens, mask_len=args.eval_mask_len)
        return mask_intervals, ["frame" for _ in mask_intervals]

    mask_intervals = []
    mask_sources = []
    for item, y_len in zip(sample_items, y_lens):
        interval = sample_word_aligned_mask_interval(
            metadata=item.get("metadata", {}),
            y_len=int(y_len),
            mask_len_min=args.mask_len_min,
            mask_len_max=args.mask_len_max,
            codec_frame_rate=codec_frame_rate,
            max_words=args.word_mask_max_words,
            deterministic=True,
        )
        if interval is None:
            if args.require_aligned_mask:
                raise RuntimeError(f"no valid word-aligned mask candidate for {item['utterance_id']}")
            fallback, _ = sample_eval_mask_intervals([y_len], mask_len=args.eval_mask_len)
            mask_intervals.append(fallback[0])
            mask_sources.append("fallback_frame")
        else:
            mask_intervals.append([interval])
            mask_sources.append("aligned")
    return mask_intervals, mask_sources


def filter_generation_indices_with_aligned_masks(
    dataset: TokenizedLibriTTSRWDSDataset,
    indices: list[int],
    mask_len_min: int,
    mask_len_max: int,
    codec_frame_rate: float,
    word_mask_max_words: int,
) -> list[int]:
    filtered = []
    for index in indices:
        item = dataset[index]
        candidates = word_aligned_mask_candidates(
            metadata=item.get("metadata", {}),
            y_len=int(item["tokens"].shape[-1]),
            mask_len_min=mask_len_min,
            mask_len_max=mask_len_max,
            codec_frame_rate=codec_frame_rate,
            max_words=word_mask_max_words,
        )
        if candidates:
            filtered.append(index)
    return filtered


def audio_only_roles_generate(
    model,
    role_embeddings,
    prompt_speech_token: torch.Tensor,
    speaker_emb: torch.Tensor,
    target_frames: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    num_codebooks = int(model.num_codebooks)
    total_steps = target_frames + num_codebooks
    hidden_states = build_roles_prefill_hidden(model, role_embeddings, prompt_speech_token, speaker_emb)
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


def load_roles_checkpoint(model, role_embeddings, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    role_state = checkpoint.get("role_embeddings")
    if role_state is None:
        raise RuntimeError(f"role_embeddings missing in checkpoint: {checkpoint_path}")
    role_embeddings.load_state_dict(role_state)
    print(
        {
            "checkpoint": str(checkpoint_path),
            "missing_model_keys": str(len(missing)),
            "unexpected_model_keys": str(len(unexpected)),
            "role_embeddings": "loaded",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
