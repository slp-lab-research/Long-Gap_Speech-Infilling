from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
import tarfile
from typing import Any, Sequence

import torch

from .online_masking import (
    build_voicecraftx_inpainting_input,
    pad_voicecraftx_inpainting_batch,
    sample_eval_mask_intervals,
    sample_mask_intervals,
)
from .text_gap_alignment import load_word_segments


@dataclass(frozen=True)
class TokenizedWDSRecord:
    shard_path: str
    key: str
    tokens_member: str
    text_member: str
    metadata_member: str
    speaker_member: str | None = None


class TokenizedLibriTTSRWDSDataset(torch.utils.data.Dataset):
    """Map-style reader for VoiceCraft-X tokenized LibriTTS-R tar shards."""

    def __init__(
        self,
        root: str | Path,
        shard_glob: str = "*.tar",
        load_speaker_embedding: bool = True,
    ) -> None:
        self.root = Path(root)
        self.load_speaker_embedding = load_speaker_embedding
        if not self.root.exists():
            raise FileNotFoundError(f"tokenized WDS directory not found: {self.root}")

        self.shards = sorted(self.root.glob(shard_glob))
        if not self.shards:
            raise RuntimeError(f"no shards matching {shard_glob!r} under {self.root}")
        self.records = self._index_shards(self.shards)
        if not self.records:
            raise RuntimeError(f"no complete tokenized samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with tarfile.open(record.shard_path, mode="r") as tar:
            tokens = _load_tensor(tar, record.tokens_member).long()
            text = _load_bytes(tar, record.text_member).decode("utf-8")
            metadata = json.loads(_load_bytes(tar, record.metadata_member).decode("utf-8"))
            speaker_embedding = None
            if self.load_speaker_embedding and record.speaker_member is not None:
                speaker_embedding = _load_tensor(tar, record.speaker_member).float()

        if tokens.dim() != 2:
            raise RuntimeError(f"expected tokens [K, T], got {tuple(tokens.shape)} for {record.key}")
        metadata.setdefault("utterance_id", record.key)
        return {
            "tokens": tokens,
            "text": text,
            "metadata": metadata,
            "speaker_embedding": speaker_embedding,
            "utterance_id": metadata.get("utterance_id", record.key),
            "speaker_id": metadata.get("speaker_id"),
            "chapter_id": metadata.get("chapter_id"),
            "wav_path": metadata.get("wav_path"),
            "num_frames": int(tokens.shape[-1]),
        }

    def _index_shards(self, shards: Sequence[Path]) -> list[TokenizedWDSRecord]:
        records: list[TokenizedWDSRecord] = []
        for shard in shards:
            with tarfile.open(shard, mode="r") as tar:
                grouped: dict[str, dict[str, str]] = {}
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    key, suffix = _split_member_name(member.name)
                    grouped.setdefault(key, {})[suffix] = member.name

            for key in sorted(grouped):
                parts = grouped[key]
                required = {"tokens.pt", "txt", "json"}
                if not required.issubset(parts):
                    continue
                records.append(
                    TokenizedWDSRecord(
                        shard_path=str(shard),
                        key=key,
                        tokens_member=parts["tokens.pt"],
                        text_member=parts["txt"],
                        metadata_member=parts["json"],
                        speaker_member=parts.get("spk.pt"),
                    )
                )
        return records


class VoiceCraftXTokenizedWDSCollator:
    """Build VoiceCraft-X online inpainting batches from pre-tokenized samples."""

    def __init__(
        self,
        speech_mask_idx: int,
        pad_token: int,
        mask_len_min: int = 10,
        mask_len_max: int = 150,
        max_n_spans: int = 1,
        mask_sample_dist: str = "poisson1",
        min_gap: int = 5,
        eval_mask_len: int = 50,
        deterministic_eval: bool = False,
        generator: torch.Generator | None = None,
        mask_mode: str = "frame",
        codec_frame_rate: float = 50.0,
        word_mask_max_words: int = 4,
        require_word_aligned_mask: bool = False,
    ) -> None:
        if mask_mode not in {"frame", "word-aligned"}:
            raise ValueError(f"unsupported mask_mode: {mask_mode}")
        self.speech_mask_idx = speech_mask_idx
        self.pad_token = pad_token
        self.mask_len_min = mask_len_min
        self.mask_len_max = mask_len_max
        self.max_n_spans = max_n_spans
        self.mask_sample_dist = mask_sample_dist
        self.min_gap = min_gap
        self.eval_mask_len = eval_mask_len
        self.deterministic_eval = deterministic_eval
        self.generator = generator
        self.mask_mode = mask_mode
        self.codec_frame_rate = codec_frame_rate
        self.word_mask_max_words = word_mask_max_words
        self.require_word_aligned_mask = require_word_aligned_mask

    def __call__(self, items: Sequence[dict]) -> dict[str, torch.Tensor | list]:
        speech_tokens = [item["tokens"].long() for item in items]
        y_lens = [tokens.shape[-1] for tokens in speech_tokens]

        mask_sources = ["frame" for _ in items]
        if self.mask_mode == "word-aligned":
            mask_intervals, mask_sources = self._sample_word_aligned_masks(items, y_lens)
        elif self.deterministic_eval:
            mask_intervals, _ = sample_eval_mask_intervals(y_lens, mask_len=self.eval_mask_len)
        else:
            mask_intervals, _ = sample_mask_intervals(
                y_lens,
                mask_len_min=self.mask_len_min,
                mask_len_max=self.mask_len_max,
                max_n_spans=self.max_n_spans,
                mask_sample_dist=self.mask_sample_dist,
                min_gap=self.min_gap,
                generator=self.generator,
            )

        examples = [
            build_voicecraftx_inpainting_input(
                tokens,
                intervals,
                speech_mask_idx=self.speech_mask_idx,
            )
            for tokens, intervals in zip(speech_tokens, mask_intervals)
        ]
        batch = pad_voicecraftx_inpainting_batch(examples, pad_token=self.pad_token)
        batch.update(
            {
                "texts": [item["text"] for item in items],
                "metadata": [item["metadata"] for item in items],
                "utterance_ids": [item["utterance_id"] for item in items],
                "speaker_ids": [item["speaker_id"] for item in items],
                "chapter_ids": [item["chapter_id"] for item in items],
                "wav_paths": [item["wav_path"] for item in items],
                "speaker_embeddings": _stack_optional_tensors(
                    [item.get("speaker_embedding") for item in items]
                ),
                "mask_intervals": [example.mask_interval for example in examples],
                "mask_sources": mask_sources,
                "y_lens": torch.tensor(y_lens, dtype=torch.long),
            }
        )
        return batch

    def _sample_word_aligned_masks(self, items: Sequence[dict], y_lens: Sequence[int]) -> tuple[list[list[tuple[int, int]]], list[str]]:
        mask_intervals: list[list[tuple[int, int]]] = []
        mask_sources: list[str] = []
        for item, y_len in zip(items, y_lens):
            interval = sample_word_aligned_mask_interval(
                metadata=item.get("metadata", {}),
                y_len=int(y_len),
                mask_len_min=self.mask_len_min,
                mask_len_max=self.mask_len_max,
                codec_frame_rate=self.codec_frame_rate,
                max_words=self.word_mask_max_words,
                deterministic=self.deterministic_eval,
                generator=self.generator,
            )
            if interval is not None:
                mask_intervals.append([interval])
                mask_sources.append("aligned")
                continue

            if self.require_word_aligned_mask:
                raise ValueError(f"no valid word-aligned mask candidate for {item.get('utterance_id', '<unknown>')}")

            if self.deterministic_eval:
                fallback_intervals, _ = sample_eval_mask_intervals([y_len], mask_len=self.eval_mask_len)
            else:
                fallback_intervals, _ = sample_mask_intervals(
                    [y_len],
                    mask_len_min=self.mask_len_min,
                    mask_len_max=self.mask_len_max,
                    max_n_spans=self.max_n_spans,
                    mask_sample_dist=self.mask_sample_dist,
                    min_gap=self.min_gap,
                    generator=self.generator,
                )
            mask_intervals.append(fallback_intervals[0])
            mask_sources.append("fallback_frame")
        return mask_intervals, mask_sources


def _split_member_name(name: str) -> tuple[str, str]:
    filename = Path(name).name
    for suffix in ("tokens.pt", "spk.pt", "txt", "json"):
        marker = f".{suffix}"
        if filename.endswith(marker):
            return filename[: -len(marker)], suffix
    return filename, ""


def _load_bytes(tar: tarfile.TarFile, member_name: str) -> bytes:
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    return extracted.read()


def _load_tensor(tar: tarfile.TarFile, member_name: str) -> torch.Tensor:
    return torch.load(io.BytesIO(_load_bytes(tar, member_name)), map_location="cpu")


def _stack_optional_tensors(tensors: Sequence[torch.Tensor | None]) -> torch.Tensor | list:
    if not tensors or any(tensor is None for tensor in tensors):
        return list(tensors)
    shapes = {tuple(tensor.shape) for tensor in tensors if tensor is not None}
    if len(shapes) != 1:
        return list(tensors)
    return torch.stack([tensor for tensor in tensors if tensor is not None], dim=0)


def sample_word_aligned_mask_interval(
    metadata: dict[str, Any],
    y_len: int,
    mask_len_min: int,
    mask_len_max: int,
    codec_frame_rate: float,
    max_words: int,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> tuple[int, int] | None:
    """Sample a mask interval whose boundaries follow word alignment times."""
    candidates = word_aligned_mask_candidates(
        metadata=metadata,
        y_len=y_len,
        mask_len_min=mask_len_min,
        mask_len_max=mask_len_max,
        codec_frame_rate=codec_frame_rate,
        max_words=max_words,
    )
    if not candidates:
        return None
    if deterministic:
        center = y_len / 2.0
        return min(candidates, key=lambda interval: abs(((interval[0] + interval[1]) / 2.0) - center))
    idx = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
    return candidates[idx]


def word_aligned_mask_candidates(
    metadata: dict[str, Any],
    y_len: int,
    mask_len_min: int,
    mask_len_max: int,
    codec_frame_rate: float,
    max_words: int,
) -> list[tuple[int, int]]:
    words = _normalized_word_segments(metadata)
    candidates: list[tuple[int, int]] = []
    for start_idx in range(len(words)):
        for end_idx in range(start_idx + 1, min(len(words), start_idx + max_words) + 1):
            start_sec = words[start_idx]["start"]
            end_sec = words[end_idx - 1]["end"]
            start_frame = max(0, int(math.floor(start_sec * codec_frame_rate)))
            end_frame = min(y_len, int(math.ceil(end_sec * codec_frame_rate)))
            length = end_frame - start_frame
            if length > mask_len_max:
                break
            if (
                mask_len_min <= length <= mask_len_max
                and start_frame < end_frame
                and _overlapped_word_indices(words, start_frame, end_frame, codec_frame_rate) == list(range(start_idx, end_idx))
            ):
                candidates.append((start_frame, end_frame))
    return sorted(set(candidates))


def _normalized_word_segments(metadata: dict[str, Any]) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    for segment in load_word_segments(metadata):
        word = str(segment.get("word") or segment.get("text") or segment.get("label") or segment.get("Label") or "").strip()
        start = _first_float(segment, ("start", "Start", "begin", "Begin", "start_time", "xmin"))
        end = _first_float(segment, ("end", "End", "end_time", "xmax"))
        if not word or start is None or end is None or end <= start:
            continue
        words.append({"word": word, "start": start, "end": end})
    return sorted(words, key=lambda value: (float(value["start"]), float(value["end"])))


def _overlapped_word_indices(
    words: Sequence[dict[str, float | str]],
    start_frame: int,
    end_frame: int,
    codec_frame_rate: float,
) -> list[int]:
    start_sec = start_frame / float(codec_frame_rate)
    end_sec = end_frame / float(codec_frame_rate)
    return [
        idx
        for idx, word in enumerate(words)
        if float(word["start"]) < end_sec and start_sec < float(word["end"])
    ]


def _first_float(data: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None
