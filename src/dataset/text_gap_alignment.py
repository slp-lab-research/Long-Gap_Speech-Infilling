from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


def encode_aligned_fim_texts(
    text_tokenizer,
    texts: Sequence[str],
    mask_intervals: Sequence[Sequence[tuple[int, int]] | tuple[int, int]],
    y_lens: torch.Tensor | Sequence[int],
    language: str,
    pad_token_id: int,
    metadata: Sequence[dict[str, Any]] | None = None,
    codec_frame_rate: int = 50,
    mode: str = "auto",
):
    """Encode FIM prompts whose text gap matches the speech mask interval.

    ``mode='auto'`` uses word-level alignment metadata when available and falls
    back to proportional text splitting otherwise.
    """
    encoded = []
    splits: list[dict[str, str]] = []
    y_lens_list = _to_int_list(y_lens)
    metadata = metadata or [{} for _ in texts]

    for text, interval_value, y_len, item_metadata in zip(texts, mask_intervals, y_lens_list, metadata):
        interval = _single_interval(interval_value)
        prefix_text, middle_text, suffix_text, source = split_text_for_gap(
            text=text,
            interval=interval,
            y_len=y_len,
            metadata=item_metadata,
            codec_frame_rate=codec_frame_rate,
            mode=mode,
        )
        transcript = build_fim_transcript(prefix_text, middle_text, suffix_text)
        transcript = text_tokenizer.text_normalize(
            transcript,
            split=False,
            text_frontend=True,
            lang=language,
        )
        token_ids, _ = text_tokenizer(transcript)
        encoded.append(token_ids.reshape(-1).long())
        splits.append(
            {
                "prefix_text": prefix_text,
                "middle_text": middle_text,
                "suffix_text": suffix_text,
                "alignment_source": source,
            }
        )

    max_len = max(ids.numel() for ids in encoded)
    batch = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for idx, ids in enumerate(encoded):
        batch[idx, : ids.numel()] = ids
        attention_mask[idx, : ids.numel()] = True
    return batch, attention_mask, splits


def encode_aligned_fim_text(
    text_tokenizer,
    text: str,
    interval: tuple[int, int],
    y_len: int,
    language: str,
    pad_token_id: int,
    metadata: dict[str, Any] | None = None,
    codec_frame_rate: int = 50,
    mode: str = "auto",
) -> tuple[torch.Tensor, dict[str, str]]:
    batch, _, splits = encode_aligned_fim_texts(
        text_tokenizer=text_tokenizer,
        texts=[text],
        mask_intervals=[interval],
        y_lens=[y_len],
        language=language,
        pad_token_id=pad_token_id,
        metadata=[metadata or {}],
        codec_frame_rate=codec_frame_rate,
        mode=mode,
    )
    return batch, splits[0]


def split_text_for_gap(
    text: str,
    interval: tuple[int, int],
    y_len: int,
    metadata: dict[str, Any] | None = None,
    codec_frame_rate: int = 50,
    mode: str = "auto",
) -> tuple[str, str, str, str]:
    if mode not in {"auto", "aligned", "proportional", "whole"}:
        raise ValueError(f"unsupported text gap mode: {mode}")
    if mode == "whole":
        return "", text, "", "whole"

    metadata = metadata or {}
    if mode in {"auto", "aligned"}:
        word_segments = load_word_segments(metadata)
        if word_segments:
            split = split_text_by_word_times(text, interval, word_segments, codec_frame_rate)
            if split is not None:
                return (*split, "aligned")
        if mode == "aligned":
            raise RuntimeError("text gap mode 'aligned' requires word-level alignment metadata")

    return (*split_text_proportionally(text, interval, y_len), "proportional")


def build_fim_transcript(prefix_text: str, middle_text: str, suffix_text: str) -> str:
    return f"<|fim_prefix|>{prefix_text}<|fim_suffix|>{suffix_text}<|fim_middle|>{middle_text}"


def split_text_by_word_times(
    text: str,
    interval: tuple[int, int],
    word_segments: Sequence[dict[str, Any]],
    codec_frame_rate: int,
) -> tuple[str, str, str] | None:
    start_frame, end_frame = interval
    gap_start_sec = start_frame / float(codec_frame_rate)
    gap_end_sec = end_frame / float(codec_frame_rate)
    words = []
    for segment in word_segments:
        word = str(segment.get("word") or segment.get("text") or segment.get("label") or "").strip()
        start = _first_present_float(segment, ("start", "begin", "Begin", "start_time"))
        end = _first_present_float(segment, ("end", "End", "end_time"))
        if not word or start is None or end is None:
            continue
        words.append({"word": word, "start": start, "end": end})
    if not words:
        return None

    middle_indices = [
        idx for idx, word in enumerate(words)
        if word["start"] < gap_end_sec and gap_start_sec < word["end"]
    ]
    if not middle_indices:
        return None

    middle_start = min(middle_indices)
    middle_end = max(middle_indices) + 1
    prefix_words = [word["word"] for word in words[:middle_start]]
    middle_words = [word["word"] for word in words[middle_start:middle_end]]
    suffix_words = [word["word"] for word in words[middle_end:]]
    return " ".join(prefix_words), " ".join(middle_words), " ".join(suffix_words)


def split_text_proportionally(text: str, interval: tuple[int, int], y_len: int) -> tuple[str, str, str]:
    words = list(_word_spans(text))
    if not words or y_len <= 0:
        return "", text, ""
    start, end = interval
    start_ratio = min(max(float(start) / float(y_len), 0.0), 1.0)
    end_ratio = min(max(float(end) / float(y_len), start_ratio), 1.0)
    n_words = len(words)
    word_start = min(max(int(start_ratio * n_words), 0), n_words)
    word_end = min(max(int(end_ratio * n_words + 0.999999), word_start), n_words)
    if word_start == word_end and start < end:
        word_end = min(word_start + 1, n_words)
    char_start = words[word_start][0] if word_start < n_words else len(text)
    char_end = words[word_end - 1][1] if word_end > word_start else char_start
    return text[:char_start], text[char_start:char_end], text[char_end:]


def load_word_segments(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("word_segments", "words", "alignment", "alignments"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    alignment_path = metadata.get("alignment_path") or metadata.get("word_alignment_path")
    if not alignment_path:
        return []
    path = Path(str(alignment_path))
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("word_segments", "words", "alignment", "alignments"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def _single_interval(interval_value: Sequence[tuple[int, int]] | tuple[int, int]) -> tuple[int, int]:
    if len(interval_value) == 2 and isinstance(interval_value[0], int):
        return int(interval_value[0]), int(interval_value[1])
    if len(interval_value) != 1:
        raise NotImplementedError("aligned FIM currently supports one speech gap")
    interval = interval_value[0]
    return int(interval[0]), int(interval[1])


def _to_int_list(values: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().flatten().tolist()]
    return [int(value) for value in values]


def _word_spans(text: str) -> Iterable[tuple[int, int]]:
    for match in re.finditer(r"\S+", text):
        yield match.span()


def _first_present_float(data: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None
