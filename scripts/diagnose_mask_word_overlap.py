from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.online_masking import sample_eval_mask_intervals, sample_mask_intervals  # noqa: E402
from dataset.text_gap_alignment import load_word_segments  # noqa: E402
from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, sample_word_aligned_mask_interval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose how frame-level speech masks overlap word alignments.")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument("--output-dir", default="exp/mask_word_overlap_diagnostics")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-len-min", type=int, default=10)
    parser.add_argument("--mask-len-max", type=int, default=30)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--codec-frame-rate", type=float, default=50.0)
    parser.add_argument("--mode", choices=["random", "deterministic", "word-aligned", "all", "both"], default="both")
    parser.add_argument("--word-mask-max-words", type=int, default=4)
    parser.add_argument("--boundary-tolerance-sec", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root, load_speaker_embedding=False)
    n = min(args.num_samples, len(dataset))
    indices = list(range(n))
    generator = torch.Generator().manual_seed(args.seed)

    if args.mode == "all":
        modes = ["random", "deterministic", "word-aligned"]
    elif args.mode == "both":
        modes = ["random", "deterministic"]
    else:
        modes = [args.mode]
    all_reports: dict[str, Any] = {}
    for mode in modes:
        rows = []
        for index in indices:
            item = dataset[index]
            y_len = int(item["tokens"].shape[-1])
            if mode == "deterministic":
                mask_intervals, _ = sample_eval_mask_intervals([y_len], mask_len=args.eval_mask_len)
            elif mode == "word-aligned":
                interval = sample_word_aligned_mask_interval(
                    metadata=item["metadata"],
                    y_len=y_len,
                    mask_len_min=args.mask_len_min,
                    mask_len_max=args.mask_len_max,
                    codec_frame_rate=args.codec_frame_rate,
                    max_words=args.word_mask_max_words,
                    deterministic=False,
                    generator=generator,
                )
                if interval is None:
                    rows.append(analyze_missing_candidate(args, mode, index, item))
                    continue
                mask_intervals = [[interval]]
            else:
                mask_intervals, _ = sample_mask_intervals(
                    [y_len],
                    mask_len_min=args.mask_len_min,
                    mask_len_max=args.mask_len_max,
                    max_n_spans=args.max_n_spans,
                    mask_sample_dist=args.mask_sample_dist,
                    min_gap=args.min_gap,
                    generator=generator,
                )
            start_frame, end_frame = mask_intervals[0][0]
            rows.append(analyze_item(args, mode, index, item, start_frame, end_frame))

        summary = summarize_rows(rows)
        all_reports[mode] = {"summary": summary, "rows": rows}
        write_csv(output_dir / f"{mode}_mask_word_overlap.csv", rows)
        write_json(output_dir / f"{mode}_mask_word_overlap.json", {"summary": summary, "rows": rows})

    write_markdown(output_dir / "mask_word_overlap_report.md", args, all_reports)
    print(
        {
            "report": str(output_dir / "mask_word_overlap_report.md"),
            "samples": str(n),
            "modes": ",".join(modes),
        },
        flush=True,
    )


def analyze_item(
    args: argparse.Namespace,
    mode: str,
    index: int,
    item: dict[str, Any],
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    gap_start = start_frame / args.codec_frame_rate
    gap_end = end_frame / args.codec_frame_rate
    gap_duration = gap_end - gap_start
    word_segments = normalize_words(load_word_segments(item["metadata"]))
    overlapping = [
        word
        for word in word_segments
        if word["start"] < gap_end and gap_start < word["end"]
    ]

    full_words = []
    partial_words = []
    overlap_durations = []
    for word in overlapping:
        overlap = max(0.0, min(gap_end, word["end"]) - max(gap_start, word["start"]))
        overlap_durations.append(overlap)
        word_duration = max(word["end"] - word["start"], 1e-8)
        if gap_start <= word["start"] + args.boundary_tolerance_sec and gap_end >= word["end"] - args.boundary_tolerance_sec:
            full_words.append(word)
        elif overlap / word_duration >= 0.98:
            full_words.append(word)
        else:
            partial_words.append(word)

    start_cut_word = find_boundary_cut_word(word_segments, gap_start, args.boundary_tolerance_sec)
    end_cut_word = find_boundary_cut_word(word_segments, gap_end, args.boundary_tolerance_sec)
    category = categorize(overlapping, full_words, partial_words, start_cut_word, end_cut_word)
    gap_word_coverage = sum(overlap_durations) / max(gap_duration, 1e-8)

    return {
        "mode": mode,
        "index": index,
        "utterance_id": item.get("utterance_id"),
        "num_frames": int(item["tokens"].shape[-1]),
        "duration_sec": round(int(item["tokens"].shape[-1]) / args.codec_frame_rate, 4),
        "mask_start_frame": int(start_frame),
        "mask_end_frame": int(end_frame),
        "mask_len_frames": int(end_frame - start_frame),
        "mask_start_sec": round(gap_start, 4),
        "mask_end_sec": round(gap_end, 4),
        "mask_duration_sec": round(gap_duration, 4),
        "num_words": len(word_segments),
        "num_overlap_words": len(overlapping),
        "num_full_words": len(full_words),
        "num_partial_words": len(partial_words),
        "gap_word_coverage": round(gap_word_coverage, 4),
        "start_cut_word": start_cut_word["word"] if start_cut_word else "",
        "end_cut_word": end_cut_word["word"] if end_cut_word else "",
        "overlap_words": " ".join(word["word"] for word in overlapping),
        "full_words": " ".join(word["word"] for word in full_words),
        "partial_words": " ".join(word["word"] for word in partial_words),
        "category": category,
        "text": item.get("text", "").strip(),
    }


def analyze_missing_candidate(args: argparse.Namespace, mode: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "index": index,
        "utterance_id": item.get("utterance_id"),
        "num_frames": int(item["tokens"].shape[-1]),
        "duration_sec": round(int(item["tokens"].shape[-1]) / args.codec_frame_rate, 4),
        "mask_start_frame": -1,
        "mask_end_frame": -1,
        "mask_len_frames": 0,
        "mask_start_sec": -1,
        "mask_end_sec": -1,
        "mask_duration_sec": 0,
        "num_words": len(normalize_words(load_word_segments(item["metadata"]))),
        "num_overlap_words": 0,
        "num_full_words": 0,
        "num_partial_words": 0,
        "gap_word_coverage": 0,
        "start_cut_word": "",
        "end_cut_word": "",
        "overlap_words": "",
        "full_words": "",
        "partial_words": "",
        "category": "no_aligned_candidate",
        "text": item.get("text", "").strip(),
    }


def normalize_words(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = []
    for item in items:
        word = item.get("word") or item.get("text") or item.get("label") or item.get("Label")
        start = first_float(item, ("start", "Start", "begin", "Begin", "xmin"))
        end = first_float(item, ("end", "End", "stop", "Stop", "xmax"))
        if word is None or start is None or end is None:
            continue
        word = str(word).strip()
        if not word or end <= start:
            continue
        words.append({"word": word, "start": float(start), "end": float(end)})
    return sorted(words, key=lambda value: (value["start"], value["end"]))


def first_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def find_boundary_cut_word(words: list[dict[str, Any]], boundary: float, tolerance: float) -> dict[str, Any] | None:
    for word in words:
        if word["start"] + tolerance < boundary < word["end"] - tolerance:
            return word
    return None


def categorize(
    overlapping: list[dict[str, Any]],
    full_words: list[dict[str, Any]],
    partial_words: list[dict[str, Any]],
    start_cut_word: dict[str, Any] | None,
    end_cut_word: dict[str, Any] | None,
) -> str:
    if not overlapping:
        return "no_word_overlap"
    if partial_words or start_cut_word or end_cut_word:
        if full_words:
            return "mixed_full_and_partial_words"
        return "partial_word_only"
    return "full_word_span"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(row["category"] for row in rows)
    total = len(rows)
    mask_lens = [int(row["mask_len_frames"]) for row in rows]
    word_coverage = [float(row["gap_word_coverage"]) for row in rows]
    overlap_counts = [int(row["num_overlap_words"]) for row in rows]
    return {
        "samples": total,
        "category_counts": dict(categories),
        "category_rates": {
            key: round(value / max(total, 1), 4)
            for key, value in sorted(categories.items())
        },
        "avg_mask_len_frames": round(statistics.mean(mask_lens), 4) if mask_lens else 0.0,
        "avg_gap_word_coverage": round(statistics.mean(word_coverage), 4) if word_coverage else 0.0,
        "avg_overlap_words": round(statistics.mean(overlap_counts), 4) if overlap_counts else 0.0,
        "partial_or_mixed_rate": round(
            sum(categories[key] for key in ("partial_word_only", "mixed_full_and_partial_words")) / max(total, 1),
            4,
        ),
        "full_word_span_rate": round(categories["full_word_span"] / max(total, 1), 4),
        "no_word_overlap_rate": round(categories["no_word_overlap"] / max(total, 1), 4),
        "no_aligned_candidate_rate": round(categories["no_aligned_candidate"] / max(total, 1), 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, args: argparse.Namespace, reports: dict[str, Any]) -> None:
    lines = [
        "# Mask Word-Overlap Diagnostics",
        "",
        "This report checks whether current frame-level masks cover complete words or cut through word interiors.",
        "",
        "## Settings",
        "",
        "```text",
        f"wds_root: {args.wds_root}",
        f"num_samples: {args.num_samples}",
        f"seed: {args.seed}",
        f"mask_len_min: {args.mask_len_min}",
        f"mask_len_max: {args.mask_len_max}",
        f"eval_mask_len: {args.eval_mask_len}",
        f"codec_frame_rate: {args.codec_frame_rate}",
        f"boundary_tolerance_sec: {args.boundary_tolerance_sec}",
        "```",
        "",
    ]
    for mode, report in reports.items():
        summary = report["summary"]
        rows = report["rows"]
        lines.extend(
            [
                f"## {mode.title()} Mask Summary",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| samples | {summary['samples']} |",
                f"| full_word_span_rate | {summary['full_word_span_rate']:.4f} |",
                f"| partial_or_mixed_rate | {summary['partial_or_mixed_rate']:.4f} |",
                f"| no_word_overlap_rate | {summary['no_word_overlap_rate']:.4f} |",
                f"| no_aligned_candidate_rate | {summary['no_aligned_candidate_rate']:.4f} |",
                f"| avg_mask_len_frames | {summary['avg_mask_len_frames']:.4f} |",
                f"| avg_gap_word_coverage | {summary['avg_gap_word_coverage']:.4f} |",
                f"| avg_overlap_words | {summary['avg_overlap_words']:.4f} |",
                "",
                "Category counts:",
                "",
                "```json",
                json.dumps(summary["category_counts"], indent=2, ensure_ascii=False),
                "```",
                "",
                "Representative partial-word cuts:",
                "",
                "| utterance | mask_sec | category | cut_start | cut_end | overlap_words | text |",
                "|---|---:|---|---|---|---|---|",
            ]
        )
        partial_rows = [
            row for row in rows
            if row["category"] in {"partial_word_only", "mixed_full_and_partial_words"}
        ][:12]
        for row in partial_rows:
            text = shorten(str(row["text"]), 90)
            words = shorten(str(row["overlap_words"]), 40)
            lines.append(
                f"| `{row['utterance_id']}` | {row['mask_start_sec']}-{row['mask_end_sec']} | "
                f"{row['category']} | {row['start_cut_word'] or '-'} | {row['end_cut_word'] or '-'} | "
                f"{words or '-'} | {text} |"
            )
        if not partial_rows:
            lines.append("| - | - | - | - | - | - | - |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def shorten(text: str, limit: int) -> str:
    clean = " ".join(text.replace("|", " ").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


if __name__ == "__main__":
    main()
