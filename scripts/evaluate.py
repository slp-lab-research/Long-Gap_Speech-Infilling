from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from jiwer import wer
from pesq import pesq
from pystoi import stoi


TARGET_SR = 16000


@dataclass
class SampleMetrics:
    method: str
    sample: str
    utterance_id: str
    pesq: float | None
    stoi: float | None
    wer: float | None
    speaker_similarity: float | None
    reference_text: str | None
    predicted_text: str | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate generated speech gap-infilling samples with PESQ/STOI, ASR WER, and speaker similarity."
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Method name and sample root. Can be repeated, e.g. --method original=exp/a --method ours=exp/b.",
    )
    parser.add_argument(
        "--samples-root",
        default=None,
        help="Single sample root fallback. Equivalent to --method method=<samples-root>.",
    )
    parser.add_argument("--method-name", default="method", help="Name used with --samples-root.")
    parser.add_argument("--generated-name", default="generated_matched.wav")
    parser.add_argument("--original-name", default="original.wav")
    parser.add_argument("--metadata-name", default="metadata.json")
    parser.add_argument("--output-dir", default="exp/quantitative_eval")
    parser.add_argument("--model-cache-dir", default=None, help="Cache directory for Whisper and SpeechBrain models.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--reference-text-source",
        choices=["metadata", "asr-original"],
        default="metadata",
        help="Use metadata text as WER reference, or transcribe original.wav with the same ASR model.",
    )
    parser.add_argument("--skip-pesq-stoi", action="store_true")
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--skip-speaker-sim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = Path(args.model_cache_dir) if args.model_cache_dir else output_dir / "model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    whisper_model = None
    speaker_model = None

    if not args.skip_wer:
        import whisper

        whisper_model = whisper.load_model(
            args.whisper_model,
            device=str(device),
            download_root=str(model_cache_dir / "whisper"),
        )

    if not args.skip_speaker_sim:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ModuleNotFoundError:
            from speechbrain.pretrained import EncoderClassifier

        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_cache_dir / "speechbrain_spkrec_ecapa_voxceleb"),
            run_opts={"device": str(device)},
        )

    all_metrics: list[SampleMetrics] = []
    for method_name, root in methods:
        sample_dirs = find_sample_dirs(root, args.metadata_name, args.original_name, args.generated_name)
        if args.limit is not None:
            sample_dirs = sample_dirs[: args.limit]
        if not sample_dirs:
            raise RuntimeError(f"no valid sample directories found for {method_name}: {root}")
        for sample_dir in sample_dirs:
            metrics = evaluate_sample(
                method_name=method_name,
                sample_dir=sample_dir,
                args=args,
                device=device,
                whisper_model=whisper_model,
                speaker_model=speaker_model,
            )
            all_metrics.append(metrics)
            print(
                {
                    "method": metrics.method,
                    "sample": metrics.sample,
                    "pesq": format_optional(metrics.pesq),
                    "stoi": format_optional(metrics.stoi),
                    "wer": format_optional(metrics.wer),
                    "speaker_similarity": format_optional(metrics.speaker_similarity),
                    "error": metrics.error,
                },
                flush=True,
            )

    summary = summarize(all_metrics)
    write_outputs(output_dir, all_metrics, summary, args)
    print({"summary_md": str(output_dir / "summary.md"), "samples_csv": str(output_dir / "samples.csv")}, flush=True)


def parse_methods(args: argparse.Namespace) -> list[tuple[str, Path]]:
    methods: list[tuple[str, Path]] = []
    for item in args.method:
        if "=" not in item:
            raise ValueError(f"--method must be NAME=DIR, got: {item}")
        name, root = item.split("=", 1)
        methods.append((name.strip(), Path(root).expanduser()))
    if args.samples_root:
        methods.append((args.method_name, Path(args.samples_root).expanduser()))
    if not methods:
        raise ValueError("provide at least one --method NAME=DIR or --samples-root")
    return methods


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda:0")
    return torch.device(device)


def find_sample_dirs(root: Path, metadata_name: str, original_name: str, generated_name: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"samples root not found: {root}")
    dirs = []
    for metadata in sorted(root.rglob(metadata_name)):
        sample_dir = metadata.parent
        if (sample_dir / original_name).exists() and (sample_dir / generated_name).exists():
            dirs.append(sample_dir)
    return dirs


def evaluate_sample(
    *,
    method_name: str,
    sample_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    whisper_model: Any,
    speaker_model: Any,
) -> SampleMetrics:
    metadata_path = sample_dir / args.metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    utterance_id = str(metadata.get("utterance_id", sample_dir.name))
    try:
        original, _ = load_audio_16k(sample_dir / args.original_name)
        generated, _ = load_audio_16k(sample_dir / args.generated_name)
        original, generated = align_pair(original, generated)

        pesq_score = None
        stoi_score = None
        if not args.skip_pesq_stoi:
            pesq_score = safe_pesq(original, generated)
            stoi_score = safe_stoi(original, generated)

        reference_text = None
        predicted_text = None
        wer_score = None
        if not args.skip_wer:
            if args.reference_text_source == "metadata":
                reference_text = str(metadata.get("text") or "").strip()
            else:
                reference_text = transcribe(whisper_model, sample_dir / args.original_name, device)
            predicted_text = transcribe(whisper_model, sample_dir / args.generated_name, device)
            if reference_text:
                wer_score = wer(normalize_text(reference_text), normalize_text(predicted_text))

        speaker_similarity = None
        if not args.skip_speaker_sim:
            speaker_similarity = compute_speaker_similarity(speaker_model, original, generated, device)

        return SampleMetrics(
            method=method_name,
            sample=sample_dir.name,
            utterance_id=utterance_id,
            pesq=pesq_score,
            stoi=stoi_score,
            wer=wer_score,
            speaker_similarity=speaker_similarity,
            reference_text=reference_text,
            predicted_text=predicted_text,
        )
    except Exception as exc:  # keep batch evaluation running and surface bad samples in outputs.
        return SampleMetrics(
            method=method_name,
            sample=sample_dir.name,
            utterance_id=utterance_id,
            pesq=None,
            stoi=None,
            wer=None,
            speaker_similarity=None,
            reference_text=str(metadata.get("text") or "").strip() or None,
            predicted_text=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)
    wav = wav.squeeze(0).contiguous()
    peak = wav.abs().max().clamp_min(1e-6)
    if float(peak) > 1.0:
        wav = wav / peak
    return wav.cpu(), TARGET_SR


def align_pair(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(reference.numel(), candidate.numel())
    if length < TARGET_SR // 4:
        raise ValueError("audio is too short for evaluation")
    return reference[:length], candidate[:length]


def safe_pesq(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    try:
        return float(pesq(TARGET_SR, reference.numpy(), candidate.numpy(), "wb"))
    except Exception:
        return None


def safe_stoi(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    try:
        return float(stoi(reference.numpy(), candidate.numpy(), TARGET_SR, extended=False))
    except Exception:
        return None


def transcribe(model: Any, wav_path: Path, device: torch.device) -> str:
    result = model.transcribe(str(wav_path), fp16=device.type == "cuda", language="en")
    return str(result.get("text", "")).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_speaker_similarity(model: Any, reference: torch.Tensor, candidate: torch.Tensor, device: torch.device) -> float:
    with torch.inference_mode():
        ref = reference.to(device).unsqueeze(0)
        cand = candidate.to(device).unsqueeze(0)
        ref_emb = model.encode_batch(ref).squeeze()
        cand_emb = model.encode_batch(cand).squeeze()
        return float(F.cosine_similarity(ref_emb.flatten(), cand_emb.flatten(), dim=0).item())


def summarize(metrics: list[SampleMetrics]) -> dict[str, dict[str, str]]:
    by_method: dict[str, list[SampleMetrics]] = {}
    for item in metrics:
        by_method.setdefault(item.method, []).append(item)

    summary = {}
    for method, rows in by_method.items():
        summary[method] = {
            "n": str(len(rows)),
            "valid_pesq": str(count_valid(row.pesq for row in rows)),
            "pesq": mean_std(row.pesq for row in rows),
            "stoi": mean_std(row.stoi for row in rows),
            "wer": mean_std(row.wer for row in rows),
            "speaker_similarity": mean_std(row.speaker_similarity for row in rows),
            "errors": str(sum(1 for row in rows if row.error)),
        }
    return summary


def count_valid(values) -> int:
    return sum(1 for value in values if value is not None and math.isfinite(float(value)))


def mean_std(values) -> str:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return "NA"
    if len(clean) == 1:
        return f"{clean[0]:.4f}"
    return f"{mean(clean):.4f} ± {stdev(clean):.4f}"


def format_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_outputs(
    output_dir: Path,
    metrics: list[SampleMetrics],
    summary: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "samples.json").write_text(
        json.dumps([asdict(item) for item in metrics], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()) if metrics else [])
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))

    lines = [
        "# Quantitative Evaluation",
        "",
        f"- Generated wav: `{args.generated_name}`",
        f"- ASR model: `{args.whisper_model}`" if not args.skip_wer else "- ASR WER: skipped",
        f"- Reference text source: `{args.reference_text_source}`" if not args.skip_wer else "",
        "",
        "| Method | N | PESQ ↑ | STOI ↑ | WER ↓ | Speaker Sim. ↑ | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| {method} | {row['n']} | {row['pesq']} | {row['stoi']} | {row['wer']} | "
            f"{row['speaker_similarity']} | {row['errors']} |"
        )
    lines.append("")
    lines.append(
        "All methods should be generated from the same validation samples, gap locations, and decoding settings "
        "for this table to be interpreted as a controlled comparison."
    )
    (output_dir / "summary.md").write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
