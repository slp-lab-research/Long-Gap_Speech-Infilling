from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SampleVerdict:
    sample: str
    utterance_id: str
    mask_start_sec: float
    mask_end_sec: float
    gap_duration_sec: float
    score: float
    verdict: str
    original_gap_rms_db: float
    generated_gap_rms_db: float
    energy_error_db: float
    waveform_corr: float
    snr_db: float
    log_spectral_distance: float
    generated_silence_ratio: float
    generated_clip_ratio: float
    boundary_jump_ratio: float
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate generated VoiceCraft-X inpainting wavs like an audio triage agent.")
    parser.add_argument("--samples-root", required=True, help="Directory containing sample subdirectories.")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--silence-db", type=float, default=-45.0)
    parser.add_argument("--clip-threshold", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_root = Path(args.samples_root)
    if not samples_root.exists():
        raise FileNotFoundError(f"samples root not found: {samples_root}")

    sample_dirs = find_sample_dirs(samples_root)
    if not sample_dirs:
        raise RuntimeError(f"no sample directories found under {samples_root}")

    verdicts = [evaluate_sample(sample_dir, args) for sample_dir in sample_dirs]
    verdicts.sort(key=lambda item: item.score)

    output_md = Path(args.output_md) if args.output_md else samples_root / "audio_agent_report.md"
    output_json = Path(args.output_json) if args.output_json else samples_root / "audio_agent_report.json"
    output_md.write_text(render_markdown(samples_root, verdicts, args.top_k), encoding="utf-8")
    output_json.write_text(json.dumps([asdict(item) for item in verdicts], indent=2), encoding="utf-8")

    summary = summarize(verdicts)
    print({**summary, "report": str(output_md), "json": str(output_json)}, flush=True)


def find_sample_dirs(root: Path) -> list[Path]:
    dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "metadata.json").exists() and (path / "original.wav").exists():
            dirs.append(path)
    return dirs


def evaluate_sample(sample_dir: Path, args: argparse.Namespace) -> SampleVerdict:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    original, sr = load_mono(sample_dir / "original.wav")
    generated, gen_sr = load_mono(sample_dir / "generated_matched.wav")
    if gen_sr != sr:
        generated = torchaudio.transforms.Resample(orig_freq=gen_sr, new_freq=sr)(generated.unsqueeze(0)).squeeze(0)

    min_len = min(original.numel(), generated.numel())
    original = original[:min_len]
    generated = generated[:min_len]

    start = clamp_index(round(float(metadata["mask_start_sec"]) * sr), min_len)
    end = clamp_index(round(float(metadata["mask_end_sec"]) * sr), min_len)
    if end <= start:
        raise RuntimeError(f"invalid mask interval in {sample_dir}")

    original_gap = original[start:end]
    generated_gap = generated[start:end]

    original_rms = rms_db(original_gap)
    generated_rms = rms_db(generated_gap)
    energy_error = generated_rms - original_rms
    corr = waveform_corr(original_gap, generated_gap)
    snr = snr_db(original_gap, generated_gap)
    spectral = log_spectral_distance(original_gap, generated_gap)
    silence = silence_ratio(generated_gap, sr, args.silence_db)
    clip = clip_ratio(generated_gap, args.clip_threshold)
    boundary = boundary_jump_ratio(generated, start, end, sr)

    notes = build_notes(
        energy_error_db=energy_error,
        corr=corr,
        snr=snr,
        spectral=spectral,
        silence=silence,
        clip=clip,
        boundary=boundary,
    )
    score = quality_score(
        energy_error_db=energy_error,
        corr=corr,
        snr=snr,
        spectral=spectral,
        silence=silence,
        clip=clip,
        boundary=boundary,
    )

    return SampleVerdict(
        sample=sample_dir.name,
        utterance_id=str(metadata.get("utterance_id", sample_dir.name)),
        mask_start_sec=float(metadata["mask_start_sec"]),
        mask_end_sec=float(metadata["mask_end_sec"]),
        gap_duration_sec=float(metadata["mask_end_sec"]) - float(metadata["mask_start_sec"]),
        score=round(score, 2),
        verdict=verdict_label(score),
        original_gap_rms_db=round(original_rms, 2),
        generated_gap_rms_db=round(generated_rms, 2),
        energy_error_db=round(energy_error, 2),
        waveform_corr=round(corr, 4),
        snr_db=round(snr, 2),
        log_spectral_distance=round(spectral, 4),
        generated_silence_ratio=round(silence, 4),
        generated_clip_ratio=round(clip, 6),
        boundary_jump_ratio=round(boundary, 3),
        notes=notes,
    )


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.ndim != 2:
        raise RuntimeError(f"unexpected wav shape for {path}: {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).float(), int(sr)


def clamp_index(value: int, length: int) -> int:
    return max(0, min(int(value), int(length)))


def rms_db(wav: torch.Tensor, eps: float = 1e-8) -> float:
    return 20.0 * math.log10(float(torch.sqrt(torch.mean(wav.square()) + eps).item()) + eps)


def waveform_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    n = min(a.numel(), b.numel())
    if n <= 1:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) < eps:
        return 0.0
    return float((a * b).sum().div(denom).clamp(-1, 1).item())


def snr_db(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-8) -> float:
    n = min(reference.numel(), candidate.numel())
    reference = reference[:n]
    candidate = candidate[:n]
    noise = reference - candidate
    ref_power = torch.mean(reference.square()) + eps
    noise_power = torch.mean(noise.square()) + eps
    return 10.0 * math.log10(float((ref_power / noise_power).item()) + eps)


def log_spectral_distance(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-7) -> float:
    n = min(reference.numel(), candidate.numel())
    reference = reference[:n]
    candidate = candidate[:n]
    if n < 32:
        return 0.0
    n_fft = min(512, max(32, 2 ** int(math.floor(math.log2(n)))))
    hop = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=reference.device)
    ref_stft = torch.stft(reference, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    cand_stft = torch.stft(candidate, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    ref_log = torch.log(ref_stft.abs().clamp_min(eps))
    cand_log = torch.log(cand_stft.abs().clamp_min(eps))
    return float(torch.sqrt(torch.mean((ref_log - cand_log).square())).item())


def silence_ratio(wav: torch.Tensor, sr: int, silence_db: float) -> float:
    frame = max(1, int(0.02 * sr))
    if wav.numel() < frame:
        return 1.0 if rms_db(wav) < silence_db else 0.0
    frames = wav.unfold(0, frame, frame)
    frame_db = 20.0 * torch.log10(torch.sqrt(torch.mean(frames.square(), dim=1) + 1e-8) + 1e-8)
    return float((frame_db < silence_db).float().mean().item())


def clip_ratio(wav: torch.Tensor, threshold: float) -> float:
    return float((wav.abs() >= threshold).float().mean().item())


def boundary_jump_ratio(wav: torch.Tensor, start: int, end: int, sr: int) -> float:
    jumps = []
    if 0 < start < wav.numel():
        jumps.append(float(abs(wav[start] - wav[start - 1]).item()))
    if 0 < end < wav.numel():
        jumps.append(float(abs(wav[end] - wav[end - 1]).item()))
    if not jumps:
        return 0.0
    context = int(0.05 * sr)
    left = wav[max(0, start - context) : max(0, start - 1)]
    right = wav[min(wav.numel(), end + 1) : min(wav.numel(), end + context)]
    diffs = []
    for span in (left, right):
        if span.numel() > 2:
            diffs.append(span.diff().abs())
    if not diffs:
        return max(jumps)
    ref = torch.cat(diffs).median().clamp_min(1e-5)
    return max(jumps) / float(ref.item())


def build_notes(
    *,
    energy_error_db: float,
    corr: float,
    snr: float,
    spectral: float,
    silence: float,
    clip: float,
    boundary: float,
) -> list[str]:
    notes = []
    if abs(energy_error_db) > 8:
        notes.append("gap energy is far from original")
    elif abs(energy_error_db) > 4:
        notes.append("gap energy mismatch")
    if corr < 0.05:
        notes.append("very low waveform correlation")
    elif corr < 0.15:
        notes.append("low waveform correlation")
    if snr < -3:
        notes.append("large waveform error")
    if spectral > 2.5:
        notes.append("large spectral mismatch")
    if silence > 0.7:
        notes.append("generated gap is mostly silent")
    if clip > 0:
        notes.append("generated gap has clipping")
    if boundary > 20:
        notes.append("likely audible boundary click")
    elif boundary > 10:
        notes.append("possible boundary discontinuity")
    return notes or ["no severe signal-level issue"]


def quality_score(
    *,
    energy_error_db: float,
    corr: float,
    snr: float,
    spectral: float,
    silence: float,
    clip: float,
    boundary: float,
) -> float:
    score = 100.0
    score -= min(30.0, abs(energy_error_db) * 3.0)
    score -= min(25.0, max(0.0, 0.25 - corr) * 70.0)
    score -= min(20.0, max(0.0, 8.0 - snr) * 1.5)
    score -= min(20.0, spectral * 5.0)
    score -= min(25.0, silence * 25.0)
    score -= min(30.0, clip * 1000.0)
    score -= min(20.0, max(0.0, boundary - 6.0) * 1.2)
    return max(0.0, min(100.0, score))


def verdict_label(score: float) -> str:
    if score >= 70:
        return "pass"
    if score >= 45:
        return "partial"
    return "fail"


def summarize(verdicts: list[SampleVerdict]) -> dict[str, str]:
    count = len(verdicts)
    avg_score = sum(item.score for item in verdicts) / max(count, 1)
    avg_loss_proxy = sum(item.log_spectral_distance for item in verdicts) / max(count, 1)
    return {
        "samples": str(count),
        "avg_score": f"{avg_score:.2f}",
        "avg_log_spectral_distance": f"{avg_loss_proxy:.4f}",
        "pass": str(sum(item.verdict == "pass" for item in verdicts)),
        "partial": str(sum(item.verdict == "partial" for item in verdicts)),
        "fail": str(sum(item.verdict == "fail" for item in verdicts)),
    }


def render_markdown(samples_root: Path, verdicts: list[SampleVerdict], top_k: int) -> str:
    summary = summarize(verdicts)
    best = sorted(verdicts, key=lambda item: item.score, reverse=True)[:top_k]
    worst = verdicts[:top_k]
    lines = [
        "# Generated Audio Evaluation Agent Report",
        "",
        f"- samples_root: `{samples_root}`",
        f"- samples: {summary['samples']}",
        f"- avg_score: {summary['avg_score']}",
        f"- avg_log_spectral_distance: {summary['avg_log_spectral_distance']}",
        f"- verdicts: pass={summary['pass']}, partial={summary['partial']}, fail={summary['fail']}",
        "",
        "## How To Listen",
        "",
        "For each selected sample, listen in this order: `original.wav`, `masked_input.wav`, `generated_matched.wav`, then `generated_gap_matched.wav`.",
        "",
        "## Worst Samples",
        "",
        sample_table(worst),
        "",
        "## Best Samples",
        "",
        sample_table(best),
        "",
        "## All Samples",
        "",
        sample_table(verdicts),
        "",
        "## Interpretation Rules",
        "",
        "- Low score means the generated gap is likely audibly wrong, silent, too loud/quiet, spectrally mismatched, or has boundary clicks.",
        "- This is a signal-level triage report, not a replacement for human listening or ASR/PESQ/STOI evaluation.",
        "- Prefer comparing runs by the same sample set and same deterministic mask length.",
        "",
    ]
    return "\n".join(lines)


def sample_table(verdicts: list[SampleVerdict]) -> str:
    lines = [
        "| sample | verdict | score | energy_err_db | corr | snr_db | spec_dist | silence | boundary | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in verdicts:
        notes = "; ".join(item.notes)
        lines.append(
            f"| `{item.sample}` | {item.verdict} | {item.score:.2f} | {item.energy_error_db:.2f} | "
            f"{item.waveform_corr:.4f} | {item.snr_db:.2f} | {item.log_spectral_distance:.4f} | "
            f"{item.generated_silence_ratio:.4f} | {item.boundary_jump_ratio:.3f} | {notes} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
