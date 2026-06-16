from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare a LibriTTS-R split as an MFA wav/lab corpus.")
    parser.add_argument("--libritts-root", default="data/LibriTTS_R")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--output-dir", default="data/mfa_corpus/train-clean-100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy-audio", action="store_true", help="Copy wav files instead of symlinking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.libritts_root) / args.split
    output_dir = Path(args.output_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"LibriTTS-R split not found: {source_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for wav_path in sorted(source_root.rglob("*.wav")):
        text_path = wav_path.with_suffix(".normalized.txt")
        if not text_path.exists():
            skipped += 1
            continue
        transcript = " ".join(text_path.read_text(encoding="utf-8").split())
        if not transcript:
            skipped += 1
            continue

        relative = wav_path.relative_to(source_root)
        target_wav = output_dir / relative
        target_lab = target_wav.with_suffix(".lab")
        target_wav.parent.mkdir(parents=True, exist_ok=True)
        if target_wav.exists() or target_wav.is_symlink():
            target_wav.unlink()
        if args.copy_audio:
            shutil.copy2(wav_path, target_wav)
        else:
            target_wav.symlink_to(wav_path.resolve())
        target_lab.write_text(transcript + "\n", encoding="utf-8")

        written += 1
        if args.limit is not None and written >= args.limit:
            break

    print(
        {
            "source": str(source_root),
            "output": str(output_dir),
            "written": str(written),
            "skipped": str(skipped),
            "audio_mode": "copy" if args.copy_audio else "symlink",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
