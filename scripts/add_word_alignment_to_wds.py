from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Copy tokenized WDS shards and add word-level alignment metadata to each sample json."
    )
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100")
    parser.add_argument("--alignment-root", required=True)
    parser.add_argument("--output-dir", default="data/tokenized_wds/train-clean-100-aligned")
    parser.add_argument(
        "--alignment-format",
        choices=["auto", "json", "textgrid"],
        default="auto",
        help="Alignment file format. auto checks json first, then TextGrid.",
    )
    parser.add_argument(
        "--missing",
        choices=["skip", "error"],
        default="skip",
        help="skip leaves samples without alignment metadata; error fails on first missing alignment.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wds_root = Path(args.wds_root)
    alignment_root = Path(args.alignment_root)
    output_dir = Path(args.output_dir)
    if not wds_root.exists():
        raise FileNotFoundError(f"WDS root not found: {wds_root}")
    if not alignment_root.exists():
        raise FileNotFoundError(f"alignment root not found: {alignment_root}")
    _prepare_output_dir(output_dir, args.overwrite)

    alignment_index = build_alignment_index(alignment_root, args.alignment_format)
    shard_paths = sorted(wds_root.glob("*.tar"))
    if not shard_paths:
        raise RuntimeError(f"no tar shards found under {wds_root}")

    total = 0
    aligned = 0
    missing = 0
    for shard_path in shard_paths:
        out_shard = output_dir / shard_path.name
        shard_total, shard_aligned, shard_missing = copy_shard_with_alignments(
            shard_path=shard_path,
            out_shard=out_shard,
            alignment_index=alignment_index,
            missing_policy=args.missing,
        )
        total += shard_total
        aligned += shard_aligned
        missing += shard_missing
        print(
            {
                "shard": shard_path.name,
                "samples": str(shard_total),
                "aligned": str(shard_aligned),
                "missing": str(shard_missing),
            },
            flush=True,
        )

    write_manifest(
        source_root=wds_root,
        alignment_root=alignment_root,
        output_dir=output_dir,
        total=total,
        aligned=aligned,
        missing=missing,
    )
    print(
        {
            "done": str(output_dir),
            "samples": str(total),
            "aligned": str(aligned),
            "missing": str(missing),
            "alignment_coverage": f"{aligned / max(total, 1):.4f}",
        },
        flush=True,
    )


def build_alignment_index(alignment_root: Path, alignment_format: str) -> dict[str, Path]:
    candidates: list[Path] = []
    if alignment_format in {"auto", "json"}:
        candidates.extend(sorted(alignment_root.rglob("*.json")))
    if alignment_format in {"auto", "textgrid"}:
        candidates.extend(sorted(alignment_root.rglob("*.TextGrid")))
        candidates.extend(sorted(alignment_root.rglob("*.textgrid")))

    index: dict[str, Path] = {}
    for path in candidates:
        key = path.stem
        index.setdefault(key, path)
    if not index:
        raise RuntimeError(f"no alignment files found under {alignment_root}")
    return index


def copy_shard_with_alignments(
    shard_path: Path,
    out_shard: Path,
    alignment_index: dict[str, Path],
    missing_policy: str,
) -> tuple[int, int, int]:
    total = 0
    aligned = 0
    missing = 0
    with tarfile.open(shard_path, mode="r") as src, tarfile.open(out_shard, mode="w") as dst:
        for member in src.getmembers():
            if not member.isfile():
                continue
            data = _read_member(src, member)
            suffix_key = _json_sample_key(member.name)
            if suffix_key is not None:
                total += 1
                alignment_path = alignment_index.get(suffix_key)
                if alignment_path is None:
                    missing += 1
                    if missing_policy == "error":
                        raise FileNotFoundError(f"missing alignment for sample {suffix_key}")
                else:
                    metadata = json.loads(data.decode("utf-8"))
                    word_segments = load_alignment(alignment_path)
                    metadata["word_segments"] = word_segments
                    metadata["alignment_path"] = str(alignment_path)
                    metadata["alignment_format"] = alignment_path.suffix.lower().lstrip(".")
                    data = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
                    aligned += 1
            _add_bytes(dst, member.name, data)
    return total, aligned, missing


def load_alignment(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_alignment(path)
    if suffix == ".textgrid":
        return load_textgrid_alignment(path)
    raise ValueError(f"unsupported alignment file: {path}")


def load_json_alignment(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("word_segments", "words", "alignment", "alignments"):
            value = data.get(key)
            if isinstance(value, list):
                return normalize_word_segments(value)
    if isinstance(data, list):
        return normalize_word_segments(data)
    raise ValueError(f"could not find word alignment list in {path}")


def load_textgrid_alignment(path: Path) -> list[dict[str, Any]]:
    praatio_segments = _load_textgrid_with_praatio(path)
    if praatio_segments:
        return praatio_segments

    text = path.read_text(encoding="utf-8", errors="ignore")
    intervals = []
    for block in re.finditer(
        r"intervals \[\d+\]:\s*xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = \"(.*?)\"",
        text,
        flags=re.DOTALL,
    ):
        word = block.group(3).strip()
        if not word:
            continue
        intervals.append(
            {
                "word": word,
                "start": float(block.group(1)),
                "end": float(block.group(2)),
            }
        )
    if not intervals:
        raise ValueError(f"no word intervals parsed from {path}")
    return intervals


def _load_textgrid_with_praatio(path: Path) -> list[dict[str, Any]]:
    try:
        from praatio import textgrid
    except ImportError:
        return []

    try:
        grid = textgrid.openTextgrid(
            str(path),
            includeEmptyIntervals=False,
            reportingMode="silence",
            duplicateNamesMode="rename",
        )
    except Exception:
        return []

    tier_names = list(getattr(grid, "tierNames", []))
    preferred_names = [
        name
        for name in tier_names
        if any(marker in name.lower() for marker in ("word", "words", "transcript"))
    ]
    for tier_name in preferred_names + tier_names:
        try:
            tier = grid.getTier(tier_name)
        except Exception:
            continue
        entries = getattr(tier, "entries", [])
        intervals = []
        for entry in entries:
            start = getattr(entry, "start", None)
            end = getattr(entry, "end", None)
            label = getattr(entry, "label", None)
            if start is None and isinstance(entry, tuple) and len(entry) >= 3:
                start, end, label = entry[:3]
            if start is None or end is None or label is None:
                continue
            word = str(label).strip()
            if not word:
                continue
            intervals.append({"word": word, "start": float(start), "end": float(end)})
        if intervals:
            return intervals
    return []


def normalize_word_segments(items: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        word = item.get("word") or item.get("text") or item.get("label") or item.get("Label")
        start = _first_present(item, ("start", "begin", "Begin", "start_time"))
        end = _first_present(item, ("end", "End", "end_time"))
        if word is None or start is None or end is None:
            continue
        word = str(word).strip()
        if not word:
            continue
        normalized.append({"word": word, "start": float(start), "end": float(end)})
    if not normalized:
        raise ValueError("alignment has no usable word segments")
    return normalized


def write_manifest(
    source_root: Path,
    alignment_root: Path,
    output_dir: Path,
    total: int,
    aligned: int,
    missing: int,
) -> None:
    source_manifest = source_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if source_manifest.exists():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["source_wds_root"] = str(source_root)
    manifest["alignment_root"] = str(alignment_root)
    manifest["word_alignment"] = {
        "samples": total,
        "aligned": aligned,
        "missing": missing,
        "coverage": aligned / max(total, 1),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _json_sample_key(member_name: str) -> str | None:
    filename = Path(member_name).name
    marker = ".json"
    if not filename.endswith(marker):
        return None
    return filename[: -len(marker)]


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member.name)
    return extracted.read()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


if __name__ == "__main__":
    main()
