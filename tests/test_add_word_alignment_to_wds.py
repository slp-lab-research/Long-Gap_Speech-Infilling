import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.add_word_alignment_to_wds import (
    build_alignment_index,
    copy_shard_with_alignments,
    load_textgrid_alignment,
)


class AddWordAlignmentToWDSTest(unittest.TestCase):
    def test_copy_shard_adds_word_segments_to_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "sample-000000.tar"
            alignment_root = root / "alignments"
            alignment_root.mkdir()
            output = root / "out.tar"

            self._write_wds_sample(shard)
            (alignment_root / "utt_a.json").write_text(
                json.dumps(
                    {
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.2},
                            {"word": "world", "start": 0.2, "end": 0.5},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            index = build_alignment_index(alignment_root, "auto")
            total, aligned, missing = copy_shard_with_alignments(shard, output, index, "error")

            self.assertEqual((total, aligned, missing), (1, 1, 0))
            with tarfile.open(output, mode="r") as tar:
                metadata = json.loads(tar.extractfile("utt_a.json").read().decode("utf-8"))
            self.assertEqual(metadata["word_segments"][1], {"word": "world", "start": 0.2, "end": 0.5})

    def test_load_textgrid_alignment_reads_word_tier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "utt_a.TextGrid"
            path.write_text(
                """
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "hello"
        intervals [2]:
            xmin = 0.2
            xmax = 0.5
            text = "world"
""",
                encoding="utf-8",
            )

            self.assertEqual(
                load_textgrid_alignment(path),
                [
                    {"word": "hello", "start": 0.0, "end": 0.2},
                    {"word": "world", "start": 0.2, "end": 0.5},
                ],
            )

    def _write_wds_sample(self, shard: Path) -> None:
        with tarfile.open(shard, mode="w") as tar:
            self._add_bytes(tar, "utt_a.tokens.pt", self._tensor_bytes(torch.zeros(4, 10, dtype=torch.long)))
            self._add_bytes(tar, "utt_a.spk.pt", self._tensor_bytes(torch.ones(192)))
            self._add_bytes(tar, "utt_a.txt", b"hello world")
            self._add_bytes(tar, "utt_a.json", json.dumps({"utterance_id": "utt_a"}).encode("utf-8"))

    def _tensor_bytes(self, tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def _add_bytes(self, tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
