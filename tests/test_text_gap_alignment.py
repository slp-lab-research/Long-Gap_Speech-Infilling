import unittest

from dataset.text_gap_alignment import split_text_for_gap


class TextGapAlignmentTest(unittest.TestCase):
    def test_uses_word_times_when_available(self):
        text = "hello brave new world"
        metadata = {
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.2},
                {"word": "brave", "start": 0.2, "end": 0.4},
                {"word": "new", "start": 0.4, "end": 0.6},
                {"word": "world", "start": 0.6, "end": 0.8},
            ]
        }

        prefix, middle, suffix, source = split_text_for_gap(
            text=text,
            interval=(10, 25),
            y_len=40,
            metadata=metadata,
            codec_frame_rate=50,
            mode="auto",
        )

        self.assertEqual((prefix, middle, suffix, source), ("hello", "brave new", "world", "aligned"))

    def test_auto_falls_back_to_proportional_without_alignment(self):
        prefix, middle, suffix, source = split_text_for_gap(
            text="hello brave new world",
            interval=(10, 25),
            y_len=40,
            metadata={},
            codec_frame_rate=50,
            mode="auto",
        )

        self.assertEqual((prefix.strip(), middle.strip(), suffix.strip(), source), ("hello", "brave new", "world", "proportional"))


if __name__ == "__main__":
    unittest.main()
