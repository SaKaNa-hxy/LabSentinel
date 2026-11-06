import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.dataset_audit import source_key, write_grouped_split


class DatasetAuditTests(unittest.TestCase):
    def test_numbered_images_with_same_prefix_share_group(self):
        first = Path("C1200_MP4-0001_jpg.rf.aaaaaaaa.jpg")
        second = Path("C1200_MP4-0421_jpg.rf.bbbbbbbb.jpg")
        self.assertEqual(source_key(first), source_key(second))

    def test_augmented_still_images_share_group(self):
        first = Path("DSC01746_JPG.rf.aaaaaaaa.jpg")
        second = Path("DSC01746_JPG.rf.bbbbbbbb.jpg")
        self.assertEqual(source_key(first), source_key(second))

    def test_frozen_split_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            frozen = output / "train.txt"
            frozen.write_text("original-membership\n", encoding="utf-8")
            with patch("scripts.dataset_audit.OUTPUT", output):
                with self.assertRaises(FileExistsError):
                    write_grouped_split([], {}, 42)
            self.assertEqual(frozen.read_text(encoding="utf-8"), "original-membership\n")


if __name__ == "__main__":
    unittest.main()
