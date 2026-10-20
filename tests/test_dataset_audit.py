import unittest
from pathlib import Path

from scripts.dataset_audit import source_key


class DatasetAuditTests(unittest.TestCase):
    def test_numbered_images_with_same_prefix_share_group(self):
        first = Path("C1200_MP4-0001_jpg.rf.aaaaaaaa.jpg")
        second = Path("C1200_MP4-0421_jpg.rf.bbbbbbbb.jpg")
        self.assertEqual(source_key(first), source_key(second))

    def test_augmented_still_images_share_group(self):
        first = Path("DSC01746_JPG.rf.aaaaaaaa.jpg")
        second = Path("DSC01746_JPG.rf.bbbbbbbb.jpg")
        self.assertEqual(source_key(first), source_key(second))


if __name__ == "__main__":
    unittest.main()
