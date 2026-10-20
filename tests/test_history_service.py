import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.backend import history_service


class HistoryServiceTests(unittest.TestCase):
    def test_delete_all_removes_every_source_and_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(history_service, "HISTORY_DIR", directory), patch.object(history_service, "RECORDS_FILE", directory / "records.json"):
                image = io.BytesIO()
                Image.new("RGB", (20, 20), "red").save(image, format="JPEG")
                records = [history_service.save_history(source, image.getvalue(), {}) for source in ("image", "video", "camera")]
                result = history_service.delete_all_history()
                self.assertEqual(result, {"deleted_count": 3, "failed_count": 0})
                self.assertEqual(history_service.list_history(), [])
                self.assertTrue(all(not (directory / record["id"]).exists() for record in records))
                self.assertEqual(history_service.delete_all_history(), {"deleted_count": 0, "failed_count": 0})

    def test_delete_all_preserves_failed_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(history_service, "HISTORY_DIR", directory), patch.object(history_service, "RECORDS_FILE", directory / "records.json"):
                image = io.BytesIO()
                Image.new("RGB", (20, 20), "red").save(image, format="JPEG")
                record = history_service.save_history("image", image.getvalue(), {})
                with patch.object(history_service, "_delete_record_files", side_effect=PermissionError("file in use")):
                    result = history_service.delete_all_history()
                self.assertEqual(result, {"deleted_count": 0, "failed_count": 1})
                self.assertIsNotNone(history_service.get_history(record["id"]))
                self.assertTrue((directory / record["id"] / "annotated.jpg").exists())

    def test_delete_all_does_not_follow_paths_outside_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "history"
            directory.mkdir()
            outside = Path(temporary) / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with patch.object(history_service, "HISTORY_DIR", directory), patch.object(history_service, "RECORDS_FILE", directory / "records.json"):
                history_service._write_records([{"id": "../outside"}])
                self.assertEqual(history_service.delete_all_history(), {"deleted_count": 0, "failed_count": 1})
                self.assertTrue(sentinel.exists())

    def test_save_get_and_delete_record(self):
        old_dir = history_service.HISTORY_DIR
        old_file = history_service.RECORDS_FILE
        try:
            with tempfile.TemporaryDirectory() as temporary:
                history_service.HISTORY_DIR = Path(temporary)
                history_service.RECORDS_FILE = Path(temporary) / "records.json"
                image = io.BytesIO()
                Image.new("RGB", (20, 20), "red").save(image, format="JPEG")
                record = history_service.save_history("image", image.getvalue(), {"risk": {"events": []}, "detections": []})
                self.assertEqual(history_service.get_history(record["id"])["source"], "image")
                self.assertEqual(len(history_service.list_history()), 1)
                self.assertTrue((history_service.HISTORY_DIR / record["id"] / "annotated.jpg").is_file())
                self.assertTrue(history_service.delete_history(record["id"]))
                self.assertEqual(history_service.list_history(), [])
        finally:
            history_service.HISTORY_DIR = old_dir
            history_service.RECORDS_FILE = old_file


if __name__ == "__main__":
    unittest.main()
