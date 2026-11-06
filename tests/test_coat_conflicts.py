import io
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.backend import inference


def detection(name, box, confidence=0.9, **extra):
    return {"class_name": name, "bbox": box, "confidence": confidence, **extra}


class CoatConflictTests(unittest.TestCase):
    def test_same_location_prefers_coat_even_if_negative_confidence_is_higher(self):
        items = [detection("no_lab_coat", [104.66, 175.46, 369, 376], 0.95),
                 detection("lab_coat", [88.83, 176.83, 368.45, 376], 0.2),
                 detection("no_mask", [200, 100, 250, 140])]
        before = deepcopy(items)
        kept, suppressed = inference.resolve_coat_conflicts(items)
        self.assertEqual(kept, [1, 2])
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["detection"], items[0])
        self.assertGreater(suppressed[0]["overlap_iou"], 0.93)
        self.assertEqual(items, before)

    def test_other_person_and_weak_overlap_remain(self):
        items = [detection("lab_coat", [0, 0, 100, 100]),
                 detection("no_lab_coat", [150, 0, 250, 100]),
                 detection("no_lab_coat", [50, 0, 150, 100])]
        self.assertEqual(inference.resolve_coat_conflicts(items), ([0, 1, 2], []))

    def test_different_frames_cannot_suppress_each_other(self):
        items = [detection("lab_coat", [0, 0, 100, 100], frame=1),
                 detection("no_lab_coat", [0, 0, 100, 100], frame=2)]
        self.assertEqual(inference.resolve_coat_conflicts(items), ([0, 1], []))

    def test_no_coat_without_positive_evidence_remains(self):
        items = [detection("no_lab_coat", [0, 0, 100, 100])]
        self.assertEqual(inference.resolve_coat_conflicts(items), ([0], []))

    def test_invalid_geometry_is_not_suppressed(self):
        for box in (None, [0, 0, 0, 100], [float("nan"), 0, 100, 100], [0, 0, 100]):
            with self.subTest(box=box):
                items = [detection("lab_coat", [0, 0, 100, 100]), detection("no_lab_coat", box)]
                self.assertEqual(inference.resolve_coat_conflicts(items), ([0, 1], []))

    def test_detection_list_plot_and_risk_all_use_resolved_boxes(self):
        class Boxes:
            def __init__(self, rows):
                self.rows = rows
                self.cls = np.array([row[0] for row in rows])
                self.conf = np.array([row[1] for row in rows])
                self.xyxy = np.array([row[2:] for row in rows])

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, indices):
                return Boxes([self.rows[index] for index in indices])

        class Result:
            names = {5: "Lab Coat", 9: "No Lab coat"}

            def __init__(self):
                self.boxes = Boxes([[9, 0.95, 1, 1, 19, 19], [5, 0.8, 1, 1, 19, 19]])
                self.plotted_classes = None

            def plot(self):
                self.plotted_classes = self.boxes.cls.tolist()
                return np.zeros((20, 20, 3), dtype=np.uint8)

        result = Result()
        class Model:
            def predict(self, **kwargs):
                return [result]

        image = io.BytesIO()
        Image.new("RGB", (20, 20)).save(image, format="PNG")
        with patch.object(inference, "_model", Model()), patch.object(inference, "_custom_model_compatible", True):
            output = inference.predict(image.getvalue())
        self.assertEqual([item["class_name"] for item in output["detections"]], ["lab_coat"])
        self.assertEqual(result.plotted_classes, [5])
        self.assertEqual(output["risk"]["counts"]["no_lab_coat"], 0)
        self.assertFalse(any(event["key"] == "no_lab_coat" for event in output["risk"]["events"]))
        self.assertEqual(output["postprocessing"]["raw_detection_count"], 2)
        self.assertEqual(len(output["postprocessing"]["suppressed_detections"]), 1)


if __name__ == "__main__":
    unittest.main()
