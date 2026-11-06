import unittest

from app.backend.risk_engine import analyze_risk


class RiskEngineTests(unittest.TestCase):
    def test_behavior_infers_no_mask_without_inventing_detection(self):
        for behavior in ("eating", "drinking"):
            with self.subTest(behavior=behavior):
                detections = [{"class_name": behavior, "confidence": 0.9}]
                result = analyze_risk(detections)
                self.assertEqual(result["counts"]["no_mask"], 0)
                self.assertNotIn("no_mask", result["confidence_by_class"])
                self.assertEqual([event["key"] for event in result["events"]], [behavior, "no_mask"])
                inferred = result["events"][-1]
                self.assertEqual(inferred["source"], "behavior_inference")
                self.assertEqual(inferred["evidence_classes"], [behavior])
                self.assertIn("行为推断", inferred["label"])
                self.assertEqual(len(detections), 1)

    def test_behaviors_and_positive_mask_still_generate_only_one_inferred_event(self):
        result = analyze_risk([{"class_name": name} for name in ("eating", "drinking", "mask")])
        events = [event for event in result["events"] if event["key"] == "no_mask"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "behavior_inference")
        self.assertEqual(events[0]["evidence_classes"], ["eating", "drinking"])
        self.assertEqual(result["counts"]["no_mask"], 0)

    def test_direct_no_mask_detection_takes_precedence_without_duplicate(self):
        result = analyze_risk([{"class_name": name} for name in ("drinking", "no_mask", "no_mask")])
        events = [event for event in result["events"] if event["key"] == "no_mask"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "visual_detection")
        self.assertEqual(result["counts"]["no_mask"], 2)

    def test_no_behavior_does_not_infer_no_mask(self):
        for names in ([], ["mask"], ["lab_coat"], ["no_gloves"]):
            with self.subTest(names=names):
                result = analyze_risk([{"class_name": name} for name in names])
                self.assertFalse(any(event["key"] == "no_mask" for event in result["events"]))

    def test_no_ppe_event_is_grounded_in_visual_detection(self):
        result = analyze_risk([{"class_name": "no_gloves", "confidence": 0.8}])
        self.assertEqual(result["events"][0]["source"], "visual_detection")
        self.assertEqual(result["missing_ppe"][0]["source"], "visual_detection")
        self.assertEqual(result["person_association"], "not_available")

    def test_ppe_classes_do_not_prove_each_person_is_fully_protected(self):
        names = ("gloves", "goggles", "head_mask", "lab_coat", "mask")
        result = analyze_risk([{"class_name": name, "confidence": 0.9} for name in names])
        self.assertEqual(result["title"], "检测到多类防护用品")
        self.assertIn("不能据此确认每个人", result["summary"])


if __name__ == "__main__":
    unittest.main()
