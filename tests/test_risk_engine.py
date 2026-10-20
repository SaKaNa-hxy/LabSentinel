import unittest

from app.backend.risk_engine import analyze_risk


class RiskEngineTests(unittest.TestCase):
    def test_eating_does_not_invent_no_mask_detection(self):
        result = analyze_risk([{"class_name": "eating", "confidence": 0.9}])
        self.assertEqual(result["counts"]["no_mask"], 0)
        self.assertEqual([event["key"] for event in result["events"]], ["eating"])
        self.assertEqual(result["missing_ppe"], [])

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
