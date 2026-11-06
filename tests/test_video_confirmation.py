import unittest
import json
from copy import deepcopy

from app.backend import risk_engine
from app.backend.llm_service import _build_prompt


class VideoConfirmationTests(unittest.TestCase):
    def resolve(self, frames, selected=None):
        detections = [{"class_name": "no_lab_coat", "frame": frame, "confidence": 0.8} for frame in frames]
        scoped = detections if selected is None else [item for item in detections if item["frame"] in selected]
        raw = risk_engine.analyze_risk(scoped)
        before = deepcopy(raw)
        result = risk_engine.apply_video_coat_confirmation(raw, scoped, detections)
        self.assertEqual(raw, before)
        return result

    def test_isolated_frame_is_pending_and_raw_count_is_preserved(self):
        result = self.resolve([3])
        self.assertFalse(any(event["key"] == "no_lab_coat" for event in result["events"]))
        self.assertEqual(result["pending_events"][0]["evidence_frames"], [3])
        self.assertEqual(result["counts"]["no_lab_coat"], 1)
        self.assertEqual(result["missing_ppe"], [])
        self.assertIn("待复核", result["summary"])
        self.assertIn('"pending_events"', _build_prompt([], result))

    def test_two_consecutive_frames_confirm_but_gaps_or_duplicate_boxes_do_not(self):
        for frames, expected in (([2, 3], True), ([1, 3], False), ([3, 3], False), ([2, 3, 4], True)):
            with self.subTest(frames=frames):
                result = self.resolve(frames)
                self.assertEqual(any(event["key"] == "no_lab_coat" for event in result["events"]), expected)

    def test_single_frame_question_can_use_explicit_neighbor_confirmation(self):
        result = self.resolve([2, 3, 7, 8], selected=[2])
        evidence = result["temporal_confirmation"]["no_lab_coat"]
        self.assertEqual(evidence["confirmed_frames"], [2])
        self.assertEqual(evidence["supporting_frames"], [2, 3])
        self.assertEqual(result["counts"]["no_lab_coat"], 1)

    def test_scope_does_not_borrow_risk_from_other_frames(self):
        result = self.resolve([2, 3], selected=[1])
        self.assertEqual(result["events"], [])
        self.assertEqual(result.get("pending_events", []), [])

    def test_report_prompt_rechecks_old_unfiltered_video_summary(self):
        detections = [{"class_name": "no_lab_coat", "frame": 3}]
        old_risk = risk_engine.analyze_risk(detections)
        facts = json.loads(_build_prompt(detections, old_risk))
        self.assertFalse(any(event["key"] == "no_lab_coat" for event in facts["events"]))
        self.assertEqual(facts["pending_events"][0]["evidence_frames"], [3])
        self.assertEqual(old_risk["events"][0]["key"], "no_lab_coat")


if __name__ == "__main__":
    unittest.main()
