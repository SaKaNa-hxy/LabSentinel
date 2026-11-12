import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.backend.main import ReportRequest, report


class SafetyReportTests(unittest.TestCase):
    def test_report_returns_clean_sections_and_plain_text(self):
        response = {
            "summary": "## **发现需要复核的防护用品风险。**",
            "findings": "画面中有`手套`缺失提示。",
            "actions": ["1. **对照画面核实手套佩戴情况。**", "- 确认缺失后补齐手套。"],
        }
        completion = {"provider": "DeepSeek", "model": "test", "content": f"```json\n{json.dumps(response)}\n```"}
        with patch("app.backend.llm_service.generate_chat_completion", return_value=completion) as generate:
            result = report(ReportRequest(detections=[{"class_name": "no_lab_coat", "frame": 3}], risk={}))

        self.assertEqual(result["sections"], {
            "summary": "发现需要复核的防护用品风险。",
            "findings": "画面中有手套缺失提示。",
            "actions": ["对照画面核实手套佩戴情况。", "确认缺失后补齐手套。"],
        })
        self.assertNotRegex(result["report"], r"[#*`]")
        self.assertIn("2. 确认缺失后补齐手套。", result["report"])
        messages = generate.call_args.args[0]
        self.assertIn("source=behavior_inference", messages[0]["content"])
        facts = json.loads(messages[1]["content"])
        self.assertEqual(facts["events"], [])
        self.assertEqual(facts["pending_events"][0]["evidence_frames"], [3])

    def test_unusable_model_output_returns_a_retry_message(self):
        for content in ('{"summary":', '{"summary":"结论", "findings":"发现", "actions":[null]}'):
            with self.subTest(content=content), patch(
                "app.backend.llm_service.generate_chat_completion", return_value={"content": content}
            ):
                with self.assertRaises(HTTPException) as error:
                    report(ReportRequest(detections=[], risk={}))
                self.assertEqual(error.exception.status_code, 502)
                self.assertIn("请重新生成", error.exception.detail)
