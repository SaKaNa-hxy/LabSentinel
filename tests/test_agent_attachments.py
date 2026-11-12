import base64
import io
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from app.backend import agent_service as agent, chat_store, history_service, llm_service
from app.backend.main import ChatRequest, chat, chat_detail


class AgentAttachmentTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        temp = stack.enter_context(TemporaryDirectory())
        root = Path(temp) / "history"
        stack.enter_context(patch.object(history_service, "HISTORY_DIR", root))
        stack.enter_context(patch.object(history_service, "RECORDS_FILE", root / "records.json"))
        stack.enter_context(patch.object(chat_store, "CHAT_DB", Path(temp) / "chats" / "test.sqlite3"))
        self.session = None
        self.first = self.save("image", "red", "lab_coat")
        self.second = self.save("video", "blue", "no_lab_coat", frame=3)
        self.third = self.save("camera", "green", "no_mask", frame=12)
        self.ids = [self.first["id"], self.second["id"]]
        self.provider = stack.enter_context(patch.object(agent, "generate_chat_completion", return_value={
            "content": "**图 2** 的截图需要复核。", "provider": "test", "model": "test",
        }))

    def save(self, source, color, category, frame=None):
        payload = io.BytesIO()
        Image.new("RGB", (1920, 1080), color).save(payload, "JPEG")
        return history_service.save_history(source, payload.getvalue(), {
            "detections": [{"class_name": category, "bbox": [50, 100, 500, 800]}],
            "risk": {
                "events": [],
                "pending_events": [{"label": "未穿实验服（未连续检出）"}] if source == "video" else [],
            },
        }, frame)

    def ask(self, message="比较两张图的实验服", **kwargs):
        result = chat(ChatRequest(message=message, session_id=self.session, **kwargs))
        self.session = result["session_id"]
        return result

    def test_only_selected_images_and_independent_record_facts_reach_vision(self):
        result = self.ask(history_record_ids=self.ids, analysis={"detections": [{"class_name": "do_not_use"}]})
        messages = self.provider.call_args.args[0]
        blocks = messages[-1]["content"]
        self.assertEqual(self.provider.call_args.kwargs, {"vision": True})
        self.assertEqual(result["evidence_mode"], "vision")
        self.assertNotIn("**", result["answer"])
        self.assertNotIn(self.third["id"], repr(messages))
        self.assertNotIn("do_not_use", repr(messages))
        self.assertEqual([block["type"] for block in blocks], ["text", "image_url", "text", "image_url", "text"])
        one, two = (json.loads(blocks[index]["text"]) for index in (0, 2))
        self.assertEqual([one["image_label"], two["image_label"]], ["图 1", "图 2"])
        self.assertEqual(one["detections"][0]["class_name"], "lab_coat")
        self.assertEqual(two["detections"][0]["class_name"], "no_lab_coat")
        self.assertEqual(two["frame_index"], 3)
        self.assertEqual(two["risk"]["events"], [])
        self.assertTrue(two["risk"]["pending_events"])
        for block, expected_channel in ((blocks[1], 0), (blocks[3], 2)):
            data = block["image_url"]["url"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                self.assertEqual(image.size, (1600, 900))
                pixel = image.getpixel((100, 100))
                self.assertGreater(pixel[expected_channel], 200)

    def test_followup_switch_and_current_analysis_use_the_right_session_context(self):
        self.ask("第一轮只问这两张图", history_record_ids=self.ids)
        self.ask("那图 2 呢？", history_record_ids=self.ids)
        messages = self.provider.call_args.args[0]
        self.assertEqual(messages[1]["content"], "第一轮只问这两张图")
        self.assertEqual(len(messages), 4)
        self.ask("交换图片顺序", history_record_ids=list(reversed(self.ids)))
        messages = self.provider.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(json.loads(messages[-1]["content"][0]["text"])["record_id"], self.second["id"])
        self.ask("当前画面有什么风险", analysis={"detections": [], "risk": {}}, analysis_id="current")
        self.assertEqual(self.provider.call_args.kwargs, {})
        self.assertEqual(len(self.provider.call_args.args[0]), 2)

    def test_text_followup_keeps_evidence_without_uploading_images_again(self):
        self.ask("比较图 1 和图 2", history_record_ids=self.ids)
        result = self.ask("那图 2 呢？")
        messages = self.provider.call_args.args[0]
        self.assertEqual(self.provider.call_args.kwargs, {})
        self.assertEqual(result["evidence_mode"], "record_only")
        self.assertTrue(result["has_analysis"])
        self.assertEqual(messages[1]["content"], "比较图 1 和图 2")
        self.assertIn(self.second["id"], messages[0]["content"])
        self.assertNotIn("data:image", repr(messages))
        detail = chat_detail(result["session_id"])
        self.assertEqual(detail["history_record_ids"], [])
        self.assertEqual(detail["turns"][0]["history_record_ids"], self.ids)
        self.provider.side_effect = RuntimeError("timeout")
        fallback = self.ask("再解释一下图 2")
        self.assertEqual(fallback["evidence_mode"], "record_only")
        self.assertIn("图 2", fallback["answer"])

    def test_missing_image_and_provider_timeout_are_explicit_record_only(self):
        self.provider.side_effect = RuntimeError("timeout")
        result = self.ask(history_record_ids=self.ids)
        self.assertEqual(result["evidence_mode"], "record_only")
        self.assertIn("仅根据检测记录回答", result["answer"])
        self.assertIn("图 2 第 3 帧", result["answer"])
        self.assertIn("待复核", result["answer"])
        self.provider.reset_mock()
        history_service.history_image_path(self.second["id"]).unlink()
        self.assertEqual(self.ask(history_record_ids=self.ids)["evidence_mode"], "record_only")
        self.provider.assert_not_called()

    def test_deleted_screenshot_remains_referenced_in_saved_transcript(self):
        result = self.ask(history_record_ids=self.ids)
        self.assertTrue(history_service.delete_history(self.second["id"]))
        detail = chat_detail(result["session_id"])
        self.assertEqual(detail["history_record_ids"], self.ids)
        self.assertEqual(detail["turns"][0]["history_record_ids"], self.ids)

    def test_deleted_invalid_duplicate_and_excess_records_never_call_provider(self):
        self.assertTrue(history_service.delete_history(self.second["id"]))
        with self.assertRaises(HTTPException) as missing:
            self.ask(history_record_ids=self.ids)
        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(missing.exception.detail["missing_record_ids"], [self.second["id"]])
        for ids in (["../annotated.jpg"], [self.ids[0]] * 2, [self.ids[0]] * 5):
            with self.assertRaises(HTTPException) as invalid:
                self.ask(history_record_ids=ids)
            self.assertEqual(invalid.exception.status_code, 400)
        self.provider.assert_not_called()


class VisionProviderTests(unittest.TestCase):
    def test_vision_uses_flash_and_text_model_override_still_works(self):
        config = {"base_url": llm_service.DEFAULT_BASE_URL, "model": "deepseek-v4-pro"}
        response = b'{"choices":[{"message":{"content":"ok"}}]}'
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}), \
             patch.object(llm_service, "_load_env"), \
             patch.object(llm_service, "llm_config", return_value=config), \
             patch.object(llm_service, "urlopen", side_effect=lambda *_a, **_kw: io.BytesIO(response)) as send:
            llm_service.generate_chat_completion([{"role": "user", "content": []}], vision=True)
            body = json.loads(send.call_args.args[0].data)
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            llm_service.generate_chat_completion([{"role": "user", "content": "hello"}])
            self.assertEqual(json.loads(send.call_args.args[0].data)["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
