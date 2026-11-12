import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from app.backend import agent_service, chat_store
from app.backend.main import ChatRequest, chat, chat_delete, chat_detail, chat_list
from app.backend.risk_engine import analyze_risk


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = patch.object(chat_store, "CHAT_DB", Path(directory.name) / "chats.sqlite3")
        database.start()
        self.addCleanup(database.stop)
        provider = patch.object(agent_service, "generate_chat_completion", return_value={
            "content": "已核对对应检测依据。", "provider": "test", "model": "test",
        })
        self.provider = provider.start()
        self.addCleanup(provider.stop)

    def ask(self, message, **kwargs):
        return chat(ChatRequest(message=message, **kwargs))

    def test_saved_chats_keep_independent_context_and_full_transcript(self):
        detections = [{"class_name": "no_goggles", "confidence": .8, "frame": 1}]
        analysis = {"detections": detections, "risk": analyze_risk(detections), "frame_count": 1}
        first = self.ask("第一帧护目镜有什么问题？", analysis=analysis, analysis_id="video-a")
        second = self.ask("你好")
        for _ in range(7):
            self.ask("谢谢", session_id=first["session_id"])
        restored = self.ask("这个判断靠谱吗？", session_id=first["session_id"])
        self.assertEqual(restored["discussion_focus"]["frames"], [1])
        prompt = self.provider.call_args.args[0]
        self.assertEqual(len(prompt), agent_service.MAX_MESSAGES + 2)
        self.assertEqual(json.loads(prompt[-1]["content"])["tool_data"]["result"]["evidence"], detections)

        first_detail = chat_detail(first["session_id"])
        self.assertEqual(len(first_detail["turns"]), 9)
        self.assertEqual(first_detail["title"], "第一帧护目镜有什么问题？")
        self.assertTrue(first_detail["has_analysis"])
        self.assertFalse(chat_detail(second["session_id"])["has_analysis"])
        self.assertEqual([item["id"] for item in chat_list()["items"]], [first["session_id"], second["session_id"]])
        chat_delete(second["session_id"])
        self.assertEqual([item["id"] for item in chat_list()["items"]], [first["session_id"]])
        self.assertEqual(len(chat_detail(first["session_id"])["turns"]), 9)

    def test_retry_after_lost_response_keeps_one_turn_and_one_chat(self):
        first = self.ask("你好", request_id="first-request")
        retry = self.ask("你好", request_id="first-request")
        self.assertEqual(first, retry)
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(len(chat_list()["items"]), 1)
        self.assertEqual(len(chat_detail(first["session_id"])["turns"]), 1)
        with self.assertRaises(HTTPException) as mismatch:
            self.ask("换一个问题", request_id="first-request")
        self.assertEqual(mismatch.exception.status_code, 400)

    def test_in_flight_reply_cannot_recreate_deleted_chat(self):
        chat_id = self.ask("你好")["session_id"]

        def delete_while_answering(*args, **kwargs):
            chat_delete(chat_id)
            return {"content": "结束。", "provider": "test", "model": "test"}

        self.provider.side_effect = delete_while_answering
        with self.assertRaises(HTTPException) as missing:
            self.ask("继续说", session_id=chat_id)
        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(chat_list()["items"], [])

    def test_failed_request_does_not_create_partial_chat(self):
        with patch.object(chat_store, "answer_turn", side_effect=ValueError("输入错误")):
            with self.assertRaises(HTTPException):
                self.ask("你好", request_id="retry-later")
        self.assertEqual(chat_list()["items"], [])
        reply = self.ask("你好", request_id="retry-later")
        self.assertEqual(len(chat_detail(reply["session_id"])["turns"]), 1)


if __name__ == "__main__":
    unittest.main()
