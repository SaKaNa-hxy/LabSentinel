import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from app.backend import agent_service as agent
from app.backend.risk_engine import analyze_risk


class AgentContextTests(unittest.TestCase):
    def setUp(self):
        self.session = agent.create_session()
        self.addCleanup(agent.clear_session, self.session)
        self.calls = []

        def complete(messages):
            self.calls.append(deepcopy(messages))
            return {"content": "我已查看对应的检测证据。", "provider": "test", "model": "test"}

        self.completion = patch.object(agent, "generate_chat_completion", side_effect=complete).start()
        self.addCleanup(patch.stopall)
        patch.object(agent, "list_history", return_value=[]).start()
        self.history = patch.object(agent, "get_history", return_value=None).start()
        detections = [
            {"class_name": "no_goggles", "confidence": 0.4, "frame": 1},
            {"class_name": "no_gloves", "confidence": 0.8, "frame": 1},
            {"class_name": "goggles", "confidence": 0.9, "frame": 2},
            {"class_name": "no_gloves", "confidence": 0.7, "frame": 2},
            {"class_name": "no_gloves", "confidence": 0.6, "frame": 3},
            {"class_name": "no_goggles", "confidence": 0.41, "frame": 5},
        ]
        self.analysis = {"detections": detections, "risk": analyze_risk(detections), "frame_count": 5, "source_type": "video"}

    def ask(self, message, **kwargs):
        return agent.chat(message, session_id=self.session, **kwargs)

    def payload(self):
        return json.loads(self.calls[-1][-1]["content"])["tool_data"]["result"]

    def begin(self, message="第一帧护目镜有什么问题？"):
        return self.ask(message, analysis=self.analysis, analysis_id="video-a")

    def test_frame_risk_explanation_and_action_share_the_same_evidence(self):
        self.begin()
        for question, tool in (("这个判断靠谱吗？", "explain_risk"), ("那怎么处理？", "generate_rectification_plan")):
            reply = self.ask(question)
            self.assertEqual(reply["tool"], tool)
            self.assertEqual(reply["discussion_focus"]["frames"], [1])
            self.assertEqual(reply["discussion_focus"]["categories"], ["护目镜"])
            self.assertEqual([(d["frame"], d["class_name"]) for d in self.payload()["evidence"]], [(1, "no_goggles")])

    def test_switching_frames_does_not_borrow_a_risk_from_other_frames(self):
        self.begin()
        reply = self.ask("那第三帧呢？")
        self.assertEqual(reply["discussion_focus"]["frames"], [3])
        self.assertEqual(reply["discussion_focus"]["categories"], ["护目镜"])
        self.assertEqual(self.payload()["detections"], [])
        self.assertEqual(self.payload()["risk"]["events"], [])
        self.assertEqual(self.payload()["discussion_focus"]["box_count_before_category_filter"], 1)
        self.assertIn("只描述所选类别", self.payload()["risk"]["summary"])
        self.ask("这个判断靠谱吗？")
        self.assertEqual(self.payload()["events"], [])

    def test_positive_equipment_does_not_turn_into_a_missing_equipment_event(self):
        self.begin("第二帧护目镜呢？")
        self.assertEqual([d["class_name"] for d in self.payload()["detections"]], ["goggles"])
        self.assertEqual(self.payload()["risk"]["events"], [])

    def test_empty_frame_is_valid_when_total_frame_count_is_known(self):
        self.begin("第四帧有什么问题？")
        self.assertEqual(self.payload()["detections"], [])
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [4])

    def test_invalid_frame_does_not_call_llm_or_replace_valid_focus(self):
        self.begin()
        call_count = self.completion.call_count
        reply = self.ask("那第六帧呢？")
        self.assertEqual(reply["tool"], "clarify_context")
        self.assertIn("共分析了5", reply["answer"])
        self.assertEqual(self.completion.call_count, call_count)
        self.assertEqual(reply["discussion_focus"]["frames"], [1])

    def test_chinese_frame_ranges_and_relative_frames(self):
        self.begin("第十一到第十三帧有什么问题？")
        self.assertEqual(self.completion.call_count, 0)
        self.begin("第1帧到第3帧的手套呢？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [1, 2, 3])
        self.assertEqual(self.payload()["evidence_summary"]["box_counts"], {"no_gloves": 3})
        self.ask("第二帧手套呢？")
        self.ask("那下一帧呢？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [3])

    def test_whole_video_request_resets_frame_and_category_focus(self):
        self.begin()
        self.ask("整个视频有哪些风险？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [])
        self.assertEqual(self.payload()["discussion_focus"]["categories"], [])
        self.assertEqual(len(self.payload()["detections"]), 6)

    def test_entire_specific_frame_expands_category_without_changing_frame(self):
        self.begin("第三帧护目镜呢？")
        self.ask("那第三帧整体有什么问题？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [3])
        self.assertEqual(self.payload()["discussion_focus"]["categories"], [])
        self.assertEqual([d["class_name"] for d in self.payload()["detections"]], ["no_gloves"])

    def test_new_analysis_resets_focus_even_for_identical_detections(self):
        self.begin()
        reply = self.ask("当前结果怎么样？", analysis=self.analysis, analysis_id="video-b")
        self.assertEqual(reply["discussion_focus"]["frames"], [])
        self.assertEqual(reply["discussion_focus"]["categories"], [])
        self.assertEqual(len(self.calls[-1]), 2)

    def test_new_file_without_results_cannot_reuse_previous_video(self):
        self.begin()
        calls = self.completion.call_count
        reply = self.ask("这个判断靠谱吗？", analysis_id="new-upload")
        self.assertEqual(reply["tool"], "clarify_context")
        self.assertIn("还没有关联", reply["answer"])
        self.assertEqual(self.completion.call_count, calls)

    def test_refreshing_same_camera_session_keeps_category_and_updates_facts(self):
        self.begin("护目镜呢？")
        updated = {"detections": [], "risk": analyze_risk([]), "source_type": "camera"}
        self.ask("这个判断靠谱吗？", analysis=updated, analysis_id="video-a")
        self.assertEqual(self.payload()["discussion_focus"]["categories"], ["护目镜"])
        self.assertEqual(self.payload()["evidence"], [])

    def test_recent_frames_are_not_routed_to_history(self):
        reply = self.begin("最近这几帧有什么风险？")
        self.assertEqual(reply["tool"], "get_current_analysis")

    def test_history_followup_does_not_silently_use_current_video(self):
        self.begin()
        self.ask("最近的历史记录呢？")
        calls = self.completion.call_count
        reply = self.ask("那怎么处理？")
        self.assertEqual(reply["tool"], "clarify_context")
        self.assertEqual(self.completion.call_count, calls)
        self.assertIn("历史记录", reply["answer"])

    def test_thanks_preserves_focus_but_unrelated_conversation_does_not(self):
        self.begin()
        self.ask("谢谢")
        self.ask("这个判断靠谱吗？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [1])
        self.ask("换个话题，给我讲一个故事")
        reply = self.ask("这个结尾有什么依据？")
        self.assertEqual(reply["tool"], "general_chat")

    def test_clearing_session_removes_evidence_and_focus(self):
        self.begin()
        agent.clear_session(self.session)
        reply = self.ask("这个判断靠谱吗？")
        self.assertEqual(reply["tool"], "clarify_context")
        self.assertEqual(reply["discussion_focus"]["frames"], [])

    def test_history_frame_metadata_and_missing_record(self):
        self.history.return_value = {"frame_index": 3, "source": "video", "risk": analyze_risk(self.analysis["detections"][:1]), "detections": self.analysis["detections"][:1]}
        self.ask("第三帧护目镜呢？", record_id="record-a")
        self.assertEqual(self.payload()["detections"][0]["frame"], 3)
        self.history.return_value = None
        reply = self.ask("这个判断靠谱吗？", record_id="record-a")
        self.assertEqual(reply["tool"], "clarify_context")

    def test_scope_selection_does_not_mutate_source_analysis(self):
        original = deepcopy(self.analysis)
        self.begin()
        self.ask("那第三帧呢？")
        self.assertEqual(self.analysis, original)

    def test_focus_survives_message_window_and_sessions_are_isolated(self):
        self.begin()
        for _ in range(8):
            self.ask("谢谢")
        self.assertEqual(len(agent._messages(self.session)), agent.MAX_MESSAGES)
        self.ask("这个判断靠谱吗？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [1])
        other = agent.create_session()
        self.addCleanup(agent.clear_session, other)
        reply = agent.chat("第三帧手套呢？", session_id=other, analysis=self.analysis, analysis_id="video-a")
        self.assertEqual(reply["discussion_focus"]["categories"], ["手套"])
        self.ask("那怎么处理？")
        self.assertEqual(self.payload()["discussion_focus"]["frames"], [1])
        self.assertEqual(self.payload()["discussion_focus"]["categories"], ["护目镜"])

    def test_sparse_summary_cannot_be_used_as_frame_evidence(self):
        summary = {"detections": [], "risk": self.analysis["risk"], "frame_count": 5}
        reply = self.ask("第三帧护目镜呢？", analysis=summary, analysis_id="summary-only")
        self.assertEqual(reply["tool"], "clarify_context")
        self.assertEqual(self.completion.call_count, 0)

    def test_unavailable_llm_still_answers_with_the_selected_focus(self):
        self.completion.side_effect = RuntimeError("offline")
        self.begin()
        reply = self.ask("这个判断靠谱吗？")
        self.assertEqual(reply["provider"], "local-fallback")
        self.assertIn("第1帧的护目镜", reply["answer"])
        self.assertNotIn("未佩戴手套", reply["answer"])

    def test_api_forwards_analysis_identity_and_frame_metadata(self):
        from app.backend.main import ChatRequest, chat
        reply = chat(ChatRequest(message="第三帧手套呢？", session_id=self.session, analysis=self.analysis, analysis_id="api-video"))
        self.assertEqual(reply["discussion_focus"]["frames"], [3])
        self.assertEqual(self.payload()["discussion_focus"]["frame_count"], 5)
        self.assertEqual(self.payload()["detections"][0]["frame"], 3)


if __name__ == "__main__":
    unittest.main()
