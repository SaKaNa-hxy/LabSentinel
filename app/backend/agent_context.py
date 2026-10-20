"""Conversation focus and evidence selection for an existing analysis."""

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .risk_engine import analyze_risk
except ImportError:
    from risk_engine import analyze_risk


CATEGORY_ALIASES = {
    "饮水": (("饮水", "喝水", "喝饮料", "drinking"), ("drinking",)),
    "进食": (("进食", "吃饭", "吃东西", "eating"), ("eating",)),
    "护目镜": (("护目镜", "眼镜", "goggles"), ("goggles", "no_goggles")),
    "手套": (("手套", "gloves"), ("gloves", "no_gloves")),
    "头套": (("头套", "头罩", "head_mask"), ("head_mask", "no_head_mask")),
    "实验服": (("实验服", "实验衣", "白大褂", "lab_coat"), ("lab_coat", "no_lab_coat")),
    "口罩": (("口罩", "no_mask"), ("mask", "no_mask")),
}
NUMBER = r"[0-9零〇一二两三四五六七八九十百]+"
FRAME_PATTERN = re.compile(rf"第\s*({NUMBER})\s*(?:个)?(?:关键)?帧")
RANGE_PATTERN = re.compile(rf"第?\s*({NUMBER})\s*(?:帧)?\s*[到至~—–\-]\s*第?\s*({NUMBER})\s*帧")


def _number(text: str) -> int:
    if text.isdigit():
        return int(text)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    value, digit = 0, 0
    for char in text:
        if char in digits:
            digit = digits[char]
        elif char in ("十", "百"):
            value += (digit or 1) * (10 if char == "十" else 100)
            digit = 0
    return value + digit


def requested_frames(message: str) -> list[int]:
    match = RANGE_PATTERN.search(message)
    if match:
        start, end = (_number(part) for part in match.groups())
        # Oversized/reversed requests stay invalid instead of silently changing
        # the requested range or allocating an unbounded list.
        if start > end or end - start > 100:
            return [0]
        return list(range(start, end + 1))
    return list(dict.fromkeys(_number(match) for match in FRAME_PATTERN.findall(message)))


def requested_categories(message: str) -> list[str]:
    text = message.lower()
    return [label for label, (aliases, _) in CATEGORY_ALIASES.items() if any(alias in text for alias in aliases)]


def detection_frame(item: dict[str, Any]) -> Optional[int]:
    value = item.get("frame", item.get("frame_index"))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def facts_identity(facts: dict[str, Any], analysis_id: Optional[str], record_id: Optional[str]) -> str:
    if analysis_id:
        return "analysis:" + analysis_id
    if record_id:
        return "record:" + record_id
    encoded = json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "snapshot:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ConversationContext:
    identity: Optional[str] = None
    facts: dict[str, Any] = field(default_factory=dict)
    frames: list[int] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    last_tool: str = "general_chat"

    def reset_focus(self) -> None:
        self.frames = []
        self.categories = []
        self.last_tool = "general_chat"

    def bind(self, facts: dict[str, Any], identity: str) -> bool:
        available = facts.get("source") != "暂无当前分析结果"
        was_available = bool(self.facts) and self.facts.get("source") != "暂无当前分析结果"
        changed = self.identity != identity or available != was_available
        if changed:
            self.reset_focus()
            if facts.get("source") != "暂无当前分析结果":
                self.last_tool = "get_current_analysis"
        self.identity = identity
        self.facts = deepcopy(facts)
        return changed

    def select(self, message: str) -> dict[str, Any]:
        explicit_frames = requested_frames(message)
        explicit_categories = requested_categories(message)
        frames = explicit_frames or list(self.frames)
        if "上一帧" in message or "下一帧" in message:
            if len(self.frames) != 1:
                return {**deepcopy(self.facts), "scope_error": "请先指定一个帧号，我才能定位它的上一帧或下一帧。"}
            frames = [self.frames[0] + (-1 if "上一帧" in message else 1)]
        categories = explicit_categories or list(self.categories)
        if not explicit_frames and any(word in message for word in ("整个视频", "全部帧", "所有帧", "整体", "全部结果")):
            frames = []
        if not explicit_categories and any(word in message for word in ("全部风险", "所有风险", "其他风险", "所有类别", "全部结果", "有什么问题", "哪些问题", "有哪些风险", "整体", "整帧", "其他问题")):
            categories = []

        facts = deepcopy(self.facts)
        detections = facts.get("detections") or []
        available = sorted({value for item in detections if (value := detection_frame(item)) is not None})
        frame_count = facts.get("frame_count")
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or not 1 <= frame_count <= 1000:
            frame_count = None
        error = None
        if frames:
            if any(frame < 1 for frame in frames):
                error = "帧号或帧范围无效，请指定有效的关键帧编号。"
            elif frame_count and any(frame > frame_count for frame in frames):
                error = f"这次视频共分析了{frame_count}个关键帧，没有你指定的帧。请在第1到第{frame_count}帧中选择。"
            elif not frame_count and not available:
                error = "当前结果没有可用的帧号信息，无法定位到你指定的帧。请先完成一次视频分析。"
            elif not frame_count and any(frame not in available for frame in frames):
                error = "当前记录没有你指定帧的证据，请提供对应的视频分析结果或历史记录。"
            elif detections and not available:
                error = "当前检测结果缺少逐帧标记，无法可靠筛选这一帧的证据。请重新分析视频。"
        if (frames or categories) and not detections and (facts.get("risk") or {}).get("events"):
            error = error or "当前只有风险摘要，没有可用于定位帧和类别的详细检测证据。请提供对应的分析结果。"
        if error:
            facts["scope_error"] = error
            return facts

        self.frames, self.categories = frames, categories
        classes = {key for label in categories for key in CATEGORY_ALIASES[label][1]}
        frame_selected = [item for item in detections if not frames or detection_frame(item) in frames]
        selected = [item for item in frame_selected if not classes or item.get("class_name") in classes]
        if frames or categories:
            facts["detections"] = selected
            facts["risk"] = analyze_risk(selected)
            if categories:
                labels = "、".join(categories)
                events = facts["risk"]["events"]
                event_note = "；相关风险提示：" + "、".join(event["label"] for event in events) if events else "；未检出对应风险类"
                facts["risk"]["summary"] = f"当前筛选的是{labels}相关类别，共有{len(selected)}个所选类别检测框{event_note}。这只描述所选类别，不能据此说明整帧没有其他目标或风险。"
        scope = {
            "frames": frames,
            "categories": categories,
            "frame_count": frame_count,
            "source": facts.get("source"),
            "record_id": facts.get("record_id"),
            "category_filter_applied": bool(categories),
            "box_count_before_category_filter": len(frame_selected),
        }
        facts["discussion_focus"] = scope
        # Computed directly from scoped evidence, not from the model's answer
        # or video risk.counts (which the frontend aggregates using a maximum).
        class_counts = Counter(item.get("class_name", "unknown") for item in selected)
        per_frame: dict[str, dict[str, int]] = {}
        summary_frames = frames or (list(range(1, frame_count + 1)) if frame_count else available)
        for frame in summary_frames:
            per_frame[str(frame)] = dict(Counter(item.get("class_name", "unknown") for item in selected if detection_frame(item) == frame))
        facts["evidence_summary"] = {
            "box_counts": dict(class_counts),
            "per_frame_box_counts": per_frame,
            "count_unit": "检测框；不是人数或独立事件数",
        }
        return facts

    def public_focus(self) -> dict[str, Any]:
        return {"frames": list(self.frames), "categories": list(self.categories),
                "source": self.facts.get("source"), "record_id": self.facts.get("record_id")}
