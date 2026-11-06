from copy import deepcopy
from typing import Any

CLASS_NAMES = (
    "drinking",
    "eating",
    "gloves",
    "goggles",
    "head_mask",
    "lab_coat",
    "mask",
    "no_gloves",
    "no_head_mask",
    "no_lab_coat",
    "no_mask",
    "no_goggles",
)

POSITIVE_PPE_CLASSES = (
    "gloves",
    "goggles",
    "head_mask",
    "lab_coat",
    "mask",
)

BEHAVIOR_LABELS = {
    "eating": "实验室内进食",
    "drinking": "实验室内饮水",
}

NEGATIVE_PPE_LABELS = {
    "no_mask": "检测到未佩戴口罩目标",
    "no_gloves": "检测到未佩戴手套目标",
    "no_goggles": "检测到未佩戴护目镜目标",
    "no_head_mask": "检测到未佩戴头套目标",
    "no_lab_coat": "检测到未穿实验服目标",
}


def _class_count(counts: dict[str, int], name: str) -> int:
    return counts.get(name, 0)


def _missing_ppe(counts: dict[str, int]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []

    for class_name, label in NEGATIVE_PPE_LABELS.items():
        count = _class_count(counts, class_name)
        if count:
            missing.append({
                "class_name": class_name,
                "label": label,
                "count": count,
                "missing": [class_name.replace("no_", "")],
                "source": "visual_detection",
            })

    behavior_evidence = [name for name in BEHAVIOR_LABELS if counts.get(name)]
    if behavior_evidence and not counts.get("no_mask"):
        missing.append({
            "class_name": "no_mask",
            "label": "未戴口罩（行为推断）",
            "count": 0,
            "missing": ["mask"],
            "source": "behavior_inference",
            "evidence_classes": behavior_evidence,
            "evidence_count": sum(counts[name] for name in behavior_evidence),
        })

    return missing


def analyze_risk(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in CLASS_NAMES}
    confidence_by_class: dict[str, list[float]] = {}

    for item in detections:
        name = item.get("class_name")
        if name in counts:
            counts[name] += 1
            confidence_by_class.setdefault(name, []).append(item.get("confidence", 0))

    eating_count = counts["eating"]
    drinking_count = counts["drinking"]
    ppe_count = sum(counts[name] for name in POSITIVE_PPE_CLASSES)
    missing_ppe = _missing_ppe(counts)
    events: list[dict[str, Any]] = []

    for key, label in BEHAVIOR_LABELS.items():
        if counts[key]:
            events.append({"key": key, "label": label, "source": "visual_detection"})
    for item in missing_ppe:
        event = {"key": item["class_name"], "label": item["label"], "source": item["source"]}
        if item.get("evidence_classes"):
            event["evidence_classes"] = item["evidence_classes"]
        events.append(event)

    if eating_count or drinking_count:
        behavior_labels = "、".join(BEHAVIOR_LABELS[key] for key in ("eating", "drinking") if counts[key])
        ppe_note = "防护风险提示：" + "、".join(item["label"] for item in missing_ppe) + "。"
        rule_note = "系统按饮食行为规则判定未戴口罩；该项为行为推断，没有新增口罩检测框，也未关联到具体人员。" if any(item["source"] == "behavior_inference" for item in missing_ppe) else ""
        title = "实验室行为风险"
        summary = f"检测到{behavior_labels}，请现场复核。{ppe_note}{rule_note}"
        recommendations = ["现场确认行为后，提醒相关人员停止在实验操作区域饮食", "复核口罩等防护装备的实际穿戴状态", "将食品和饮品移出实验台及样品、试剂存放区域", "结合实验室管理制度记录并复核该行为"]
    elif missing_ppe:
        labels = "、".join(item["label"] for item in missing_ppe)
        title = "防护装备未完整"
        summary = f"PPE 模型输出：{labels}。当前尚未关联到具体人员，需现场复核。"
        recommendations = ["现场确认目标与人员的对应关系后补齐防护装备", "按实验室实际要求确认所需 PPE", "复核遮挡、漏检或误检情况"]
    elif all(_class_count(counts, name) > 0 for name in POSITIVE_PPE_CLASSES):
        title = "检测到多类防护用品"
        summary = "画面中检测到实验服、口罩、手套、护目镜和头套，但不能据此确认每个人均穿戴完整。"
        recommendations = ["逐人复核防护用品穿戴状态", "持续观察实验操作区域"]
    elif ppe_count:
        title = "暂未发现明确安全事件"
        summary = "当前只识别到部分防护用品目标，未检测到进食、饮水或 No 类 PPE 风险。"
        recommendations = ["结合现场巡检确认 PPE 是否完整", "调整拍摄角度减少遮挡和漏检", "进入实验区前按要求佩戴完整防护用品"]
    else:
        title = "未发现明确安全事件"
        summary = "当前画面未识别到进食、饮水或防护用品，不能据此判定实验室绝对安全。"
        recommendations = ["调整拍摄角度或提高图像清晰度后重新分析", "结合现场巡检确认实验区域和人员状态", "确认人员是否按要求穿戴防护装备"]

    return {
        "title": title,
        "summary": summary,
        "events": events,
        "detected_hazards": eating_count + drinking_count,
        "missing_ppe": missing_ppe,
        "recommendations": recommendations,
        "counts": counts,
        "confidence_by_class": confidence_by_class,
        "person_association": "not_available",
    }


def apply_video_coat_confirmation(risk: dict[str, Any], detections: list[dict[str, Any]],
                                  evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Require adjacent sampled frames for a video NoCoat risk, retaining raw boxes.

    ``detections`` is the requested scope; ``evidence`` supplies explicit neighbor
    context from this same video. Frame numbers are sampling indices, not seconds.
    """
    def frame(item):
        value = item.get("frame", item.get("frame_index"))
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    all_frames = {frame(item) for item in evidence if item.get("class_name") == "no_lab_coat" and frame(item)}
    confirmed = {value for value in all_frames if value-1 in all_frames or value+1 in all_frames}
    scoped = {frame(item) for item in detections if item.get("class_name") == "no_lab_coat" and frame(item)}
    scoped_confirmed, pending = scoped & confirmed, scoped - confirmed
    raw_event = next((item for item in risk.get("events", []) if item.get("key") == "no_lab_coat"), None)
    unknown = any(item.get("class_name") == "no_lab_coat" and not frame(item) for item in detections)
    if not scoped and not raw_event and not unknown:
        return deepcopy(risk)
    supporting = set()
    for value in scoped_confirmed:
        start = end = value
        while start-1 in all_frames:
            start -= 1
        while end+1 in all_frames:
            end += 1
        supporting.update(range(start, end+1))
    result = deepcopy(risk)
    result["events"] = [item for item in result.get("events", []) if item.get("key") != "no_lab_coat"]
    result["missing_ppe"] = [item for item in result.get("missing_ppe", []) if item.get("class_name") != "no_lab_coat"]
    result["pending_events"] = [item for item in result.get("pending_events", []) if item.get("key") != "no_lab_coat"]
    if scoped_confirmed:
        event = {**(raw_event or {}), "key": "no_lab_coat", "label": "未穿实验服（连续多帧检出）",
                 "source": "visual_detection", "confirmation": "consecutive_sampled_frames", "evidence_frames": sorted(supporting)}
        result["events"].append(event)
        result["missing_ppe"].append({"class_name": "no_lab_coat", "label": event["label"],
            "count": sum(item.get("class_name") == "no_lab_coat" and frame(item) in scoped_confirmed for item in detections),
            "missing": ["lab_coat"], "source": "visual_detection", "evidence_frames": sorted(supporting)})
    if pending or unknown or (raw_event and not scoped):
        result["pending_events"].append({"key": "no_lab_coat", "label": "未穿实验服（未连续检出）",
            "status": "pending_temporal_confirmation", "source": "visual_detection", "evidence_frames": sorted(pending)})
    result.setdefault("temporal_confirmation", {})["no_lab_coat"] = {
        "minimum_consecutive_frames": 2, "detected_frames": sorted(scoped), "confirmed_frames": sorted(scoped_confirmed),
        "pending_frames": sorted(pending), "supporting_frames": sorted(supporting), "unit": "sampled_keyframe", "person_tracking": "not_available"}
    labels = "、".join(item["label"] for item in result["events"])
    result["title"] = "相关风险提示" if labels else "实验服状态待复核"
    result["summary"] = f"当前分析保留{len(result['events'])}类风险提示：{labels}。" if labels else "当前筛选范围尚未形成确认的风险提示。"
    if result["pending_events"]:
        where = "第" + "、".join(map(str, sorted(pending))) + "个采样关键帧" if pending else "缺少连续帧证据的画面"
        result["summary"] += f"未穿实验服仅在{where}出现提示，未满足连续2个采样关键帧检出条件，列为待复核。"
    if not result["events"] and result["pending_events"]:
        result["recommendations"] = ["复核待确认关键帧及相邻画面的实际衣物", "结合其他角度确认衣领、袖子和衣服款式"]
    return result
