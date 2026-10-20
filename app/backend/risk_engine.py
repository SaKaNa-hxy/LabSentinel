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
    events: list[dict[str, str]] = []

    for key, label in BEHAVIOR_LABELS.items():
        if counts[key]:
            events.append({"key": key, "label": label, "source": "visual_detection"})
    for item in missing_ppe:
        events.append({"key": item["class_name"], "label": item["label"], "source": "visual_detection"})

    if eating_count or drinking_count:
        behavior_labels = "、".join(BEHAVIOR_LABELS[key] for key in ("eating", "drinking") if counts[key])
        ppe_note = "另有 PPE 模型输出：" + "、".join(item["label"] for item in missing_ppe) + "。" if missing_ppe else ""
        title = "实验室行为风险"
        summary = f"检测到{behavior_labels}，请现场复核。{ppe_note}进食或饮水本身不构成未戴口罩的视觉证据。"
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
