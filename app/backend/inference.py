import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from .risk_engine import analyze_risk
except ImportError:
    from risk_engine import analyze_risk

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "datasets" / "laboratory_safety"
CUSTOM_MODEL = MODEL_DIR / "laboratory_safety" / "weights" / "best.pt"
FALLBACK_MODEL = MODEL_DIR / "yolo11s.pt"
DEPLOYMENT_MANIFEST = MODEL_DIR / "laboratory_safety" / "weights" / "deployment.json"
DATASET_CLASS_NAMES = [
    "Drinking",
    "Eating",
    "Gloves",
    "Googles",
    "Head Mask",
    "Lab Coat",
    "Mask",
    "No Gloves",
    "No Head Mask",
    "No Lab coat",
    "No Mask",
    "No googles",
]
CLASS_NAME_MAP = {
    "Drinking": "drinking",
    "Eating": "eating",
    "Gloves": "gloves",
    "Googles": "goggles",
    "Head Mask": "head_mask",
    "Lab Coat": "lab_coat",
    "Mask": "mask",
    "No Gloves": "no_gloves",
    "No Head Mask": "no_head_mask",
    "No Lab coat": "no_lab_coat",
    "No Mask": "no_mask",
    "No googles": "no_goggles",
}
CLASS_NAMES = list(CLASS_NAME_MAP.values())

_model = None
_model_path: Optional[Path] = None
_model_error: Optional[str] = None
_custom_model_compatible = False
_model_sha256: Optional[str] = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deployment_info() -> Optional[dict[str, Any]]:
    if not _custom_model_compatible or not DEPLOYMENT_MANIFEST.is_file():
        return None
    try:
        manifest = json.loads(DEPLOYMENT_MANIFEST.read_text(encoding="utf-8"))
        return manifest if manifest.get("deployed_sha256") == _model_sha256 else None
    except (OSError, json.JSONDecodeError):
        return None


def load_model() -> None:
    global _model, _model_path, _model_error, _custom_model_compatible, _model_sha256
    try:
        from ultralytics import YOLO
        selected = FALLBACK_MODEL
        if CUSTOM_MODEL.exists():
            custom = YOLO(str(CUSTOM_MODEL))
            custom_names = getattr(custom, "names", {}) or {}
            ordered_names = [str(custom_names[index]) for index in range(len(DATASET_CLASS_NAMES))] if len(custom_names) == len(DATASET_CLASS_NAMES) else []
            if ordered_names == DATASET_CLASS_NAMES:
                _model = custom
                selected = CUSTOM_MODEL
                _custom_model_compatible = True
            else:
                _model_error = "专用模型类别名称或顺序与当前 12 类数据集不一致，请核对权重。"
        if _model is None:
            if not selected.exists():
                _model_error = _model_error or f"模型文件不存在：{selected}"
                return
            _model = YOLO(str(selected))
        _model_path = selected
        _model_sha256 = _file_sha256(selected)
        if _custom_model_compatible:
            _model_error = None
    except Exception as exc:
        _model_error = str(exc)


def model_info() -> dict[str, Any]:
    if _model is None and _model_error is None:
        load_model()
    return {
        "loaded": _model is not None,
        "path": str(_model_path.relative_to(ROOT)) if _model_path else None,
        "custom_model_ready": _custom_model_compatible,
        "deployment_verified": _deployment_info() is not None,
        "model_sha256": _model_sha256,
        "evaluation": (_deployment_info() or {}).get("grouped_test_metrics"),
        "error": _model_error,
        "classes": CLASS_NAMES,
        "dataset": "7217 images / 12 classes / detection labels",
        "note": "当前数据集识别进食、饮水和 PPE 状态；风险判断直接基于行为类和 No 类检测结果。",
    }


def _encode_image(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def predict(image_bytes: bytes, confidence: float = 0.15) -> dict[str, Any]:
    if _model is None and _model_error is None:
        load_model()
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        if source_image.width * source_image.height > 20_000_000:
            raise ValueError("图片像素过大，最多支持 2000 万像素")
        image = source_image.convert("RGB")
    detections: list[dict[str, Any]] = []
    warning = None
    annotated = image
    if _model is None:
        warning = _model_error or "模型未加载"
    else:
        results = _model.predict(source=image, conf=confidence, max_det=300, verbose=False)
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is not None:
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index].item())
                score = float(boxes.conf[index].item())
                xyxy = [round(float(value), 2) for value in boxes.xyxy[index].tolist()]
                model_name = str(names.get(class_id, class_id))
                class_name = CLASS_NAME_MAP.get(model_name, model_name.lower().replace(" ", "_"))
                detections.append({"class_id": class_id, "class_name": class_name, "confidence": round(score, 4), "bbox": xyxy})
        try:
            annotated = Image.fromarray(result.plot()[:, :, ::-1])
        except Exception:
            annotated = image
        if not _custom_model_compatible:
            warning = "当前未加载与 12 类数据集匹配的专用权重；请先重新训练 laboratory_safety 模型。"

    risk = analyze_risk(detections)
    return {
        "image": _encode_image(annotated),
        "detections": detections,
        "risk": risk,
        "model": model_info(),
        "warning": warning,
        "image_size": {"width": image.width, "height": image.height},
    }
