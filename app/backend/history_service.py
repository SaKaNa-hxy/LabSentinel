import io
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "storage" / "history"
RECORDS_FILE = HISTORY_DIR / "records.json"
RECORDS_LOCK = RLock()

SOURCE_LABELS = {
    "image": "图片分析",
    "video": "视频分析",
    "camera": "摄像头捕捉",
}


def _ensure_storage() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not RECORDS_FILE.exists():
        RECORDS_FILE.write_text("[]", encoding="utf-8")


def _read_records() -> List[Dict[str, Any]]:
    _ensure_storage()
    try:
        value = json.loads(RECORDS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _write_records(records: List[Dict[str, Any]]) -> None:
    _ensure_storage()
    temporary = RECORDS_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(RECORDS_FILE)


def list_history() -> List[Dict[str, Any]]:
    with RECORDS_LOCK:
        return sorted(
            _read_records(),
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )


def get_history(record_id: str) -> Optional[Dict[str, Any]]:
    with RECORDS_LOCK:
        return next(
            (item for item in _read_records() if item.get("id") == record_id),
            None,
        )


def save_history(
    source_type: str,
    image_bytes: bytes,
    analysis: Dict[str, Any],
    frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    if source_type not in SOURCE_LABELS:
        raise ValueError("来源类型必须是 image、video 或 camera")
    if not image_bytes:
        raise ValueError("历史记录图片不能为空")

    record_id = uuid.uuid4().hex
    record_dir = HISTORY_DIR / record_id
    record_dir.mkdir(parents=True, exist_ok=False)
    image_path = record_dir / "annotated.jpg"
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(image_path, format="JPEG", quality=92)
    except Exception as exc:
        if image_path.exists():
            image_path.unlink()
        record_dir.rmdir()
        raise ValueError("历史记录图片格式无效") from exc

    risk = analysis.get("risk") or {}
    record = {
        "id": record_id,
        "source": source_type,
        "source_label": SOURCE_LABELS[source_type],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_url": f"/api/history-files/{record_id}/annotated.jpg",
        "frame_index": frame_index,
        "risk": risk,
        "detections": analysis.get("detections", []),
        "warning": analysis.get("warning"),
        "image_size": analysis.get("image_size"),
    }
    with RECORDS_LOCK:
        records = _read_records()
        records.append(record)
        _write_records(records)
    return record


def _delete_record_files(record_id: str) -> None:
    if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{32}", record_id):
        raise ValueError("历史记录 ID 无效")
    record_dir = HISTORY_DIR / record_id
    if record_dir.is_symlink() or record_dir.resolve().parent != HISTORY_DIR.resolve():
        raise ValueError("历史记录目录超出存储范围")
    if record_dir.exists():
        for path in record_dir.iterdir():
            if path.is_file():
                path.unlink()
        record_dir.rmdir()


def delete_all_history() -> Dict[str, int]:
    """Delete indexed records under the same lock as saves; retain failures."""
    with RECORDS_LOCK:
        records = _read_records()
        remaining = []
        for record in records:
            try:
                _delete_record_files(record.get("id"))
            except (OSError, ValueError):
                remaining.append(record)
        _write_records(remaining)
        return {"deleted_count": len(records) - len(remaining), "failed_count": len(remaining)}


def delete_history(record_id: str) -> bool:
    with RECORDS_LOCK:
        records = _read_records()
        record = next((item for item in records if item.get("id") == record_id), None)
        if record is None:
            return False

        _delete_record_files(record_id)
        _write_records([item for item in records if item.get("id") != record_id])
        return True
