import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .history_service import delete_all_history, delete_history, get_history, history_image_path, list_history, save_history
except ImportError:
    from history_service import delete_all_history, delete_history, get_history, history_image_path, list_history, save_history

try:
    from .agent_service import MissingHistoryRecords
    from .chat_store import ChatChanged, ChatNotFound, clear_chat_context, delete_chat, get_chat, list_chats, reply as agent_chat
    from .inference import model_info, predict
    from .llm_service import generate_safety_report, llm_config
except ImportError:
    from agent_service import MissingHistoryRecords
    from chat_store import ChatChanged, ChatNotFound, clear_chat_context, delete_chat, get_chat, list_chats, reply as agent_chat
    from inference import model_info, predict
    from llm_service import generate_safety_report, llm_config

ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ReportRequest(BaseModel):
    detections: list[dict]
    risk: dict


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    record_id: Optional[str] = None
    analysis: Optional[dict] = None
    analysis_id: Optional[str] = None
    history_record_ids: Optional[list[str]] = None
    request_id: Optional[str] = None


class ClearChatRequest(BaseModel):
    session_id: str


FRONTEND_DIR = ROOT / "app" / "frontend"

app = FastAPI(title="LabSentinel API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"], allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])


@app.get("/api/health")
def health() -> dict:
    info = model_info()
    return {"status": "ok", "model_loaded": info["loaded"], "model": info}


@app.get("/api/model-info")
def get_model_info() -> dict:
    info = model_info()
    info["llm"] = llm_config()
    return info


@app.post("/api/report")
def report(payload: ReportRequest) -> dict:
    try:
        return generate_safety_report(payload.detections, payload.risk)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"智能报告生成失败：{exc}") from exc


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return agent_chat(
            payload.message,
            payload.session_id,
            payload.record_id,
            payload.analysis,
            payload.analysis_id,
            history_record_ids=payload.history_record_ids,
            request_id=payload.request_id,
        )
    except MissingHistoryRecords as exc:
        raise HTTPException(status_code=404, detail={
            "message": "部分历史记录已删除，请刷新记录后重试。",
            "missing_record_ids": exc.record_ids,
        }) from exc
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatChanged as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="无法保存本地对话，请稍后重试。") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent 请求失败：{exc}") from exc


@app.post("/api/chat/clear")
def clear_chat(payload: ClearChatRequest) -> dict:
    try:
        clear_chat_context(payload.session_id)
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="暂时无法更新本地对话。") from exc
    return {"cleared": True, "session_id": payload.session_id}


@app.get("/api/chats")
def chat_list() -> dict:
    try:
        return {"items": list_chats()}
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="暂时无法读取对话历史。") from exc


@app.get("/api/chats/{chat_id}")
def chat_detail(chat_id: str) -> dict:
    try:
        return get_chat(chat_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="暂时无法读取对话。") from exc


@app.delete("/api/chats/{chat_id}")
def chat_delete(chat_id: str) -> dict:
    try:
        delete_chat(chat_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="暂时无法删除对话。") from exc
    return {"deleted": True, "id": chat_id}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), confidence: float = 0.15) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传 JPG、PNG 或 WEBP 图片")
    payload = await file.read(MAX_IMAGE_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 10 MB")
    try:
        return predict(payload, max(0.05, min(confidence, 0.95)))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"图片分析失败：{exc}") from exc


@app.post("/api/history")
async def create_history(
    file: UploadFile = File(...),
    source: str = Form("image"),
    frame_index: Optional[int] = Form(None),
    analysis: str = Form(...),
) -> dict:
    if source not in {"image", "video", "camera"}:
        raise HTTPException(status_code=400, detail="来源必须是 image、video 或 camera")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="历史图片必须是图片文件")
    payload = await file.read(MAX_IMAGE_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="历史图片为空")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="历史图片不能超过 10 MB")
    if len(analysis) > 500_000:
        raise HTTPException(status_code=413, detail="分析结果过大")
    try:
        analysis_data = json.loads(analysis)
        if not isinstance(analysis_data, dict):
            raise ValueError("分析结果格式无效")
        return save_history(source, payload, analysis_data, frame_index)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"历史分析结果无效：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"历史记录保存失败：{exc}") from exc


@app.get("/api/history")
def history_list() -> dict:
    return {"items": list_history()}


@app.delete("/api/history")
def history_delete_all() -> dict:
    try:
        return delete_all_history()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="历史记录删除失败，请刷新记录后重试") from exc


@app.get("/api/history/{record_id}")
def history_detail(record_id: str) -> dict:
    record = get_history(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="历史记录不存在")
    return record


@app.delete("/api/history/{record_id}")
def history_delete(record_id: str) -> dict:
    if not delete_history(record_id):
        raise HTTPException(status_code=404, detail="历史记录不存在")
    return {"deleted": True, "id": record_id}


@app.get("/api/history-files/{record_id}/{filename}")
def history_file(record_id: str, filename: str) -> FileResponse:
    if filename != "annotated.jpg":
        raise HTTPException(status_code=404, detail="历史文件不存在")
    try:
        path = history_image_path(record_id)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail="历史文件不存在") from exc
    return FileResponse(path, media_type="image/jpeg")


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
