import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from .agent_context import ConversationContext
    from .agent_service import MAX_MESSAGES, answer_turn
except ImportError:
    from agent_context import ConversationContext
    from agent_service import MAX_MESSAGES, answer_turn


CHAT_DB = Path(__file__).resolve().parents[2] / "storage" / "chats" / "conversations.sqlite3"


class ChatNotFound(LookupError):
    pass


class ChatChanged(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    CHAT_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_DB)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_turns (
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                request_id TEXT UNIQUE,
                request_signature TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY (chat_id, position)
            );
        """)
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _summary(row: sqlite3.Row) -> dict[str, str]:
    return {key: row[key] for key in ("id", "title", "created_at", "updated_at")}


def _chat_row(connection: sqlite3.Connection, chat_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        raise ChatNotFound("这条对话已不存在。")
    return row


def _has_analysis(context: ConversationContext) -> bool:
    return context.facts.get("source") in ("当前分析结果", "历史分析记录", "历史图片复核")


def _encode_state(
    context: ConversationContext, messages: list[dict[str, str]], attachment_ids: list[str],
) -> str:
    return json.dumps({
        "context": asdict(context), "messages": messages, "history_record_ids": attachment_ids,
    }, ensure_ascii=False)


def list_chats() -> list[dict[str, str]]:
    with closing(_connect()) as connection:
        return [_summary(row) for row in connection.execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC, id DESC"
        )]


def get_chat(chat_id: str) -> dict[str, Any]:
    with closing(_connect()) as connection, connection:
        connection.execute("BEGIN")
        row = _chat_row(connection, chat_id)
        state = json.loads(row["state"])
        turns = [json.loads(item["payload"]) for item in connection.execute(
            "SELECT payload FROM chat_turns WHERE chat_id = ? ORDER BY position", (chat_id,)
        )]
    for turn in turns:
        result = turn.pop("reply")
        turn.update(answer=result["answer"], evidence_mode=result.get("evidence_mode"))
    return {
        **_summary(row),
        "turns": turns,
        "history_record_ids": state["history_record_ids"],
        "has_analysis": _has_analysis(ConversationContext(**state["context"])),
    }


def delete_chat(chat_id: str) -> None:
    with closing(_connect()) as connection, connection:
        cursor = connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        if not cursor.rowcount:
            raise ChatNotFound("这条对话已不存在。")


def clear_chat_context(chat_id: str) -> None:
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE chats SET state = ?, revision = revision + 1 WHERE id = ?",
            (_encode_state(ConversationContext(), [], []), chat_id),
        )


def _saved_reply(connection: sqlite3.Connection, request_id: Optional[str], signature: Optional[str]) -> Optional[dict[str, Any]]:
    if request_id is None:
        return None
    row = connection.execute(
        "SELECT request_signature, payload FROM chat_turns WHERE request_id = ?", (request_id,)
    ).fetchone()
    if row is None:
        return None
    if row["request_signature"] != signature:
        raise ValueError("重试请求与原消息不一致。")
    return json.loads(row["payload"])["reply"]


def reply(
    message: str,
    session_id: Optional[str] = None,
    record_id: Optional[str] = None,
    analysis: Optional[dict[str, Any]] = None,
    analysis_id: Optional[str] = None,
    *,
    history_record_ids: Optional[list[str]] = None,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    content = message.strip()
    if not content:
        raise ValueError("消息不能为空")
    if request_id is not None and not 1 <= len(request_id) <= 80:
        raise ValueError("请求标识无效。")
    attachment_ids = history_record_ids or []
    signature = None
    if request_id:
        body = json.dumps(
            [content, session_id, record_id, analysis, analysis_id, attachment_ids],
            sort_keys=True, separators=(",", ":"),
        )
        signature = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if session_id or request_id:
        with closing(_connect()) as connection:
            saved = _saved_reply(connection, request_id, signature)
            if saved is not None:
                return saved
            row = _chat_row(connection, session_id) if session_id else None
    else:
        row = None
    if row is not None:
        state = json.loads(row["state"])
        context = ConversationContext(**state["context"])
        messages = state["messages"]
        summary = _summary(row)
        revision = row["revision"]
    else:
        context, messages = ConversationContext(), []
        title = " ".join(content.split())
        summary = {"id": uuid.uuid4().hex, "title": title[:40] + ("…" if len(title) > 40 else "")}
        revision = 0

    chat_id = summary["id"]
    # Let the remote call run outside the SQLite write transaction.
    result = answer_turn(
        content, chat_id, messages, context, record_id, analysis, analysis_id,
        history_record_ids=attachment_ids,
    )
    messages.extend((
        {"role": "user", "content": content},
        {"role": "assistant", "content": result["answer"]},
    ))
    state_json = _encode_state(context, messages[-MAX_MESSAGES:], attachment_ids)
    now = datetime.now().astimezone().isoformat(timespec="microseconds")
    summary["updated_at"] = now
    if row is None:
        summary["created_at"] = now
    response = {**result, "conversation": summary, "has_analysis": _has_analysis(context)}
    turn = {
        "question": content, "history_record_ids": attachment_ids, "created_at": now, "reply": response,
    }
    try:
        with closing(_connect()) as connection, connection:
            if row is not None:
                cursor = connection.execute(
                    "UPDATE chats SET updated_at = ?, state = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
                    (now, state_json, chat_id, revision),
                )
                if not cursor.rowcount:
                    saved = _saved_reply(connection, request_id, signature)
                    if saved is not None:
                        return saved
                    _chat_row(connection, chat_id)
                    raise ChatChanged("对话已在其他窗口更新，请重新打开后再发送。")
            else:
                connection.execute(
                    "INSERT INTO chats (id, title, created_at, updated_at, revision, state) VALUES (?, ?, ?, ?, 1, ?)",
                    (chat_id, summary["title"], now, now, state_json),
                )
            connection.execute(
                "INSERT INTO chat_turns (chat_id, position, request_id, request_signature, payload) VALUES (?, ?, ?, ?, ?)",
                (chat_id, revision + 1, request_id, signature, json.dumps(turn, ensure_ascii=False)),
            )
    except sqlite3.IntegrityError:
        with closing(_connect()) as connection:
            saved = _saved_reply(connection, request_id, signature)
        if saved is None:
            raise
        return saved
    return response
