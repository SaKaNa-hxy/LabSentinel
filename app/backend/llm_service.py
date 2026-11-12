import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .risk_engine import apply_video_coat_confirmation
except ImportError:
    from risk_engine import apply_video_coat_confirmation

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
_env_loaded = False


def _load_env() -> None:
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def llm_config() -> dict[str, Any]:
    _load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
    return {
        "configured": bool(api_key),
        "base_url": base_url,
        "model": model,
    }


def _build_prompt(detections: list[dict[str, Any]], risk: dict[str, Any]) -> str:
    if any(isinstance(item.get("frame"), int) and item["frame"] > 0 for item in detections):
        risk = apply_video_coat_confirmation(risk, detections, detections)
    facts = {
        "detections": detections,
        "counts": risk.get("counts", {}),
        "events": risk.get("events", []),
        "pending_events": risk.get("pending_events", []),
        "temporal_confirmation": risk.get("temporal_confirmation", {}),
        "rule_summary": risk.get("summary", ""),
        "rule_recommendations": risk.get("recommendations", []),
    }
    return json.dumps(facts, ensure_ascii=False, indent=2)


def generate_chat_completion(
    messages: list[dict[str, Any]],
    max_tokens: int = 700,
    *,
    vision: bool = False,
) -> dict[str, Any]:
    _load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先在系统环境变量或项目根目录 .env 文件中设置 DeepSeek API Key")

    config = llm_config()
    if vision and config["base_url"] != DEFAULT_BASE_URL:
        raise RuntimeError("视觉问答需要 DeepSeek 官方接口")
    endpoint = f"{config['base_url']}/chat/completions"
    payload = {
        "model": DEFAULT_MODEL if vision else config["model"],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if config["base_url"] == DEFAULT_BASE_URL and payload["model"] == DEFAULT_MODEL:
        payload["thinking"] = {"type": "disabled"}
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek 接口请求失败（HTTP {exc.code}）：{detail[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 DeepSeek 接口：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("DeepSeek 接口请求超时，请稍后重试") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError("DeepSeek 接口连接或响应异常") from exc

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek 返回结果格式异常") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek 返回结果为空")
    return {"provider": "DeepSeek", "model": payload["model"], "content": content.strip()}


def generate_safety_report(
    detections: list[dict[str, Any]],
    risk: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "以实验室巡检员的口吻写一份简体中文记录。只输出一个 JSON 对象，"
        '格式为 {"summary":"一句话总体判断", "findings":"两到三句主要发现", '
        '"actions":["一条可执行的建议", "另一条建议"]}。'
        "正文约 150 至 250 字，使用日常中文和短句。各字段只写内容，不重复标题，"
        "不要添加 Markdown、表格、编号、英文类别名、字段名或逐项置信度。"
        "总体判断先说明是否发现需要复核的风险。主要发现只写输入支持的事实，"
        "注明影响判断的不确定项。处置建议给出一至三项，按现场复核和确认后处理的顺序写。"
        "不要猜测设备供应、人员身份、人数、具体原因或现场操作。"
        "未检出风险时，写明本次画面的发现范围，不保证现场安全。"
        "source=behavior_inference 的未戴口罩事件是由进食或饮水规则推断，必须明确其来源，"
        "不能当作模型直接检测到未戴口罩，也不能据此编造检测框、人数或具体人员。"
        "仅在这类推断实际出现时解释来源。视频未穿实验服须连续两个采样关键帧检出才纳入风险。"
        "pending_events 中的事项只能写为待复核。不能根据单帧检测框或 counts 确认视频未穿实验服，"
        "采样帧号不代表秒数。只在有待复核事件时说明，不逐项罗列未触发的规则。"
    )
    result = generate_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_prompt(detections, risk)},
        ],
        max_tokens=900,
    )
    content = re.sub(r"\A```(?:json)?[ \t]*\r?\n|\r?\n```[ \t]*\Z", "", result["content"], flags=re.IGNORECASE)
    try:
        sections = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("报告格式不完整，请重新生成") from exc
    if not isinstance(sections, dict) or not isinstance(sections.get("actions"), list):
        raise RuntimeError("报告格式不完整，请重新生成")

    summary = _plain_report_text(sections.get("summary"))
    findings = _plain_report_text(sections.get("findings"))
    actions = [re.sub(r"^(?:\d+[.、．)）]|[-–•])\s*", "", _plain_report_text(action)) for action in sections["actions"]]
    if not summary or not findings or not actions or not all(actions):
        raise RuntimeError("报告格式不完整，请重新生成")

    numbered_actions = "\n".join(f"{index}. {action}" for index, action in enumerate(actions, start=1))
    return {
        "provider": result["provider"],
        "model": result["model"],
        "report": f"实验室安全分析报告\n\n总体判断\n{summary}\n\n主要发现\n{findings}\n\n处置建议\n{numbered_actions}",
        "sections": {"summary": summary, "findings": findings, "actions": actions},
    }


def _plain_report_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"\\?[#*`]+|__", "", value).split())
