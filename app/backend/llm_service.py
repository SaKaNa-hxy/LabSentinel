import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
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
    facts = {
        "detections": detections,
        "counts": risk.get("counts", {}),
        "events": risk.get("events", []),
        "rule_summary": risk.get("summary", ""),
        "rule_recommendations": risk.get("recommendations", []),
    }
    return json.dumps(facts, ensure_ascii=False, indent=2)


def generate_chat_completion(
    messages: list[dict[str, str]],
    max_tokens: int = 700,
) -> dict[str, Any]:
    _load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先在系统环境变量或项目根目录 .env 文件中设置 DeepSeek API Key")

    config = llm_config()
    endpoint = f"{config['base_url']}/chat/completions"
    payload = {
        "model": config["model"],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": messages,
    }
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

    choices = response_data.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not content:
        raise RuntimeError("DeepSeek 返回结果为空")
    return {"provider": "DeepSeek", "model": config["model"], "content": content.strip()}


def generate_safety_report(
    detections: list[dict[str, Any]],
    risk: dict[str, Any],
) -> dict[str, Any]:
    _load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先在系统环境变量或项目根目录 .env 文件中设置 DeepSeek API Key")

    config = llm_config()
    endpoint = f"{config['base_url']}/chat/completions"
    system_prompt = (
        "你是实验室安全分析专家。你只能基于输入中的视觉检测结果和规则事件进行判断，"
        "不能编造未被检测到的目标。请用简体中文输出结构化安全报告，包含：总体判断、事件解释、"
        "可能原因、优先处置措施、后续整改建议。明确区分‘检测事实’与‘推测’，语言简洁、可执行。"
    )
    user_prompt = _build_prompt(detections, risk)
    payload = {
        "model": config["model"],
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
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

    choices = response_data.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("DeepSeek 返回结果为空")
    return {
        "provider": "DeepSeek",
        "model": config["model"],
        "report": choices[0]["message"]["content"].strip(),
    }
