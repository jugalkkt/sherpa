"""Optional Langfuse tracing, attached through LangChain callbacks.

Agents never import this module. Tracing is switched on purely by passing
``config["callbacks"]`` to ``graph.invoke``: LangGraph hands the callback
handler to every node, LLM call and tool call, which become nested spans in
one trace. With no keys in the environment everything runs identically, just
untraced.

Langfuse 4.x: ``CallbackHandler()`` reads LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY and LANGFUSE_HOST from the environment, and its
constructor accepts only ``public_key`` and ``trace_context``. Session, user
and tags travel as ``langfuse_*`` keys in ``config["metadata"]``.
"""

from __future__ import annotations

import os

import httpx

TAGS = ["sherpa", "local-inference"]
HEALTH_TIMEOUT = 2.0


def _langfuse_configured() -> bool:
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public and secret)


def _langfuse_host() -> str:
    # Same precedence as the langfuse client itself.
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")


def _langfuse_reachable(host: str) -> bool:
    """Quick health check, so a stopped server means "tracing off" instead of
    span-export errors flooding the terminal for the whole session."""
    try:
        httpx.get(f"{host}/api/public/health", timeout=HEALTH_TIMEOUT)
        return True  # any HTTP answer means something is listening
    except httpx.HTTPError:
        return False


def get_langfuse_handler():
    """A Langfuse CallbackHandler, or None if tracing is off or unavailable."""
    if not _langfuse_configured():
        return None
    host = _langfuse_host()
    if not _langfuse_reachable(host):
        print(f"[Tracing] Langfuse keys are set but {host} isn't answering (is `make langfuse` running?). "
              "Continuing without tracing.")
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except ImportError as exc:
        print(f"[Tracing] Langfuse keys are set but its LangChain integration won't import ({exc}). "
              "Install requirements.txt (it needs the `langchain` package).")
    except Exception as exc:
        print(f"[Tracing] Could not start Langfuse ({type(exc).__name__}: {exc}). Continuing without tracing.")
    return None


def get_langfuse_config(session_id: str, user_id: str = "local", extra_config: dict | None = None) -> dict:
    """The config for graph.invoke: thread_id always, Langfuse callbacks when available."""
    config: dict = {"configurable": {"thread_id": session_id}}
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
        config["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": TAGS,
        }
        print(f"[Tracing] On: Langfuse at {_langfuse_host()} (session {session_id})")
    elif not _langfuse_configured():
        print("[Tracing] Off (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable)")
    # else: get_langfuse_handler already printed why tracing couldn't start
    for key, value in (extra_config or {}).items():
        # Merge nested dicts so e.g. extra "configurable" keys don't drop thread_id.
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def flush_langfuse() -> None:
    """Send any buffered spans before the process exits. Best effort, never raises."""
    if not _langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        print(f"[Tracing] Could not flush traces ({type(exc).__name__}).")
