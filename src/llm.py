"""The one place a ChatOllama is built.

Works with a local Ollama (``OLLAMA_BASE_URL=http://localhost:11434``) or a
remote one behind an ngrok tunnel protected by basic auth (set
``OLLAMA_BASIC_AUTH=user:pass``). Every setting is read from env at call
time, so tests can monkeypatch them and switching backends is a .env change.
"""

from __future__ import annotations

import base64
import os

import httpx
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
# A dropped tunnel would otherwise hang a call forever.
REQUEST_TIMEOUT = 300.0


def model_name() -> str:
    return os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL


def base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def auth_headers() -> dict[str, str]:
    """HTTP headers for a remote Ollama: basic auth plus ngrok's skip-warning flag.

    Kept separate from build_llm so CrewAI/LiteLLM (Phase 9) can send the same
    headers.
    """
    headers: dict[str, str] = {}
    creds = (os.getenv("OLLAMA_BASIC_AUTH") or "").strip()
    if creds:
        headers["Authorization"] = "Basic " + base64.b64encode(creds.encode()).decode()
    if "ngrok" in base_url():
        headers["ngrok-skip-browser-warning"] = "true"
    return headers


def build_llm(
    temperature: float,
    json_mode: bool = False,
    *,
    model: str | None = None,
    json_schema: dict | None = None,
) -> ChatOllama:
    """A ChatOllama for the configured server.

    json_mode: any valid JSON. json_schema: Ollama structured outputs, i.e.
    generation is constrained to that exact JSON schema (takes precedence).
    model: overrides OLLAMA_MODEL (e.g. a separate eval judge).
    """
    return ChatOllama(
        model=model or model_name(),
        base_url=base_url(),
        temperature=temperature,
        format=json_schema or ("json" if json_mode else None),
        client_kwargs={"headers": auth_headers(), "timeout": REQUEST_TIMEOUT},
    )


def describe_llm_error(exc: Exception) -> str:
    """Turn a failed LLM call into a message that says what to do about it."""
    text = str(exc)
    status = getattr(exc, "status_code", None)
    if "ERR_NGROK_3200" in text:
        return (f"Ollama tunnel is offline ({base_url()}). "
                "Start the Kaggle notebook, then try again.")
    if status == 401:
        return "Ollama rejected the credentials. Check OLLAMA_BASIC_AUTH in .env."
    if status == 404 and "model" in text.lower():
        return f"Model '{model_name()}' isn't pulled on the Ollama server ({text[:120]})."
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return f"Can't reach Ollama at {base_url()}. Is it running?"
    if isinstance(exc, httpx.TimeoutException):
        return f"Ollama at {base_url()} timed out after {REQUEST_TIMEOUT:.0f}s."
    return f"LLM call failed: {type(exc).__name__}: {text[:200]}"
