"""Single point of contact with Ollama (served from Kaggle through ngrok).

- `get_chat(role)` / `get_embeddings()` build authenticated clients.
- `with_retry(fn)` retries transient tunnel/server failures, then raises LLMUnavailableError.
- `healthcheck()` powers `python -m sherpa doctor`.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Callable, Literal, TypeVar

import httpx
import ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import SecretStr
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from sherpa.config import Settings, get_settings

Role = Literal["planner", "explainer", "quiz_gen", "quiz_grade", "coach"]
T = TypeVar("T")

RECOVERY_HINT = (
    "Is the Kaggle notebook running (Ollama server + ngrok cells)? "
    "Check with `python -m sherpa doctor`, then continue with `python -m sherpa resume <id>`."
)


class LLMUnavailableError(RuntimeError):
    """The endpoint stayed unreachable after all retries."""


class LLMAuthError(RuntimeError):
    """ngrok rejected the basic-auth credentials (HTTP 401)."""


def auth_headers(basic_auth: SecretStr | None) -> dict[str, str]:
    headers = {"ngrok-skip-browser-warning": "1"}
    if basic_auth is not None:
        token = base64.b64encode(basic_auth.get_secret_value().encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return headers


def describe_http_error(status: int | None, body: str) -> str:
    """Turn an ngrok HTML error page / Ollama error body into one readable line."""
    if m := re.search(r"ERR_NGROK_\d+", body):
        hints = {
            "ERR_NGROK_8012": "tunnel is up but Ollama is not running in Kaggle (run the `ollama serve` cell)",
            "ERR_NGROK_3200": "tunnel is offline (run the ngrok cell in Kaggle)",
        }
        return f"{m.group(0)}: {hints.get(m.group(0), 'ngrok error')}"
    if "ngrok Cloud Endpoint" in body:
        return "a dashboard-created ngrok Cloud Endpoint owns this domain; delete it in the ngrok dashboard"
    if status == 401:
        return "401 Unauthorized: OLLAMA_BASIC_AUTH does not match the Kaggle secret"
    snippet = re.sub(r"\s+", " ", body).strip()[:160]
    return f"HTTP {status}: {snippet}" if status else snippet


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code in (429, 502, 503, 504) or exc.status_code >= 500
    return False


def with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap `fn` so transient endpoint failures are retried with backoff (2s, 4s, 8s...)."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        s = get_settings()
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(s.llm_max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=8),
                retry=retry_if_exception(_is_transient),
                reraise=True,
            ):
                with attempt:
                    return fn(*args, **kwargs)
        except ollama.ResponseError as e:
            if e.status_code == 401:
                raise LLMAuthError(describe_http_error(401, e.error)) from e
            if _is_transient(e):
                raise LLMUnavailableError(f"{describe_http_error(e.status_code, e.error)}. {RECOVERY_HINT}") from e
            raise
        except Exception as e:
            if _is_transient(e):
                raise LLMUnavailableError(f"LLM endpoint unreachable ({type(e).__name__}). {RECOVERY_HINT}") from e
            raise
        raise AssertionError("unreachable")  # pragma: no cover

    return wrapper


def _temperature(s: Settings, role: Role) -> float:
    return {
        "planner": s.temp_planner,
        "explainer": s.temp_explainer,
        "quiz_gen": s.temp_quiz_gen,
        "quiz_grade": s.temp_quiz_grade,
        "coach": s.temp_coach,
    }[role]


@lru_cache
def get_chat(role: Role) -> ChatOllama:
    s = get_settings()
    return ChatOllama(
        model=s.chat_model,
        base_url=s.ollama_base_url,
        temperature=_temperature(s, role),
        num_ctx=s.num_ctx,
        client_kwargs={"headers": auth_headers(s.ollama_basic_auth), "timeout": s.request_timeout_s},
    )


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    s = get_settings()
    return OllamaEmbeddings(
        model=s.embed_model,
        base_url=s.effective_embed_base_url,
        client_kwargs={"headers": auth_headers(s.effective_embed_basic_auth), "timeout": s.request_timeout_s},
    )


# --------------------------------------------------------------------------- health check


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _model_present(name: str, available: set[str]) -> bool:
    return name in available or f"{name}:latest" in available or name.removesuffix(":latest") in available


def _get_json(client: httpx.Client, path: str) -> dict:
    r = client.get(path)
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        raise RuntimeError(describe_http_error(r.status_code, r.text))
    return r.json()


def healthcheck(*, quick: bool = False) -> list[Check]:
    """Probe the endpoint step by step; stops at the first failure that blocks later checks.

    quick=True skips the chat and embedding calls (version + model list only).
    """
    s = get_settings()
    checks: list[Check] = []
    with httpx.Client(
        base_url=s.ollama_base_url, headers=auth_headers(s.ollama_basic_auth), timeout=15, follow_redirects=True
    ) as client:
        try:
            v = _get_json(client, "/api/version")
            checks.append(Check("endpoint + auth", True, f"{s.ollama_base_url} · Ollama {v.get('version', '?')}"))
        except Exception as e:
            checks.append(Check("endpoint + auth", False, f"{s.ollama_base_url}: {e}"))
            return checks

        try:
            tags = _get_json(client, "/api/tags")
            available = {m["name"] for m in tags.get("models", [])}
        except Exception as e:
            checks.append(Check("models", False, str(e)))
            return checks

    for label, model in (("chat model", s.chat_model), ("embed model", s.embed_model)):
        if label == "embed model" and s.embed_base_url:
            checks.append(Check(label, True, f"{model} (separate endpoint {s.embed_base_url}, checked below)"))
            continue
        present = _model_present(model, available)
        detail = model if present else f"{model} not pulled; run `ollama pull {model}` in Kaggle"
        checks.append(Check(label, present, detail))

    if quick or not all(c.ok for c in checks):
        return checks

    try:
        t0 = time.perf_counter()
        reply = with_retry(get_chat("planner").invoke)("Reply with the single word: ready")
        dt = time.perf_counter() - t0
        checks.append(Check("chat call", True, f"{dt:.1f}s · reply {str(reply.content).strip()[:30]!r}"))
    except Exception as e:
        checks.append(Check("chat call", False, str(e)))

    try:
        t0 = time.perf_counter()
        vec = with_retry(get_embeddings().embed_query)("search_query: hello")
        dt = time.perf_counter() - t0
        checks.append(Check("embedding call", True, f"{dt:.1f}s · dimension {len(vec)}"))
    except Exception as e:
        checks.append(Check("embedding call", False, str(e)))

    return checks
