import base64

import httpx
import pytest
from ollama import ResponseError

from llm import auth_headers, build_llm, describe_llm_error

pytestmark = pytest.mark.unit

NGROK_URL = "https://example-tunnel.ngrok-free.dev"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_BASIC_AUTH"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_local_ollama_without_auth():
    llm = build_llm(temperature=0.3)
    assert llm.model == "qwen2.5:7b"
    assert llm.base_url == "http://localhost:11434"
    assert llm.format is None
    assert auth_headers() == {}


def test_reads_env_at_call_time(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", NGROK_URL + "/")
    llm = build_llm(temperature=0.1, json_mode=True)
    assert llm.model == "qwen3:8b"
    assert llm.base_url == NGROK_URL  # trailing slash stripped
    assert llm.format == "json" and llm.temperature == 0.1


def test_model_and_json_schema_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}
    llm = build_llm(0.0, json_mode=True, model="qwen3:8b", json_schema=schema)
    assert llm.model == "qwen3:8b"
    assert llm.format == schema  # the schema wins over plain JSON mode


def test_basic_auth_header(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASIC_AUTH", "sherpa:s3cret")
    expected = "Basic " + base64.b64encode(b"sherpa:s3cret").decode()
    assert auth_headers()["Authorization"] == expected
    assert build_llm(0.3).client_kwargs["headers"]["Authorization"] == expected


def test_ngrok_url_gets_skip_warning_header(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", NGROK_URL)
    assert auth_headers() == {"ngrok-skip-browser-warning": "true"}


def test_auth_header_reaches_the_http_request(monkeypatch):
    """End to end through ChatOllama → ollama.Client → httpx, with a fake transport."""
    monkeypatch.setenv("OLLAMA_BASE_URL", NGROK_URL)
    monkeypatch.setenv("OLLAMA_BASIC_AUTH", "u:p")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={
            "model": "qwen2.5:7b", "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"}, "done": True,
        })

    llm = build_llm(0.3)
    llm._client._client = httpx.Client(base_url=NGROK_URL, headers=llm._client._client.headers,
                                       transport=httpx.MockTransport(handler))
    assert llm.invoke("hello").content == "hi"
    assert seen["authorization"] == "Basic " + base64.b64encode(b"u:p").decode()


def test_error_tunnel_offline():
    exc = ResponseError("<html>The endpoint is offline. (ERR_NGROK_3200)</html>", 404)
    assert "tunnel is offline" in describe_llm_error(exc)


def test_error_bad_credentials():
    assert "OLLAMA_BASIC_AUTH" in describe_llm_error(ResponseError("Unauthorized", 401))


def test_error_model_not_pulled(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen9:1b")
    msg = describe_llm_error(ResponseError("model 'qwen9:1b' not found", 404))
    assert "qwen9:1b" in msg and "pulled" in msg


def test_error_unreachable():
    assert "Can't reach Ollama" in describe_llm_error(ConnectionError("refused"))


def test_error_timeout():
    assert "timed out" in describe_llm_error(httpx.ReadTimeout("slow"))


def test_error_fallback():
    assert describe_llm_error(RuntimeError("boom")) == "LLM call failed: RuntimeError: boom"
