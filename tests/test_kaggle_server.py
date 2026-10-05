import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import hosting.kaggle_server as ks
from hosting.kaggle_server import (
    kernel_metadata,
    render_server_script,
    secrets_payload,
    server_status,
    start_server,
    upload_secrets,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def load_server_script():
    spec = importlib.util.spec_from_file_location("sherpa_server", ROOT / "deploy" / "kaggle" / "sherpa_ollama_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: main() only runs under __main__
    return module


server = load_server_script()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_SLUG", "jugal/sherpa-ollama-server")
    monkeypatch.setenv("KAGGLE_SECRETS_DATASET", "jugal/sherpa-server-secrets")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://sherpa.ngrok-free.dev")
    monkeypatch.setenv("OLLAMA_BASIC_AUTH", "user:pa:ss")
    monkeypatch.setenv("NGROK_AUTHTOKEN", "ngrok-token")


@pytest.fixture
def unreachable(monkeypatch):
    monkeypatch.setattr(ks, "model_reachable", lambda timeout=5.0: False)


class FakeKaggle:
    def __init__(self, status="COMPLETE", failure="", push_error=None, status_exc=None, create_error=None):
        self.status, self.failure, self.push_error, self.status_exc = status, failure, push_error, status_exc
        self.create_error = create_error
        self.pushed = None
        self.dataset_calls = []

    def kernels_status(self, slug):
        if self.status_exc:
            raise self.status_exc
        return SimpleNamespace(status=SimpleNamespace(name=self.status), failure_message=self.failure)

    def kernels_push(self, folder, timeout=None):
        self.pushed = {
            "timeout": timeout,
            "files": {p.name: p.read_text() for p in Path(folder).iterdir()},
        }
        return SimpleNamespace(error=self.push_error, url="https://www.kaggle.com/code/jugal/sherpa-ollama-server")

    def dataset_create_new(self, folder, public=False, quiet=False):
        self.dataset_calls.append(("new", public, {p.name: p.read_text() for p in Path(folder).iterdir()}))
        return SimpleNamespace(error=self.create_error)

    def dataset_create_version(self, folder, notes, quiet=False, delete_old_versions=False):
        self.dataset_calls.append(("version", delete_old_versions, {p.name: p.read_text() for p in Path(folder).iterdir()}))
        return SimpleNamespace(error=None)


# --- the Kaggle-side script's pure helpers ------------------------------------------------------

def test_find_secrets_searches_the_mounted_datasets(tmp_path):
    nested = tmp_path / "datasets" / "jugal" / "sherpa-server-secrets"
    nested.mkdir(parents=True)
    (nested / "sherpa_secrets.json").write_text(json.dumps(
        {"ngrok_authtoken": "t", "ngrok_url": "https://x.ngrok-free.dev", "basic_auth": "u:p"}))
    assert server.find_secrets(str(tmp_path))["basic_auth"] == "u:p"


def test_find_secrets_missing_file_explains_the_fix(tmp_path):
    with pytest.raises(SystemExit, match="upload-secrets"):
        server.find_secrets(str(tmp_path))


@pytest.mark.parametrize("secrets, message", [
    ({"ngrok_authtoken": "t", "ngrok_url": "https://x"}, "missing: basic_auth"),
    ({"ngrok_authtoken": "t", "ngrok_url": "https://x", "basic_auth": "nocolon"}, "user:password"),
])
def test_find_secrets_validates(tmp_path, secrets, message):
    (tmp_path / "sherpa_secrets.json").write_text(json.dumps(secrets))
    with pytest.raises(SystemExit, match=message):
        server.find_secrets(str(tmp_path))


def test_traffic_policy_requires_auth_then_rewrites_host():
    policy = json.loads(server.traffic_policy('we"ird:pa\\ss'))
    basic, headers = policy["on_http_request"][0]["actions"]
    assert basic == {"type": "basic-auth", "config": {"credentials": ['we"ird:pa\\ss'], "enforce": True}}
    assert headers == {"type": "add-headers", "config": {"headers": {"host": "localhost"}}}


def test_stop_reason():
    start = 1000.0
    assert server.stop_reason([{"name": "qwen2.5:7b"}], start, start + 60, max_hours=3) is None
    assert "idle" in server.stop_reason([], start, start + 60, max_hours=3)
    assert "cap" in server.stop_reason([{"name": "m"}], start, start + 3 * 3600, max_hours=3)


# --- rendering and metadata ------------------------------------------------------------------------

def test_render_fills_the_three_settings():
    text = render_server_script("qwen3:8b", 15, 1.5)
    assert 'MODEL = "qwen3:8b"' in text and "IDLE_MINUTES = 15" in text and "MAX_HOURS = 1.5" in text
    compile(text, "server", "exec")


def test_render_fails_loudly_if_template_changed():
    with pytest.raises(ValueError, match="MODEL"):
        render_server_script("m", 1, 1, template="print('no settings here')")


def test_kernel_metadata(monkeypatch):
    monkeypatch.delenv("KAGGLE_ACCELERATOR", raising=False)
    meta = kernel_metadata("jugal/sherpa-ollama-server", "jugal/sherpa-server-secrets")
    assert meta["title"] == "sherpa-ollama-server" and meta["kernel_type"] == "script"
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["jugal/sherpa-server-secrets"]


# --- server_status ---------------------------------------------------------------------------------

def test_reachable_model_is_ready_even_without_kaggle_settings(monkeypatch):
    monkeypatch.setattr(ks, "model_reachable", lambda timeout=5.0: True)
    assert server_status(api=FakeKaggle()).state == "ready"


def test_not_configured(monkeypatch, unreachable):
    for var in ks.REQUIRED_SETTINGS:
        monkeypatch.delenv(var, raising=False)
    status = server_status(api=FakeKaggle())
    assert status.state == "not_configured" and "KAGGLE_KERNEL_SLUG" in status.detail


@pytest.mark.parametrize("kaggle_status, state", [
    ("QUEUED", "starting"), ("RUNNING", "starting"), ("COMPLETE", "stopped"),
    ("CANCEL_ACKNOWLEDGED", "stopped"), ("ERROR", "error"), ("SOMETHING_NEW", "error"),
])
def test_kernel_status_mapping(settings, unreachable, kaggle_status, state):
    assert server_status(api=FakeKaggle(status=kaggle_status)).state == state


def test_error_includes_kaggle_failure_message(settings, unreachable):
    assert "CUDA out of memory" in server_status(api=FakeKaggle(status="ERROR", failure="CUDA out of memory")).detail


def test_never_started_kernel_is_stopped(settings, unreachable):
    api = FakeKaggle(status_exc=RuntimeError("404 Client Error: Not Found"))
    assert server_status(api=api).state == "stopped"


def test_kernel_that_was_never_pushed_is_stopped_not_an_error(settings, unreachable):
    """Seen live: Kaggle answers 403 for a kernel that doesn't exist yet."""
    exc = ValueError("Cannot access kernel 'jugal/sherpa-ollama-server' (Permission 'kernels.get' was denied). "
                     "The most likely cause is a wrong kernel slug.")
    assert server_status(api=FakeKaggle(status_exc=exc)).state == "stopped"


def test_status_never_raises(settings, unreachable):
    status = server_status(api=FakeKaggle(status_exc=RuntimeError("connection reset")))
    assert status.state == "error" and "connection reset" in status.detail


def test_model_reachable_sends_auth(settings, monkeypatch):
    seen = {}

    def fake_get(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(ks.httpx, "get", fake_get)
    assert ks.model_reachable()
    assert seen["url"] == "https://sherpa.ngrok-free.dev/api/tags"
    assert seen["headers"]["Authorization"].startswith("Basic ")


def test_model_unreachable_on_http_error(settings, monkeypatch):
    def fail(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(ks.httpx, "get", fail)
    assert not ks.model_reachable()


# --- start_server ----------------------------------------------------------------------------------

def test_start_pushes_script_and_metadata(settings, unreachable, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("SERVER_IDLE_MINUTES", "30")
    monkeypatch.setenv("SERVER_MAX_HOURS", "2")
    api = FakeKaggle(status="COMPLETE")
    ok, message = start_server(api=api)
    assert ok and "kaggle.com/code" in message
    files = api.pushed["files"]
    assert set(files) == {"sherpa_ollama_server.py", "kernel-metadata.json"}
    assert "IDLE_MINUTES = 30" in files["sherpa_ollama_server.py"]
    assert json.loads(files["kernel-metadata.json"])["id"] == "jugal/sherpa-ollama-server"
    assert api.pushed["timeout"] == str(2 * 3600 + 900)  # Kaggle's own hard stop
    assert "ngrok-token" not in files["sherpa_ollama_server.py"]  # secrets never go in the code


@pytest.mark.parametrize("kaggle_status", ["QUEUED", "RUNNING"])
def test_start_refuses_when_already_starting(settings, unreachable, kaggle_status):
    api = FakeKaggle(status=kaggle_status)
    ok, message = start_server(api=api)
    assert not ok and "Already starting" in message and api.pushed is None


def test_start_reports_kaggle_rejection(settings, unreachable):
    ok, message = start_server(api=FakeKaggle(push_error="Phone verification required for GPU"))
    assert not ok and "Phone verification" in message


def test_start_without_settings(monkeypatch, unreachable):
    monkeypatch.delenv("KAGGLE_KERNEL_SLUG", raising=False)
    ok, message = start_server(api=FakeKaggle())
    assert not ok and "KAGGLE_KERNEL_SLUG" in message


# --- secrets dataset ---------------------------------------------------------------------------------

def test_secrets_payload(settings):
    assert secrets_payload() == {"ngrok_authtoken": "ngrok-token", "ngrok_url": "https://sherpa.ngrok-free.dev",
                                 "basic_auth": "user:pa:ss"}


@pytest.mark.parametrize("var, value, message", [
    ("NGROK_AUTHTOKEN", "", "NGROK_AUTHTOKEN"),
    ("OLLAMA_BASIC_AUTH", "nocolon", "user:password"),
    ("OLLAMA_BASE_URL", "http://localhost:11434", "https://"),
])
def test_secrets_payload_validates(settings, monkeypatch, var, value, message):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=message):
        secrets_payload()


def test_upload_creates_private_dataset(settings):
    api = FakeKaggle()
    upload_secrets(api=api)
    kind, public, files = api.dataset_calls[0]
    assert kind == "new" and public is False
    assert json.loads(files["sherpa_secrets.json"])["ngrok_authtoken"] == "ngrok-token"
    assert json.loads(files["dataset-metadata.json"])["id"] == "jugal/sherpa-server-secrets"


def test_upload_versions_existing_dataset_and_drops_old_versions(settings):
    api = FakeKaggle(create_error="The requested title is already in use")
    upload_secrets(api=api)
    assert [c[0] for c in api.dataset_calls] == ["new", "version"]
    assert api.dataset_calls[1][1] is True  # delete_old_versions: they hold old secrets


# --- a missing Kaggle token must not kill the page (found by rehearsing a Cloud deploy) --------------------------

def exit_on_authenticate(monkeypatch):
    """The real library prints help text, then calls sys.exit() when there is no token."""
    # (import the class, not the module path: the package replaces its `api` submodule with an object)
    from kaggle.api.kaggle_api_extended import KaggleApi

    def authenticate(self):
        print("Authentication required to call the Kaggle API.\n  export KAGGLE_API_TOKEN=...")
        raise SystemExit(1)
    monkeypatch.setattr(KaggleApi, "authenticate", authenticate)


def test_missing_kaggle_token_becomes_an_error_state_not_a_crash(settings, unreachable, monkeypatch, capsys):
    exit_on_authenticate(monkeypatch)
    status = server_status()  # api=None: builds the real client
    assert status.state == "error" and "KAGGLE_API_TOKEN" in status.detail
    assert "Authentication required" not in capsys.readouterr().out  # the library's help text is swallowed


def test_start_without_a_kaggle_token_reports_instead_of_exiting(settings, unreachable, monkeypatch):
    exit_on_authenticate(monkeypatch)
    ok, message = start_server()
    assert not ok and "KAGGLE_API_TOKEN" in message


def test_upload_secrets_without_a_token_raises_a_normal_error(settings, monkeypatch):
    exit_on_authenticate(monkeypatch)
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        upload_secrets()
