import sys
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import Command

import observability.langfuse_setup as lf
from graph.state import initial_state
from graph.workflow import build_graph
from observability.langfuse_setup import (
    _langfuse_configured,
    flush_langfuse,
    get_langfuse_config,
    get_langfuse_handler,
)

pytestmark = pytest.mark.unit

REAL_REACHABLE = lf._langfuse_reachable  # captured before the autouse fixture patches it


@pytest.fixture(autouse=True)
def server_up(monkeypatch):
    """No real health checks in unit tests: assume the server answers unless a test says otherwise."""
    monkeypatch.setattr(lf, "_langfuse_reachable", lambda host: True)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")


class FakeHandler:
    instances = []

    def __init__(self, *args, **kwargs):
        # The real 4.x constructor rejects session_id, tags, etc.
        assert not args and not kwargs, "CallbackHandler must be built with no arguments"
        FakeHandler.instances.append(self)


@pytest.fixture
def fake_handler(monkeypatch):
    FakeHandler.instances = []
    import langfuse.langchain
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", FakeHandler)
    return FakeHandler


def must_not_be_called(*_a, **_k):
    raise AssertionError("should not be called when Langfuse is not configured")


# --- _langfuse_configured -----------------------------------------------------------

def test_configured_needs_both_keys(monkeypatch, keys):
    assert _langfuse_configured()
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    assert not _langfuse_configured()


def test_unconfigured_by_default():
    assert not _langfuse_configured()


@pytest.mark.parametrize("public, secret", [("  ", "sk"), ("pk", "\t\n"), ("", "")])
def test_whitespace_only_keys_count_as_unconfigured(monkeypatch, public, secret):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret)
    assert not _langfuse_configured()


# --- get_langfuse_handler -------------------------------------------------------------

def test_no_handler_when_unconfigured(monkeypatch):
    import langfuse.langchain
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", must_not_be_called)
    assert get_langfuse_handler() is None


def test_handler_built_without_arguments(keys, fake_handler):
    handler = get_langfuse_handler()
    assert isinstance(handler, FakeHandler) and len(fake_handler.instances) == 1


def test_import_error_returns_none_with_hint(keys, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "langfuse.langchain", None)  # makes the import fail
    assert get_langfuse_handler() is None
    assert "langchain" in capsys.readouterr().out


def test_constructor_failure_returns_none(keys, monkeypatch, capsys):
    import langfuse.langchain

    def broken(**_):
        raise ValueError("bad host")
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", broken)
    assert get_langfuse_handler() is None
    assert "Continuing without tracing" in capsys.readouterr().out


def test_unreachable_server_means_no_handler(keys, monkeypatch, capsys):
    import langfuse.langchain
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", must_not_be_called)
    monkeypatch.setattr(lf, "_langfuse_reachable", lambda host: False)
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")
    assert get_langfuse_handler() is None
    assert "http://localhost:3000 isn't answering" in capsys.readouterr().out


def test_health_check_hits_the_health_endpoint(monkeypatch):
    seen = []
    monkeypatch.setattr(lf.httpx, "get", lambda url, timeout: seen.append((url, timeout)))
    assert REAL_REACHABLE("http://localhost:3000") is True
    assert seen == [("http://localhost:3000/api/public/health", lf.HEALTH_TIMEOUT)]


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_health_check_failure(monkeypatch, error):
    def fail(url, timeout):
        raise error
    monkeypatch.setattr(lf.httpx, "get", fail)
    assert REAL_REACHABLE("http://localhost:3000") is False


def test_host_precedence(monkeypatch):
    assert lf._langfuse_host() == "https://cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")
    assert lf._langfuse_host() == "http://localhost:3000"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://other:3000")
    assert lf._langfuse_host() == "http://other:3000"


def test_real_handler_constructs_with_env_only(keys, monkeypatch):
    """The installed langfuse accepts our zero-argument construction (gotcha #4)."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    from langfuse.langchain import CallbackHandler
    assert isinstance(get_langfuse_handler(), CallbackHandler)


# --- get_langfuse_config ------------------------------------------------------------------

def test_config_unconfigured_has_only_thread_id(capsys):
    assert get_langfuse_config("abc12345") == {"configurable": {"thread_id": "abc12345"}}
    assert "Tracing] Off" in capsys.readouterr().out


def test_config_configured_has_callbacks_and_metadata(keys, fake_handler, capsys):
    config = get_langfuse_config("abc12345", user_id="jugal")
    assert config["configurable"] == {"thread_id": "abc12345"}
    assert config["callbacks"] == fake_handler.instances
    assert config["metadata"] == {
        "langfuse_session_id": "abc12345",
        "langfuse_user_id": "jugal",
        "langfuse_tags": ["sherpa", "local-inference"],
    }
    assert "Tracing] On" in capsys.readouterr().out


def test_config_reports_the_host(keys, fake_handler, monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    get_langfuse_config("s")
    assert "http://localhost:3000" in capsys.readouterr().out


def test_config_default_user_id(keys, fake_handler):
    assert get_langfuse_config("s")["metadata"]["langfuse_user_id"] == "local"


def test_extra_config_merges_without_dropping_thread_id():
    config = get_langfuse_config("s1", extra_config={"configurable": {"checkpoint_ns": ""}, "recursion_limit": 50})
    assert config["configurable"] == {"thread_id": "s1", "checkpoint_ns": ""}
    assert config["recursion_limit"] == 50


def test_handler_failure_still_gives_a_working_config(keys, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "langfuse.langchain", None)
    assert get_langfuse_config("s1") == {"configurable": {"thread_id": "s1"}}
    out = capsys.readouterr().out
    assert "set LANGFUSE_PUBLIC_KEY" not in out  # keys ARE set; don't tell the user to set them


# --- flush_langfuse -------------------------------------------------------------------

def test_flush_is_a_noop_when_unconfigured(monkeypatch):
    import langfuse
    monkeypatch.setattr(langfuse, "get_client", must_not_be_called)
    flush_langfuse()


def test_flush_calls_client_when_configured(keys, monkeypatch):
    import langfuse
    flushed = []
    monkeypatch.setattr(langfuse, "get_client", lambda: SimpleNamespace(flush=lambda: flushed.append(True)))
    flush_langfuse()
    assert flushed == [True]


def test_flush_never_raises(keys, monkeypatch):
    import langfuse

    def boom():
        raise ConnectionError("langfuse down")
    monkeypatch.setattr(langfuse, "get_client", boom)
    flush_langfuse()


# --- callbacks reach the whole graph ---------------------------------------------------------

class Recorder(BaseCallbackHandler):
    """A plain LangChain handler: the same hook Langfuse's handler plugs into."""

    def __init__(self):
        self.started: list[str] = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name:
            self.started.append(name)


def test_callbacks_in_config_see_every_node(db_path, mocked_llms, answers, monkeypatch):
    """Agents never import tracing, yet every node is visible to the handler."""
    recorder = Recorder()
    monkeypatch.setattr(lf, "get_langfuse_handler", lambda: recorder)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    config = get_langfuse_config("t1")

    graph = build_graph(db_path=db_path)
    graph.invoke(initial_state("Learn closures", "t1"), config)
    answers.extend(["good"] * 4)
    graph.invoke(Command(resume="yes"), config)

    for node in ("curriculum_planner", "human_approval", "explainer", "quiz_generator", "progress_coach"):
        assert node in recorder.started, node
    assert recorder.started.count("explainer") == 2
