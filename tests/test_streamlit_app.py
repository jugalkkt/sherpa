"""The web UI, driven headlessly with Streamlit's AppTest (no browser, no model)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import hosting.kaggle_server as ks
import mcp_servers.memory_server as mem
from hosting.kaggle_server import ServerStatus

pytestmark = pytest.mark.unit

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def server(monkeypatch):
    """Control the model-server panel: server.state = "ready" | "stopped" | ..."""
    class Ctl:
        state = "ready"
        started = []

    monkeypatch.setattr(ks, "server_status", lambda api=None: ServerStatus(Ctl.state, f"({Ctl.state})"))
    monkeypatch.setattr(ks, "start_server", lambda api=None: (Ctl.started.append(1) or True, "Started."))
    return Ctl


@pytest.fixture
def app(monkeypatch, tmp_path, server):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # never read the real .env
    # ...nor the real .streamlit/secrets.toml (it holds the developer's actual password and URLs)
    monkeypatch.setattr("streamlit.runtime.secrets.Secrets.to_dict", lambda self: {})
    monkeypatch.setenv("UI_CHECKPOINT_DB", str(tmp_path / "ui.db"))
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    st.cache_resource.clear()  # fresh graph on the temp DB
    yield lambda: AppTest.from_file(APP, default_timeout=30).run()
    st.cache_resource.clear()


def button(at, label):
    return next(b for b in at.button if label in b.label)


def answer(at, text):
    at.text_area[0].input(text)
    return button(at, "Submit").click().run()


def test_goal_screen_waits_for_the_model_server(app, server):
    server.state = "stopped"
    at = app()
    assert not at.exception
    assert button(at, "Start session").disabled
    assert any("Start the model server" in i.value for i in at.info)
    button(at, "Start model server").click().run()
    assert server.started == [1]


def test_not_configured_disables_start(app, server):
    server.state = "not_configured"
    at = app()
    assert button(at, "Start model server").disabled


def test_full_session_in_the_browser(app, mocked_llms):
    at = app()
    assert not button(at, "Start session").disabled
    at = button(at, "Start session").click().run()

    # roadmap approval
    assert any("Closures Explained" in m.value for m in at.markdown)
    at = button(at, "Looks good").click().run()

    # topic 1: pass both questions
    assert at.title[0].value == "Closures Explained"
    for i, label in enumerate(["Next question", "Finish quiz"]):
        at = answer(at, "good")
        assert at.success, "expected a ✓ result"
        at = button(at, label).click().run()

    # topic 2: fail both questions; the coach's note for topic 1 is shown
    assert at.title[0].value == "Practical Closure Patterns"
    assert any("Coach on Closures Explained" in i.value for i in at.info)
    for label in ["Next question", "Finish quiz"]:
        at = answer(at, "bad")
        assert at.error, "expected a ✗ result"
        at = button(at, label).click().run()

    assert at.title[0].value == "🎉 Session complete"
    page = " ".join(m.value for m in at.markdown)
    assert "✓ **Closures Explained**: 90%" in page
    assert "✗ **Practical Closure Patterns**: 20%" in page
    assert "nonlocal" in page  # weak area
    assert mocked_llms.planner.invoke.call_count == 1
    assert mocked_llms.explainer.invoke.call_count == 2
    assert any(k.startswith("progress_topic_") for s in mem._store.values() for k in s)


def test_reject_roadmap_makes_a_new_one(app, mocked_llms):
    at = button(app(), "Start session").click().run()
    at = button(at, "Make a new one").click().run()
    assert mocked_llms.planner.invoke.call_count == 2
    assert at.title[0].value == "Your roadmap"


def test_code_is_shown_with_the_question(app, mocked_llms, monkeypatch):
    import agents.quiz_generator as quiz
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3, allow_code=True: [
        {"question": "What does this print?", "code": "print(1 + 1)", "expected_answer": "2", "difficulty": "easy"}])
    at = button(app(), "Start session").click().run()
    at = button(at, "Looks good").click().run()
    assert at.code[0].value == "print(1 + 1)"


def test_planner_failure_returns_to_goal_with_the_error(app, monkeypatch):
    from unittest.mock import MagicMock
    import agents.curriculum_planner as planner
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("refused")
    monkeypatch.setattr(planner, "build_planner_llm", lambda: llm)
    at = button(app(), "Start session").click().run()
    assert at.title[0].value == "🏔️ Sherpa"
    assert any("Can't reach Ollama" in e.value for e in at.error)


def test_password_gate(app, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = app()
    assert not [b for b in at.button if "Start session" in b.label]
    at.text_input[0].input("wrong")
    at = button(at, "Enter").click().run()
    assert any("Wrong password" in e.value for e in at.error)
    at.text_input[0].input("s3cret")
    at = button(at, "Enter").click().run()
    assert button(at, "Start session")


def test_warns_when_no_password_is_set(app):
    at = app()
    assert any("No APP_PASSWORD" in c.value for c in at.caption)


# --- show the expected answer, and dispute a grade (fix C) -----------------------------------------------

def reach_first_graded_question(app, answer_text="bad"):
    at = button_click(app(), "Start session")
    at = button_click(at, "Looks good")
    return answer(at, answer_text)


def button_click(at, label):
    return button(at, label).click().run()


def test_expected_answer_is_shown_after_grading(app, mocked_llms, monkeypatch):
    import agents.quiz_generator as quiz
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3, allow_code=True: [
        {"question": "What does this print?", "code": "print(1)", "difficulty": "easy", "verified": "1",
         "expected_answer": "Verified output (from running the code):\n1\n\nWhy: it prints one."}])
    at = reach_first_graded_question(app)
    shown = " ".join(t.value for t in at.text)
    assert "Verified output (from running the code):\n1" in shown and "it prints one" in shown
    assert [e.label for e in at.expander if "Expected answer" in e.label]


def test_disputed_question_does_not_count(app, mocked_llms, answers):
    """Q1 is graded 'bad' (0.2) and disputed; Q2 is graded 'good' (0.9): the score is 0.9, not 0.55."""
    at = reach_first_graded_question(app, "bad")
    at = button_click(at, "This grade is wrong")
    assert any("won't count" in w.value for w in at.warning)
    at = button_click(at, "Next question")
    at = answer(at, "good")
    at = button_click(at, "Finish quiz")

    # topic 2 now: fail both, so the finished topic-1 coach note is on screen
    note = next(i.value for i in at.info if "Coach on Closures Explained" in i.value)
    assert "(90%)" in note, note


def test_dispute_can_be_taken_back(app, mocked_llms):
    at = reach_first_graded_question(app, "bad")
    at = button_click(at, "This grade is wrong")
    at = button_click(at, "Count this question after all")
    assert not at.warning
    assert [b for b in at.button if "This grade is wrong" in b.label]


def test_disputed_weak_area_is_not_reported(app, mocked_llms):
    """Q1 (bad -> weak area 'nonlocal') is disputed, Q2 is good: no weak areas for topic 1."""
    at = reach_first_graded_question(app, "bad")
    at = button_click(at, "This grade is wrong")
    at = button_click(at, "Next question")
    at = answer(at, "good")
    at = button_click(at, "Finish quiz")
    # finish the second topic with all-good answers so only topic-1 weak areas could appear
    for label in ["Next question", "Finish quiz"]:
        at = answer(at, "good")
        at = button_click(at, label)
    assert at.title[0].value == "🎉 Session complete"
    assert not any("Review:" in m.value for m in at.markdown)


# --- settings come from Streamlit secrets when hosted --------------------------------------------------------

def test_streamlit_secrets_become_environment_variables(app, monkeypatch):
    """On Community Cloud there is no .env: the app reads st.secrets into os.environ."""
    import os
    monkeypatch.setattr("streamlit.runtime.secrets.Secrets.to_dict", lambda self: {
        "APP_PASSWORD": "from-secrets", "SERVER_IDLE_MINUTES": 45, "A_TABLE": {"ignored": 1}})
    monkeypatch.delenv("SERVER_IDLE_MINUTES", raising=False)
    at = app()
    assert not at.exception
    assert os.environ["APP_PASSWORD"] == "from-secrets" and os.environ["SERVER_IDLE_MINUTES"] == "45"
    assert "A_TABLE" not in os.environ  # nested tables are not settings
    assert not [b for b in at.button if "Start session" in b.label]  # the password gate is up


def test_real_environment_wins_over_secrets(app, monkeypatch):
    import os
    monkeypatch.setattr("streamlit.runtime.secrets.Secrets.to_dict", lambda self: {"OLLAMA_MODEL": "from-secrets"})
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    app()
    assert os.environ["OLLAMA_MODEL"] == "from-env"


def test_roadmap_from_notes_shows_each_topics_sections(app, tmp_path):
    from graph.state import StudyRoadmap, Topic

    app()  # builds the cached graph on the temp DB, as the fixture intends
    (tmp_path / "agents.md").write_text("# Agents\nAn agent loops.\n## Tools\nTools act.\n")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["screen"] = "approve"
    at.session_state["notes_dir"] = str(tmp_path)
    at.session_state["roadmap"] = StudyRoadmap(goal="label", total_weeks=1, topics=[
        Topic("Agent basics", "What an agent is.", 30, sources=["agents.md#1", "agents.md#2"]),
    ])
    at.run()
    text = "\n".join(m.value for m in at.markdown)
    assert "from: agents.md › Agents, agents.md › Tools" in text
    assert any("Built from your notes" in c.value for c in at.caption)


def test_roadmap_from_the_goal_shows_no_sources(app, mocked_llms):
    at = app()
    at.button[0].click().run()  # Start session
    assert not any("from:" in m.value for m in at.markdown)
    assert not any("Built from your notes" in c.value for c in at.caption)


def test_uploader_hint_shows_the_real_limit_not_streamlits(app):
    at = app()
    css = next(m.value for m in at.markdown if "stFileUploaderDropzoneInstructions" in m.value)
    assert 'content: "Limit 15KB per file, up to 5 files' in css
