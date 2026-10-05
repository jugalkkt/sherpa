import os
import sys
from pathlib import Path

import pytest

# Before anything can import deepeval: otherwise its import looks up the public
# IP and starts Sentry/PostHog telemetry.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLOSURES_NOTE = ROOT / "study_materials" / "sample_notes" / "closures.md"

# Used only if closures.md has been moved or deleted.
_CLOSURES_FALLBACK = """\
# Closures

A closure is a nested function that remembers variables from its enclosing
scope even after the enclosing function has returned.

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment
```
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests that never call an LLM")
    config.addinivalue_line("markers", "eval: LLM-quality evals that need a running Ollama")


@pytest.fixture(autouse=True)
def no_real_ollama(request, monkeypatch):
    """Unit tests must never reach a model: point Ollama at a closed port.

    If a test forgets to mock an LLM, it fails fast with "connection refused"
    instead of silently calling the real (possibly remote) server.
    """
    if request.node.get_closest_marker("eval"):
        return
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("OLLAMA_BASIC_AUTH", raising=False)


@pytest.fixture(autouse=True)
def no_real_langfuse(monkeypatch):
    """Tracing is off in tests unless a test sets the keys itself."""
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


# graph.state lands in Phase 1, so these fixtures import it lazily: test
# collection must not fail before that module exists.

@pytest.fixture
def sample_roadmap():
    from graph.state import StudyRoadmap, Topic

    return StudyRoadmap(
        goal="Learn Python closures",
        total_weeks=1,
        weekly_hours=5,
        topics=[
            Topic(
                title="Closures Explained",
                description="What a closure is and how Python creates one.",
                estimated_minutes=45,
            ),
            Topic(
                title="Practical Closure Patterns",
                description="Counters, factories and callbacks built with closures.",
                estimated_minutes=60,
                prerequisites=["Closures Explained"],
            ),
        ],
    )


@pytest.fixture
def sample_state(sample_roadmap) -> dict:
    from graph.state import initial_state

    state = initial_state(goal=sample_roadmap.goal, session_id="test1234")
    state["roadmap"] = sample_roadmap
    return state


@pytest.fixture
def closures_note_content() -> str:
    try:
        return CLOSURES_NOTE.read_text(encoding="utf-8")
    except OSError:
        return _CLOSURES_FALLBACK


# --- shared fakes for running the whole graph without a model ------------------------

ROADMAP_JSON = """{
  "goal": "Learn closures", "total_weeks": 1, "weekly_hours": 5,
  "topics": [
    {"title": "Closures Explained", "description": "d", "estimated_minutes": 45},
    {"title": "Practical Closure Patterns", "description": "d", "estimated_minutes": 60,
     "prerequisites": ["Closures Explained"]}
  ]
}"""

QUESTIONS = [
    {"question": "Q1?", "expected_answer": "A1", "difficulty": "easy"},
    {"question": "Q2?", "expected_answer": "A2", "difficulty": "hard"},
]


def fake_grade(question, expected, answer):
    """Answer "good" scores 0.9; anything else scores 0.2 with a weak area."""
    if answer == "good":
        return {"correct": True, "score": 0.9, "feedback": "Nice.", "missing_concept": ""}
    return {"correct": False, "score": 0.2, "feedback": "Not quite.", "missing_concept": "nonlocal"}


@pytest.fixture(autouse=True)
def clear_memory():
    """The memory MCP server is an in-process dict: reset it around every test."""
    import mcp_servers.memory_server as mem

    mem._store.clear()
    yield
    mem._store.clear()


@pytest.fixture
def answers(monkeypatch):
    """Feed input() from a queue. Raising an exception from the queue is allowed:
    put an exception instance in it to simulate e.g. Ctrl+C at that prompt."""
    queue: list = []

    def fake_input(_prompt=""):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    return queue


@pytest.fixture
def mocked_llms(monkeypatch):
    """Every agent's model, mocked. Returns the planner and explainer mocks so
    tests can count how often each agent ran."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from langchain_core.messages import AIMessage

    import agents.curriculum_planner as planner
    import agents.explainer as explainer
    import agents.progress_coach as coach
    import agents.quiz_generator as quiz

    planner_llm = MagicMock()
    planner_llm.invoke.return_value = AIMessage(content=ROADMAP_JSON)
    monkeypatch.setattr(planner, "build_planner_llm", lambda: planner_llm)
    explainer_llm = MagicMock()
    explainer_llm.invoke.return_value = AIMessage(content="An explanation.")
    monkeypatch.setattr(explainer, "build_explainer_llm", lambda: explainer_llm)
    monkeypatch.setattr(quiz, "generate_questions", lambda topic, explanation, n=3: QUESTIONS)
    monkeypatch.setattr(quiz, "grade_answer", fake_grade)
    monkeypatch.setattr(coach, "get_coaching_message",
                        lambda topic, score, weak: {"summary": f"Coached {topic}", "encouragement": ""})
    return SimpleNamespace(planner=planner_llm, explainer=explainer_llm)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "nested" / "checkpoints.db")
