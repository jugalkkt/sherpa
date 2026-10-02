import sys
from pathlib import Path

import pytest

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
