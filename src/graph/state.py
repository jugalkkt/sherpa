"""Shared state for the Sherpa LangGraph workflow.

Every node reads from and writes partial updates to ``AgentState``. The
dataclasses here are the domain model; the helpers at the bottom are the
only sanctioned way to read them out of state.

Dict-or-dataclass tolerance: after a SQLite checkpoint round-trip, the
dataclasses below may come back as plain dicts. Every helper (and every
``from_dict``) therefore accepts either form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

PASS_SCORE = 0.5
STRONG_PASS_SCORE = 0.8

# Topic status lifecycle: pending → in_progress → completed | needs_review
TOPIC_STATUSES = ("pending", "in_progress", "completed", "needs_review")
FINISHED_STATUSES = frozenset({"completed", "needs_review"})


def _known_fields(cls: type, data: dict) -> dict:
    """Drop keys the dataclass doesn't define, so stray keys never crash."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Topic:
    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Topic | dict) -> Topic:
        if isinstance(data, cls):
            return data
        d = _known_fields(cls, data)
        d["prerequisites"] = list(d.get("prerequisites") or [])
        return cls(**d)


@dataclass
class StudyRoadmap:
    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5

    def completed_count(self) -> int:
        return sum(1 for t in self.topics if t.status == "completed")

    def is_complete(self) -> bool:
        """True when every topic is finished (completed or needs_review)."""
        return all(t.status in FINISHED_STATUSES for t in self.topics)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: StudyRoadmap | dict) -> StudyRoadmap:
        if isinstance(data, cls):
            return data
        d = _known_fields(cls, data)
        d["topics"] = [Topic.from_dict(t) for t in d.get("topics") or []]
        return cls(**d)


@dataclass
class QuizQuestion:
    question: str
    expected_answer: str
    user_answer: str = ""
    correct: bool = False
    feedback: str = ""
    score: float = 0.0
    disputed: bool = False  # the learner says the question or its grade is wrong: it doesn't count

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: QuizQuestion | dict) -> QuizQuestion:
        if isinstance(data, cls):
            return data
        return cls(**_known_fields(cls, data))


@dataclass
class QuizResult:
    topic: str
    questions: list[QuizQuestion]
    score: float
    weak_areas: list[str]
    timestamp: str = ""

    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    def strong_pass(self) -> bool:
        return self.score >= STRONG_PASS_SCORE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: QuizResult | dict) -> QuizResult:
        if isinstance(data, cls):
            return data
        d = _known_fields(cls, data)
        d["questions"] = [QuizQuestion.from_dict(q) for q in d.get("questions") or []]
        d["weak_areas"] = list(d.get("weak_areas") or [])
        return cls(**d)


class AgentState(TypedDict):
    # add_messages is the only reducer: nodes return *new* messages and they
    # are appended. Every other key is overwritten by whatever a node returns.
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    goal: str
    roadmap: StudyRoadmap | None
    approved: bool
    current_topic_index: int
    quiz_results: list[QuizResult]
    weak_areas: list[str]
    # Folder of notes the learner uploaded (web app). "" means none: the
    # Explainer reads NOTES_PATH and the Planner plans from the goal alone.
    study_materials_path: str
    error: str | None


def initial_state(
    goal: str,
    session_id: str,
    study_materials_path: str = "",
) -> dict:
    """The ONLY way to create a fresh state. Sets every AgentState key."""
    return {
        "messages": [],
        "session_id": session_id,
        "goal": goal,
        "roadmap": None,
        "approved": False,
        "current_topic_index": 0,
        "quiz_results": [],
        "weak_areas": [],
        "study_materials_path": study_materials_path,
        "error": None,
    }


def _get_roadmap(state: dict) -> StudyRoadmap | None:
    roadmap: Any = state.get("roadmap")
    if not roadmap:
        return None
    return StudyRoadmap.from_dict(roadmap)


def get_current_topic(state: dict) -> Topic | None:
    roadmap = _get_roadmap(state)
    if roadmap is None:
        return None
    idx = state.get("current_topic_index", 0)
    if not 0 <= idx < len(roadmap.topics):
        return None
    return roadmap.topics[idx]


def get_latest_quiz_result(state: dict) -> QuizResult | None:
    results = state.get("quiz_results") or []
    if not results:
        return None
    return QuizResult.from_dict(results[-1])


def session_is_complete(state: dict) -> bool:
    """True when there's no roadmap or the index has moved past the last topic."""
    roadmap = _get_roadmap(state)
    if roadmap is None:
        return True
    return state.get("current_topic_index", 0) >= len(roadmap.topics)
