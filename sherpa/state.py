"""Graph state. Values are JSON-serializable (dicts/lists/primitives) so the SQLite
checkpointer never has to deserialize custom classes; use `schemas` models to validate."""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from sherpa.schemas import Topic

OPEN_STATUSES = ("pending", "in_progress")


class SherpaState(TypedDict, total=False):
    # session
    session_id: str
    goal: str
    learner_level: str  # "beginner" | "intermediate" | "advanced"
    note_titles: list[str]  # snapshot of indexed docs for the planner

    # approval loop
    curriculum: list[dict]  # list[Topic]
    curriculum_rationale: str
    approval_status: Literal["pending", "approved", "rejected"]
    approval_feedback: str | None
    revision_count: int

    # study loop
    current_topic_idx: int
    focus_areas: list[str]  # set by progress_coach on retry; [] on a fresh topic
    explanation: str
    sources: list[dict]  # list[Source]
    quiz: list[dict]  # list[MCQ]
    answers: list[int]  # learner's chosen option indices, aligned with quiz
    last_grade: dict | None  # GradeReport
    asked_questions: dict[str, list[str]]  # topic_id -> question stems already asked
    coach_message: str

    # append-only audit log
    events: Annotated[list[dict], operator.add]


def current_topic(state: SherpaState) -> Topic:
    return Topic.model_validate(state["curriculum"][state["current_topic_idx"]])


def next_open_topic_idx(curriculum: list[dict]) -> int | None:
    """Index of the first topic still pending or in progress, or None when all are done."""
    for i, t in enumerate(curriculum):
        if t.get("status", "pending") in OPEN_STATUSES:
            return i
    return None


def log_event(node: str, type: str, **data: Any) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node": node,
        "type": type,
        "data": data,
    }
