"""Pydantic models.

Two groups:
- LLM output contracts (`*Draft`, `KeyCheck`, `GradeFeedback`): what the model must return.
  Their JSON Schema is sent to Ollama as `format=` by `invoke_structured`.
- State shapes (`Topic`, `MCQ`, `Source`, `GradeReport`): stored in graph state as plain
  dicts via `.model_dump()`, re-validated with `Model.model_validate(d)` where needed.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from sherpa.config import get_settings

TopicStatus = Literal["pending", "in_progress", "passed", "needs_review"]

_TOPIC_ID = re.compile(r"^t[1-9]\d*$")


def _ctx(info: ValidationInfo, key: str, default: int) -> int:
    return (info.context or {}).get(key, default)


# --------------------------------------------------------------------------- LLM contracts


class TopicDraft(BaseModel):
    id: str = Field(description='Sequential id: "t1", "t2", ...')
    title: str = Field(description="Short topic title, at most 80 characters")
    objectives: list[str] = Field(
        min_length=2, max_length=4, description="2-4 measurable objectives, each starting with an action verb"
    )
    prerequisites: list[str] = Field(default_factory=list, description="ids of EARLIER topics only")
    est_minutes: int = Field(description="Estimated study time, 10-45 minutes")

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not _TOPIC_ID.match(v):
            raise ValueError('topic id must look like "t1", "t2", ...')
        return v

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty")
        return v[:80]

    @field_validator("objectives")
    @classmethod
    def _objectives(cls, v: list[str]) -> list[str]:
        v = [o.strip() for o in v]
        if any(not o for o in v):
            raise ValueError("objectives must not be empty strings")
        return v

    @field_validator("prerequisites")
    @classmethod
    def _prereqs(cls, v: list[str]) -> list[str]:
        return [p.strip().lower() for p in v if p.strip()]

    @field_validator("est_minutes")
    @classmethod
    def _clamp_minutes(cls, v: int) -> int:
        return max(10, min(45, v))


class CurriculumDraft(BaseModel):
    rationale: str = Field(description="1-3 sentences explaining scope and ordering")
    topics: list[TopicDraft]

    @model_validator(mode="after")
    def _check_structure(self, info: ValidationInfo) -> "CurriculumDraft":
        s = get_settings()
        lo = _ctx(info, "min_topics", s.min_topics)
        hi = _ctx(info, "max_topics", s.max_topics)
        n = len(self.topics)
        if not lo <= n <= hi:
            raise ValueError(f"need between {lo} and {hi} topics, got {n}")

        expected = [f"t{i}" for i in range(1, n + 1)]
        actual = [t.id for t in self.topics]
        if actual != expected:
            raise ValueError(f"topic ids must be {expected} in order, got {actual}")

        for idx, t in enumerate(self.topics):
            earlier = set(expected[:idx])
            bad = [p for p in t.prerequisites if p not in earlier]
            if bad:
                raise ValueError(f"{t.id} has prerequisites {bad} that are not earlier topics")
        return self


class MCQDraft(BaseModel):
    objective: str = Field(description="The topic objective this question tests")
    question: str
    correct_answer: str
    distractors: list[str] = Field(min_length=3, max_length=3, description="Exactly 3 plausible wrong options")
    explanation: str = Field(description="1-2 sentences on why the correct answer is right")


class QuizDraft(BaseModel):
    questions: list[MCQDraft] = Field(min_length=1)


class KeyCheck(BaseModel):
    answers: list[Literal["A", "B", "C", "D"]]


class QuestionFeedback(BaseModel):
    question_id: str
    feedback: str = Field(description="1-3 sentences addressed to the learner")
    misconception: str | None = Field(default=None, description="Belief the chosen option suggests, or null")


class GradeFeedback(BaseModel):
    per_question: list[QuestionFeedback] = Field(description="Only for incorrectly answered questions")
    weak_objectives: list[str]
    summary: str


# --------------------------------------------------------------------------- state shapes


class Topic(TopicDraft):
    status: TopicStatus = "pending"
    attempts: int = 0
    scores: list[float] = Field(default_factory=list)


class MCQ(BaseModel):
    id: str  # f"{topic_id}-a{attempt}-q{n}"
    objective: str
    question: str
    options: list[str] = Field(min_length=4, max_length=4)  # shuffled in code
    correct_index: int = Field(ge=0, le=3)  # computed in code after the shuffle
    explanation: str


class Source(BaseModel):
    sid: str  # "S1"
    kind: Literal["note", "web"]
    title: str
    location: str  # file path + page/heading, or URL


class GradeReport(BaseModel):
    topic_id: str
    attempt: int
    score: float = Field(ge=0.0, le=1.0)
    correct: list[str]
    incorrect: list[str]
    feedback: list[QuestionFeedback]
    weak_objectives: list[str]
    misconceptions: list[str]
    summary: str
