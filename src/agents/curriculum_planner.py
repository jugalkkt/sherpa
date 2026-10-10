"""Curriculum Planner: turns a learning goal into a StudyRoadmap.

Single JSON-mode call at low temperature. Parsing lives in
``parse_roadmap_json`` so it can be tested without an LLM.

When the learner uploaded notes (state["study_materials_path"]), the call
also carries an outline of them (``notes_digest``), and the topics must come
from the notes: the Explainer can only teach what they cover. The outline is
capped so it fits the hosted model's 8,192-token context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from graph.state import StudyRoadmap, Topic
from llm import build_llm, describe_llm_error
from mcp_servers.filesystem_server import list_study_files, read_study_file, use_notes_dir

PLANNER_SYSTEM_PROMPT = """\
You are a curriculum planner. Turn the learner's goal into a study roadmap.

Respond with a single JSON object and nothing else: no prose, no markdown,
no code fences. Use exactly this schema:

{
  "goal": "<the learner's goal, restated>",
  "total_weeks": <integer 1-12>,
  "weekly_hours": <integer 3-10>,
  "topics": [
    {
      "title": "<3-6 words>",
      "description": "<one sentence>",
      "estimated_minutes": <integer 30-120>,
      "prerequisites": ["<exact title of an EARLIER topic>"],
      "status": "pending"
    }
  ]
}

Rules:
- Include 4 to 6 topics, ordered from foundational to advanced.
- "prerequisites" may only contain titles of topics that appear earlier in
  the list, copied exactly. The first topic has no prerequisites.
- Every number is a JSON integer, not a string.
"""

NOTES_RULE = """\
The learner uploaded their own notes; an outline of them follows. Build the
topics from what the notes cover, in service of the goal. Do not add topics
the notes don't cover."""

# About 1,500-2,000 tokens, leaving room for the prompt and the roadmap JSON.
NOTES_DIGEST_CHARS = 6000

REQUIRED_KEYS = ("goal", "total_weeks", "topics")
REQUIRED_TOPIC_KEYS = ("title", "description", "estimated_minutes")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _headings(text: str) -> list[str]:
    """Markdown heading lines, skipping code blocks (where "# ..." is a comment)."""
    found, in_code = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        elif not in_code and re.match(r"#{1,6}\s", stripped):
            found.append(stripped)
    return found


def notes_digest(base: str | Path, budget_chars: int = NOTES_DIGEST_CHARS) -> str:
    """An outline of the notes in `base`: per file, its headings, then its opening
    text. The budget is split evenly across files, so no file crowds out the rest."""
    with use_notes_dir(base):
        names = list_study_files()
        texts = [read_study_file(name) for name in names]
    if not names:
        return ""
    per_file = budget_chars // len(names)
    sections = []
    for name, text in zip(names, texts):
        if text.startswith("Error:"):
            continue
        section = f"--- {name} ---\n" + "\n".join(_headings(text))
        room = per_file - len(section) - len("\nStart of the file:\n")
        if room > 0:
            section += "\nStart of the file:\n" + text.strip()[:room]
        sections.append(section[:per_file])
    return "\n\n".join(sections)


def build_planner_llm() -> ChatOllama:
    return build_llm(temperature=0.1, json_mode=True)


def _to_int(value, field: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Field '{field}' must be an integer, got {value!r}.") from None


def parse_roadmap_json(raw: str) -> StudyRoadmap:
    """Parse and validate the planner's JSON. Raises ValueError with a helpful message."""
    # JSON mode should prevent fences, but small models add them anyway.
    text = _FENCE.sub("", (raw or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON ({exc.msg}). Raw output: {raw[:300]!r}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Planner JSON must be an object, got {type(data).__name__}.")

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Planner JSON is missing required field(s): {', '.join(missing)}.")

    raw_topics = data["topics"]
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("Planner JSON 'topics' must be a non-empty list.")

    topics = []
    for i, t in enumerate(raw_topics):
        if not isinstance(t, dict):
            raise ValueError(f"Topic {i} must be an object, got {type(t).__name__}.")
        missing = [k for k in REQUIRED_TOPIC_KEYS if k not in t]
        if missing:
            raise ValueError(f"Topic {i} ({t.get('title', '?')!r}) is missing: {', '.join(missing)}.")
        prereqs = t.get("prerequisites") or []
        if isinstance(prereqs, str):
            prereqs = [prereqs]
        topics.append(Topic(
            title=str(t["title"]).strip(),
            description=str(t["description"]).strip(),
            estimated_minutes=_to_int(t["estimated_minutes"], f"topics[{i}].estimated_minutes"),
            prerequisites=[str(p) for p in prereqs],
            status="pending",  # a new roadmap always starts fresh, whatever the LLM says
        ))

    return StudyRoadmap(
        goal=str(data["goal"]),
        total_weeks=_to_int(data["total_weeks"], "total_weeks"),
        weekly_hours=_to_int(data.get("weekly_hours", 5), "weekly_hours"),
        topics=topics,
    )


def curriculum_planner_node(state: dict) -> dict:
    """Generate a study roadmap for the goal.

    Reads: goal, study_materials_path
    Writes: roadmap, messages, error
    """
    goal = (state.get("goal") or "").strip()
    if not goal:
        return {"error": "No learning goal provided."}

    request = f"Learning goal: {goal}"
    uploaded = state.get("study_materials_path") or ""
    digest = notes_digest(uploaded) if uploaded else ""
    if digest:
        request += f"\n\n{NOTES_RULE}\n\n{digest}"

    print(f"[Planner] Building a roadmap for: {goal}" + (" (from the learner's notes)" if digest else ""))
    try:
        response = build_planner_llm().invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=request),
        ])
    except Exception as exc:
        error = describe_llm_error(exc)
        print(f"[Planner] {error}")
        return {"error": error}

    try:
        roadmap = parse_roadmap_json(response.content)
    except ValueError as exc:
        print(f"[Planner] {exc}")
        return {"error": str(exc), "messages": [response]}

    print(f"[Planner] Roadmap ready: {len(roadmap.topics)} topics over {roadmap.total_weeks} week(s).")
    return {"roadmap": roadmap, "messages": [response], "error": None}
