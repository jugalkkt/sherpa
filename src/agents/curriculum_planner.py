"""Curriculum Planner: turns a learning goal into a StudyRoadmap.

Single JSON-mode call at low temperature. Parsing lives in
``parse_roadmap_json`` so it can be tested without an LLM.

When the learner uploaded notes (state["study_materials_path"]), the roadmap
comes from the notes alone and the goal is only its label. The model groups
the notes' sections (see notes_sections.py) into topics:
  - the JSON schema lets each topic name only real section ids (an enum);
  - ``parse_notes_roadmap`` then drops unknown and repeated ids, and adds any
    section the model left out to the topic of its nearest neighbour, so
    every section is taught;
  - if the output is unusable, ``fallback_notes_roadmap`` splits the
    sections in document order instead.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from graph.state import StudyRoadmap, Topic
from llm import build_llm, describe_llm_error
from notes_sections import Section, load_sections

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

NOTES_PLANNER_PROMPT = """\
You are a curriculum planner. The learner uploaded their own study notes,
split into sections. Group the sections into a study roadmap that teaches
exactly what the notes contain.

Rules:
- Make {min_topics} to {max_topics} topics, ordered from foundational to advanced.
- Every section id must be used, in exactly one topic.
- Group sections that belong together; keep a topic's sections in the same
  order as in the notes where you can.
- Title (3-6 words) and description (one sentence) must describe what those
  sections say. Do not add anything the notes don't cover.
- "sections" lists the ids exactly as given, e.g. "closures.md#3".

Respond with a single JSON object and nothing else:
{{"total_weeks": <integer 1-12>, "weekly_hours": <integer 3-10>,
  "topics": [{{"title": "...", "description": "...", "estimated_minutes": <integer 30-120>,
              "sections": ["<section id>", ...]}}]}}
"""

MAX_NOTES_TOPICS = 6
MIN_NOTES_TOPICS = 4
# Budget for the section list sent to the planner: about 2,000 tokens.
NOTES_LIST_CHARS = 6000
MAX_PREVIEW_CHARS = 200

REQUIRED_KEYS = ("goal", "total_weeks", "topics")
REQUIRED_TOPIC_KEYS = ("title", "description", "estimated_minutes")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def build_planner_llm(json_schema: dict | None = None) -> ChatOllama:
    if json_schema is not None:
        return build_llm(temperature=0.1, json_schema=json_schema)
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


# --- planning from uploaded notes ------------------------------------------------------

def _clamp(value, low: int, high: int, default: int) -> int:
    try:
        return min(high, max(low, int(float(value))))
    except (TypeError, ValueError):
        return default


def topic_count_range(n_sections: int) -> tuple[int, int]:
    """4-6 topics, or fewer when the notes have fewer than 4 sections."""
    return min(MIN_NOTES_TOPICS, n_sections), min(MAX_NOTES_TOPICS, n_sections)


def notes_roadmap_schema(section_ids: list[str]) -> dict:
    """Ollama structured output: a topic can only name sections that exist,
    and the number of topics must be in range (the model sometimes made one
    topic per file instead)."""
    low, high = topic_count_range(len(section_ids))
    return {
        "type": "object",
        "properties": {
            "total_weeks": {"type": "integer"},
            "weekly_hours": {"type": "integer"},
            "topics": {
                "type": "array",
                "minItems": low,
                "maxItems": high,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "estimated_minutes": {"type": "integer"},
                        "sections": {"type": "array", "minItems": 1,
                                     "items": {"type": "string", "enum": section_ids}},
                    },
                    "required": ["title", "description", "estimated_minutes", "sections"],
                },
            },
        },
        "required": ["total_weeks", "weekly_hours", "topics"],
    }


def section_list(sections: list[Section], budget: int = NOTES_LIST_CHARS) -> str:
    """One line per section for the planner: id, file and heading, then the start of its text."""
    heads = [f"{s.id} | {s.label}" for s in sections]
    room = budget - sum(len(h) + 4 for h in heads)
    preview_len = max(0, min(MAX_PREVIEW_CHARS, room // max(1, len(sections))))
    lines = []
    for head, s in zip(heads, sections):
        body = s.text.split("\n", 1)[1] if s.text.startswith("#") and "\n" in s.text else s.text
        preview = " ".join(body.split())[:preview_len]
        lines.append(f"{head} | {preview}" if preview else head)
    return "\n".join(lines)


def _covers(ids: list[str], by_id: dict[str, Section]) -> str:
    return "Covers: " + ", ".join(by_id[i].heading for i in ids) + "."


def _notes_topic(title: str, description: str, minutes, ids: list[str], by_id: dict[str, Section]) -> Topic:
    return Topic(
        title=title or by_id[ids[0]].heading,
        description=description or _covers(ids, by_id),
        estimated_minutes=_clamp(minutes, 30, 120, default=_clamp(15 * len(ids), 30, 120, 45)),
        sources=list(ids),
    )


def parse_notes_roadmap(raw: str, sections: list[Section], goal: str) -> StudyRoadmap:
    """Turn the model's grouping into a roadmap that covers every section exactly once.

    Raises ValueError when nothing usable came back (the caller then falls back).
    """
    text = _FENCE.sub("", (raw or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON ({exc.msg}).") from None
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise ValueError("Planner JSON has no 'topics' list.")

    position = {s.id: i for i, s in enumerate(sections)}
    by_id = {s.id: s for s in sections}
    groups: list[dict] = []
    used: set[str] = set()
    for t in data["topics"]:
        if not isinstance(t, dict):
            continue
        ids = []
        for sid in t.get("sections") or []:
            sid = str(sid)
            if sid in position and sid not in used:  # unknown ids dropped; a repeat keeps its first topic
                ids.append(sid)
                used.add(sid)
        if ids:
            groups.append({"title": str(t.get("title") or "").strip(),
                           "description": str(t.get("description") or "").strip(),
                           "minutes": t.get("estimated_minutes"), "ids": ids})
    if not groups:
        raise ValueError("Planner's topics named none of the sections.")

    low, high = topic_count_range(len(sections))
    groups = groups[:high]  # sections of dropped topics are re-homed below

    # Every section left out joins the topic of its nearest neighbour in the
    # notes: the previous section first, else the next one.
    owner = {sid: g for g, group in enumerate(groups) for sid in group["ids"]}
    order = [s.id for s in sections]
    for pos, sid in enumerate(order):
        if sid in owner:
            continue
        before = next((owner[order[j]] for j in range(pos - 1, -1, -1) if order[j] in owner), None)
        after = next((owner[order[j]] for j in range(pos + 1, len(order)) if order[j] in owner), None)
        g = before if before is not None else after
        owner[sid] = g
        groups[g]["ids"].append(sid)
    for group in groups:
        group["ids"].sort(key=position.__getitem__)

    # Too few topics (the schema asks for the minimum, but check): split the
    # biggest one in half, in document order, until the minimum is met. The
    # parts are numbered: "Agentic AI (1 of 2)", "Agentic AI (2 of 2)".
    while len(groups) < low:
        g = max(range(len(groups)), key=lambda i: len(groups[i]["ids"]))
        ids = groups[g]["ids"]
        if len(ids) < 2:
            break
        half = len(ids) // 2
        part = {**groups[g], "title": groups[g]["title"] or by_id[ids[0]].heading,
                "description": "", "minutes": None, "split": True}
        groups[g:g + 1] = [{**part, "ids": ids[:half]}, {**part, "ids": ids[half:]}]
    totals = Counter(g["title"] for g in groups if g.get("split"))
    seen: Counter[str] = Counter()
    for group in groups:
        if group.pop("split", False):
            title = group["title"]
            seen[title] += 1
            group["title"] = f"{title} ({seen[title]} of {totals[title]})"

    return StudyRoadmap(
        goal=goal,
        total_weeks=_clamp(data.get("total_weeks"), 1, 12, 1),
        weekly_hours=_clamp(data.get("weekly_hours"), 3, 10, 5),
        topics=[_notes_topic(g["title"], g["description"], g["minutes"], g["ids"], by_id) for g in groups],
    )


def fallback_notes_roadmap(sections: list[Section], goal: str) -> StudyRoadmap:
    """No usable model output: split the sections, in document order, into evenly sized topics."""
    n = len(sections)
    low, high = topic_count_range(n)
    k = max(low, min(high, math.ceil(n / 3)))
    by_id = {s.id: s for s in sections}
    chunks = [sections[i * n // k:(i + 1) * n // k] for i in range(k)]
    return StudyRoadmap(
        goal=goal, total_weeks=1, weekly_hours=5,
        topics=[_notes_topic("", "", None, [s.id for s in chunk], by_id) for chunk in chunks],
    )


def _plan_from_notes(goal: str, notes_dir: str) -> dict:
    sections = load_sections(notes_dir)
    if not sections:
        return {"error": "Your uploaded notes have no text to plan from."}
    low, high = topic_count_range(len(sections))
    print(f"[Planner] Building a roadmap from your notes ({len(sections)} sections).")
    try:
        response = build_planner_llm(notes_roadmap_schema([s.id for s in sections])).invoke([
            SystemMessage(content=NOTES_PLANNER_PROMPT.format(min_topics=low, max_topics=high)),
            HumanMessage(content=f"Sections of the learner's notes:\n{section_list(sections)}"),
        ])
    except Exception as exc:
        error = describe_llm_error(exc)
        print(f"[Planner] {error}")
        return {"error": error}

    try:
        roadmap = parse_notes_roadmap(response.content, sections, goal)
    except ValueError as exc:
        print(f"[Planner] {exc} Splitting the notes in order instead.")
        roadmap = fallback_notes_roadmap(sections, goal)

    print(f"[Planner] Roadmap ready: {len(roadmap.topics)} topics covering all {len(sections)} sections.")
    return {"roadmap": roadmap, "messages": [response], "error": None}


def curriculum_planner_node(state: dict) -> dict:
    """Generate a study roadmap: from the uploaded notes if there are any, else from the goal.

    Reads: goal, study_materials_path
    Writes: roadmap, messages, error
    """
    goal = (state.get("goal") or "").strip()
    if not goal:
        return {"error": "No learning goal provided."}
    uploaded = state.get("study_materials_path") or ""
    if uploaded:
        return _plan_from_notes(goal, uploaded)  # the goal is only the roadmap's label here

    print(f"[Planner] Building a roadmap for: {goal}")
    try:
        response = build_planner_llm().invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Learning goal: {goal}"),
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
