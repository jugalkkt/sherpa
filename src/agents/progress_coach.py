"""Progress Coach: gives feedback, marks the topic, saves progress, moves on.

Routing (next topic vs. end) is decided in Python by ``route_after_coach``
from the ``current_topic_index`` this node writes, never by the LLM.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.quiz_generator import extract_explanation
from graph.state import PASS_SCORE, QuizResult, StudyRoadmap, get_latest_quiz_result
from llm import build_llm
from mcp_servers.memory_server import memory_set

PASS_THRESHOLD = PASS_SCORE

COACHING_PROMPT = """\
You are a warm, practical study coach. The learner just took a quiz.

Write a short, specific message:
- Name the topic and how they did.
- If there are weak areas, name them and suggest one concrete way to review.
- Never be discouraging; a low score is information, not failure.

Respond with a single JSON object and nothing else:
{"summary": "<2-3 sentences of specific feedback>", "encouragement": "<1 short sentence>"}
"""


def _fallback_coaching(topic: str, score: float, weak_areas: list[str]) -> dict:
    if score >= PASS_THRESHOLD:
        summary = f"You scored {score:.0%} on {topic}. Nicely done."
    else:
        summary = f"You scored {score:.0%} on {topic}. It's worth another look before moving on."
    if weak_areas:
        summary += f" Review: {', '.join(weak_areas)}."
    return {"summary": summary, "encouragement": "Keep going, every topic builds on the last."}


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> dict:
    """Return {summary, encouragement}. Falls back to a template on any failure."""
    llm = build_llm(temperature=0.4, json_mode=True)
    try:
        response = llm.invoke([
            SystemMessage(content=COACHING_PROMPT),
            HumanMessage(content=(
                f"Topic: {topic}\nScore: {score:.0%}\n"
                f"Weak areas: {', '.join(weak_areas) if weak_areas else 'none'}"
            )),
        ])
        data = json.loads(response.content)
        summary, encouragement = str(data["summary"]).strip(), str(data["encouragement"]).strip()
        if not summary:
            raise ValueError("empty summary")
        return {"summary": summary, "encouragement": encouragement}
    except Exception as exc:
        print(f"[Coach] Using a template message ({type(exc).__name__}).")
        return _fallback_coaching(topic, score, weak_areas)


def try_study_buddy_assistance(topic: str, explanation: str, weak_areas: list[str]) -> str | None:
    """Extra help from the CrewAI Study Buddy over A2A. Placeholder until Phase 9."""
    return None


def progress_coach_node(state: dict) -> dict:
    """Coach the learner on the latest quiz and advance to the next topic.

    Reads: quiz_results, roadmap, current_topic_index, session_id, messages, error
    Writes: roadmap, current_topic_index, messages, error
    """
    latest = get_latest_quiz_result(state)
    if latest is None or not state.get("roadmap"):
        return {"error": "Coach: no quiz result or roadmap to coach on."}

    # Work on a copy: never mutate objects that are still part of the old state.
    roadmap = copy.deepcopy(StudyRoadmap.from_dict(state["roadmap"]))
    idx = state.get("current_topic_index", 0)
    if not 0 <= idx < len(roadmap.topics):
        return {"error": f"Coach: topic index {idx} is out of range."}
    session_id = state.get("session_id", "")

    coaching = get_coaching_message(latest.topic, latest.score, latest.weak_areas)

    # Update THIS topic before moving the index. Doing it after would mark the
    # next topic instead (the ordering bug the tests guard against).
    status = "completed" if latest.score >= PASS_THRESHOLD else "needs_review"
    roadmap.topics[idx].status = status
    next_idx = idx + 1

    memory_set(session_id, f"progress_topic_{idx}", json.dumps({
        "topic": latest.topic,
        "score": latest.score,
        "weak_areas": latest.weak_areas,
        "status": status,
        "timestamp": latest.timestamp or datetime.now(timezone.utc).isoformat(),
    }))

    print(f"\n🧭 Coach: {coaching['summary']}")
    if coaching["encouragement"]:
        print(f"   {coaching['encouragement']}")
    if next_idx < len(roadmap.topics):
        print(f"\nNext up: {roadmap.topics[next_idx].title} ({next_idx + 1}/{len(roadmap.topics)})")
    else:
        results = [QuizResult.from_dict(r) for r in state.get("quiz_results") or []]
        passed = sum(r.passed() for r in results)
        avg = sum(r.score for r in results) / len(results)
        print(f"\n🎉 Session complete: {passed}/{len(results)} topics passed, average {avg:.0%}.")

    if latest.score < PASS_THRESHOLD and latest.weak_areas:
        try:
            help_text = try_study_buddy_assistance(latest.topic, extract_explanation(state), latest.weak_areas)
        except Exception:
            help_text = None  # extra help is optional; never let it break coaching
        if help_text:
            print(f"\n🤝 Study Buddy:\n{help_text}")

    return {
        "roadmap": roadmap,
        "current_topic_index": next_idx,
        "messages": [AIMessage(content=coaching["summary"])],
        "error": None,
    }
