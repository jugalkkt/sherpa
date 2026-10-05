"""Human Approval: pauses the graph so the learner can accept or reject the roadmap.

``interrupt()`` saves the run to the checkpointer and returns control to the
caller with the payload. When the caller resumes with
``graph.invoke(Command(resume=answer), config)``, this node runs again FROM
THE TOP and ``interrupt()`` returns ``answer`` this time. So nothing with a
side effect may come before ``interrupt()``: it would happen twice.
"""

from __future__ import annotations

from langgraph.types import interrupt

APPROVE_WORDS = {"yes", "y", "ok", "approve"}


def human_approval_node(state: dict) -> dict:
    """Ask the learner to approve the roadmap.

    Reads: roadmap (+ every key it passes through)
    Writes: approved, error (+ roadmap, goal, session_id, current_topic_index,
        quiz_results, weak_areas, study_materials_path, passed through)
    """
    roadmap = state.get("roadmap")
    if not roadmap:
        return {"approved": True}

    decision = interrupt({
        "type": "roadmap_approval",
        "roadmap": roadmap,
        "prompt": "Approve this roadmap? Type yes to start, or no for a new one.",
    })
    approved = str(decision).lower().strip() in APPROVE_WORDS
    print(f"[Approval] Roadmap {'approved' if approved else 'rejected, generating a new one'}.")

    # The plan returns the full state here, defensively (it reports keys going
    # missing after Command(resume=...) on langgraph 1.1.0). See DEVIATIONS.md.
    return {
        "approved": approved,
        "roadmap": roadmap,
        "goal": state.get("goal", ""),
        "session_id": state.get("session_id", ""),
        "current_topic_index": state.get("current_topic_index", 0),
        "quiz_results": state.get("quiz_results") or [],
        "weak_areas": state.get("weak_areas") or [],
        "study_materials_path": state.get("study_materials_path", ""),
        "error": None,
    }
