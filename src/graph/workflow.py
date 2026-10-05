"""The Sherpa LangGraph: nodes, edges, routing and the SQLite checkpointer."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.curriculum_planner import curriculum_planner_node
from agents.explainer import explainer_node
from agents.human_approval import human_approval_node
from agents.progress_coach import progress_coach_node
from agents.quiz_generator import quiz_generator_node
from graph.state import AgentState, session_is_complete

DEFAULT_DB_PATH = "data/checkpoints.db"

# Our dataclasses, registered so checkpoints restore them as dataclasses
# instead of warning (or, in strict mode, returning dicts). See DEVIATIONS.md.
CHECKPOINT_TYPES = [
    ("graph.state", name) for name in ("Topic", "StudyRoadmap", "QuizQuestion", "QuizResult")
]


# --- routing: pure Python, never an LLM decision -------------------------------

def route_after_planner(state: dict) -> str:
    # A failed plan ends the run, so its error is what the user sees instead
    # of being overwritten by later nodes that have nothing to work on.
    return "end" if state.get("error") else "human_approval"


def route_after_approval(state: dict) -> str:
    return "explainer" if state.get("approved") else "curriculum_planner"


def route_after_coach(state: dict) -> str:
    return "end" if session_is_complete(state) else "explainer"


# --- graph ----------------------------------------------------------------------

def build_checkpointer(db_path: str) -> SqliteSaver:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # Not a context manager: the connection must outlive this function.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def build_graph(db_path: str | None = None, interrupt_before: list[str] | None = None):
    db_path = db_path or os.getenv("CHECKPOINT_DB") or DEFAULT_DB_PATH

    builder = StateGraph(AgentState)
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    builder.add_edge(START, "curriculum_planner")
    builder.add_conditional_edges(
        "curriculum_planner",
        route_after_planner,
        {"human_approval": "human_approval", "end": END},
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"explainer": "explainer", "curriculum_planner": "curriculum_planner"},
    )
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {"explainer": "explainer", "end": END},
    )

    return builder.compile(
        checkpointer=build_checkpointer(db_path),
        interrupt_before=interrupt_before or [],
    )


_graph = None


def __getattr__(name: str):
    """`from graph.workflow import graph` builds the default graph on first use.

    Lazy (PEP 562) so that importing this module, e.g. in tests, doesn't open
    data/checkpoints.db as a side effect.
    """
    global _graph
    if name == "graph":
        if _graph is None:
            _graph = build_graph()
        return _graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
