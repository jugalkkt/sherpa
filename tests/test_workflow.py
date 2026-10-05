from unittest.mock import MagicMock

import pytest
from langgraph.types import Command

import agents.curriculum_planner as planner
import mcp_servers.memory_server as mem
from graph.state import StudyRoadmap, initial_state
from graph.workflow import build_graph, route_after_approval, route_after_coach, route_after_planner

pytestmark = pytest.mark.unit


def config(thread_id="t1"):
    return {"configurable": {"thread_id": thread_id}}


# --- routing -------------------------------------------------------------------

@pytest.mark.parametrize("error, expected", [(None, "human_approval"), ("boom", "end")])
def test_route_after_planner(error, expected):
    assert route_after_planner({"error": error}) == expected


@pytest.mark.parametrize("approved, expected", [(True, "explainer"), (False, "curriculum_planner")])
def test_route_after_approval(approved, expected):
    assert route_after_approval({"approved": approved}) == expected


def test_route_after_coach(sample_state):
    assert route_after_coach(sample_state) == "explainer"
    sample_state["current_topic_index"] = 2
    assert route_after_coach(sample_state) == "end"


def test_route_after_coach_dict_roadmap(sample_state):
    sample_state["roadmap"] = sample_state["roadmap"].to_dict()
    sample_state["current_topic_index"] = 1
    assert route_after_coach(sample_state) == "explainer"


# --- graph ---------------------------------------------------------------------

def test_graph_has_the_five_nodes(db_path):
    nodes = set(build_graph(db_path=db_path).get_graph().nodes)
    assert {"curriculum_planner", "human_approval", "explainer",
            "quiz_generator", "progress_coach"} <= nodes


def test_creates_db_directory_and_honours_env(tmp_path, monkeypatch):
    path = tmp_path / "from_env" / "cp.db"
    monkeypatch.setenv("CHECKPOINT_DB", str(path))
    build_graph()
    assert path.exists()


def run_approved(graph, thread_id="t1"):
    """Start a session, then approve the roadmap when the graph pauses for it."""
    first = graph.invoke(initial_state("Learn closures", thread_id), config(thread_id))
    assert "__interrupt__" in first
    return graph.invoke(Command(resume="yes"), config(thread_id))


def test_full_session_end_to_end(db_path, mocked_llms, answers, capsys):
    answers[:] = ["good", "good", "bad", "bad"]  # pass topic 1, fail topic 2
    result = run_approved(build_graph(db_path=db_path))

    assert result["error"] is None and result["approved"] is True
    assert result["current_topic_index"] == 2
    assert [t.status for t in result["roadmap"].topics] == ["completed", "needs_review"]
    assert [round(r.score, 2) for r in result["quiz_results"]] == [0.9, 0.2]
    assert [r.topic for r in result["quiz_results"]] == ["Closures Explained", "Practical Closure Patterns"]
    assert result["weak_areas"] == ["nonlocal"]
    assert mem.memory_list_keys("t1") == ["explained_topics", "progress_topic_0", "progress_topic_1"]
    assert answers == []  # every question was asked

    out = capsys.readouterr().out
    assert out.count("[Explainer] Topic:") == 2
    assert "Session complete: 1/2 topics passed" in out


def test_state_is_checkpointed_and_readable_from_a_new_graph(db_path, mocked_llms, answers):
    answers.extend(["good"] * 4)
    run_approved(build_graph(db_path=db_path))
    snapshot = build_graph(db_path=db_path).get_state(config())
    assert snapshot.next == ()  # finished
    roadmap = snapshot.values["roadmap"]
    assert isinstance(roadmap, StudyRoadmap)  # registered type survives SQLite
    assert roadmap.topics[1].prerequisites == ["Closures Explained"]


def test_planner_failure_does_not_crash_the_graph(db_path, monkeypatch):
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("refused")
    monkeypatch.setattr(planner, "build_planner_llm", lambda: llm)
    graph = build_graph(db_path=db_path)
    result = graph.invoke(initial_state("Learn closures", "t1"), config())
    assert "Can't reach Ollama" in result["error"]  # the real cause survives
    assert result["roadmap"] is None
    assert result["approved"] is False  # stopped before approval
    assert graph.get_state(config()).next == ()


def test_interrupt_before_pauses_there(db_path, mocked_llms):
    graph = build_graph(db_path=db_path, interrupt_before=["explainer"])
    run_approved(graph)
    assert graph.get_state(config()).next == ("explainer",)
    assert mocked_llms.explainer.invoke.call_count == 0
