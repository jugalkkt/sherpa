import sys
from types import SimpleNamespace

import pytest
from langgraph.types import Command

import agents.human_approval as approval
import graph.workflow as wf
import mcp_servers.memory_server as mem
from agents.human_approval import human_approval_node
from graph.state import StudyRoadmap, get_current_topic, get_latest_quiz_result, initial_state, session_is_complete
from graph.workflow import build_graph

pytestmark = pytest.mark.unit


def config(thread_id="t1"):
    return {"configurable": {"thread_id": thread_id}}


def start(graph, thread_id="t1"):
    return graph.invoke(initial_state("Learn closures", thread_id), config(thread_id))


# --- human_approval_node on its own -----------------------------------------------------

def test_no_roadmap_approves_without_interrupting():
    # interrupt() outside a running graph would raise, so this proves it isn't called.
    assert human_approval_node(initial_state("g", "s")) == {"approved": True}


@pytest.mark.parametrize("decision, approved", [
    ("yes", True), ("y", True), ("OK", True), ("approve", True), ("  Yes \n", True),
    ("no", False), ("", False), ("maybe", False), ("yess", False), (None, False),
])
def test_decision_parsing(sample_state, monkeypatch, decision, approved):
    monkeypatch.setattr(approval, "interrupt", lambda payload: decision)
    assert human_approval_node(sample_state)["approved"] is approved


def test_payload_and_full_state_returned(sample_state, monkeypatch):
    seen = {}
    monkeypatch.setattr(approval, "interrupt", lambda payload: seen.update(payload) or "yes")
    sample_state["quiz_results"] = ["kept"]
    out = human_approval_node(sample_state)
    assert seen["type"] == "roadmap_approval" and "yes" in seen["prompt"]
    assert seen["roadmap"] is sample_state["roadmap"]
    assert set(out) == {"approved", "roadmap", "goal", "session_id", "current_topic_index",
                        "quiz_results", "weak_areas", "study_materials_path", "error"}
    assert out["quiz_results"] == ["kept"] and out["error"] is None


# --- pausing and resuming in the graph ----------------------------------------------------

def test_fresh_run_pauses_at_approval(db_path, mocked_llms):
    graph = build_graph(db_path=db_path)
    result = start(graph)
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "roadmap_approval"
    assert [t.title for t in StudyRoadmap.from_dict(payload["roadmap"]).topics] == [
        "Closures Explained", "Practical Closure Patterns"]
    assert graph.get_state(config()).next == ("human_approval",)
    assert mocked_llms.explainer.invoke.call_count == 0


def test_yes_runs_to_the_end_without_re_planning(db_path, mocked_llms, answers):
    graph = build_graph(db_path=db_path)
    start(graph)
    answers.extend(["good"] * 4)
    result = graph.invoke(Command(resume="yes"), config())
    assert result["approved"] is True and "__interrupt__" not in result
    assert result["current_topic_index"] == 2
    assert graph.get_state(config()).next == ()
    assert mocked_llms.planner.invoke.call_count == 1  # approval re-ran, the planner didn't


def test_no_regenerates_the_roadmap_then_asks_again(db_path, mocked_llms, answers):
    graph = build_graph(db_path=db_path)
    start(graph)
    again = graph.invoke(Command(resume="no"), config())
    assert "__interrupt__" in again
    assert mocked_llms.planner.invoke.call_count == 2
    assert graph.get_state(config()).next == ("human_approval",)
    answers.extend(["good"] * 4)
    assert graph.invoke(Command(resume="yes"), config())["current_topic_index"] == 2


def test_new_graph_instance_resumes_a_pending_approval(db_path, mocked_llms, answers):
    """Simulates quitting while the roadmap is on screen and coming back later."""
    start(build_graph(db_path=db_path))
    restarted = build_graph(db_path=db_path)  # new process, same SQLite file
    assert restarted.get_state(config()).next == ("human_approval",)
    answers.extend(["good"] * 4)
    assert restarted.invoke(Command(resume="yes"), config())["current_topic_index"] == 2


def test_crash_after_explainer_resumes_at_the_quiz(db_path, mocked_llms, answers):
    """The plan's manual check, automated: approve, Ctrl+C during the first quiz,
    then resume with a new graph. The explainer must not run again for topic 1."""
    graph = build_graph(db_path=db_path)
    start(graph)
    answers.append(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        graph.invoke(Command(resume="yes"), config())
    assert mocked_llms.explainer.invoke.call_count == 1

    restarted = build_graph(db_path=db_path)
    snapshot = restarted.get_state(config())
    assert snapshot.next == ("quiz_generator",)
    assert snapshot.values["approved"] is True
    assert get_current_topic(snapshot.values).title == "Closures Explained"

    answers.extend(["good"] * 4)
    result = restarted.invoke(None, config())
    assert result["current_topic_index"] == 2
    assert mocked_llms.explainer.invoke.call_count == 2  # topic 2 only, topic 1 not repeated
    assert len(result["quiz_results"]) == 2


def test_invoke_none_on_a_finished_session_changes_nothing(db_path, mocked_llms, answers):
    graph = build_graph(db_path=db_path)
    start(graph)
    answers.extend(["good"] * 4)
    graph.invoke(Command(resume="yes"), config())
    result = graph.invoke(None, config())
    assert result["current_topic_index"] == 2 and len(result["quiz_results"]) == 2
    assert mocked_llms.planner.invoke.call_count == 1


def test_thread_ids_are_isolated(db_path, mocked_llms, answers):
    graph = build_graph(db_path=db_path)
    start(graph, "alice")
    start(graph, "bob")
    answers.extend(["good"] * 4)
    graph.invoke(Command(resume="yes"), config("alice"))
    assert graph.get_state(config("alice")).next == ()
    assert graph.get_state(config("bob")).next == ("human_approval",)
    assert graph.get_state(config("bob")).values["quiz_results"] == []
    assert mem.memory_list_keys("bob") == []


def test_every_node_writes_a_checkpoint(db_path, mocked_llms, answers):
    graph = build_graph(db_path=db_path)
    start(graph)
    answers.extend(["good"] * 4)
    graph.invoke(Command(resume="yes"), config())
    writers = [s.metadata.get("source") for s in graph.get_state_history(config())]
    assert len(writers) >= 9  # input + planner + approval (x2) + 2 × (explainer, quiz, coach)


def test_whole_session_works_when_checkpoints_return_dicts(db_path, mocked_llms, answers, monkeypatch):
    """Strict mode: no registered types, so state comes back from SQLite as plain
    dicts. Every agent and accessor must cope (the dict-or-dataclass rule)."""
    monkeypatch.setattr(wf, "CHECKPOINT_TYPES", None)
    graph = build_graph(db_path=db_path)
    start(graph)
    assert isinstance(graph.get_state(config()).values["roadmap"], dict)
    answers.extend(["good", "good", "bad", "bad"])
    result = graph.invoke(Command(resume="yes"), config())
    values = graph.get_state(config()).values
    assert result["current_topic_index"] == 2
    assert [t["status"] for t in values["roadmap"]["topics"]] == ["completed", "needs_review"]
    assert get_latest_quiz_result(values).score == pytest.approx(0.2)
    assert session_is_complete(values)


# --- the CLI (main.py) ---------------------------------------------------------------------

@pytest.fixture
def cli(monkeypatch, tmp_path, mocked_llms):
    """Import main.py fresh, with no .env loading and a temp checkpoint DB.

    Calling it again simulates restarting the program.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cli.db"))
    monkeypatch.setattr("uuid.uuid4", lambda: SimpleNamespace(hex="cafe1234" + "0" * 24))

    def load():
        monkeypatch.setattr(wf, "_graph", None)
        monkeypatch.delitem(sys.modules, "main", raising=False)
        import main
        return main
    return load


def test_cli_full_session(cli, answers, capsys):
    answers.extend(["yes", "good", "good", "good", "good"])
    cli().run_session("Learn closures")
    out = capsys.readouterr().out
    assert "session cafe1234" in out
    assert "1. Closures Explained" in out and "Approve this roadmap?" in out
    assert "Quizzes: 2/2 passed, average 90%" in out
    assert answers == []


def test_cli_reject_then_approve(cli, mocked_llms, answers, capsys):
    answers.extend(["no", "y", "good", "good", "good", "good"])
    cli().run_session("Learn closures")
    out = capsys.readouterr().out
    assert out.count("Approve this roadmap?") == 2
    assert mocked_llms.planner.invoke.call_count == 2


def test_cli_ctrl_c_then_resume(cli, mocked_llms, answers, capsys):
    answers.extend(["yes", KeyboardInterrupt()])
    cli().run_session("Learn closures")
    assert "Resume with: python main.py --resume cafe1234" in capsys.readouterr().out

    answers.extend(["good"] * 4)
    cli().run_session("ignored", session_id="cafe1234")  # a fresh import = a restart
    out = capsys.readouterr().out
    assert "resuming" in out and "Quizzes: 2/2 passed" in out
    assert mocked_llms.explainer.invoke.call_count == 2
    assert mocked_llms.planner.invoke.call_count == 1


def test_cli_resume_unknown_session(cli, capsys):
    cli().run_session("ignored", session_id="nope0000")
    assert "No saved session with id 'nope0000'" in capsys.readouterr().out
