import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agents.explainer as explainer
import mcp_servers.memory_server as mem
from agents.explainer import (
    EXPLAINER_TOOLS,
    MAX_ITERATIONS,
    TOOL_MAP,
    build_explainer_llm,
    execute_tool_call,
    explainer_node,
    tool_list_files,
    tool_memory_get,
    tool_memory_set,
    tool_read_file,
    tool_search_notes,
)
from graph.state import initial_state

pytestmark = pytest.mark.unit


class ScriptedLLM:
    """Returns scripted responses in order and snapshots what it was sent each call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))  # copy: the node keeps appending to the list
        return self.responses.pop(0)


def call(name, args=None, id_="call-1"):
    return {"name": name, "args": args or {}, "id": id_, "type": "tool_call"}



@pytest.fixture
def use_llm(monkeypatch):
    def install(llm):
        monkeypatch.setattr(explainer, "build_explainer_llm", lambda: llm)
        return llm
    return install


# --- tool wrappers ----------------------------------------------------------------

def test_tool_names_the_model_sees():
    assert set(TOOL_MAP) == {"list_files", "read_file", "search_notes", "memory_get", "memory_set"}
    assert all(t.description for t in EXPLAINER_TOOLS)


def test_wrappers_delegate_to_mcp_functions():
    assert "closures.md" in tool_list_files.invoke({})
    assert tool_read_file.invoke({"filename": "closures.md"}).startswith("# Closures")
    assert "access denied" in tool_read_file.invoke({"filename": "../../.env"})
    tool_memory_set.invoke({"session_id": "s1", "key": "k", "value": "v"})
    assert tool_memory_get.invoke({"session_id": "s1", "key": "k"}) == "v"


def test_search_wrapper_returns_json_or_no_matches():
    hits = json.loads(tool_search_notes.invoke({"query": "nonlocal"}))
    assert any(h["file"] == "closures.md" for h in hits)
    assert tool_search_notes.invoke({"query": "zzz-not-there"}) == "No matches found."


def test_llm_is_bound_to_all_tools():
    bound = build_explainer_llm()
    assert [t["function"]["name"] for t in bound.kwargs["tools"]] == list(TOOL_MAP)
    assert bound.bound.temperature == 0.3


# --- execute_tool_call ----------------------------------------------------------------

def test_execute_unknown_tool():
    out = execute_tool_call(call("delete_everything"))
    assert out.startswith("Error: unknown tool 'delete_everything'") and "read_file" in out


def test_execute_list_result_is_json():
    assert "closures.md" in json.loads(execute_tool_call(call("list_files")))


def test_execute_bad_args_returns_error_string():
    out = execute_tool_call(call("read_file", {}))  # missing filename
    assert out.startswith("Error executing read_file: ValidationError")


def test_execute_exception_returns_error_string(monkeypatch):
    def boom(*_):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(explainer, "list_study_files", boom)
    assert execute_tool_call(call("list_files")) == "Error executing list_files: RuntimeError: disk on fire"


# --- explainer_node ---------------------------------------------------------------------

def test_direct_answer_without_tools(sample_state, use_llm):
    llm = use_llm(ScriptedLLM(AIMessage(content="**Analogy**: a backpack.")))
    out = explainer_node(sample_state)
    assert out["error"] is None
    system, human, answer = out["messages"]
    assert isinstance(system, SystemMessage) and "memory_get" in system.content
    assert isinstance(human, HumanMessage)
    assert "Closures Explained" in human.content and "test1234" in human.content
    assert answer.content == "**Analogy**: a backpack."
    assert len(llm.calls) == 1


def test_one_tool_round_then_answer(sample_state, use_llm):
    llm = use_llm(ScriptedLLM(
        AIMessage(content="", tool_calls=[call("read_file", {"filename": "closures.md"}, id_="abc-123")]),
        AIMessage(content="Final explanation."),
    ))
    out = explainer_node(sample_state)
    assert out["error"] is None
    tool_msg = out["messages"][3]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "abc-123"  # must match the request
    assert tool_msg.content.startswith("# Closures")
    # the second LLM call saw the tool result
    assert llm.calls[1][-1] is tool_msg
    assert out["messages"][-1].content == "Final explanation."


def test_parallel_tool_calls_each_get_a_matching_result(sample_state, use_llm):
    use_llm(ScriptedLLM(
        AIMessage(content="", tool_calls=[
            call("list_files", id_="a"),
            call("search_notes", {"query": "nonlocal"}, id_="b"),
        ]),
        AIMessage(content="Done."),
    ))
    out = explainer_node(sample_state)
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]


def test_tool_error_is_fed_back_not_raised(sample_state, use_llm):
    use_llm(ScriptedLLM(
        AIMessage(content="", tool_calls=[call("read_file", {"filename": "generators.md"})]),
        AIMessage(content="Recovered."),
    ))
    out = explainer_node(sample_state)
    assert out["error"] is None
    assert "Available files" in out["messages"][3].content


def test_max_iterations_returns_error(sample_state, use_llm):
    loop = AIMessage(content="", tool_calls=[call("list_files")])
    llm = use_llm(ScriptedLLM(*[loop] * (MAX_ITERATIONS + 5)))
    out = explainer_node(sample_state)
    assert out["error"] == f"Explainer reached max iterations ({MAX_ITERATIONS})."
    assert len(llm.calls) == MAX_ITERATIONS


def test_no_topic_returns_error(use_llm):
    llm = use_llm(ScriptedLLM())
    out = explainer_node(initial_state("goal", "s1"))
    assert out == {"error": "Explainer: no current topic to explain."}
    assert llm.calls == []


def test_index_past_end_returns_error(sample_state, use_llm):
    use_llm(ScriptedLLM())
    sample_state["current_topic_index"] = 2
    assert "no current topic" in explainer_node(sample_state)["error"]


def test_dict_roadmap_is_handled(sample_state, use_llm):
    use_llm(ScriptedLLM(AIMessage(content="ok")))
    sample_state["roadmap"] = sample_state["roadmap"].to_dict()
    sample_state["current_topic_index"] = 1
    out = explainer_node(sample_state)
    assert "Practical Closure Patterns" in out["messages"][1].content


def test_llm_failure_returns_error(sample_state):
    # No mock: conftest points Ollama at a closed port, so the real client fails.
    out = explainer_node(sample_state)
    assert "Can't reach Ollama" in out["error"]


def test_explained_topics_recorded_in_memory(sample_state, use_llm):
    use_llm(ScriptedLLM(AIMessage(content="one"), AIMessage(content="two"), AIMessage(content="again")))
    explainer_node(sample_state)
    sample_state["current_topic_index"] = 1
    explainer_node(sample_state)
    sample_state["current_topic_index"] = 0
    explainer_node(sample_state)  # re-explaining doesn't duplicate
    assert mem.memory_get("test1234", "explained_topics") == "Closures Explained, Practical Closure Patterns"


def test_nothing_recorded_on_failure(sample_state, use_llm):
    use_llm(ScriptedLLM(*[AIMessage(content="", tool_calls=[call("list_files")])] * MAX_ITERATIONS))
    explainer_node(sample_state)
    assert mem.memory_get("test1234", "explained_topics") == "null"


def test_uploaded_notes_replace_the_default_folder(sample_state, use_llm, tmp_path):
    (tmp_path / "sorting.md").write_text("# Sorting\n")
    llm = use_llm(ScriptedLLM(
        AIMessage(content="", tool_calls=[call("list_files")]),
        AIMessage(content="Final explanation."),
    ))
    out = explainer_node({**sample_state, "study_materials_path": str(tmp_path)})
    assert out["error"] is None
    assert json.loads(out["messages"][3].content) == ["sorting.md"]
    assert "closures.md" in tool_list_files.invoke({})  # only for that call
    assert len(llm.calls) == 2
