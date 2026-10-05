import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import agents.curriculum_planner as planner
from agents.curriculum_planner import curriculum_planner_node, parse_roadmap_json
from graph.state import StudyRoadmap, Topic, initial_state

pytestmark = pytest.mark.unit


def roadmap_dict(**overrides) -> dict:
    data = {
        "goal": "Learn closures",
        "total_weeks": 2,
        "weekly_hours": 4,
        "topics": [
            {"title": "Scope and LEGB", "description": "How names resolve.",
             "estimated_minutes": 45, "prerequisites": [], "status": "pending"},
            {"title": "Closures Explained", "description": "Functions that remember.",
             "estimated_minutes": 60, "prerequisites": ["Scope and LEGB"], "status": "pending"},
        ],
    }
    data.update(overrides)
    return data


def mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=content)
    return llm


# --- parse_roadmap_json ---------------------------------------------------------

def test_valid_json_parses():
    r = parse_roadmap_json(json.dumps(roadmap_dict()))
    assert isinstance(r, StudyRoadmap)
    assert [t.title for t in r.topics] == ["Scope and LEGB", "Closures Explained"]
    assert isinstance(r.topics[0], Topic)
    assert r.topics[1].prerequisites == ["Scope and LEGB"]
    assert r.weekly_hours == 4


@pytest.mark.parametrize("field", ["goal", "total_weeks", "topics"])
def test_missing_top_level_field_raises(field):
    data = roadmap_dict()
    del data[field]
    with pytest.raises(ValueError, match=field):
        parse_roadmap_json(json.dumps(data))


@pytest.mark.parametrize("field", ["title", "description", "estimated_minutes"])
def test_missing_topic_field_raises(field):
    data = roadmap_dict()
    del data["topics"][1][field]
    with pytest.raises(ValueError, match=f"Topic 1.*{field}"):
        parse_roadmap_json(json.dumps(data))


def test_bad_json_raises_with_raw_output():
    raw = "Sure! Here is your roadmap: {goal: oops" + "x" * 500
    with pytest.raises(ValueError, match="invalid JSON") as exc:
        parse_roadmap_json(raw)
    assert "Sure! Here is" in str(exc.value)
    assert "x" * 400 not in str(exc.value)  # raw output is truncated to 300 chars


@pytest.mark.parametrize("topics", [[], "not a list", None])
def test_empty_or_non_list_topics_raises(topics):
    with pytest.raises(ValueError, match="non-empty list"):
        parse_roadmap_json(json.dumps(roadmap_dict(topics=topics)))


def test_non_object_json_raises():
    with pytest.raises(ValueError, match="object"):
        parse_roadmap_json("[1, 2, 3]")


def test_string_ints_are_coerced():
    data = roadmap_dict(total_weeks="3", weekly_hours="6")
    data["topics"][0]["estimated_minutes"] = "45"
    data["topics"][1]["estimated_minutes"] = 60.0
    r = parse_roadmap_json(json.dumps(data))
    assert (r.total_weeks, r.weekly_hours) == (3, 6)
    assert [t.estimated_minutes for t in r.topics] == [45, 60]


def test_non_numeric_int_raises():
    data = roadmap_dict(total_weeks="a few")
    with pytest.raises(ValueError, match="total_weeks"):
        parse_roadmap_json(json.dumps(data))


def test_weekly_hours_defaults_to_5():
    data = roadmap_dict()
    del data["weekly_hours"]
    assert parse_roadmap_json(json.dumps(data)).weekly_hours == 5


def test_status_is_always_pending_and_prereq_string_wrapped():
    data = roadmap_dict()
    data["topics"][0]["status"] = "completed"
    data["topics"][1]["prerequisites"] = "Scope and LEGB"
    r = parse_roadmap_json(json.dumps(data))
    assert {t.status for t in r.topics} == {"pending"}
    assert r.topics[1].prerequisites == ["Scope and LEGB"]


def test_code_fences_are_tolerated():
    raw = "```json\n" + json.dumps(roadmap_dict()) + "\n```"
    assert len(parse_roadmap_json(raw).topics) == 2


# --- curriculum_planner_node ------------------------------------------------------

@pytest.mark.parametrize("goal", ["", "   "])
def test_node_empty_goal_returns_error_without_llm(goal, monkeypatch):
    build = MagicMock()
    monkeypatch.setattr(planner, "build_planner_llm", build)
    out = curriculum_planner_node(initial_state(goal, "s1"))
    assert out == {"error": "No learning goal provided."}
    build.assert_not_called()


def test_node_with_mocked_llm_returns_roadmap(monkeypatch):
    llm = mock_llm(json.dumps(roadmap_dict()))
    monkeypatch.setattr(planner, "build_planner_llm", lambda: llm)
    out = curriculum_planner_node(initial_state("Learn closures", "s1"))
    assert out["error"] is None
    assert isinstance(out["roadmap"], StudyRoadmap) and len(out["roadmap"].topics) == 2
    assert out["messages"] == [llm.invoke.return_value]
    system, human = llm.invoke.call_args.args[0]
    assert "JSON" in system.content and "Learn closures" in human.content


def test_node_parse_failure_returns_error_and_messages(monkeypatch):
    monkeypatch.setattr(planner, "build_planner_llm", lambda: mock_llm("not json"))
    out = curriculum_planner_node(initial_state("Learn closures", "s1"))
    assert "invalid JSON" in out["error"]
    assert "roadmap" not in out
    assert out["messages"][0].content == "not json"


def test_node_llm_exception_returns_error(monkeypatch):
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("Failed to connect to Ollama")
    monkeypatch.setattr(planner, "build_planner_llm", lambda: llm)
    out = curriculum_planner_node(initial_state("Learn closures", "s1"))
    assert "Can't reach Ollama" in out["error"]


def test_planner_llm_is_json_mode_low_temp():
    llm = planner.build_planner_llm()
    assert llm.format == "json" and llm.temperature == 0.1
