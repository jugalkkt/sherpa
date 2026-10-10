import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import agents.curriculum_planner as planner
from agents.curriculum_planner import (
    NOTES_PLANNER_PROMPT,
    curriculum_planner_node,
    fallback_notes_roadmap,
    notes_roadmap_schema,
    parse_notes_roadmap,
    parse_roadmap_json,
    section_list,
)
from notes_sections import load_sections
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


# --- planning from uploaded notes --------------------------------------------------------

@pytest.fixture
def notes(tmp_path):
    """Two files, 6 sections: a.md#1-3, b.md#1-3."""
    (tmp_path / "a.md").write_text("# Agents\nAn agent loops.\n## Tools\nTools act.\n## Memory\nMemory persists.\n")
    (tmp_path / "b.md").write_text("# Planning\nPlans first.\n## Reflection\nCheck work.\n## Evaluation\nMeasure it.\n")
    return tmp_path


def ids_of(roadmap):
    return [t.sources for t in roadmap.topics]


def notes_json(*topics, **extra):
    return json.dumps({"total_weeks": 2, "weekly_hours": 4, **extra, "topics": [
        {"title": f"T{i}", "description": f"D{i}", "estimated_minutes": 45, "sections": list(ids)}
        for i, ids in enumerate(topics)
    ]})


def test_schema_only_allows_real_section_ids_and_the_topic_count(notes):
    ids = [s.id for s in load_sections(notes)]
    schema = notes_roadmap_schema(ids)
    topics = schema["properties"]["topics"]
    assert topics["items"]["properties"]["sections"]["items"]["enum"] == ids == [
        "a.md#1", "a.md#2", "a.md#3", "b.md#1", "b.md#2", "b.md#3"]
    assert (topics["minItems"], topics["maxItems"]) == (4, 6)
    assert notes_roadmap_schema(ids[:2])["properties"]["topics"]["minItems"] == 2


def test_section_list_fits_budget_and_names_every_section(notes):
    text = section_list(load_sections(notes), budget=400)
    assert len(text) <= 400
    assert all(f"{sid} |" in text for sid in ["a.md#1", "a.md#3", "b.md#3"])
    assert "a.md › Tools | Tools act." in text


def test_valid_grouping_is_kept(notes):
    sections = load_sections(notes)
    raw = notes_json(["a.md#1"], ["a.md#2", "a.md#3"], ["b.md#1", "b.md#2"], ["b.md#3"])
    r = parse_notes_roadmap(raw, sections, "my label")
    assert ids_of(r) == [["a.md#1"], ["a.md#2", "a.md#3"], ["b.md#1", "b.md#2"], ["b.md#3"]]
    assert r.goal == "my label" and [t.title for t in r.topics] == ["T0", "T1", "T2", "T3"]
    assert r.total_weeks == 2 and r.weekly_hours == 4


def test_unknown_and_repeated_ids_are_dropped_and_empty_topics_removed(notes):
    raw = notes_json(["a.md#1", "zzz.md#9"], ["a.md#1", "a.md#2"], ["nope"], ["a.md#3"], ["b.md#1", "b.md#2", "b.md#3"])
    r = parse_notes_roadmap(raw, load_sections(notes), "g")
    assert ids_of(r) == [["a.md#1"], ["a.md#2"], ["a.md#3"], ["b.md#1", "b.md#2", "b.md#3"]]


def test_left_out_sections_join_their_nearest_neighbours_topic(notes):
    sections = load_sections(notes)
    # a.md#2 and b.md#3 are missing: each joins the topic of the section before it
    raw = notes_json(["a.md#1"], ["a.md#3"], ["b.md#1"], ["b.md#2"])
    r = parse_notes_roadmap(raw, sections, "g")
    covered = [sid for t in r.topics for sid in t.sources]
    assert sorted(covered) == sorted(s.id for s in sections) and len(covered) == len(set(covered))
    assert ids_of(r) == [["a.md#1", "a.md#2"], ["a.md#3"], ["b.md#1"], ["b.md#2", "b.md#3"]]


def test_section_at_the_start_joins_the_next_topic(notes):
    r = parse_notes_roadmap(notes_json(["a.md#2", "a.md#3"], ["b.md#1"], ["b.md#2"], ["b.md#3"]), load_sections(notes), "g")
    assert ids_of(r)[0] == ["a.md#1", "a.md#2", "a.md#3"]


def test_sections_are_put_in_document_order(notes):
    r = parse_notes_roadmap(notes_json(["a.md#3", "a.md#1"], ["a.md#2"], ["b.md#1"], ["b.md#3", "b.md#2"]), load_sections(notes), "g")
    assert ids_of(r) == [["a.md#1", "a.md#3"], ["a.md#2"], ["b.md#1"], ["b.md#2", "b.md#3"]]


def test_more_than_six_topics_are_cut_and_their_sections_rehomed(tmp_path):
    (tmp_path / "n.md").write_text("".join(f"## S{i}\nbody {i}\n" for i in range(8)))
    sections = load_sections(tmp_path)
    r = parse_notes_roadmap(notes_json(*[[s.id] for s in sections]), sections, "g")
    assert len(r.topics) == 6
    assert ids_of(r)[-1] == ["n.md#6", "n.md#7", "n.md#8"]  # the 7th and 8th join their neighbour


def test_too_few_topics_are_split_to_reach_four(notes):
    r = parse_notes_roadmap(notes_json([s.id for s in load_sections(notes)]), load_sections(notes), "g")
    assert len(r.topics) == 4
    assert [sid for t in r.topics for sid in t.sources] == ["a.md#1", "a.md#2", "a.md#3", "b.md#1", "b.md#2", "b.md#3"]
    assert [t.title for t in r.topics] == ["T0 (1 of 4)", "T0 (2 of 4)", "T0 (3 of 4)", "T0 (4 of 4)"]
    assert r.topics[1].description == "Covers: Tools, Memory."  # split parts list their sections


def test_fewer_than_four_sections_gives_one_topic_per_section(tmp_path):
    (tmp_path / "one.md").write_text("just text, no headings")
    sections = load_sections(tmp_path)
    r = parse_notes_roadmap(notes_json([sections[0].id]), sections, "g")
    assert ids_of(r) == [["one.md#1"]]


@pytest.mark.parametrize("raw", ["not json", "[]", json.dumps({"topics": "x"}), notes_json(["nope"])])
def test_unusable_output_raises(notes, raw):
    with pytest.raises(ValueError):
        parse_notes_roadmap(raw, load_sections(notes), "g")


def test_fallback_splits_in_order_and_covers_everything(notes):
    sections = load_sections(notes)
    r = fallback_notes_roadmap(sections, "g")
    assert ids_of(r) == [["a.md#1"], ["a.md#2", "a.md#3"], ["b.md#1"], ["b.md#2", "b.md#3"]]
    assert r.topics[1].title == "Tools" and r.topics[1].description == "Covers: Tools, Memory."


def test_node_plans_from_notes_without_sending_the_goal(monkeypatch, notes):
    llm = mock_llm(notes_json(["a.md#1"], ["a.md#2", "a.md#3"], ["b.md#1", "b.md#2"], ["b.md#3"]))
    schemas = []
    monkeypatch.setattr(planner, "build_planner_llm", lambda json_schema=None: schemas.append(json_schema) or llm)
    out = curriculum_planner_node(initial_state("SECRET-GOAL-TEXT", "s1", study_materials_path=str(notes)))
    assert out["error"] is None and out["roadmap"].goal == "SECRET-GOAL-TEXT"
    system, human = llm.invoke.call_args.args[0]
    assert system.content == NOTES_PLANNER_PROMPT.format(min_topics=4, max_topics=6)
    assert "SECRET-GOAL-TEXT" not in system.content + human.content
    assert "a.md#1 | a.md › Agents" in human.content
    assert schemas[0]["properties"]["topics"]["items"]["properties"]["sections"]["items"]["enum"][0] == "a.md#1"


def test_node_falls_back_when_the_notes_output_is_unusable(monkeypatch, notes):
    monkeypatch.setattr(planner, "build_planner_llm", lambda json_schema=None: mock_llm("not json"))
    out = curriculum_planner_node(initial_state("g", "s1", study_materials_path=str(notes)))
    assert out["error"] is None and len(out["roadmap"].topics) == 4


def test_node_with_empty_notes_folder_returns_error(monkeypatch, tmp_path):
    build = MagicMock()
    monkeypatch.setattr(planner, "build_planner_llm", build)
    out = curriculum_planner_node(initial_state("g", "s1", study_materials_path=str(tmp_path)))
    assert "no text" in out["error"]
    build.assert_not_called()


def test_node_without_uploads_still_plans_from_the_goal(monkeypatch):
    llm = mock_llm(json.dumps(roadmap_dict()))
    monkeypatch.setattr(planner, "build_planner_llm", lambda: llm)
    curriculum_planner_node(initial_state("Learn sorting", "s1"))
    _, human = llm.invoke.call_args.args[0]
    assert human.content == "Learning goal: Learn sorting"
