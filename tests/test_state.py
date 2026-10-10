import pytest

from graph.state import (
    AgentState,
    QuizQuestion,
    QuizResult,
    StudyRoadmap,
    Topic,
    get_current_topic,
    get_latest_quiz_result,
    initial_state,
    session_is_complete,
)

pytestmark = pytest.mark.unit


def make_result(score: float, topic: str = "Closures Explained") -> QuizResult:
    return QuizResult(
        topic=topic,
        questions=[QuizQuestion(question="Q?", expected_answer="A", user_answer="A", correct=True, score=score)],
        score=score,
        weak_areas=["nonlocal"],
        timestamp="2026-10-02T12:00:00",
    )


# --- round-trips -------------------------------------------------------------

def test_topic_round_trip():
    t = Topic("Closures", "desc", 45, ["Basics"], "in_progress")
    assert Topic.from_dict(t.to_dict()) == t


def test_topic_defaults():
    t = Topic.from_dict({"title": "T", "description": "d", "estimated_minutes": 30})
    assert t.prerequisites == [] and t.status == "pending"


def test_topic_default_prerequisites_not_shared():
    a = Topic("A", "d", 30)
    b = Topic("B", "d", 30)
    a.prerequisites.append("X")
    assert b.prerequisites == []


def test_roadmap_round_trip_converts_nested_topics(sample_roadmap):
    d = sample_roadmap.to_dict()
    assert isinstance(d["topics"][0], dict)
    back = StudyRoadmap.from_dict(d)
    assert back == sample_roadmap
    assert all(isinstance(t, Topic) for t in back.topics)


def test_roadmap_from_dict_defaults_weekly_hours():
    r = StudyRoadmap.from_dict({"goal": "g", "total_weeks": 2, "topics": []})
    assert r.weekly_hours == 5


def test_quiz_question_round_trip():
    q = QuizQuestion("Q?", "A", "my answer", True, "good", 0.9)
    assert QuizQuestion.from_dict(q.to_dict()) == q


def test_quiz_result_round_trip_converts_nested_questions():
    r = make_result(0.7)
    back = QuizResult.from_dict(r.to_dict())
    assert back == r
    assert isinstance(back.questions[0], QuizQuestion)


def test_from_dict_passes_through_instances(sample_roadmap):
    assert StudyRoadmap.from_dict(sample_roadmap) is sample_roadmap


def test_from_dict_ignores_unknown_keys():
    t = Topic.from_dict({"title": "T", "description": "d", "estimated_minutes": 30, "extra": 1})
    assert t.title == "T"


# --- roadmap progress --------------------------------------------------------

def test_completed_count_counts_only_completed(sample_roadmap):
    sample_roadmap.topics[0].status = "completed"
    sample_roadmap.topics[1].status = "needs_review"
    assert sample_roadmap.completed_count() == 1


def test_is_complete_accepts_needs_review(sample_roadmap):
    assert not sample_roadmap.is_complete()
    sample_roadmap.topics[0].status = "completed"
    assert not sample_roadmap.is_complete()
    sample_roadmap.topics[1].status = "needs_review"
    assert sample_roadmap.is_complete()


# --- quiz thresholds ---------------------------------------------------------

@pytest.mark.parametrize(
    "score, passed, strong",
    [(0.0, False, False), (0.49, False, False), (0.5, True, False),
     (0.79, True, False), (0.8, True, True), (1.0, True, True)],
)
def test_pass_thresholds(score, passed, strong):
    r = make_result(score)
    assert r.passed() is passed
    assert r.strong_pass() is strong


# --- initial_state -----------------------------------------------------------

def test_initial_state_has_every_key():
    state = initial_state("goal", "abc12345")
    assert set(state) == set(AgentState.__annotations__)


def test_initial_state_values():
    state = initial_state("goal", "abc12345")
    assert state["goal"] == "goal" and state["session_id"] == "abc12345"
    assert state["roadmap"] is None and state["approved"] is False
    assert state["current_topic_index"] == 0 and state["error"] is None
    assert state["messages"] == [] and state["quiz_results"] == [] and state["weak_areas"] == []
    assert state["study_materials_path"] == ""


def test_initial_state_lists_not_shared():
    a = initial_state("g", "a")
    b = initial_state("g", "b")
    a["weak_areas"].append("x")
    assert b["weak_areas"] == []


# --- helpers, dataclass and dict roadmaps ------------------------------------

@pytest.fixture(params=["dataclass", "dict"])
def state_any_form(request, sample_state):
    """The same state, with the roadmap either as a dataclass or as a plain dict."""
    if request.param == "dict":
        sample_state["roadmap"] = sample_state["roadmap"].to_dict()
    return sample_state


def test_get_current_topic(state_any_form):
    topic = get_current_topic(state_any_form)
    assert isinstance(topic, Topic) and topic.title == "Closures Explained"
    state_any_form["current_topic_index"] = 1
    assert get_current_topic(state_any_form).prerequisites == ["Closures Explained"]


@pytest.mark.parametrize("idx", [2, 99, -1])
def test_get_current_topic_out_of_range(state_any_form, idx):
    state_any_form["current_topic_index"] = idx
    assert get_current_topic(state_any_form) is None


def test_get_current_topic_without_roadmap():
    assert get_current_topic(initial_state("g", "s")) is None


@pytest.mark.parametrize("idx, complete", [(0, False), (1, False), (2, True), (3, True)])
def test_session_is_complete_boundaries(state_any_form, idx, complete):
    state_any_form["current_topic_index"] = idx
    assert session_is_complete(state_any_form) is complete


def test_session_is_complete_without_roadmap():
    assert session_is_complete(initial_state("g", "s")) is True


@pytest.mark.parametrize("form", ["dataclass", "dict"])
def test_get_latest_quiz_result(sample_state, form):
    results = [make_result(0.4, "First"), make_result(0.9, "Second")]
    sample_state["quiz_results"] = results if form == "dataclass" else [r.to_dict() for r in results]
    latest = get_latest_quiz_result(sample_state)
    assert isinstance(latest, QuizResult)
    assert latest.topic == "Second" and latest.strong_pass()


def test_get_latest_quiz_result_empty(sample_state):
    assert get_latest_quiz_result(sample_state) is None


# --- real checkpoint serializer ----------------------------------------------

def test_helpers_survive_strict_checkpoint_round_trip(sample_state):
    """In strict mode, unregistered dataclasses deserialize as plain dicts.

    allowed_msgpack_modules=None is what LANGGRAPH_STRICT_MSGPACK=true selects
    (the env var itself is read at import time, so it can't be monkeypatched).
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    sample_state["quiz_results"] = [make_result(0.6)]
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    back = serde.loads_typed(serde.dumps_typed(sample_state))

    assert isinstance(back["roadmap"], dict)
    assert get_current_topic(back).title == "Closures Explained"
    assert get_latest_quiz_result(back).passed()
    assert not session_is_complete(back)


def test_topic_sources_round_trip_and_old_dicts_still_load():
    t = Topic("T", "d", 30, sources=["a.md#1", "a.md#2"])
    assert Topic.from_dict(t.to_dict()) == t
    assert Topic.from_dict({"title": "T", "description": "d", "estimated_minutes": 30}).sources == []
