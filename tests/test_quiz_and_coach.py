import copy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents.progress_coach as coach
import agents.quiz_generator as quiz
import mcp_servers.memory_server as mem
from agents.progress_coach import get_coaching_message, progress_coach_node
from agents.quiz_generator import (
    GENERATION_PROMPT,
    NO_ANSWER,
    dedupe,
    extract_explanation,
    generate_questions,
    grade_answer,
    quiz_generator_node,
    run_quiz,
)
from graph.state import QuizQuestion, QuizResult, StudyRoadmap, initial_state

pytestmark = pytest.mark.unit


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=self.content if isinstance(self.content, str) else json.dumps(self.content))


@pytest.fixture
def llm_returns(monkeypatch):
    """llm_returns(module, content) makes that module's build_llm return a FakeLLM."""
    def install(module, content):
        fake = FakeLLM(content)
        monkeypatch.setattr(module, "build_llm", lambda **_: fake)
        return fake
    return install




def result(score, weak=(), topic="Closures Explained"):
    return QuizResult(topic=topic, questions=[QuizQuestion("q", "a", score=score)],
                      score=score, weak_areas=list(weak), timestamp="2026-10-04T10:00:00+00:00")


# --- generate_questions -------------------------------------------------------------

def test_generation_prompt_formats_with_literal_braces():
    prompt = GENERATION_PROMPT.format(n=4)
    assert "exactly 4 questions" in prompt
    assert '{"questions": [{"question"' in prompt


def test_generate_parses_caps_and_normalises(llm_returns):
    llm = llm_returns(quiz, {"questions": [
        {"question": "Why nonlocal?", "expected_answer": "To rebind.", "difficulty": "HARD"},
        {"question": "Q2", "expected_answer": "A2", "difficulty": "impossible"},
        {"question": "Q3", "expected_answer": "A3"},
        {"question": "Q4", "expected_answer": "A4"},
    ]})
    qs = generate_questions("Closures", "explanation text", n=3)
    assert len(qs) == 3
    assert qs[0] == {"question": "Why nonlocal?", "code": "", "expected_answer": "To rebind.", "difficulty": "hard"}
    assert qs[1]["difficulty"] == "medium" and qs[2]["difficulty"] == "medium"
    system, human = llm.seen[0]
    assert "exactly 3" in system.content and "explanation text" in human.content
    assert "CANNOT see the explanation" in system.content


def test_generation_uses_structured_output(monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return FakeLLM({"questions": []})
    monkeypatch.setattr(quiz, "build_llm", fake_build)
    generate_questions("Closures", "x")
    assert seen["json_schema"] is quiz.QUESTIONS_SCHEMA
    assert seen["json_schema"]["properties"]["questions"]["items"]["required"] == [
        "question", "code", "expected_answer", "difficulty"]


def test_code_is_kept_and_fences_stripped(llm_returns):
    llm_returns(quiz, {"questions": [{
        "question": "What does this print, and why?",
        "code": "```python\ndef f():\n    x = 1\n```",
        "expected_answer": "Nothing; f is never called.", "difficulty": "easy"}]})
    assert generate_questions("Scope", "x")[0]["code"] == "def f():\n    x = 1"


@pytest.mark.parametrize("question", [
    "Explain why the following code results in both a and b being [1, 2, 3, 4].",
    "Explain why b and a refer to the same list after b = a in the example provided.",
    "Why does y still refer to 10 after x = 20 in the second example?",
    "What does this code print?",
])
def test_question_referring_to_unseen_code_is_dropped(llm_returns, question, capsys):
    llm_returns(quiz, {"questions": [
        {"question": question, "code": "", "expected_answer": "A", "difficulty": "easy"},
        {"question": "Why use None as a default?", "code": "", "expected_answer": "B", "difficulty": "easy"},
    ]})
    qs = generate_questions("Basics", "x")
    assert [q["question"] for q in qs] == ["Why use None as a default?"]
    assert "Dropped a question" in capsys.readouterr().out


def test_same_reference_is_fine_when_code_is_attached(llm_returns):
    llm_returns(quiz, {"questions": [{"question": "Explain why the following code prints [1, 2].",
                                      "code": "a = [1]\nb = a\nb.append(2)\nprint(a)",
                                      "expected_answer": "Same list.", "difficulty": "easy"}]})
    assert len(generate_questions("Basics", "x")) == 1


def test_code_pasted_in_question_and_code_field_is_shown_once(llm_returns):
    """Seen live: the model filled `code` AND pasted the same block into the question."""
    llm_returns(quiz, {"questions": [{
        "question": "Explain why a and b match in the following code:\n```python\na = [1]\nb = a\n```",
        "code": "a = [1]\nb = a", "expected_answer": "Same list.", "difficulty": "easy"}]})
    q = generate_questions("Basics", "x")[0]
    assert q["question"] == "Explain why a and b match in the following code"
    assert q["code"] == "a = [1]\nb = a"


def test_code_only_in_question_is_moved_to_code_field(llm_returns):
    llm_returns(quiz, {"questions": [{
        "question": "What does this print?\n```python\nprint(1 + 1)\n```",
        "code": "", "expected_answer": "2", "difficulty": "easy"}]})
    q = generate_questions("Basics", "x")[0]
    assert (q["question"], q["code"]) == ("What does this print?", "print(1 + 1)")


def test_all_questions_dropped_falls_back(llm_returns):
    llm_returns(quiz, {"questions": [{"question": "What does this code print?", "code": "",
                                      "expected_answer": "A", "difficulty": "easy"}]})
    assert "own words" in generate_questions("Basics", "x")[0]["question"]


@pytest.mark.parametrize("content", [
    "not json",
    "[1, 2]",
    {"questions": []},
    {"questions": [{"question": "no expected answer"}, {"expected_answer": "no question"}]},
    {"something_else": True},
])
def test_generate_falls_back_on_bad_output(llm_returns, content):
    llm_returns(quiz, content)
    qs = generate_questions("Closures", "x")
    assert len(qs) == 1 and "Explain Closures in your own words" in qs[0]["question"]


def test_generate_falls_back_when_llm_unreachable():
    # No mock: conftest points Ollama at a closed port.
    assert "own words" in generate_questions("Closures", "x")[0]["question"]


# --- grade_answer -------------------------------------------------------------------

def verdict(v, score, feedback="", missing=""):
    return {"verdict": v, "score": score, "feedback": feedback, "missing_concept": missing}


def test_grade_parses_response(llm_returns):
    llm = llm_returns(quiz, verdict("minor_gaps", 0.8, "Good.", "edge case"))
    g = grade_answer("Why?", "Because.", "Because of scope.")
    assert g == {"correct": True, "score": 0.8, "feedback": "Good.", "missing_concept": "edge case"}
    system, human = llm.seen[0]
    assert "First decide the verdict" in system.content
    assert "Student answer: Because of scope." in human.content


def test_grading_uses_structured_output(monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return FakeLLM(verdict("wrong", 0.0))
    monkeypatch.setattr(quiz, "build_llm", fake_build)
    grade_answer("q", "e", "a")
    assert seen["json_schema"] is quiz.GRADE_SCHEMA and seen["temperature"] == 0.1
    assert seen["json_schema"]["properties"]["verdict"]["enum"] == ["correct", "minor_gaps", "partial", "wrong"]


@pytest.mark.parametrize("v, expected_correct", [
    ("correct", True), ("minor_gaps", True), ("partial", False), ("wrong", False)])
def test_correct_comes_from_the_verdict(llm_returns, v, expected_correct):
    low, high = quiz.VERDICT_RANGES[v]
    llm_returns(quiz, verdict(v, (low + high) / 2))
    assert grade_answer("q", "e", "a")["correct"] is expected_correct


def test_model_cannot_contradict_itself(llm_returns):
    """The seen-live failure shape: a wrong answer with a middling score and correct=True."""
    llm_returns(quiz, {"verdict": "wrong", "score": 0.6, "correct": True, "feedback": "", "missing_concept": ""})
    g = grade_answer("q", "e", "a")
    assert g["correct"] is False and g["score"] == 0.2  # clamped into the "wrong" range


@pytest.mark.parametrize("v, raw, expected", [
    ("wrong", 0.5, 0.2), ("wrong", -1, 0.0),
    ("partial", 0.1, 0.4), ("partial", 0.9, 0.7), ("partial", 0.5, 0.5),
    ("correct", 0.3, 0.85), ("correct", 1.7, 1.0), ("correct", "0.95", 0.95),
    ("minor_gaps", 0.99, 0.85),
])
def test_score_is_clamped_into_the_verdict_range(llm_returns, v, raw, expected):
    llm_returns(quiz, verdict(v, raw))
    assert grade_answer("q", "e", "a")["score"] == pytest.approx(expected)


def test_verdict_is_case_and_space_insensitive_and_nulls_become_empty(llm_returns):
    llm_returns(quiz, {"verdict": " Wrong ", "score": 0.1, "feedback": "No.", "missing_concept": None})
    g = grade_answer("q", "e", "a")
    assert g["correct"] is False and g["missing_concept"] == ""


@pytest.mark.parametrize("content", [
    "oops",                                                                  # not JSON
    {"score": 0.9, "feedback": "", "missing_concept": ""},                   # no verdict
    {"verdict": "great", "score": 0.9, "feedback": "", "missing_concept": ""},  # unknown verdict
    {"verdict": "correct", "score": "high", "feedback": "", "missing_concept": ""},
    {"correct": True, "score": 0.9, "feedback": "", "missing_concept": ""},  # the old format
])
def test_grade_fallback(llm_returns, content):
    llm_returns(quiz, content)
    g = grade_answer("q", "e", "a")
    assert g["correct"] is False and g["score"] == 0.5
    assert g["feedback"].startswith("Could not grade automatically")
    assert g["missing_concept"] == ""


def test_grade_falls_back_when_llm_unreachable():
    assert grade_answer("q", "e", "a")["score"] == 0.5


# --- run_quiz -----------------------------------------------------------------------

def test_dedupe_keeps_order_and_ignores_case_and_blanks():
    assert dedupe(["nonlocal", "", "late binding", "Nonlocal", "  ", "late binding"]) == ["nonlocal", "late binding"]


def test_run_quiz_flow(monkeypatch, answers, capsys):
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3: [
        {"question": "Q1", "expected_answer": "A1", "difficulty": "easy"},
        {"question": "Q2", "expected_answer": "A2", "difficulty": "hard"},
    ])
    graded = []

    def fake_grade(q, e, a):
        graded.append(a)
        return ({"correct": True, "score": 1.0, "feedback": "Yes.", "missing_concept": ""} if a == "right"
                else {"correct": False, "score": 0.2, "feedback": "No.", "missing_concept": "nonlocal"})
    monkeypatch.setattr(quiz, "grade_answer", fake_grade)
    answers.extend(["right", "   "])

    r = run_quiz("Closures", "explanation")
    assert graded == ["right", NO_ANSWER]  # empty answer is replaced
    assert r.score == pytest.approx(0.6)
    assert r.weak_areas == ["nonlocal"]
    assert [q.user_answer for q in r.questions] == ["right", NO_ANSWER]
    assert r.timestamp
    out = capsys.readouterr().out
    assert "Q2 [hard] Q2" in out and "✓ 100%" in out and "✗ 20%" in out


def test_run_quiz_shows_code_and_grades_with_it(monkeypatch, answers, capsys):
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3: [
        {"question": "What does this print?", "code": "a = [1]\nb = a\nb.append(2)\nprint(a)",
         "expected_answer": "[1, 2]", "difficulty": "easy"},
    ])
    graded_questions = []
    monkeypatch.setattr(quiz, "grade_answer", lambda q, e, a: graded_questions.append(q) or
                        {"correct": True, "score": 1.0, "feedback": "Yes.", "missing_concept": ""})
    answers.append("[1, 2]")

    r = run_quiz("Basics", "x")
    out = capsys.readouterr().out
    assert "\n    a = [1]\n    b = a\n    b.append(2)\n    print(a)\n" in out  # shown, indented
    assert graded_questions[0] == "What does this print?\n\n```python\na = [1]\nb = a\nb.append(2)\nprint(a)\n```"
    assert r.questions[0].question == graded_questions[0]  # stored as the learner saw it


# --- extract_explanation -------------------------------------------------------------

def test_extract_skips_tool_call_messages(sample_state):
    sample_state["messages"] = [
        HumanMessage(content="explain"),
        AIMessage(content="The real explanation."),
        AIMessage(content="Let me check one more file", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        ToolMessage(content="file text", tool_call_id="1"),
    ]
    assert extract_explanation(sample_state) == "The real explanation."


def test_extract_ignores_stale_explanation_when_explainer_failed(sample_state):
    """Regression: a failed Explainer must not reuse the PREVIOUS topic's explanation."""
    sample_state["messages"] = [AIMessage(content="Explanation of topic 1.")]
    sample_state["current_topic_index"] = 1
    sample_state["error"] = "Explainer reached max iterations (8)."
    assert extract_explanation(sample_state) == "Practical Closure Patterns: Counters, factories and callbacks built with closures."


def test_extract_falls_back_without_ai_message(sample_state):
    assert extract_explanation(sample_state).startswith("Closures Explained:")


# --- quiz_generator_node ---------------------------------------------------------------

def test_quiz_node_accumulates_results_and_dedupes_weak_areas(sample_state, monkeypatch):
    earlier = result(0.9, ["scope"], topic="Earlier")
    sample_state.update(quiz_results=[earlier], weak_areas=["scope", "nonlocal"],
                        messages=[AIMessage(content="explained")])
    seen = {}

    def fake_run_quiz(title, explanation):
        seen.update(title=title, explanation=explanation)
        return result(0.4, ["Nonlocal", "late binding"])
    monkeypatch.setattr(quiz, "run_quiz", fake_run_quiz)

    out = quiz_generator_node(sample_state)
    assert seen == {"title": "Closures Explained", "explanation": "explained"}
    assert out["quiz_results"][0] is earlier and len(out["quiz_results"]) == 2
    assert out["weak_areas"] == ["scope", "nonlocal", "late binding"]
    assert out["error"] is None
    assert out["roadmap"] is sample_state["roadmap"]
    assert out["current_topic_index"] == 0 and out["session_id"] == "test1234"


def test_quiz_node_without_topic():
    assert quiz_generator_node(initial_state("g", "s"))["error"] == "Quiz: no current topic."


# --- get_coaching_message -------------------------------------------------------------

def test_coaching_message_parsed(llm_returns):
    llm = llm_returns(coach, {"summary": "Great work on closures.", "encouragement": "Onward!"})
    assert get_coaching_message("Closures", 0.8, ["nonlocal"]) == {
        "summary": "Great work on closures.", "encouragement": "Onward!"}
    assert "Weak areas: nonlocal" in llm.seen[0][1].content


@pytest.mark.parametrize("content", ["nope", {"summary": "only summary"}, {"summary": "", "encouragement": "x"}])
def test_coaching_message_fallback(llm_returns, content):
    llm_returns(coach, content)
    msg = get_coaching_message("Closures", 0.3, ["nonlocal"])
    assert "30%" in msg["summary"] and "Closures" in msg["summary"] and "nonlocal" in msg["summary"]
    assert msg["encouragement"]


# --- progress_coach_node ----------------------------------------------------------------

@pytest.fixture
def quiet_coach(monkeypatch):
    monkeypatch.setattr(coach, "get_coaching_message",
                        lambda t, s, w: {"summary": f"Summary for {t}", "encouragement": "Go on."})


def test_coach_updates_the_current_topic_not_the_next(sample_state, quiet_coach):
    """Regression for the ordering bug: status must land on topics[idx], then idx moves."""
    sample_state["quiz_results"] = [result(0.9)]
    out = progress_coach_node(sample_state)
    assert [t.status for t in out["roadmap"].topics] == ["completed", "pending"]
    assert out["current_topic_index"] == 1
    assert out["error"] is None
    assert out["messages"][0].content == "Summary for Closures Explained"


@pytest.mark.parametrize("score, status", [(0.49, "needs_review"), (0.5, "completed"), (0.2, "needs_review")])
def test_coach_status_threshold(sample_state, quiet_coach, score, status):
    sample_state["quiz_results"] = [result(score)]
    assert progress_coach_node(sample_state)["roadmap"].topics[0].status == status


def test_coach_writes_progress_to_memory(sample_state, quiet_coach):
    sample_state["current_topic_index"] = 1
    sample_state["quiz_results"] = [result(0.3, ["nonlocal"], topic="Practical Closure Patterns")]
    progress_coach_node(sample_state)
    saved = json.loads(mem.memory_get("test1234", "progress_topic_1"))
    assert saved == {"topic": "Practical Closure Patterns", "score": 0.3, "weak_areas": ["nonlocal"],
                     "status": "needs_review", "timestamp": "2026-10-04T10:00:00+00:00"}


def test_coach_handles_dict_roadmap_without_mutating_state(sample_state, quiet_coach):
    sample_state["roadmap"] = sample_state["roadmap"].to_dict()
    sample_state["quiz_results"] = [result(0.9).to_dict()]
    before = copy.deepcopy(sample_state["roadmap"])
    out = progress_coach_node(sample_state)
    assert isinstance(out["roadmap"], StudyRoadmap)
    assert out["roadmap"].topics[0].status == "completed"
    assert sample_state["roadmap"] == before


def test_coach_does_not_mutate_dataclass_roadmap_in_state(sample_state, quiet_coach):
    sample_state["quiz_results"] = [result(0.9)]
    progress_coach_node(sample_state)
    assert sample_state["roadmap"].topics[0].status == "pending"


@pytest.mark.parametrize("change", [{"quiz_results": []}, {"roadmap": None}, {"current_topic_index": 5}])
def test_coach_errors(sample_state, quiet_coach, change):
    sample_state["quiz_results"] = [result(0.9)]
    sample_state.update(change)
    assert progress_coach_node(sample_state)["error"].startswith("Coach:")


def test_coach_prints_next_topic_then_session_totals(sample_state, quiet_coach, capsys):
    sample_state["quiz_results"] = [result(0.9)]
    progress_coach_node(sample_state)
    assert "Next up: Practical Closure Patterns (2/2)" in capsys.readouterr().out

    sample_state["current_topic_index"] = 1
    sample_state["quiz_results"] = [result(0.9), result(0.3, topic="Practical Closure Patterns")]
    progress_coach_node(sample_state)
    assert "Session complete: 1/2 topics passed, average 60%" in capsys.readouterr().out


@pytest.mark.parametrize("score, weak, called", [(0.3, ["nonlocal"], True), (0.3, [], False), (0.8, ["x"], False)])
def test_study_buddy_only_on_low_score_with_weak_areas(sample_state, quiet_coach, monkeypatch, score, weak, called):
    calls = []
    monkeypatch.setattr(coach, "try_study_buddy_assistance", lambda *a: calls.append(a) or "Try this analogy.")
    sample_state["quiz_results"] = [result(score, weak)]
    progress_coach_node(sample_state)
    assert bool(calls) is called


def test_study_buddy_failure_is_silent(sample_state, quiet_coach, monkeypatch):
    def boom(*_):
        raise RuntimeError("A2A down")
    monkeypatch.setattr(coach, "try_study_buddy_assistance", boom)
    sample_state["quiz_results"] = [result(0.2, ["nonlocal"])]
    assert progress_coach_node(sample_state)["error"] is None
