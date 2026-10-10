import copy
import json
from unittest.mock import MagicMock

import httpx
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

def test_the_prompt_asks_for_both_kinds_of_question():
    assert "Include at least one of each" in GENERATION_PROMPT.format(n=3)


def test_code_questions_ask_only_for_the_output():
    prompt = GENERATION_PROMPT.format(n=3)
    assert "Ask ONLY for the output or the error, never for an explanation" in prompt
    assert 'set "expected_answer" to ""' in prompt


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
    assert qs[0] == {"question": "Why nonlocal?", "code": "", "expected_answer": "To rebind.",
                     "difficulty": "hard", "verified": ""}
    assert qs[1]["difficulty"] == "medium" and qs[2]["difficulty"] == "medium"
    system, human = llm.seen[0]
    assert f"exactly {3 + quiz.EXTRA_QUESTIONS}" in system.content  # over-generate: some get dropped
    assert "explanation text" in human.content
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
        "code": "```python\ndef f():\n    return 1\nprint(f())\n```",
        "expected_answer": "1.", "difficulty": "easy"}]})
    assert generate_questions("Scope", "x")[0]["code"] == "def f():\n    return 1\nprint(f())"


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
        "question": "Explain why a and b match in the following code:\n```python\na = [1]\nb = a\nprint(a is b)\n```",
        "code": "a = [1]\nb = a\nprint(a is b)", "expected_answer": "Same list.", "difficulty": "easy"}]})
    q = generate_questions("Basics", "x")[0]
    assert q["code"] == "a = [1]\nb = a\nprint(a is b)"
    assert "```" not in q["question"] and "a = [1]" not in q["question"]  # the block is shown once, as code
    assert q["verified"] == "True"


def test_code_only_in_question_is_moved_to_code_field(llm_returns):
    llm_returns(quiz, {"questions": [{
        "question": "What does this print?\n```python\nprint(1 + 1)\n```",
        "code": "", "expected_answer": "2", "difficulty": "easy"}]})
    q = generate_questions("Basics", "x")[0]
    assert q["code"] == "print(1 + 1)" and "```" not in q["question"]
    assert q["verified"] == "2"


# --- code questions are verified by running the code (fix A + B) ----------------------------------

ADD_ITEM = ("def add_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n"
            "    bucket.append(item)\n    return bucket\n")


def code_q(code, expected="WRONG model guess", question="What does this print, and why?"):
    return {"question": question, "code": code, "expected_answer": expected, "difficulty": "hard"}


def test_expected_answer_comes_from_running_the_code_not_the_model(llm_returns):
    """The live bug: the model said [5, 10, 15]; the code really prints [5], [10], [15]."""
    code = ADD_ITEM + "print(add_item(5))\nprint(add_item(10))\nprint(add_item(15))"
    llm_returns(quiz, {"questions": [code_q(code, expected="The output is [5, 10, 15].")]})
    q = generate_questions("Basics", "x")[0]
    assert q["verified"] == "[5]\n[10]\n[15]"
    assert q["expected_answer"] == "Verified output (from running the code):\n[5]\n[10]\n[15]"
    assert "[5, 10, 15]" not in q["expected_answer"]


@pytest.mark.parametrize("model_wording", [
    "What will the following code raise when run?",                  # but the code prints
    "Why does the following code raise an UnboundLocalError?",       # leaks the answer
    "How does Python resolve the name x? Explain.",                  # asks for an explanation
])
def test_question_text_is_written_from_what_the_code_really_does(llm_returns, model_wording):
    llm_returns(quiz, {"questions": [code_q("print(1 + 1)", question=model_wording)]})
    assert generate_questions("x", "x")[0]["question"] == "What does this code print?"


@pytest.mark.parametrize("code, wording", [
    ("print(1)", "What does this code print?"),
    ("count = 0\ndef bump():\n    count += 1\nbump()", "What error does this code raise?"),
    ("print('a')\nraise KeyError('k')", "What does this code print, and what error does it then raise?"),
])
def test_question_wording_follows_the_behaviour(llm_returns, code, wording):
    llm_returns(quiz, {"questions": [code_q(code, question="Whatever the model wrote.")]})
    assert generate_questions("x", "x")[0]["question"] == wording


def test_concept_question_wording_is_left_alone(llm_returns):
    llm_returns(quiz, {"questions": [{"question": "Why is LEGB ordered that way?", "code": "",
                                      "expected_answer": "Inner scopes shadow outer ones.", "difficulty": "easy"}]})
    assert generate_questions("x", "x")[0]["question"] == "Why is LEGB ordered that way?"


def test_code_questions_need_no_expected_answer_from_the_model(llm_returns):
    llm_returns(quiz, {"questions": [code_q("print(6 * 7)", expected="")]})
    assert generate_questions("Math", "x")[0]["verified"] == "42"


def test_concept_question_without_an_expected_answer_is_dropped(llm_returns):
    llm_returns(quiz, {"questions": [{"question": "Why LEGB?", "code": "", "expected_answer": "", "difficulty": "easy"}]})
    assert "own words" in generate_questions("Scope", "x")[0]["question"]


def test_code_that_raises_is_described_as_raising(llm_returns):
    llm_returns(quiz, {"questions": [code_q("count = 0\ndef bump():\n    count += 1\nbump()")]})
    q = generate_questions("Scope", "x")[0]
    assert q["verified"].startswith("Raises UnboundLocalError")


def test_prints_then_raises(llm_returns):
    llm_returns(quiz, {"questions": [code_q("print('a')\nraise ValueError('boom')")]})
    assert generate_questions("x", "x")[0]["verified"] == "a\n...then raises ValueError: boom"


def test_code_question_that_prints_nothing_is_dropped(llm_returns, capsys):
    """Fix B: the live question had three bare calls and no print, so there was nothing to ask."""
    llm_returns(quiz, {"questions": [
        code_q(ADD_ITEM + "add_item(5)\nadd_item(10)"),
        {"question": "Why use None as a default?", "code": "", "expected_answer": "B", "difficulty": "easy"},
    ]})
    qs = generate_questions("Basics", "x")
    assert [q["question"] for q in qs] == ["Why use None as a default?"]
    assert "prints nothing" in capsys.readouterr().out


@pytest.mark.parametrize("code, why", [
    ("import os\nprint(os.getcwd())", "imports os"),
    ("print(open('/etc/passwd').read())", "uses open()"),
    ("while True:\n    pass", "crashed"),
    ("print(", "syntax error"),
])
def test_unrunnable_code_questions_are_dropped(llm_returns, capsys, code, why):
    llm_returns(quiz, {"questions": [code_q(code)]})
    qs = generate_questions("Basics", "x")
    assert "own words" in qs[0]["question"]  # nothing left: the generic fallback
    assert why in capsys.readouterr().out


def test_no_model_explanation_is_added_to_verified_answers(monkeypatch):
    """A 7B model explained subtle behaviour wrongly, and the grader followed it (seen live)."""
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return FakeLLM({"questions": [code_q("print(2 ** 10)", expected="")]})
    monkeypatch.setattr(quiz, "build_llm", build)
    q = generate_questions("Math", "x", n=1)[0]
    assert len(calls) == 1 and calls[0].get("json_schema")  # only the question-writing call
    assert q["expected_answer"] == "Verified output (from running the code):\n1024"
    assert "Why" not in q["expected_answer"]


def test_stops_verifying_once_enough_questions_are_collected(llm_returns, monkeypatch):
    llm_returns(quiz, {"questions": [code_q(f"print({i})") for i in range(5)]})
    ran = []
    real = quiz.run_snippet
    monkeypatch.setattr(quiz, "run_snippet", lambda code: (ran.append(code), real(code))[1])
    assert len(generate_questions("x", "x", n=3)) == 3
    assert len(ran) == 3


def test_concept_questions_are_not_run(llm_returns, monkeypatch):
    llm_returns(quiz, {"questions": [{"question": "Why is LEGB ordered that way?", "code": "",
                                      "expected_answer": "Inner scopes shadow outer ones.", "difficulty": "easy"}]})
    monkeypatch.setattr(quiz, "run_snippet", lambda code: pytest.fail("no code, nothing to run"))
    q = generate_questions("Scope", "x")[0]
    assert q["expected_answer"] == "Inner scopes shadow outer ones." and q["verified"] == ""


def test_grading_prompt_treats_verified_output_as_fact():
    assert 'starts with "Verified output"' in quiz.GRADING_PROMPT
    assert "actually running the code" in quiz.GRADING_PROMPT
    assert "Ignore their explanation" in quiz.GRADING_PROMPT


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
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3, allow_code=True: [
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
    monkeypatch.setattr(quiz, "generate_questions", lambda t, e, n=3, allow_code=True: [
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

    def fake_run_quiz(title, explanation, allow_code=True):
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


# --- disputed questions don't count (fix C) -----------------------------------------------------------------

def graded(score, disputed=False):
    return QuizQuestion("q", "a", user_answer="x", score=score, disputed=disputed)


def test_quiz_score_ignores_disputed_questions():
    assert quiz.quiz_score([graded(1.0), graded(0.0), graded(0.5)]) == pytest.approx(0.5)
    assert quiz.quiz_score([graded(1.0), graded(0.0, disputed=True), graded(0.5)]) == pytest.approx(0.75)


def test_quiz_score_when_everything_is_disputed_is_neutral():
    assert quiz.quiz_score([graded(0.0, disputed=True), graded(0.0, disputed=True)]) == quiz.NEUTRAL_SCORE


def test_weak_areas_skip_disputed_questions():
    qs = [graded(0.0), graded(0.0, disputed=True), graded(0.0)]
    assert quiz.weak_areas_of(qs, ["nonlocal", "bad question topic", "Nonlocal"]) == ["nonlocal"]


def test_disputed_flag_round_trips_and_defaults_to_false():
    q = QuizQuestion("q", "a", disputed=True)
    assert QuizQuestion.from_dict(q.to_dict()).disputed is True
    assert QuizQuestion.from_dict({"question": "q", "expected_answer": "a"}).disputed is False  # old checkpoints


# --- a dropped connection is retried once -----------------------------------------------------------------

class FlakyLLM:
    def __init__(self, content, failures):
        self.content, self.failures, self.calls = content, failures, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return AIMessage(content=json.dumps(self.content))


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("llm.time.sleep", lambda s: None)


def test_grading_survives_one_dropped_connection(monkeypatch, no_sleep):
    """Seen live: a RemoteProtocolError turned a 0.0 into the neutral 0.5."""
    flaky = FlakyLLM(verdict("wrong", 0.0, "No."), failures=1)
    monkeypatch.setattr(quiz, "build_llm", lambda **_: flaky)
    g = grade_answer("q", "e", "a")
    assert g["score"] == 0.0 and flaky.calls == 2


def test_grading_falls_back_after_two_dropped_connections(monkeypatch, no_sleep):
    flaky = FlakyLLM(verdict("wrong", 0.0), failures=5)
    monkeypatch.setattr(quiz, "build_llm", lambda **_: flaky)
    assert grade_answer("q", "e", "a")["score"] == 0.5 and flaky.calls == 2


def test_question_generation_survives_one_dropped_connection(monkeypatch, no_sleep):
    flaky = FlakyLLM({"questions": [{"question": "Why LEGB?", "code": "", "expected_answer": "Scopes.",
                                     "difficulty": "easy"}]}, failures=1)
    monkeypatch.setattr(quiz, "build_llm", lambda **_: flaky)
    assert generate_questions("Scope", "x")[0]["question"] == "Why LEGB?"


# --- comments are stripped from code (they leak answers) ------------------------------------------------------

@pytest.mark.parametrize("code, expected", [
    ("x = 1  # one\nprint(x)  # prints 1", "x = 1\nprint(x)"),
    ("# What does this print?\nprint(1)", "print(1)"),
    ("print('# not a comment')  # real comment", "print('# not a comment')"),
    ("a = 1\n\nb = 2  # keep the blank line above", "a = 1\n\nb = 2"),
    ("s = \"\"\"line\n\n# inside a string\n\"\"\"\nprint(s)", "s = \"\"\"line\n\n# inside a string\n\"\"\"\nprint(s)"),
    ("def f():\n    # explain\n    return 1\nprint(f())", "def f():\n    return 1\nprint(f())"),
])
def test_strip_comments(code, expected):
    assert quiz.strip_comments(code) == expected


def test_strip_comments_leaves_untokenizable_code_alone():
    assert quiz.strip_comments("x = (  # unclosed") == "x = (  # unclosed"


def test_comments_are_stripped_before_the_code_is_shown_and_run(llm_returns):
    llm_returns(quiz, {"questions": [code_q("def f():\n    return 2  # the answer is 2\nprint(f())  # What does this print?")]})
    q = generate_questions("x", "x")[0]
    assert q["code"] == "def f():\n    return 2\nprint(f())" and q["verified"] == "2"


# --- multiple-choice wording without options is dropped ------------------------------------------------------------

def test_which_of_the_following_without_options_is_dropped(llm_returns, capsys):
    llm_returns(quiz, {"questions": [
        {"question": "Which of the following is true about the default argument in add_item?", "code": "",
         "expected_answer": "It is shared.", "difficulty": "hard"},
        {"question": "Why use None as a default?", "code": "", "expected_answer": "B", "difficulty": "easy"}]})
    assert [q["question"] for q in generate_questions("x", "x")] == ["Why use None as a default?"]
    assert "Dropped a question" in capsys.readouterr().out


# --- a second attempt covers dropped questions ------------------------------------------------------------------------

class SeqLLM:
    """Returns each scripted payload in turn (a dict becomes JSON; an Exception is raised)."""

    def __init__(self, *payloads):
        self.payloads, self.calls = list(payloads), 0

    def invoke(self, messages):
        self.calls += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return AIMessage(content=payload if isinstance(payload, str) else json.dumps(payload))


def concept(text):
    return {"question": text, "code": "", "expected_answer": "A", "difficulty": "easy"}


def seq(monkeypatch, *payloads):
    llm = SeqLLM(*payloads)
    monkeypatch.setattr(quiz, "build_llm", lambda **_: llm)
    return llm


def test_second_attempt_makes_up_the_shortfall(monkeypatch):
    llm = seq(monkeypatch,
              {"questions": [code_q("print(1)"), code_q("x = 1")]},            # 1 usable, 1 prints nothing
              {"questions": [concept("Why A?"), concept("Why B?"), concept("Why C?")]})
    qs = generate_questions("x", "x", n=3)
    assert llm.calls == 2 and len(qs) == 3
    assert [q["question"] for q in qs] == ["What does this code print?", "Why A?", "Why B?"]


def test_no_second_attempt_when_the_first_is_enough(monkeypatch):
    llm = seq(monkeypatch, {"questions": [concept("Why A?"), concept("Why B?"), concept("Why C?")]})
    assert len(generate_questions("x", "x", n=3)) == 3 and llm.calls == 1


def test_second_attempt_asks_only_for_what_is_missing(monkeypatch):
    asked = []

    def build(**kwargs):
        return type("L", (), {"invoke": lambda self, m: (asked.append(m[0].content), AIMessage(
            content=json.dumps({"questions": [concept("Why A?")]})))[1]})()
    monkeypatch.setattr(quiz, "build_llm", build)
    generate_questions("x", "x", n=3)
    assert f"exactly {3 + quiz.EXTRA_QUESTIONS}" in asked[0]
    assert f"exactly {2 + quiz.EXTRA_QUESTIONS}" in asked[1]  # one question was already collected


def test_duplicate_questions_across_attempts_are_skipped(monkeypatch):
    seq(monkeypatch, {"questions": [concept("Why A?")]}, {"questions": [concept("Why A?"), concept("Why B?")]})
    assert [q["question"] for q in generate_questions("x", "x", n=2)] == ["Why A?", "Why B?"]


def test_at_most_two_attempts(monkeypatch):
    llm = seq(monkeypatch, {"questions": []}, {"questions": []}, {"questions": [concept("never asked")]})
    assert "own words" in generate_questions("x", "x")[0]["question"] and llm.calls == 2


def test_a_failure_on_the_second_attempt_keeps_the_first_questions(monkeypatch):
    seq(monkeypatch, {"questions": [concept("Why A?")]}, ConnectionError("tunnel dropped"))
    assert [q["question"] for q in generate_questions("x", "x", n=3)] == ["Why A?"]


def test_unusable_first_attempt_is_not_retried(monkeypatch):
    llm = seq(monkeypatch, "not json at all", {"questions": [concept("never asked")]})
    assert "own words" in generate_questions("x", "x")[0]["question"] and llm.calls == 1


# --- notes without code: concept questions only ------------------------------------------------

def test_concept_only_mode_drops_code_questions(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"questions": [
        {"question": "What does this print?", "code": "print(1)", "expected_answer": "", "difficulty": "easy"},
        {"question": "Why do agents need memory?", "code": "", "expected_answer": "To keep state.", "difficulty": "medium"},
    ]}))
    monkeypatch.setattr(quiz, "build_llm", lambda **kw: llm)
    questions = quiz.generate_questions("Agents", "explanation", n=1, allow_code=False)
    assert [q["question"] for q in questions] == ["Why do agents need memory?"]
    system = llm.invoke.call_args.args[0][0].content
    assert "All questions are CONCEPT questions" in system and "CODE questions" not in system


def test_code_mode_prompt_is_unchanged(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"questions": []}))
    monkeypatch.setattr(quiz, "build_llm", lambda **kw: llm)
    quiz.generate_questions("Closures", "explanation", n=1)
    assert llm.invoke.call_args.args[0][0].content == quiz.GENERATION_PROMPT.format(n=3)


@pytest.mark.parametrize("sources, file_text, expected", [
    ([], "## A\nno code\n", True),                                   # roadmap from the goal: as before
    (["n.md#1"], "## A\nno code here\n", False),
    (["n.md#1"], "## A\n```python\nprint(1)\n```\n", True),
    (["n.md#2"], "## A\n```python\nprint(1)\n```\n## B\ntext\n", False),  # only this topic's sections count
])
def test_topic_allows_code(sample_state, tmp_path, sources, file_text, expected):
    (tmp_path / "n.md").write_text(file_text)
    roadmap = sample_state["roadmap"]
    roadmap.topics[0].sources = sources
    state = {**sample_state, "roadmap": roadmap, "study_materials_path": str(tmp_path)}
    assert quiz.topic_allows_code(state) is expected
