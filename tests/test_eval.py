"""LLM-quality evals with DeepEval and a local Ollama judge.

These hit a REAL model (they read .env, so the Kaggle/ngrok tunnel works).
Run with: make eval    (= pytest tests/test_eval.py -m eval -s -v)

Thresholds are deliberately conservative for 7B models. If Ollama isn't
reachable, every eval is skipped with the reason instead of failing.

The judge defaults to OLLAMA_MODEL; set OLLAMA_JUDGE_MODEL to grade with a
different model (a model tends to go easy on its own output).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

deepeval = pytest.importorskip("deepeval")
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402

from agents.explainer import explainer_node  # noqa: E402
from agents.progress_coach import _fallback_coaching, get_coaching_message  # noqa: E402
from agents.quiz_generator import full_question, generate_questions, grade_answer  # noqa: E402
from graph.state import StudyRoadmap, Topic, initial_state  # noqa: E402
from llm import build_llm, describe_llm_error, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.6

CLOSURES_TITLE = "Closures Explained"
CLOSURES_DESC = "What a closure is, how nonlocal works, and the late-binding gotcha."

GRADING_QUESTION = (
    "Why does `count += 1` inside a nested function raise UnboundLocalError "
    "when count is defined in the enclosing function, and how do you fix it?"
)
GRADING_EXPECTED = (
    "Assigning to count makes it local to the inner function, so it is read "
    "before it has a value. Declaring `nonlocal count` in the inner function "
    "rebinds the enclosing variable instead."
)


# --- the judge -------------------------------------------------------------------------

class OllamaJudge(DeepEvalBaseLLM):
    """DeepEval's evaluator model, backed by our Ollama (local or tunnelled).

    DeepEval calls generate(prompt, schema=SomePydanticModel). We hand the
    schema to Ollama as a structured-output format, then validate. If the
    output doesn't validate, we return the raw text and DeepEval falls back
    to its own JSON parsing.
    """

    def __init__(self, model: str | None = None):
        self.model_id = model or os.getenv("OLLAMA_JUDGE_MODEL") or model_name()
        super().__init__(self.model_id)

    def load_model(self):
        return build_llm(temperature=0.0, model=self.model_id)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return str(self.load_model().invoke(prompt).content)
        llm = build_llm(temperature=0.0, model=self.model_id, json_schema=schema.model_json_schema())
        content = str(llm.invoke(prompt).content)
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            return content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_id}"


def measure(metric, test_case: LLMTestCase) -> float:
    metric.measure(test_case)
    print(f"\n  {metric.__class__.__name__ if not getattr(metric, 'name', None) else metric.name}: "
          f"{metric.score:.2f} (threshold {metric.threshold})\n  reason: {metric.reason}")
    return metric.score


# --- live-model fixtures (module scope: each one costs several LLM calls) -----------------

@pytest.fixture(scope="module")
def live_ollama():
    """Load .env (unit tests never do) and skip everything if the model is unreachable."""
    with pytest.MonkeyPatch.context() as mp:
        for key, value in dotenv_values(ROOT / ".env").items():
            if value is not None and key not in os.environ:
                mp.setenv(key, value)
        try:
            build_llm(temperature=0.0).invoke("Reply with the single word OK.")
        except Exception as exc:
            pytest.skip(f"Ollama not reachable, evals skipped: {describe_llm_error(exc)}")
        yield


@pytest.fixture(scope="module")
def judge(live_ollama) -> OllamaJudge:
    return OllamaJudge()


def run_explainer(title: str, description: str, session_id: str) -> tuple[str, list[str]]:
    """Run the real Explainer on a one-topic roadmap.

    Returns (final explanation, contents of every note it read).
    """
    state = initial_state(goal=title, session_id=session_id)
    state["roadmap"] = StudyRoadmap(goal=title, total_weeks=1,
                                    topics=[Topic(title=title, description=description, estimated_minutes=45)])
    out = explainer_node(state)
    messages = out.get("messages") or []
    final = next((str(m.content) for m in reversed(messages)
                  if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
    read = [str(m.content) for m in messages
            if isinstance(m, ToolMessage) and m.name == "read_file" and not str(m.content).startswith("Error")]
    return final, read


@pytest.fixture(scope="module")
def closures_explanation(live_ollama, closures_note_content_module):
    explanation, notes_read = run_explainer(CLOSURES_TITLE, CLOSURES_DESC, f"eval-{uuid.uuid4().hex[:6]}")
    if not explanation.strip():
        pytest.skip("Explainer produced no final explanation (see its log above).")
    return {"text": explanation, "context": notes_read or [closures_note_content_module]}


@pytest.fixture(scope="module")
def closures_note_content_module() -> str:
    return (ROOT / "study_materials" / "sample_notes" / "closures.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def quiz_questions(live_ollama, closures_note_content_module):
    return generate_questions("Python closures", closures_note_content_module, n=3)


# --- evals -------------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.usefixtures("live_ollama")
class TestExplainerQuality:
    def test_faithful_to_the_notes_it_read(self, judge, closures_explanation):
        case = LLMTestCase(input=f"Explain: {CLOSURES_TITLE}. {CLOSURES_DESC}",
                           actual_output=closures_explanation["text"],
                           retrieval_context=closures_explanation["context"])
        metric = FaithfulnessMetric(threshold=THRESHOLD, model=judge, async_mode=False)
        assert measure(metric, case) >= THRESHOLD

    def test_relevant_to_the_topic(self, judge, closures_explanation):
        case = LLMTestCase(input=f"Explain {CLOSURES_TITLE}: {CLOSURES_DESC}",
                           actual_output=closures_explanation["text"])
        metric = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False)
        assert measure(metric, case) >= THRESHOLD

    def test_long_enough(self, closures_explanation):
        assert len(closures_explanation["text"]) >= 300, closures_explanation["text"]

    def test_mentions_key_concepts(self, closures_explanation):
        text = closures_explanation["text"].lower()
        assert "closure" in text
        assert "nonlocal" in text or "enclosing" in text, text[:500]

    def test_follows_the_required_structure(self, closures_explanation):
        text = closures_explanation["text"].lower()
        sections = ["analogy", "core concept", "example", "gotcha"]
        found = [s for s in sections if s in text]
        assert len(found) >= 3, f"only found {found} in:\n{text[:500]}"


@pytest.mark.eval
@pytest.mark.usefixtures("live_ollama")
class TestQuizGeneratorQuality:
    def test_questions_have_required_keys(self, quiz_questions):
        assert len(quiz_questions) == 3, f"generation fell back: {quiz_questions}"
        for q in quiz_questions:
            assert q["question"].strip() and q["expected_answer"].strip()
            assert q["difficulty"] in {"easy", "medium", "hard"}

    def test_code_questions_are_verified_by_really_running_the_code(self, quiz_questions):
        """No judge involved: the stored answer must equal what the code actually does."""
        from code_runner import run_snippet

        for q in quiz_questions:
            if not q["code"]:
                continue
            real = run_snippet(q["code"])
            assert real.ran, f"unrunnable code was kept: {q['code']}"
            assert q["verified"] == real.describe()
            assert q["expected_answer"] == f"Verified output (from running the code):\n{real.describe()}"
            assert "#" not in q["code"], "comments can leak the answer"
            assert q["question"] in {"What does this code print?", "What error does this code raise?",
                                     "What does this code print, and what error does it then raise?"}

    def test_questions_test_understanding_not_recall(self, judge, quiz_questions):
        metric = GEval(
            name="Tests understanding",
            criteria=(
                "The actual output is a quiz about the topic in the input. Judge whether it tests UNDERSTANDING "
                "rather than recall of a definition. Predicting what a piece of code prints or which error it "
                "raises is an application question, so 'What does this code print?' is acceptable on its own: "
                "judge the code itself. Give a high score when the code exercises the topic and is non-trivial, "
                "and especially when at least one question probes a common mistake or an edge case. Give a low "
                "score for code unrelated to the topic, trivial code, or questions that only ask for a definition."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge, threshold=THRESHOLD, async_mode=False,
        )
        case = LLMTestCase(input="Quiz topic: Python closures (nonlocal, late binding, __closure__)",
                           actual_output="\n\n".join(full_question(q) for q in quiz_questions))
        assert measure(metric, case) >= THRESHOLD


@pytest.mark.eval
@pytest.mark.usefixtures("live_ollama")
class TestGradingQuality:
    def test_correct_answer_scores_high(self):
        g = grade_answer(GRADING_QUESTION, GRADING_EXPECTED,
                         "Because the assignment makes count a local variable of the inner function, so Python "
                         "reads it before it's assigned. Adding `nonlocal count` makes it refer to the outer one.")
        print(f"\n  correct answer: {g}")
        assert g["score"] >= 0.65

    def test_wrong_answer_scores_low(self):
        g = grade_answer(GRADING_QUESTION, GRADING_EXPECTED,
                         "Python doesn't allow nested functions, so move the function to the top of the module.")
        print(f"\n  wrong answer: {g}")
        assert g["score"] <= 0.35

    def test_partial_answer_scores_in_between(self):
        g = grade_answer(GRADING_QUESTION, GRADING_EXPECTED, "You need to write nonlocal count.")
        print(f"\n  partial answer: {g}")
        assert 0.3 <= g["score"] <= 0.75

    def test_grade_fields_and_types(self):
        g = grade_answer(GRADING_QUESTION, GRADING_EXPECTED, "Use nonlocal.")
        assert g["feedback"] != "Could not grade automatically. Compare your answer with the expected one.", \
            "the judge call failed and returned the fallback"
        assert isinstance(g["correct"], bool) and isinstance(g["score"], float)
        assert isinstance(g["feedback"], str) and g["feedback"]
        assert isinstance(g["missing_concept"], str)


@pytest.mark.eval
@pytest.mark.usefixtures("live_ollama")
class TestProgressCoachQuality:
    TOPIC, SCORE, WEAK = "Python closures", 0.4, ["nonlocal keyword", "late binding in loops"]

    @pytest.fixture(scope="class")
    def coaching(self, live_ollama):
        return get_coaching_message(self.TOPIC, self.SCORE, self.WEAK)

    def test_returns_summary_and_encouragement(self, coaching):
        assert coaching != _fallback_coaching(self.TOPIC, self.SCORE, self.WEAK), "LLM failed; got the template"
        assert coaching["summary"].strip() and coaching["encouragement"].strip()

    def test_encouraging_specific_actionable_concise(self, judge, coaching):
        metric = GEval(
            name="Good coaching",
            criteria=(
                "The actual output is feedback to a learner described in the input. It must be "
                "encouraging (never discouraging), specific (names the topic and the weak areas), "
                "actionable (suggests a concrete way to review), and concise (a few sentences)."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge, threshold=THRESHOLD, async_mode=False,
        )
        case = LLMTestCase(
            input=f"Topic: {self.TOPIC}. Quiz score: {self.SCORE:.0%}. Weak areas: {', '.join(self.WEAK)}.",
            actual_output=f"{coaching['summary']} {coaching['encouragement']}",
        )
        assert measure(metric, case) >= THRESHOLD


# --- judge plumbing: unit tests, no model needed ----------------------------------------------

class Verdict(BaseModel):
    score: float
    reason: str


class FakeChat:
    def __init__(self, content):
        self.content = content
        self.formats = []

    def __call__(self, temperature, json_mode=False, *, model=None, json_schema=None):
        self.formats.append((model, json_schema))
        return self

    def invoke(self, prompt):
        return AIMessage(content=self.content)


@pytest.mark.unit
class TestOllamaJudgePlumbing:
    @pytest.fixture
    def fake(self, monkeypatch):
        def install(content):
            chat = FakeChat(content)
            monkeypatch.setattr(f"{__name__}.build_llm", chat)
            return chat
        return install

    def test_schema_output_is_validated(self, fake):
        chat = fake('{"score": 0.8, "reason": "ok"}')
        result = OllamaJudge(model="judge-model").generate("grade this", schema=Verdict)
        assert result == Verdict(score=0.8, reason="ok")
        assert chat.formats[-1] == ("judge-model", Verdict.model_json_schema())

    def test_invalid_schema_output_falls_back_to_raw_text(self, fake):
        fake('{"score": "high"}')
        assert OllamaJudge(model="m").generate("p", schema=Verdict) == '{"score": "high"}'

    def test_plain_generation_returns_text(self, fake):
        fake("plain answer")
        assert OllamaJudge(model="m").generate("p") == "plain answer"

    async def test_async_delegates(self, fake):
        fake('{"score": 1, "reason": "r"}')
        assert (await OllamaJudge(model="m").a_generate("p", schema=Verdict)).score == 1.0

    def test_model_name_and_judge_override(self, fake, monkeypatch):
        fake("x")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        assert OllamaJudge().get_model_name() == "ollama/qwen2.5:7b"
        monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "qwen3:8b")
        assert OllamaJudge().get_model_name() == "ollama/qwen3:8b"
