"""Quiz Generator: writes questions about the topic, asks them, grades the answers.

``generate_questions`` and ``grade_answer`` are pure, importable functions:
the A2A quiz service (Phase 9) and the Streamlit UI (Phase 10) reuse them.
``run_quiz`` is the terminal flow (it calls input()).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.state import QuizQuestion, QuizResult, get_current_topic
from llm import build_llm, describe_llm_error

NO_ANSWER = "(no answer provided)"
MAX_EXPLANATION_CHARS = 6000  # keep the prompt well inside a 7B model's context

# Literal braces are doubled because this template goes through str.format.
GENERATION_PROMPT = """\
You write quiz questions that check whether a learner UNDERSTANDS a topic,
not whether they memorised it.

Write exactly {n} questions about the topic, based on the explanation given.
- Test application, "why" reasoning, edge cases, and comparisons.
- At least one question must be about a common mistake or gotcha.
- No yes/no questions. No questions answerable by copying one sentence.
- Each expected_answer is 1-3 sentences: what a correct answer must
  contain. It must be technically correct.

The learner CANNOT see the explanation or their notes during the quiz.
Every question must be self-contained:
- If a question is about code, put the complete code in "code" (plain
  Python, real line breaks, no markdown fences). Otherwise "code" is "".
- Never write "the example", "the following code", "the notes", or name a
  function or variable from the explanation unless "code" defines it.

Respond with a single JSON object and nothing else:
{{"questions": [{{"question": "...", "code": "...", "expected_answer": "...", "difficulty": "easy|medium|hard"}}]}}
"""

# Ollama structured output: generation is constrained to exactly this shape.
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "code": {"type": "string"},
                    "expected_answer": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["question", "code", "expected_answer", "difficulty"],
            },
        },
    },
    "required": ["questions"],
}

# A question that points at something the learner can't see. Only a problem
# when no code is attached.
_UNSEEN_REFERENCE = re.compile(
    r"\b(the|this) (following|above|below|given|provided|first|second) (code|example|snippet|function)\b"
    r"|\b(in|from) the (example|notes|explanation)\b"
    r"|\bthe code (below|above)\b|\bthis (code|snippet)\b",
    re.IGNORECASE,
)
_FENCES = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_FENCED_BLOCK = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


def _split_code(question: str, code: str) -> tuple[str, str]:
    """Models sometimes paste the code into the question text too (or only
    there). Keep the question as prose and the code in one place."""
    code = _FENCES.sub("", code).strip()
    blocks = _FENCED_BLOCK.findall(question)
    if blocks:
        question = _FENCED_BLOCK.sub("", question).strip()
        if not code:
            code = "\n\n".join(b.strip() for b in blocks)
    return question.rstrip(":").strip() if blocks else question, code

GRADING_PROMPT = """\
You are a fair grader. Compare the student's answer with the expected answer.
Judge the meaning, not the wording.

First decide the verdict, then pick a score inside that verdict's range:
- "correct":    right and complete (score 0.85-1.0)
- "minor_gaps": right, missing a detail (0.7-0.85)
- "partial":    right idea, a key part missing or vague (0.4-0.7)
- "wrong":      contradicts the expected answer, misunderstands it, or is off-topic (0.0-0.2)

An answer that states the OPPOSITE of the key fact is "wrong", even if it uses
the right vocabulary. Do not give credit for effort.

"feedback": 1-2 sentences to the student. "missing_concept": the most
important missing idea in a few words, or "" if nothing is missing.
"""

# verdict -> allowed score range. The model picks the verdict (the easy part);
# the score is clamped into its range so the two can never disagree.
VERDICT_RANGES = {
    "correct": (0.85, 1.0),
    "minor_gaps": (0.7, 0.85),
    "partial": (0.4, 0.7),
    "wrong": (0.0, 0.2),
}
PASSING_VERDICTS = {"correct", "minor_gaps"}

# Ollama structured output: "verdict" can only be one of the four words.
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICT_RANGES)},
        "score": {"type": "number"},
        "feedback": {"type": "string"},
        "missing_concept": {"type": "string"},
    },
    "required": ["verdict", "score", "feedback", "missing_concept"],
}


def _fallback_questions(topic: str) -> list[dict]:
    return [{
        "question": f"Explain {topic} in your own words, with a short example.",
        "code": "",
        "expected_answer": f"A clear, correct explanation of {topic} with a relevant example.",
        "difficulty": "medium",
    }]


def full_question(q: dict) -> str:
    """The question as the learner sees it: text plus its code, if any.
    This is what gets graded and stored, so the grader sees what the learner saw."""
    code = (q.get("code") or "").strip()
    return f"{q['question']}\n\n```python\n{code}\n```" if code else q["question"]


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[dict]:
    """Ask the LLM for n self-contained questions: {question, code, expected_answer, difficulty}.

    Falls back to one generic question on any failure.
    """
    llm = build_llm(temperature=0.4, json_schema=QUESTIONS_SCHEMA)
    try:
        response = llm.invoke([
            SystemMessage(content=GENERATION_PROMPT.format(n=n)),
            HumanMessage(content=f"Topic: {topic}\n\nExplanation:\n{explanation[:MAX_EXPLANATION_CHARS]}"),
        ])
    except Exception as exc:
        print(f"[Quiz] {describe_llm_error(exc)} Using a generic question.")
        return _fallback_questions(topic)
    try:
        raw = json.loads(response.content).get("questions") or []
    except (ValueError, AttributeError):  # not JSON, or JSON that isn't an object
        print("[Quiz] The model's questions weren't valid JSON; using a generic question.")
        return _fallback_questions(topic)

    questions = []
    for q in raw if isinstance(raw, list) else []:
        if not (isinstance(q, dict) and str(q.get("question", "")).strip()
                and str(q.get("expected_answer", "")).strip()):
            continue
        question, code = _split_code(str(q["question"]).strip(), str(q.get("code") or ""))
        if not code and _UNSEEN_REFERENCE.search(question):
            print(f"[Quiz] Dropped a question that refers to code it doesn't show: {question[:80]}")
            continue
        difficulty = str(q.get("difficulty", "medium")).lower()
        questions.append({
            "question": question,
            "code": code,
            "expected_answer": str(q["expected_answer"]).strip(),
            "difficulty": difficulty if difficulty in {"easy", "medium", "hard"} else "medium",
        })
    if not questions:
        print("[Quiz] The model returned no usable questions; using a generic question.")
        return _fallback_questions(topic)
    return questions[:n]


def grade_answer(question: str, expected: str, student_answer: str) -> dict:
    """LLM-as-judge for one answer. Returns {correct, score, feedback, missing_concept}.

    The model gives a verdict; `correct` and the score range come from the
    verdict in code, not from the model's own (possibly contradicting) fields.
    """
    llm = build_llm(temperature=0.1, json_schema=GRADE_SCHEMA)
    try:
        response = llm.invoke([
            SystemMessage(content=GRADING_PROMPT),
            HumanMessage(content=(
                f"Question: {question}\n"
                f"Expected answer: {expected}\n"
                f"Student answer: {student_answer}"
            )),
        ])
        data = json.loads(response.content)
        verdict = str(data["verdict"]).strip().lower()
        low, high = VERDICT_RANGES[verdict]  # KeyError on an unknown verdict
        score = min(high, max(low, float(data["score"])))
        return {
            "correct": verdict in PASSING_VERDICTS,
            "score": score,
            "feedback": str(data.get("feedback", "")).strip(),
            "missing_concept": str(data.get("missing_concept") or "").strip(),
        }
    except Exception as exc:
        print(f"[Quiz] Grading failed ({type(exc).__name__}); giving neutral credit.")
        return {
            "correct": False,
            "score": 0.5,
            "feedback": "Could not grade automatically. Compare your answer with the expected one.",
            "missing_concept": "",
        }


def dedupe(items: list[str]) -> list[str]:
    """Remove blanks and duplicates (case-insensitive), keeping first-seen order."""
    seen: dict[str, str] = {}
    for item in items:
        item = (item or "").strip()
        if item and item.lower() not in seen:
            seen[item.lower()] = item
    return list(seen.values())


def run_quiz(topic: str, explanation: str) -> QuizResult:
    """Terminal quiz: ask each question with input(), grade it, print feedback."""
    questions = generate_questions(topic, explanation)
    print(f"\n📝 Quiz: {topic} ({len(questions)} question{'s' if len(questions) != 1 else ''})")

    graded: list[QuizQuestion] = []
    weak: list[str] = []
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i} [{q['difficulty']}] {q['question']}")
        code = (q.get("code") or "").strip()
        if code:
            print("\n" + "\n".join(f"    {line}" for line in code.splitlines()) + "\n")
        answer = input("Your answer: ").strip() or NO_ANSWER
        question_text = full_question(q)
        g = grade_answer(question_text, q["expected_answer"], answer)
        print(f"{'✓' if g['correct'] else '✗'} {g['score']:.0%}  {g['feedback']}")
        graded.append(QuizQuestion(
            question=question_text, expected_answer=q["expected_answer"], user_answer=answer,
            correct=g["correct"], feedback=g["feedback"], score=g["score"],
        ))
        weak.append(g["missing_concept"])

    score = sum(q.score for q in graded) / len(graded) if graded else 0.0
    return QuizResult(
        topic=topic, questions=graded, score=score, weak_areas=dedupe(weak),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def extract_explanation(state: dict) -> str:
    """The Explainer's final answer for the current topic, or a fallback.

    The plan's rule is "last AIMessage with content and no tool_calls". On its
    own that is wrong when the Explainer just failed: the last such message
    then belongs to the previous topic (or the Coach). So a set `error` means
    "no explanation this round" and we fall back to the topic itself.
    """
    topic = get_current_topic(state)
    fallback = f"{topic.title}: {topic.description}" if topic else ""
    if state.get("error"):
        return fallback
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return str(message.content)
    return fallback


def quiz_generator_node(state: dict) -> dict:
    """Quiz the learner on the current topic and record the result.

    Reads: roadmap, current_topic_index, messages, error, quiz_results, weak_areas, session_id
    Writes: quiz_results, weak_areas, error (+ roadmap, current_topic_index, session_id passed through)
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "Quiz: no current topic."}
    if state.get("error"):
        print(f"[Quiz] Explainer had a problem ({state['error']}); quizzing on the topic description.")

    result = run_quiz(topic.title, extract_explanation(state))
    print(f"\n[Quiz] Score: {result.score:.0%}")

    return {
        "quiz_results": list(state.get("quiz_results") or []) + [result],
        "weak_areas": dedupe(list(state.get("weak_areas") or []) + result.weak_areas),
        "error": None,
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
