"""Sherpa web UI: the same LangGraph as the CLI, with a browser front end.

Run locally:  make streamlit   (or: streamlit run streamlit_app.py)

The graph pauses before the Explainer and before the Quiz Generator
(interrupt_before), so input() never runs inside it: the UI asks the quiz
questions itself with the same generate_questions / grade_answer functions,
then writes the result back with update_state(as_node="quiz_generator").

The sidebar starts the Kaggle model server (see src/hosting/kaggle_server.py).

The learner may upload their own notes on the goal screen (see
src/notes_upload.py). They are saved to a temp folder for this browser
session, and that folder goes into the graph state as study_materials_path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import hmac  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Must be the first Streamlit call (reading st.secrets counts as one).
st.set_page_config(page_title="Sherpa", page_icon="🏔️")


def load_settings() -> None:
    """.env locally; Streamlit secrets when hosted. Project code reads os.environ."""
    load_dotenv(ROOT / ".env")
    try:
        for key, value in st.secrets.to_dict().items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:
        pass  # no secrets.toml: fine locally


load_settings()

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.quiz_generator import (  # noqa: E402
    NO_ANSWER,
    dedupe,
    extract_explanation,
    full_question,
    generate_questions,
    grade_answer,
    quiz_score,
    topic_allows_code,
    weak_areas_of,
)
from graph.state import QuizQuestion, QuizResult, StudyRoadmap, get_current_topic, initial_state  # noqa: E402
from graph.workflow import build_graph  # noqa: E402
from hosting.kaggle_server import idle_minutes, server_status, start_server  # noqa: E402
from notes_sections import load_sections  # noqa: E402
from notes_upload import MAX_FILE_BYTES, MAX_FILES, remove_uploads, save_uploads, validate_uploads  # noqa: E402
from observability.langfuse_setup import flush_langfuse, get_langfuse_config  # noqa: E402

# Same reasons as main.py: FastMCP turns on INFO logging for everything.
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

STATUS_ICONS = {"ready": "🟢", "starting": "🟡", "stopped": "⚪", "error": "🔴", "not_configured": "⚪"}


@st.cache_resource
def get_graph():
    db = os.getenv("UI_CHECKPOINT_DB") or "data/checkpoints_ui.db"
    return build_graph(db_path=db, interrupt_before=["explainer", "quiz_generator"])


S = st.session_state


def reset_session() -> None:
    remove_uploads(S.get("notes_dir"))
    for key in list(S.keys()):
        if key not in ("authed", "server_ready"):
            del S[key]
    S.screen = "goal"


# --- access ----------------------------------------------------------------------------------

def password_ok() -> bool:
    expected = os.getenv("APP_PASSWORD", "")
    if not expected or S.get("authed"):
        return True
    st.title("🏔️ Sherpa")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(attempt, expected):
                S.authed = True
                st.rerun()
            st.error("Wrong password.")
    return False


# --- model server (sidebar) ---------------------------------------------------------------------

@st.fragment(run_every=15)
def server_panel() -> None:
    status = server_status()
    st.markdown(f"**Model server** {STATUS_ICONS.get(status.state, '⚪')} {status.state.replace('_', ' ')}")
    st.caption(status.detail)
    if status.state in ("stopped", "error", "not_configured"):
        if st.button("▶ Start model server", use_container_width=True,
                     disabled=status.state == "not_configured"):
            with st.spinner("Asking Kaggle for a GPU..."):
                ok, message = start_server()
            (st.success if ok else st.error)(message)
    elif status.state == "starting":
        st.caption("Usually takes a few minutes. This panel refreshes on its own.")
    else:
        st.caption(f"Stops by itself after {idle_minutes()} minutes without requests.")
    was_ready = S.get("server_ready", False)
    S.server_ready = status.ready
    if status.ready != was_ready:
        st.rerun()  # update the main page (e.g. enable "Start session")


# --- graph driving ------------------------------------------------------------------------------

def last_ai_text(values: dict) -> str:
    for message in reversed(values.get("messages") or []):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return str(message.content)
    return ""


def advance(result: dict | None = None) -> None:
    """Run the graph until it needs the learner, and pick the screen to show."""
    graph, config = get_graph(), S.config
    while True:
        if result is not None and result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            S.roadmap = StudyRoadmap.from_dict(payload["roadmap"])
            S.screen = "approve"
            return
        snapshot = graph.get_state(config)
        values, nxt = snapshot.values, snapshot.next
        if nxt == ("explainer",):
            if values.get("quiz_results"):  # coming from the Coach
                S.coach_notes.append((values["quiz_results"][-1], last_ai_text(values)))
            topic = get_current_topic(values)
            with st.spinner(f"Explaining **{topic.title}** from your notes..."):
                result = graph.invoke(None, config)
            continue
        if nxt == ("quiz_generator",):
            topic = get_current_topic(values)
            S.topic, S.explanation = topic, extract_explanation(values)
            S.explainer_error = values.get("error")
            with st.spinner("Writing quiz questions..."):
                S.questions = generate_questions(topic.title, S.explanation, allow_code=topic_allows_code(values))
            S.q_index, S.graded, S.weak, S.feedback = 0, [], [], None
            S.screen = "learn"
            return
        # Finished (or stopped early, e.g. the planner failed).
        if values.get("quiz_results") and S.get("screen") != "complete":
            S.coach_notes.append((values["quiz_results"][-1], last_ai_text(values)))
        S.final = values
        S.screen = "complete" if values.get("roadmap") else "goal"
        S.error = values.get("error")
        flush_langfuse()
        return


def start_session(goal: str, uploads: list[tuple[str, bytes]]) -> None:
    remove_uploads(S.get("notes_dir"))  # from an earlier attempt that failed to plan
    S.notes_dir = str(save_uploads(uploads)) if uploads else ""
    S.notes_files = sorted(p.name for p in Path(S.notes_dir).glob("*.md")) if uploads else []
    S.session_id = uuid.uuid4().hex[:8]
    S.config = get_langfuse_config(S.session_id)
    S.coach_notes, S.error = [], None
    state = initial_state(goal, S.session_id, study_materials_path=S.notes_dir)
    with st.spinner("Planning your roadmap..."):
        result = get_graph().invoke(state, S.config)
    advance(result)


def decide_roadmap(approved: bool) -> None:
    with st.spinner("Starting..." if approved else "Planning a new roadmap..."):
        result = get_graph().invoke(Command(resume="yes" if approved else "no"), S.config)
    advance(result)


def submit_answer(answer: str) -> None:
    q = S.questions[S.q_index]
    answer = answer.strip() or NO_ANSWER
    question_text = full_question(q)
    with st.spinner("Grading..."):
        g = grade_answer(question_text, q["expected_answer"], answer)
    S.graded.append(QuizQuestion(question=question_text, expected_answer=q["expected_answer"],
                                 user_answer=answer, correct=g["correct"], feedback=g["feedback"],
                                 score=g["score"]))
    S.weak.append(g["missing_concept"])
    S.feedback = g


def toggle_dispute() -> None:
    """The learner says this question or its grade is wrong (or takes that back)."""
    S.graded[-1].disputed = not S.graded[-1].disputed


def finish_quiz() -> None:
    graph, config = get_graph(), S.config
    result = QuizResult(
        topic=S.topic.title, questions=S.graded,
        score=quiz_score(S.graded), weak_areas=weak_areas_of(S.graded, S.weak),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    values = graph.get_state(config).values
    graph.update_state(config, {
        "quiz_results": list(values.get("quiz_results") or []) + [result],
        "weak_areas": dedupe(list(values.get("weak_areas") or []) + result.weak_areas),
        "roadmap": values.get("roadmap"),
        "current_topic_index": values.get("current_topic_index", 0),
        "error": None,
    }, as_node="quiz_generator")
    with st.spinner("Your coach is reviewing the quiz..."):
        out = graph.invoke(None, config)
    advance(out)


# --- screens ----------------------------------------------------------------------------------

def show_roadmap(roadmap: StudyRoadmap) -> None:
    st.markdown(f"**{roadmap.goal}** · {roadmap.total_weeks} week(s), {roadmap.weekly_hours} h/week")
    labels = {}
    if S.get("notes_dir") and any(t.sources for t in roadmap.topics):
        st.caption("Built from your notes: each topic lists the sections it teaches.")
        labels = {s.id: s.label for s in load_sections(S.notes_dir)}
    for i, t in enumerate(roadmap.topics, 1):
        needs = f"  \n*needs: {', '.join(t.prerequisites)}*" if t.prerequisites else ""
        sources = f"  \n*from: {', '.join(labels.get(sid, sid) for sid in t.sources)}*" if t.sources else ""
        st.markdown(f"{i}. **{t.title}** ({t.estimated_minutes} min): {t.description}{needs}{sources}")


def screen_goal() -> None:
    st.title("🏔️ Sherpa")
    st.write("Tell me what you want to learn. I'll plan a roadmap, explain each topic from "
             "your notes, quiz you, and coach you on what to review.")
    if S.get("error"):
        st.error(S.error)
    goal = st.text_input("Learning goal", value="Learn Python closures and decorators from scratch")
    st.markdown(f"**You can upload up to {MAX_FILES} `.md` or `.txt` files, max {MAX_FILE_BYTES // 1024} KB "
                "each, related to your learning goal.** Without uploads, Sherpa uses its built-in "
                "sample notes (Python closures and decorators).")
    files = st.file_uploader("Your notes (optional)", type=["md", "txt"], accept_multiple_files=True,
                             help="Fixed once the session starts. To change them, use Start over.")
    uploads = [(f.name, f.getvalue()) for f in files or []]
    problems = validate_uploads(uploads)
    for problem in problems:
        st.error(problem)
    ready = S.get("server_ready", False)
    if not ready:
        st.info("Start the model server in the sidebar first. It takes a few minutes to come up.")
    if st.button("Start session", type="primary", disabled=not ready or not goal.strip() or bool(problems)):
        start_session(goal.strip(), uploads)
        st.rerun()


def screen_approve() -> None:
    st.title("Your roadmap")
    show_roadmap(S.roadmap)
    left, right = st.columns(2)
    if left.button("✅ Looks good, start", type="primary", use_container_width=True):
        decide_roadmap(True)
        st.rerun()
    if right.button("🔄 Make a new one", use_container_width=True):
        decide_roadmap(False)
        st.rerun()


def coach_notes() -> None:
    for result, note in S.get("coach_notes", [])[-1:]:
        result = QuizResult.from_dict(result)
        st.info(f"**Coach on {result.topic}** ({result.score:.0%}): {note}")


def screen_learn() -> None:
    roadmap = StudyRoadmap.from_dict(get_graph().get_state(S.config).values["roadmap"])
    index = next(i for i, t in enumerate(roadmap.topics) if t.title == S.topic.title)
    st.progress(index / len(roadmap.topics), text=f"Topic {index + 1} of {len(roadmap.topics)}")
    coach_notes()
    st.title(S.topic.title)
    if S.get("explainer_error"):
        st.warning(f"The explainer had a problem ({S.explainer_error}); the quiz uses the topic summary.")
    with st.expander("Explanation", expanded=S.q_index == 0 and S.feedback is None):
        st.markdown(S.explanation)

    q = S.questions[S.q_index]
    st.subheader(f"Question {S.q_index + 1} of {len(S.questions)} · {q['difficulty']}")
    st.markdown(q["question"])
    if q.get("code"):
        st.code(q["code"], language="python")

    if S.feedback is None:
        with st.form(f"answer-{S.topic.title}-{S.q_index}"):
            answer = st.text_area("Your answer")
            if st.form_submit_button("Submit", type="primary"):
                submit_answer(answer)
                st.rerun()
    else:
        g, graded = S.feedback, S.graded[-1]
        (st.success if g["correct"] else st.error)(f"{'✓' if g['correct'] else '✗'} {g['score']:.0%} · {g['feedback']}")
        with st.expander("Expected answer"):
            st.text(graded.expected_answer)  # monospace: it may contain real program output
            st.caption("For code questions, the output above comes from actually running the code.")
        if graded.disputed:
            st.warning("Marked as a bad question or grade: it won't count toward your score.")
            if st.button("Count this question after all", key=f"undispute-{S.q_index}"):
                toggle_dispute()
                st.rerun()
        elif st.button("🚩 This grade is wrong: don't count this question", key=f"dispute-{S.q_index}"):
            toggle_dispute()
            st.rerun()
        last = S.q_index + 1 == len(S.questions)
        if st.button("Finish quiz" if last else "Next question", type="primary"):
            if last:
                finish_quiz()
            else:
                S.q_index, S.feedback = S.q_index + 1, None
            st.rerun()


def screen_complete() -> None:
    values = S.final
    st.title("🎉 Session complete")
    coach_notes()
    if S.get("error"):
        st.warning(S.error)
    roadmap = StudyRoadmap.from_dict(values["roadmap"])
    results = {QuizResult.from_dict(r).topic: QuizResult.from_dict(r) for r in values.get("quiz_results") or []}
    for t in roadmap.topics:
        r = results.get(t.title)
        mark = "✓" if t.status == "completed" else ("✗" if t.status == "needs_review" else "·")
        st.markdown(f"{mark} **{t.title}**: {f'{r.score:.0%}' if r else 'not quizzed'} ({t.status})")
    if results:
        avg = sum(r.score for r in results.values()) / len(results)
        st.metric("Average score", f"{avg:.0%}")
    if values.get("weak_areas"):
        st.markdown("**Review:** " + ", ".join(values["weak_areas"]))
    if st.button("New session", type="primary"):
        reset_session()
        st.rerun()


SCREENS = {"goal": screen_goal, "approve": screen_approve, "learn": screen_learn, "complete": screen_complete}


def main() -> None:
    if not password_ok():
        return
    with st.sidebar:
        server_panel()
        if S.get("session_id"):
            st.caption(f"Session {S.session_id}")
            st.caption("📄 Notes: " + (", ".join(S.get("notes_files") or []) or "built-in samples"))
            if st.button("Start over"):
                reset_session()
                st.rerun()
        if not os.getenv("APP_PASSWORD"):
            st.caption("⚠️ No APP_PASSWORD set: anyone with the link can start your Kaggle GPU.")
    S.setdefault("screen", "goal")
    SCREENS[S.screen]()


main()
