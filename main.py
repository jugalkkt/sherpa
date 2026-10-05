"""Sherpa CLI: python main.py ["learning goal"] [--resume SESSION_ID]"""

import sys
from pathlib import Path

# Order matters: make src/ importable, load .env, THEN import project modules
# (some read env when they build things).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402

from langgraph.types import Command  # noqa: E402

from graph.state import QuizResult, StudyRoadmap, initial_state  # noqa: E402
from graph.workflow import graph  # noqa: E402
from observability.langfuse_setup import flush_langfuse, get_langfuse_config  # noqa: E402

# Creating a FastMCP server (imported via the Explainer) sets the root logger
# to INFO, which would print every HTTP request to Ollama. The CLI talks to
# the user with print(); only warnings and errors should be logged.
logging.getLogger().setLevel(logging.WARNING)
# Tracing is best effort: if Langfuse goes down mid-session, OpenTelemetry
# would log a full traceback for every failed span export.
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

DEFAULT_GOAL = "Learn Python closures and decorators from scratch"


def print_roadmap(roadmap) -> None:
    if roadmap is None:
        print("  (no roadmap)")
        return
    roadmap = StudyRoadmap.from_dict(roadmap)
    print(f"\n📚 {roadmap.goal}")
    print(f"   {roadmap.total_weeks} week(s), {roadmap.weekly_hours} h/week\n")
    for i, t in enumerate(roadmap.topics, 1):
        print(f"  {i}. {t.title}  ({t.estimated_minutes} min)  [{t.status}]")
        print(f"     {t.description}")
        if t.prerequisites:
            print(f"     needs: {', '.join(t.prerequisites)}")
    print()


def print_session_summary(result: dict) -> None:
    print("\n" + "=" * 60 + "\nSession summary\n" + "=" * 60)
    if result.get("error"):
        print(f"⚠️  {result['error']}")
    print_roadmap(result.get("roadmap"))
    results = [QuizResult.from_dict(r) for r in result.get("quiz_results") or []]
    if results:
        avg = sum(r.score for r in results) / len(results)
        passed = sum(r.passed() for r in results)
        print(f"Quizzes: {passed}/{len(results)} passed, average {avg:.0%}")
    if result.get("weak_areas"):
        print(f"Review: {', '.join(result['weak_areas'])}")


def run_session(goal: str, session_id: str | None = None) -> None:
    resuming = session_id is not None
    session_id = session_id or uuid.uuid4().hex[:8]

    print("=" * 60)
    print(f"🏔️  Sherpa  |  session {session_id}  |  {'resuming' if resuming else goal}")
    print("=" * 60)
    config = get_langfuse_config(session_id)

    try:
        _drive(goal, session_id, resuming, config)
    finally:
        # Also on Ctrl+C or errors: those are the traces most worth reading.
        flush_langfuse()


def _drive(goal: str, session_id: str, resuming: bool, config: dict) -> None:
    if resuming and not graph.get_state(config).values:
        print(f"No saved session with id '{session_id}'.")
        return

    try:
        result = graph.invoke(None if resuming else initial_state(goal, session_id), config=config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            if isinstance(payload, dict) and payload.get("roadmap") is not None:
                print_roadmap(payload["roadmap"])
            prompt = payload.get("prompt", "Continue?") if isinstance(payload, dict) else str(payload)
            print(prompt)
            user_input = input("> ")
            result = graph.invoke(Command(resume=user_input), config=config)
    except KeyboardInterrupt:
        print(f"\n\nPaused. Resume with: python main.py --resume {session_id}")
        return
    except Exception as exc:
        if not resuming:
            raise
        print(f"Couldn't resume session '{session_id}': {exc}")
        return

    print_session_summary(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sherpa: a local multi-agent study companion.")
    parser.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="what you want to learn")
    parser.add_argument("--resume", metavar="SESSION_ID", help="continue a saved session")
    args = parser.parse_args()
    run_session(args.goal, session_id=args.resume)


if __name__ == "__main__":
    main()
