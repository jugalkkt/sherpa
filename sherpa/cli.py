"""Command-line interface: `python -m sherpa <command>`."""

from __future__ import annotations

import tempfile

import typer
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from sherpa.llm import Check, get_chat, healthcheck

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Sherpa: AI study companion.")
console = Console()


@app.callback()
def main() -> None:
    """Sherpa: AI study companion."""


def _load_settings():
    from sherpa.config import get_settings

    try:
        return get_settings()
    except ValidationError as e:
        console.print("[red]Invalid configuration in .env:[/red]")
        for err in e.errors():
            console.print(f"  {'.'.join(map(str, err['loc'])).upper()}: {err['msg']}")
        raise typer.Exit(1)


def _check_data_dir(settings) -> Check:
    try:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.chroma_dir):
            pass
        return Check("data dir", True, f"{settings.data_dir.resolve()} writable")
    except OSError as e:
        return Check("data dir", False, f"{settings.data_dir}: {e}")


def _structured_check() -> Check:
    from sherpa.schemas import CurriculumDraft
    from sherpa.structured import invoke_structured

    s = _load_settings()
    messages = [
        SystemMessage(
            content=(
                "You are a curriculum designer. Return a learning plan as JSON. "
                f"Use {s.min_topics}-{s.max_topics} topics with ids t1, t2, ... in order; "
                "prerequisites may only list earlier ids; each topic has 2-4 objectives "
                "starting with an action verb; est_minutes between 10 and 45."
            )
        ),
        HumanMessage(content="Learning goal: Learn linear regression. Learner level: beginner."),
    ]
    try:
        plan = invoke_structured(get_chat("planner"), CurriculumDraft, messages)
    except Exception as e:
        return Check("structured output", False, str(e))
    titles = " → ".join(f"{t.id} {t.title}" for t in plan.topics)
    return Check("structured output", True, f"{len(plan.topics)} topics: {titles}")


@app.command()
def doctor(
    quick: bool = typer.Option(False, "--quick", help="Only check endpoint, auth and models (no LLM calls)."),
    structured: bool = typer.Option(
        False, "--structured", help="Also generate a sample curriculum to test structured output."
    ),
) -> None:
    """Check the LLM endpoint, models and local data directory."""
    s = _load_settings()
    console.print(
        f"Endpoint [bold]{s.ollama_base_url}[/bold] · chat [bold]{s.chat_model}[/bold] · "
        f"embed [bold]{s.embed_model}[/bold] · auth {'set' if s.ollama_basic_auth else '[yellow]not set[/yellow]'}"
    )

    with console.status("Checking endpoint..."):
        checks = healthcheck(quick=quick)
    checks.append(_check_data_dir(s))
    if structured and all(c.ok for c in checks):
        with console.status("Generating a sample curriculum..."):
            checks.append(_structured_check())

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for c in checks:
        table.add_row(c.name, "[green]OK[/green]" if c.ok else "[red]FAIL[/red]", c.detail)
    console.print(table)

    if not all(c.ok for c in checks):
        raise typer.Exit(1)
