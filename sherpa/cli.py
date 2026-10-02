"""Command-line interface: `python -m sherpa <command>`."""

from __future__ import annotations

import tempfile
from pathlib import Path

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


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(None, help="Files or folders to ingest (default: NOTES_DIR)."),
    reset: bool = typer.Option(False, "--reset", help="Delete the index and re-embed everything."),
    prune: bool = typer.Option(False, "--prune", help="Remove chunks of files that no longer exist."),
) -> None:
    """Load, chunk and embed your notes (.md, .txt, .pdf) into the local vector index."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    from sherpa.ingest.index import EmbeddingModelMismatch
    from sherpa.ingest.pipeline import run_ingest
    from sherpa.llm import LLMAuthError, LLMUnavailableError

    s = _load_settings()
    paths = paths or [s.notes_dir]
    missing = [p for p in paths if not p.exists()]
    if missing:
        console.print(f"[red]Not found:[/red] {', '.join(map(str, missing))}")
        raise typer.Exit(1)

    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        task = None

        def on_file(source: str, n_chunks: int) -> None:
            nonlocal task
            if task is not None:
                progress.remove_task(task)
            task = progress.add_task(f"Embedding {source}", total=n_chunks)

        def on_progress(n: int) -> None:
            progress.advance(task, n)

        try:
            report = run_ingest(paths, reset=reset, prune=prune, on_file=on_file, on_progress=on_progress)
        except (EmbeddingModelMismatch, LLMUnavailableError, LLMAuthError) as e:
            progress.stop()
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    table = Table(title="Ingest summary", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("files scanned", str(report.scanned))
    table.add_row("new", ", ".join(report.new) or "-")
    table.add_row("changed", ", ".join(report.changed) or "-")
    table.add_row("unchanged (skipped)", str(len(report.unchanged)))
    if report.pruned:
        table.add_row("pruned", ", ".join(report.pruned))
    if report.failed:
        table.add_row("[red]no text extracted[/red]", ", ".join(report.failed))
    table.add_row("chunks added / removed", f"{report.chunks_added} / {report.chunks_removed}")
    console.print(table)
    for w in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if report.new or report.changed:
        console.print(f"{len(report.new) + len(report.changed)} file(s) changed")
    else:
        console.print("0 changed")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for in your notes."),
    k: int = typer.Option(None, "-k", help="Number of results (default: RETRIEVAL_TOP_K)."),
) -> None:
    """Debug retrieval: show the note chunks most similar to QUERY."""
    from sherpa.ingest.index import EmbeddingModelMismatch, NotesIndex
    from sherpa.llm import LLMAuthError, LLMUnavailableError

    s = _load_settings()
    try:
        hits = NotesIndex(s).query(query, k)
    except (EmbeddingModelMismatch, LLMUnavailableError, LLMAuthError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not hits:
        console.print("Index is empty. Run `python -m sherpa ingest` first.")
        raise typer.Exit(1)

    for rank, h in enumerate(hits, start=1):
        weak = h.similarity < s.retrieval_min_similarity
        colour = "dim" if weak else "green"
        label = " (below RETRIEVAL_MIN_SIMILARITY)" if weak else ""
        console.rule(f"[{colour}]#{rank}  sim {h.similarity:.2f}{label}[/{colour}]  {h.location}", align="left")
        preview = h.text if len(h.text) <= 400 else h.text[:400] + " …"
        console.print(preview, markup=False, highlight=False)
