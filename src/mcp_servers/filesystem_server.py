"""MCP server exposing the learner's Markdown notes, read-only.

Each tool is also a plain Python function (FastMCP's decorators return the
function unchanged), so the Explainer can import and call them in-process.
Tools never raise: every failure comes back as an "Error: ..." string.

Run standalone (stdio transport): python src/mcp_servers/filesystem_server.py
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SEARCH_RESULTS = 20


def _notes_base() -> Path:
    # Relative paths are anchored at the project root, not the current
    # directory, so the server finds the notes however it is launched.
    path = Path(os.getenv("NOTES_PATH", "study_materials/sample_notes"))
    return path if path.is_absolute() else PROJECT_ROOT / path


NOTES_BASE = _notes_base()

mcp = FastMCP("Filesystem Server")


def _md_files() -> list[Path]:
    if not NOTES_BASE.is_dir():
        return []
    return sorted(p for p in NOTES_BASE.rglob("*.md") if p.is_file())


def _relative(path: Path) -> str:
    return path.relative_to(NOTES_BASE).as_posix()


@mcp.tool()
def list_study_files() -> list[str]:
    """List every Markdown study note available. Call this first.

    Returns file paths relative to the notes folder (e.g. "closures.md"),
    which you can pass to read_study_file. Returns an empty list if there
    are no notes.
    """
    return [_relative(p) for p in _md_files()]


@mcp.tool()
def read_study_file(filename: str) -> str:
    """Read the full text of one Markdown study note.

    Args:
        filename: a path exactly as returned by list_study_files,
            e.g. "closures.md".
    """
    base = NOTES_BASE.resolve()
    # resolve() collapses "..", follows symlinks and makes absolute paths win,
    # so checking the result stays under base blocks every escape route.
    target = (base / filename).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return f"Error: access denied. '{filename}' is outside the notes folder."

    if target.suffix.lower() != ".md":
        return f"Error: only Markdown (.md) notes can be read, not '{filename}'."
    if not target.is_file():
        available = ", ".join(list_study_files()) or "(none)"
        return f"Error: '{filename}' not found. Available files: {available}"
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error: could not read '{filename}': {exc}"


@mcp.tool()
def search_notes(query: str) -> list[dict]:
    """Find lines in the notes that contain the query (case-insensitive).

    Use a short keyword or phrase such as "nonlocal" or "late binding".
    Returns up to 20 matches as {file, line_number, line}; use the file
    names with read_study_file to read the surrounding context.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return []
    results: list[dict] = []
    for path in _md_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            if needle in line.lower():
                results.append({"file": _relative(path), "line_number": number, "line": line.strip()})
                if len(results) >= MAX_SEARCH_RESULTS:
                    return results
    return results


@mcp.resource("notes://index")
def notes_index() -> str:
    """A Markdown index of all study notes with their sizes."""
    files = _md_files()
    if not files:
        return f"# Study notes\n\nNo notes found in `{NOTES_BASE}`."
    rows = [f"- `{_relative(p)}` ({p.stat().st_size / 1024:.1f} KB)" for p in files]
    return f"# Study notes ({len(files)} files)\n\n" + "\n".join(rows)


if __name__ == "__main__":
    mcp.run()
