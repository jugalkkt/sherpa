"""The learner's own notes, uploaded in the web app for one browser session.

Uploads are checked here (no Streamlit code, so it is testable), then saved
to a fresh temp folder whose path goes into the graph state as
``study_materials_path``. The Explainer reads from that folder, and the
Planner builds the roadmap from it.

Limits are small because the hosted model runs with an 8,192-token context
(CONTEXT_LENGTH in deploy/kaggle/sherpa_ollama_server.py): one whole file
must fit next to the Explainer's prompt and reply. Raise them together with
the context once a bigger model is hosted.

.txt files are saved as .md, so the notes server keeps its Markdown-only rule.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

MAX_FILES = 5
MAX_FILE_BYTES = 15 * 1024
ALLOWED_SUFFIXES = (".md", ".txt")


def saved_name(name: str) -> str:
    """The file name on disk: no folders, and .txt becomes .md."""
    path = Path(name.replace("\\", "/")).name
    stem, suffix = Path(path).stem, Path(path).suffix.lower()
    return f"{stem}.md" if suffix == ".txt" else path


def validate_uploads(files: list[tuple[str, bytes]]) -> list[str]:
    """One readable error per problem; an empty list means the files can be saved."""
    errors: list[str] = []
    if len(files) > MAX_FILES:
        errors.append(f"Too many files: {len(files)}. Upload at most {MAX_FILES}.")
    seen: dict[str, str] = {}
    for name, data in files:
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            errors.append(f"{name}: only .md and .txt files are allowed.")
            continue
        if len(data) > MAX_FILE_BYTES:
            errors.append(f"{name}: {len(data) / 1024:.1f} KB, the limit is {MAX_FILE_BYTES // 1024} KB.")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: not UTF-8 text.")
            continue
        if not text.strip():
            errors.append(f"{name}: the file is empty.")
            continue
        target = saved_name(name)
        if target in seen:
            errors.append(f"{name}: clashes with {seen[target]} (both would be saved as {target}).")
            continue
        seen[target] = name
    return errors


def save_uploads(files: list[tuple[str, bytes]]) -> Path:
    """Write already-validated files to a new temp folder and return it."""
    folder = Path(tempfile.mkdtemp(prefix="sherpa-notes-"))
    for name, data in files:
        (folder / saved_name(name)).write_bytes(data)
    return folder


def remove_uploads(folder: str | Path | None) -> None:
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
