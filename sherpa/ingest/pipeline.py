"""The ingest pipeline: files -> load -> chunk -> embed -> Chroma, idempotently.

A manifest (data/ingest_manifest.json) remembers, per file, the SHA-256 of its bytes
and the chunk ids it produced:

    {"embed_model": "nomic-embed-text",
     "files": {"notes/a.md": {"sha256": "...", "title": "...", "chunk_ids": ["..."]}}}

So re-running ingest only re-embeds files whose content changed, and a changed file's
old chunks are deleted first (no stale duplicates in search results).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sherpa.config import Settings, get_settings
from sherpa.ingest.chunking import chunk_documents
from sherpa.ingest.index import EmbeddingModelMismatch, NotesIndex
from sherpa.ingest.loaders import display_path, iter_files, load_file


@dataclass
class IngestReport:
    scanned: int = 0
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    warnings: list[str] = field(default_factory=list)


def load_manifest(settings: Settings) -> dict:
    path = settings.manifest_path
    if path.exists():
        return json.loads(path.read_text())
    return {"embed_model": settings.embed_model, "files": {}}


def save_manifest(settings: Settings, manifest: dict) -> None:
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings.manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(settings.manifest_path)  # atomic: a crash never leaves half a manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_titles(settings: Settings | None = None) -> list[str]:
    """Distinct document titles in the index (fed to the curriculum planner)."""
    files = load_manifest(settings or get_settings())["files"]
    return sorted({entry["title"] for entry in files.values()})


def run_ingest(
    paths: list[Path],
    *,
    reset: bool = False,
    prune: bool = False,
    on_file: Callable[[str, int], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Ingest `paths` (files or folders). `on_file(source, n_chunks)` fires before a file
    is embedded and `on_progress(n)` after each embedded batch (for progress bars)."""
    settings = settings or get_settings()
    index = NotesIndex(settings)
    report = IngestReport()

    if reset:
        index.reset()
        manifest = {"embed_model": settings.embed_model, "files": {}}
        save_manifest(settings, manifest)
    else:
        manifest = load_manifest(settings)
        if manifest["files"] and manifest.get("embed_model") != settings.embed_model:
            raise EmbeddingModelMismatch(
                f"notes were embedded with {manifest.get('embed_model')!r} but EMBED_MODEL is "
                f"{settings.embed_model!r}; run `python -m sherpa ingest --reset`"
            )
    index.collection  # creates the collection and runs the model check

    files = iter_files(paths)
    report.scanned = len(files)

    for path in files:
        source = display_path(path)
        sha = _sha256(path)
        previous = manifest["files"].get(source)

        # Unchanged content AND its chunks really are in Chroma -> nothing to do.
        if previous and previous["sha256"] == sha and index.has_ids(previous["chunk_ids"]):
            report.unchanged.append(source)
            continue

        loaded = load_file(path)
        report.warnings += loaded.warnings
        chunks = chunk_documents(loaded.documents)
        if not chunks:
            report.failed.append(source)
            continue

        if previous:
            index.delete_ids(previous["chunk_ids"])
            report.chunks_removed += len(previous["chunk_ids"])

        if on_file:
            on_file(source, len(chunks))
        index.add_chunks(chunks, on_progress=on_progress)
        report.chunks_added += len(chunks)
        (report.changed if previous else report.new).append(source)

        manifest["files"][source] = {
            "sha256": sha,
            "title": loaded.documents[0].metadata["title"],
            "chunk_ids": [c.id for c in chunks],
        }
        save_manifest(settings, manifest)  # after every file: a crash loses at most one file

    if prune:
        for source in [s for s in manifest["files"] if not Path(s).exists()]:
            ids = manifest["files"].pop(source)["chunk_ids"]
            index.delete_ids(ids)
            report.chunks_removed += len(ids)
            report.pruned.append(source)
        save_manifest(settings, manifest)

    return report
