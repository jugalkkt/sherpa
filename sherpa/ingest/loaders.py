"""Step 1 of ingestion: turn files on disk into `Document`s (text + metadata).

One Markdown/text file -> one Document.  One PDF -> one Document per page, so
citations can point at a page number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)  # source, title, kind, page?


@dataclass
class LoadResult:
    documents: list[Document]
    warnings: list[str]


def display_path(path: Path) -> str:
    """Path relative to the working directory when possible (used as the `source` id)."""
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def iter_files(paths: list[Path]) -> list[Path]:
    """Expand directories recursively; keep supported, non-hidden files; stable order."""
    found: set[Path] = set()
    for p in paths:
        if p.is_dir():
            candidates = (f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            candidates = iter([p])
        else:
            continue
        for f in candidates:
            hidden = any(part.startswith(".") for part in f.relative_to(p if p.is_dir() else f.parent).parts)
            if f.suffix.lower() in SUPPORTED_SUFFIXES and not hidden:
                found.add(f.resolve())
    return sorted(found)


def _tidy(text: str) -> str:
    """Strip trailing spaces and collapse 3+ blank lines. Leading indentation is kept
    (Markdown code blocks depend on it)."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_markdown(path: Path) -> LoadResult:
    text = _tidy(path.read_text(encoding="utf-8", errors="replace"))
    h1 = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = h1.group(1).strip() if h1 else path.stem
    meta = {"source": display_path(path), "title": title, "kind": "md"}
    return LoadResult([Document(text, meta)] if text else [], [] if text else [f"{path.name}: empty file"])


def _load_text(path: Path) -> LoadResult:
    text = _tidy(path.read_text(encoding="utf-8", errors="replace"))
    meta = {"source": display_path(path), "title": path.stem, "kind": "txt"}
    return LoadResult([Document(text, meta)] if text else [], [] if text else [f"{path.name}: empty file"])


def _load_pdf(path: Path) -> LoadResult:
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    warnings: list[str] = []
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            return LoadResult([], [f"{path.name}: encrypted PDF, skipped"])
        raw_pages = [page.extract_text() or "" for page in reader.pages]
        title = ((reader.metadata.title if reader.metadata else None) or "").strip() or path.stem
    except Exception as e:  # corrupt / unsupported PDF must not stop the whole ingest
        return LoadResult([], [f"{path.name}: could not read PDF ({type(e).__name__}: {e})"])

    pages = [_tidy(re.sub(r"[ \t]+", " ", p)) for p in raw_pages]
    docs: list[Document] = []
    for number, text in enumerate(pages, start=1):
        text = text.strip()
        if not text:
            warnings.append(f"{path.name}: page {number} has no extractable text (scanned image?)")
            continue
        meta = {"source": display_path(path), "title": title, "kind": "pdf", "page": number}
        docs.append(Document(text, meta))
    return LoadResult(docs, warnings)


def load_file(path: Path) -> LoadResult:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return _load_markdown(path)
    if suffix == ".txt":
        return _load_text(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    return LoadResult([], [f"{path.name}: unsupported file type"])
