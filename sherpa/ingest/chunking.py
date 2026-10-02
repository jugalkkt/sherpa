"""Step 2 of ingestion: split Documents into retrieval-sized Chunks.

Why chunk at all? An embedding is one vector per text. A whole file squeezed into
one vector blurs every topic in it together; small, focused chunks give sharp
vectors, and only the relevant pieces get pasted into the LLM's limited context.

Markdown is first cut at headings (so a chunk never mixes two sections), then any
long section is cut further by size. Every chunk starts with its heading path,
e.g. "Regression Basics > Gradient descent > The learning rate", so both the
embedding and the LLM know where the text came from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sherpa.ingest.loaders import Document

CHUNK_SIZE = 1200  # characters, about 300 tokens
CHUNK_OVERLAP = 150  # repeated between neighbours so a sentence cut in half survives in one of them

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict  # Document metadata + chunk_index (+ section for Markdown)


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_path, body) pairs, cutting at #, ## and ### headings.

    Headings inside ``` code fences are ignored. A heading with no body of its own
    (like a top-level title right before a ## heading) is folded into the next section.
    """
    sections: list[tuple[str, list[str]]] = []
    stack: list[str] = []  # current heading path, index = level - 1
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in lines):
            sections.append((" > ".join(stack), lines.copy()))
        lines.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1] + [m.group(2)]
            continue  # the heading itself lives in the section path, not the body
        lines.append(line)
    flush()
    return [(path, "\n".join(body).strip()) for path, body in sections]


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    """Deterministic id: same file content -> same ids, so re-ingesting is idempotent."""
    return hashlib.sha1(f"{source}|{page}|{index}|{text}".encode()).hexdigest()[:16]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # try to cut at paragraph, then line, then sentence, then word boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc in documents:
        if doc.metadata.get("kind") == "md":
            pieces = [
                (section, part)
                for section, body in _split_markdown_sections(doc.text)
                for part in splitter.split_text(body)
            ]
        else:
            pieces = [(None, part) for part in splitter.split_text(doc.text)]

        for index, (section, part) in enumerate(pieces):
            text = f"{section}\n\n{part}" if section else part
            meta = {**doc.metadata, "chunk_index": index}
            if section:
                meta["section"] = section
            chunk_id = _chunk_id(meta["source"], meta.get("page"), index, text)
            chunks.append(Chunk(chunk_id, text, meta))
    return chunks
