"""The learner's uploaded notes, split into sections a roadmap can point at.

A section starts at a "#" or "##" heading outside code blocks; deeper
headings stay inside their section. Text before the first heading is a
section of its own (named after the file), and a file with no headings is a
single section. Ids look like "closures.md#3" and are stable for a session,
because uploads can't change once it starts.

The Planner may only group these ids into topics (enforced by the JSON
schema and again in code), and the Explainer is given the topic's sections
directly instead of searching for them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from mcp_servers.filesystem_server import list_study_files, read_study_file, use_notes_dir

# ~3.5k tokens: one topic's notes must fit next to the Explainer's prompt and
# reply inside the hosted model's 8,192-token context.
MAX_MATERIAL_CHARS = 12_000

_HEADING = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
_FENCED_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = {"python", "py", "python3"}
# An untagged block counts as Python only if it does something, so that a
# bare word or a sentence that happens to parse isn't mistaken for code.
_CODE_NODES = (ast.Call, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.ClassDef, ast.Import, ast.ImportFrom, ast.For, ast.While, ast.With, ast.Return)


@dataclass
class Section:
    id: str
    file: str
    heading: str
    text: str
    has_python_code: bool

    @property
    def label(self) -> str:
        return f"{self.file} › {self.heading}"


def has_python_code(text: str) -> bool:
    for tag, body in _FENCED_BLOCK.findall(text):
        tag = tag.strip().lower()
        if tag in _PYTHON_TAGS:
            return True
        if not tag:
            try:
                tree = ast.parse(body)
            except SyntaxError:
                continue
            if any(isinstance(node, _CODE_NODES) for node in ast.walk(tree)):
                return True
    return False


def parse_sections(file: str, text: str) -> list[Section]:
    """Split one file into sections. Sections with no text besides a heading are dropped."""
    # (heading, heading line or "", body lines); the first one holds the text before any heading.
    parts: list[tuple[str, str, list[str]]] = [(Path(file).stem, "", [])]
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line.strip())
        if match:
            parts.append((match.group(2).strip(), line, []))
        else:
            parts[-1][2].append(line)

    sections: list[Section] = []
    for heading, heading_line, body in parts:
        if not "\n".join(body).strip():
            continue
        content = "\n".join([heading_line, *body] if heading_line else body).strip()
        sections.append(Section(
            id=f"{file}#{len(sections) + 1}", file=file, heading=heading,
            text=content, has_python_code=has_python_code(content),
        ))
    return sections


def load_sections(base: str | Path) -> list[Section]:
    """Every section of the notes in `base`, in file then document order.

    Reads through the notes server's own functions, so its rules (Markdown
    only, nothing outside the folder) apply here too.
    """
    with use_notes_dir(base):
        sections: list[Section] = []
        for name in list_study_files():
            text = read_study_file(name)
            if not text.startswith("Error:"):
                sections.extend(parse_sections(name, text))
    return sections


def topic_material(base: str | Path, ids: list[str], max_chars: int = MAX_MATERIAL_CHARS) -> tuple[str, bool, bool]:
    """The text of the given sections, for the Explainer.

    Returns (text, truncated, has_python_code). Unknown ids are skipped.
    """
    wanted = set(ids)
    chosen = [s for s in load_sections(base) if s.id in wanted]
    text = "\n\n".join(f"[{s.label}]\n{s.text}" for s in chosen)
    truncated = len(text) > max_chars
    return text[:max_chars], truncated, any(s.has_python_code for s in chosen)
