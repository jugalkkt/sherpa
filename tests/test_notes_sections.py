import pytest

from notes_sections import has_python_code, load_sections, parse_sections, topic_material

pytestmark = pytest.mark.unit


def headings(sections):
    return [s.heading for s in sections]


def test_splits_at_h1_and_h2_and_keeps_deeper_headings_inside():
    text = "# Agents\nIntro.\n## Tools\nTools act.\n### Detail\nMore.\n## Memory\nKeeps state.\n"
    sections = parse_sections("a.md", text)
    assert headings(sections) == ["Agents", "Tools", "Memory"]
    assert [s.id for s in sections] == ["a.md#1", "a.md#2", "a.md#3"]
    assert sections[1].text == "## Tools\nTools act.\n### Detail\nMore."


def test_hash_lines_inside_code_blocks_are_not_headings():
    text = "## Loops\n```python\n# a comment\nfor i in range(3):\n    print(i)\n```\n"
    [section] = parse_sections("a.md", text)
    assert section.heading == "Loops" and "# a comment" in section.text


def test_text_before_the_first_heading_is_named_after_the_file():
    sections = parse_sections("notes.md", "Some intro text.\n## Part\nBody.\n")
    assert headings(sections) == ["notes", "Part"]


def test_file_without_headings_is_one_section():
    [section] = parse_sections("plain.md", "line one\nline two\n")
    assert section.heading == "plain" and section.text == "line one\nline two"


def test_heading_with_no_body_is_dropped_and_ids_stay_sequential():
    sections = parse_sections("a.md", "# Title\n## One\nBody 1.\n## Empty\n\n## Two\nBody 2.\n")
    assert headings(sections) == ["One", "Two"] and [s.id for s in sections] == ["a.md#1", "a.md#2"]


@pytest.mark.parametrize("text, expected", [
    ("```python\nx = 1\n```", True),
    ("```py\nprint(1)\n```", True),
    ("```\ndef f():\n    return 1\n```", True),       # untagged, parses, does something
    ("```\nhello\n```", False),                        # parses, but is just a name
    ("```\nThis is a sentence.\n```", False),          # doesn't parse
    ("```javascript\nconst x = 1;\n```", False),
    ("No code at all.", False),
])
def test_has_python_code(text, expected):
    assert has_python_code(text) is expected


def test_load_sections_reads_every_markdown_file_in_order(tmp_path):
    (tmp_path / "b.md").write_text("## B\nb body\n")
    (tmp_path / "a.md").write_text("## A\na body\n")
    (tmp_path / "skip.txt").write_text("not markdown")
    assert [s.id for s in load_sections(tmp_path)] == ["a.md#1", "b.md#1"]


def test_topic_material_joins_the_chosen_sections_and_reports_code(tmp_path):
    (tmp_path / "a.md").write_text("## One\nfirst\n## Two\n```python\nprint(2)\n```\n## Three\nthird\n")
    text, truncated, code = topic_material(tmp_path, ["a.md#3", "a.md#2", "missing#1"])
    assert text.startswith("[a.md › Two]\n## Two") and "[a.md › Three]" in text and "first" not in text
    assert not truncated and code


def test_topic_material_is_capped(tmp_path):
    (tmp_path / "a.md").write_text("## Long\n" + "x" * 500)
    text, truncated, code = topic_material(tmp_path, ["a.md#1"], max_chars=100)
    assert len(text) == 100 and truncated and not code
