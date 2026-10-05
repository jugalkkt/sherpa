import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

import mcp_servers.filesystem_server as fs
import mcp_servers.memory_server as mem
from mcp_servers.filesystem_server import list_study_files, notes_index, read_study_file, search_notes
from mcp_servers.memory_server import (
    memory_delete,
    memory_get,
    memory_list_keys,
    memory_set,
    session_summary,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def notes(tmp_path, monkeypatch) -> Path:
    base = tmp_path / "notes"
    (base / "advanced").mkdir(parents=True)
    (base / "closures.md").write_text("# Closures\nA closure remembers.\nUse NONLOCAL to rebind.\n")
    (base / "basics.md").write_text("# Basics\nnames and scope\n")
    (base / "advanced" / "decorators.md").write_text("# Decorators\nwrap a closure\n")
    (base / "secret.txt").write_text("not markdown")
    (tmp_path / "outside.md").write_text("TOP SECRET")
    monkeypatch.setattr(fs, "NOTES_BASE", base)
    return base



# --- list_study_files ---------------------------------------------------------------

def test_list_is_sorted_relative_and_recursive(notes):
    assert list_study_files() == ["advanced/decorators.md", "basics.md", "closures.md"]


def test_list_ignores_non_markdown(notes):
    assert "secret.txt" not in list_study_files()


def test_list_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "NOTES_BASE", tmp_path)
    assert list_study_files() == []


def test_list_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "NOTES_BASE", tmp_path / "nope")
    assert list_study_files() == []


def test_default_notes_path_is_anchored_at_project_root(monkeypatch):
    monkeypatch.delenv("NOTES_PATH", raising=False)
    assert fs._notes_base() == ROOT / "study_materials" / "sample_notes"
    monkeypatch.setenv("NOTES_PATH", "/abs/notes")
    assert fs._notes_base() == Path("/abs/notes")


def test_real_sample_notes_are_listed():
    assert {"python_basics.md", "closures.md", "decorators.md"} <= set(list_study_files())


# --- read_study_file ----------------------------------------------------------------

def test_read_existing_file(notes):
    assert read_study_file("closures.md").startswith("# Closures")


def test_read_nested_file(notes):
    assert "wrap a closure" in read_study_file("advanced/decorators.md")


def test_read_missing_file_lists_available(notes):
    out = read_study_file("generators.md")
    assert out.startswith("Error:") and "not found" in out
    assert "closures.md" in out and "basics.md" in out


@pytest.mark.parametrize("attack", [
    "../outside.md", "../../.env", "advanced/../../outside.md", "/etc/passwd",
])
def test_path_traversal_blocked(notes, attack):
    out = read_study_file(attack)
    assert out.startswith("Error: access denied")
    assert "TOP SECRET" not in out


def test_symlink_escape_blocked(notes, tmp_path):
    (notes / "link.md").symlink_to(tmp_path / "outside.md")
    assert read_study_file("link.md").startswith("Error: access denied")


def test_non_markdown_rejected(notes):
    out = read_study_file("secret.txt")
    assert out.startswith("Error:") and ".md" in out
    assert "not markdown" not in out


def test_directory_is_not_a_file(notes, tmp_path):
    (notes / "folder.md").mkdir()
    assert "not found" in read_study_file("folder.md")


def test_read_never_raises_on_bad_encoding(notes):
    (notes / "binary.md").write_bytes(b"\xff\xfe\x00bad")
    assert read_study_file("binary.md").startswith("Error: could not read")


# --- search_notes ------------------------------------------------------------------

def test_search_hit_shape(notes):
    hits = search_notes("remembers")
    assert hits == [{"file": "closures.md", "line_number": 2, "line": "A closure remembers."}]


def test_search_is_case_insensitive(notes):
    assert [h["line_number"] for h in search_notes("nonlocal")] == [3]
    assert search_notes("NoNlOcAl") == search_notes("nonlocal")


def test_search_spans_files(notes):
    files = {h["file"] for h in search_notes("closure")}
    assert files == {"closures.md", "advanced/decorators.md"}


def test_search_capped_at_20(notes):
    (notes / "many.md").write_text("\n".join(f"match {i}" for i in range(50)))
    assert len(search_notes("match")) == 20


@pytest.mark.parametrize("query", ["generators", "", "   "])
def test_search_no_matches(notes, query):
    assert search_notes(query) == []


# --- notes://index ------------------------------------------------------------------

def test_index_lists_files_with_sizes(notes):
    index = notes_index()
    assert "3 files" in index
    assert "`closures.md`" in index and "`advanced/decorators.md`" in index
    assert "KB)" in index


def test_index_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "NOTES_BASE", tmp_path)
    assert "No notes found" in notes_index()


# --- memory -------------------------------------------------------------------------

def test_set_then_get():
    assert "Saved 'topic'" in memory_set("s1", "topic", "closures")
    assert memory_get("s1", "topic") == "closures"


def test_set_overwrites():
    memory_set("s1", "k", "old")
    memory_set("s1", "k", "new")
    assert memory_get("s1", "k") == "new"


def test_set_stores_timestamp():
    memory_set("s1", "k", "v")
    assert mem._store["s1"]["k"]["updated_at"].startswith("20")


def test_missing_key_returns_null_string():
    assert memory_get("s1", "nope") == "null"
    assert memory_get("no-such-session", "k") == "null"


def test_json_values_round_trip():
    payload = json.dumps({"topic": "Closures", "score": 0.7})
    memory_set("s1", "progress_topic_0", payload)
    assert json.loads(memory_get("s1", "progress_topic_0"))["score"] == 0.7


def test_list_keys_sorted():
    memory_set("s1", "b", "1")
    memory_set("s1", "a", "2")
    assert memory_list_keys("s1") == ["a", "b"]
    assert memory_list_keys("empty") == []


def test_delete():
    memory_set("s1", "k", "v")
    assert "Deleted 'k'" in memory_delete("s1", "k")
    assert memory_get("s1", "k") == "null"
    assert "not found" in memory_delete("s1", "k")


def test_sessions_are_isolated():
    memory_set("s1", "k", "one")
    memory_set("s2", "k", "two")
    memory_delete("s1", "k")
    assert memory_get("s2", "k") == "two"
    assert memory_list_keys("s1") == []


def test_session_summary_resource():
    memory_set("s1", "explained_topics", "Closures Explained")
    memory_set("s1", "long", "x" * 500)
    summary = session_summary("s1")
    assert summary.startswith("# Session s1 (2 keys)")
    assert "**explained_topics**" in summary and "Closures Explained" in summary
    assert "x" * 201 not in summary  # long values are truncated
    assert "No memory stored yet" in session_summary("other")


# --- over the MCP protocol (in-memory client session) -------------------------------

async def test_filesystem_tools_over_mcp(notes):
    async with create_connected_server_and_client_session(fs.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"list_study_files", "read_study_file", "search_notes"}
        assert "Call this first" in tools["list_study_files"].description
        assert tools["read_study_file"].inputSchema["required"] == ["filename"]

        result = await client.call_tool("read_study_file", {"filename": "closures.md"})
        assert not result.isError and result.content[0].text.startswith("# Closures")

        listing = await client.call_tool("list_study_files", {})
        assert listing.structuredContent["result"] == list_study_files()

        index = await client.read_resource("notes://index")
        assert "3 files" in index.contents[0].text


async def test_memory_tools_and_template_over_mcp():
    async with create_connected_server_and_client_session(mem.mcp) as client:
        await client.call_tool("memory_set", {"session_id": "s9", "key": "k", "value": "v"})
        got = await client.call_tool("memory_get", {"session_id": "s9", "key": "k"})
        assert got.content[0].text == "v"
        templates = (await client.list_resource_templates()).resourceTemplates
        assert [t.uriTemplate for t in templates] == ["notes://session/{session_id}"]
        summary = await client.read_resource("notes://session/s9")
        assert "**k**" in summary.contents[0].text


async def test_filesystem_server_runs_as_a_stdio_subprocess(notes):
    """The real transport: launch the server script and talk to it over stdin/stdout."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "src" / "mcp_servers" / "filesystem_server.py")],
        env={**os.environ, "NOTES_PATH": str(notes)},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        result = await client.call_tool("search_notes", {"query": "nonlocal"})
        assert result.structuredContent["result"][0]["file"] == "closures.md"
