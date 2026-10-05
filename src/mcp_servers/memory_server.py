"""MCP server giving agents a per-session key/value memory.

Values are strings (agents JSON-encode anything structured). Tools never
raise, and a missing key returns the string "null", not None.

Run standalone (stdio transport): python src/mcp_servers/memory_server.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Memory Server")

# In-process store: {session_id: {key: {"value": str, "updated_at": iso}}}.
# This can be swapped for Redis or Postgres without touching any agent:
# they only see the four tools below.
_store: dict[str, dict[str, dict[str, str]]] = {}


@mcp.tool()
def memory_set(session_id: str, key: str, value: str) -> str:
    """Save a value under a key for this study session, replacing any old value.

    Args:
        session_id: the current session's id.
        key: a short name, e.g. "explained_topics".
        value: the text to store (JSON-encode structured data).
    """
    _store.setdefault(session_id, {})[key] = {
        "value": str(value),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return f"Saved '{key}' for session {session_id}."


@mcp.tool()
def memory_get(session_id: str, key: str) -> str:
    """Read the value saved under a key for this session.

    Returns the stored text, or the string "null" if nothing is saved yet.
    """
    entry = _store.get(session_id, {}).get(key)
    return entry["value"] if entry else "null"


@mcp.tool()
def memory_list_keys(session_id: str) -> list[str]:
    """List every key saved for this session."""
    return sorted(_store.get(session_id, {}))


@mcp.tool()
def memory_delete(session_id: str, key: str) -> str:
    """Delete a key from this session's memory."""
    if _store.get(session_id, {}).pop(key, None) is None:
        return f"Key '{key}' not found for session {session_id}; nothing deleted."
    return f"Deleted '{key}' for session {session_id}."


@mcp.resource("notes://session/{session_id}")
def session_summary(session_id: str) -> str:
    """A Markdown summary of everything stored for one session."""
    entries = _store.get(session_id, {})
    if not entries:
        return f"# Session {session_id}\n\nNo memory stored yet."
    lines = [f"# Session {session_id} ({len(entries)} keys)", ""]
    for key in sorted(entries):
        e = entries[key]
        value = e["value"] if len(e["value"]) <= 200 else e["value"][:200] + "…"
        lines.append(f"- **{key}** (updated {e['updated_at']}): {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
