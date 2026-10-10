"""Explainer: teaches the current topic, grounded in the learner's own notes.

This is the project's retrieval ("agentic RAG" without vectors): the model
calls MCP-backed tools to list, search and read the notes, then explains.

When the topic came from uploaded notes (it has `sources`), there is nothing
to find: the node puts those sections into the prompt itself, and the model
gets no read_file tool (a whole file on top would overflow the context).

The MCP server functions are imported directly and called in-process.
Production would run the servers as separate processes and load their tools
with langchain-mcp-adapters' MultiServerMCPClient; only this wrapping changes.
"""

from __future__ import annotations

import json
from contextlib import nullcontext

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from graph.state import Topic, get_current_topic
from llm import build_llm, describe_llm_error
from mcp_servers.filesystem_server import list_study_files, read_study_file, search_notes, use_notes_dir
from mcp_servers.memory_server import memory_get, memory_set
from notes_sections import MAX_MATERIAL_CHARS, topic_material

MAX_ITERATIONS = 8
EXPLAINED_TOPICS_KEY = "explained_topics"


# --- tools: thin LangChain wrappers around the MCP functions ------------------------
# The model sees the names in @tool(...) and the docstrings below.

@tool("list_files")
def tool_list_files() -> list[str]:
    """List every Markdown study note available. Call this first."""
    return list_study_files()


@tool("read_file")
def tool_read_file(filename: str) -> str:
    """Read the full text of one study note, using a name from list_files."""
    return read_study_file(filename)


@tool("search_notes")
def tool_search_notes(query: str) -> str:
    """Search all notes for a keyword or short phrase (case-insensitive).

    Returns matching lines with their file and line number.
    """
    hits = search_notes(query)
    return json.dumps(hits) if hits else "No matches found."


@tool("memory_get")
def tool_memory_get(session_id: str, key: str) -> str:
    """Read a value saved for this study session. Returns "null" if unset."""
    return memory_get(session_id, key)


@tool("memory_set")
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """Save a text value for this study session under a key."""
    return memory_set(session_id, key, value)


EXPLAINER_TOOLS = [tool_list_files, tool_read_file, tool_search_notes, tool_memory_get, tool_memory_set]
TOOL_MAP = {t.name: t for t in EXPLAINER_TOOLS}
# Topics from uploaded notes: the material is already in the prompt.
NOTES_TOOLS = [tool_search_notes, tool_memory_get, tool_memory_set]
NOTES_TOOL_MAP = {t.name: t for t in NOTES_TOOLS}


EXPLAINER_SYSTEM_PROMPT = f"""\
You are a patient tutor. Explain ONE topic to the learner, grounded in their
own study notes. Do not rely on general knowledge when the notes cover it.

Follow these steps, using the tools:
1. Call list_files to see which notes exist.
2. Call search_notes with the topic's key term to find where it is covered.
3. Call read_file on the ONE most relevant file.
4. Call memory_get with the session id and key "{EXPLAINED_TOPICS_KEY}" to see
   what was already explained. Build on those topics; don't repeat them.
5. Write the explanation as your final reply, with no further tool calls.

The explanation must have these four parts, with these headings:
**Analogy**: a real-world analogy in 1-2 sentences.
**Core concept**: the idea in 2-3 sentences.
**Example**: a short code example taken from the notes, in a ```python block.
**Common gotcha**: one mistake learners make, and how to avoid it.
"""


NOTES_EXPLAINER_PROMPT = f"""\
You are a patient tutor. Explain ONE topic to the learner using ONLY the
learner's own notes for it, which are given in the message. Do not add facts
the notes don't contain; if the notes are brief, keep the explanation brief.

You may use these tools before answering:
- memory_get with the session id and key "{EXPLAINED_TOPICS_KEY}": topics already
  explained. Build on them; don't repeat them.
- search_notes: where a term appears elsewhere in the learner's notes.
Then write the explanation as your final reply, with no further tool calls.

The explanation must have these four parts, with these headings:
**Analogy**: a real-world analogy in 1-2 sentences.
**Core concept**: what the notes say, as they put it. Cover EVERY section
given, in order, in 1-2 sentences each, naming each section's idea.
**Example**: {{example}}
**Common gotcha**: one mistake learners make, and how to avoid it.
"""
EXAMPLE_FROM_CODE = "a short code example taken from the notes, in a fenced code block."
EXAMPLE_IN_WORDS = "a concrete worked example in words, based on the notes. No code."
TRUNCATED_NOTE = ("\n\n_Note: this topic's notes were longer than the model can read at once "
                  f"({MAX_MATERIAL_CHARS:,} characters), so the end was left out._")


def build_explainer_llm() -> Runnable:
    return build_llm(temperature=0.3).bind_tools(EXPLAINER_TOOLS)


def build_notes_explainer_llm() -> Runnable:
    return build_llm(temperature=0.3).bind_tools(NOTES_TOOLS)


def execute_tool_call(tool_call: dict, tool_map: dict | None = None) -> str:
    """Run one tool call from the model and return its result as text. Never raises.

    tool_map: the tools this model was offered (default: all of them); any
    other name is refused, even if the model makes one up.
    """
    tool_map = TOOL_MAP if tool_map is None else tool_map
    name = tool_call.get("name", "")
    tool_fn = tool_map.get(name)
    if tool_fn is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tool_map)}"
    try:
        result = tool_fn.invoke(tool_call.get("args") or {})
    except Exception as exc:
        return f"Error executing {name}: {type(exc).__name__}: {exc}"
    if isinstance(result, (list, dict)):
        return json.dumps(result)
    return str(result)


def _record_explained(session_id: str, title: str) -> None:
    """Append the topic to the session's explained_topics (bookkeeping in Python,
    not left to the model, so it always happens)."""
    current = memory_get(session_id, EXPLAINED_TOPICS_KEY)
    titles = [] if current == "null" else [t.strip() for t in current.split(",") if t.strip()]
    if title not in titles:
        titles.append(title)
    memory_set(session_id, EXPLAINED_TOPICS_KEY, ", ".join(titles))


def explainer_node(state: dict) -> dict:
    """Explain the current topic using the notes, via a tool-calling loop.

    Reads: roadmap, current_topic_index, session_id, study_materials_path
    Writes: messages, error
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "Explainer: no current topic to explain."}
    session_id = state.get("session_id", "")
    uploaded = state.get("study_materials_path") or ""
    # The learner's uploaded notes, if any, replace NOTES_PATH for this call only.
    with use_notes_dir(uploaded) if uploaded else nullcontext():
        if uploaded and topic.sources:
            material, truncated, has_code = topic_material(uploaded, topic.sources)
            if material:
                prompt = NOTES_EXPLAINER_PROMPT.format(example=EXAMPLE_FROM_CODE if has_code else EXAMPLE_IN_WORDS)
                if truncated:
                    print(f"[Explainer] This topic's notes were cut at {MAX_MATERIAL_CHARS:,} characters.")
                return _explain(topic, session_id, prompt, build_notes_explainer_llm(), NOTES_TOOL_MAP,
                                material=material, footer=TRUNCATED_NOTE if truncated else "")
            print("[Explainer] This topic's sections weren't found; searching the notes instead.")
        return _explain(topic, session_id, EXPLAINER_SYSTEM_PROMPT, build_explainer_llm(), TOOL_MAP)


def _explain(topic: Topic, session_id: str, system_prompt: str, llm: Runnable, tool_map: dict,
             material: str = "", footer: str = "") -> dict:
    print(f"\n[Explainer] Topic: {topic.title}")
    request = (f"Topic: {topic.title}\n"
               f"Description: {topic.description}\n"
               f"Session id (for memory tools): {session_id}")
    if material:
        request += f"\n\nYour notes for this topic:\n\n{material}"
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=request)]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"[Explainer] LLM call {i}/{MAX_ITERATIONS}...")
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            error = describe_llm_error(exc)
            print(f"[Explainer] {error}")
            return {"messages": messages, "error": error}
        messages.append(response)

        if not response.tool_calls:
            if footer:
                response.content = f"{response.content}{footer}"
            _record_explained(session_id, topic.title)
            print(f"\n{response.content}\n")
            return {"messages": messages, "error": None}

        for call in response.tool_calls:
            print(f"  → {call['name']}({json.dumps(call.get('args') or {})})")
            result = execute_tool_call(call, tool_map)
            print(f"    ← {result[:100]}{'...' if len(result) > 100 else ''}")
            # The id must match the call, or the model can't pair result with request.
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))

    error = f"Explainer reached max iterations ({MAX_ITERATIONS})."
    print(f"[Explainer] {error}")
    return {"messages": messages, "error": error}
