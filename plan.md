# Sherpa: Multi-Agent Study Companion (LangGraph + MCP + A2A + Ollama)

> **How to use this plan.** Build it **phase by phase, in order**. After each
> phase, run the listed tests and confirm the acceptance criteria before moving
> on. Where this plan and an installed library disagree, **trust the installed
> library**: check the actual signature (`python -c "import inspect; ..."`),
> fix the code, and note the deviation in `docs/DEVIATIONS.md`. Section 12
> lists library gotchas already verified against the pinned versions.

---

## 1. What we are building

Sherpa is a local, zero-API-key study companion. The user gives a learning
goal. Four specialised agents then:

1. **Curriculum Planner** produces a structured study roadmap (4–6 topics).
2. **Human Approval** pauses for the user to accept or reject the roadmap.
   Rejecting it regenerates the plan.
3. **Explainer** reads the user's own Markdown notes through MCP tools and
   explains the current topic, grounded in those notes.
4. **Quiz Generator** generates 3 questions, collects answers, and grades
   them (LLM-as-judge).
5. **Progress Coach** gives feedback, marks the topic `completed` or
   `needs_review`, saves progress to MCP memory, asks the CrewAI Study
   Buddy (over A2A) for extra help on low scores, and decides whether to
   loop to the next topic or end.

Infrastructure concerns are first-class: crash-safe checkpointing, resume by
session ID, human-in-the-loop, tracing, LLM-as-judge evals, and
cross-framework agent calls.

### Non-goals
- No cloud LLMs, no paid APIs.
- No vector DB or embeddings in the core build (see section 13, stretch goal).
- No auth or multi-user support.

---

## 2. Architecture

```
                         ┌──────────────── Orchestration (LangGraph) ────────────────┐
 User ──goal/answers──►  │ START → curriculum_planner → human_approval ──(approved)──┐│
                         │              ▲                  │ (rejected)              ││
                         │              └──────────────────┘                         ▼│
                         │   ┌──────► explainer → quiz_generator → progress_coach ───┐│
                         │   └──────────────(more topics)──────────────────┘   (done)→END
                         │                SqliteSaver checkpoint after every node     │
                         └────────────────────────────────────────────────────────────┘
        Tool layer (MCP)                    A2A layer (JSON-RPC 2.0 over HTTP)
   ┌────────────────────────┐        ┌─────────────────────────┐ ┌───────────────────────┐
   │ filesystem_server      │        │ Quiz Service  :9001     │ │ CrewAI Study Buddy    │
   │ memory_server          │        │ (LangGraph-side logic)  │ │ :9002 (CrewAI)        │
   └────────────────────────┘        └─────────────────────────┘ └───────────────────────┘
                     Inference: Ollama @ localhost:11434 (qwen2.5:7b default)
                     Observability: Langfuse (self-hosted, Docker) via callbacks
                     Evaluation: DeepEval with a local Ollama judge
```

### Why four agents (keep this reasoning in `docs/ARCHITECTURE.md`)

| Agent | Call pattern | Temp | Output | Tools |
|---|---|---|---|---|
| Curriculum Planner | single call, JSON mode | 0.1 | `StudyRoadmap` | none |
| Explainer | multi-turn tool loop (max 8) | 0.3 | free-text explanation | 5 MCP-backed tools |
| Quiz Generator | 2 call types: generate (0.4, JSON) + grade (0.1, JSON) | 0.4 / 0.1 | `QuizResult` | none (interactive) |
| Progress Coach | single call, JSON mode | 0.4 | coaching msg + routing state | MCP memory, A2A client |

Rule: **routing is pure Python, never an LLM decision.**

---

## 3. Tech stack (pin exactly; do NOT upgrade)

`requirements.txt`:
```
langgraph==1.1.0
langgraph-checkpoint-sqlite==3.0.3
langchain-core==1.0.0
langchain-ollama==1.0.0
langchain                      # REQUIRED by langfuse.langchain; pin to whatever version
                               # pip resolves compatibly with langchain-core==1.0.0, then freeze it

mcp==1.26.0
a2a-sdk==0.3.25
crewai==1.13.0

langfuse==4.0.1
deepeval==3.9.1

litellm==1.82.4
openai==2.8.0
httpx==0.28.1
fastapi==0.115.0
uvicorn==0.34.0
streamlit==1.43.2

pydantic==2.11.9
python-dotenv==1.1.1
tenacity==8.5.0

pytest==8.3.0
pytest-asyncio==0.25.0
```
If pip reports an unresolvable conflict (crewai and langchain pins are the
usual suspects), resolve it minimally, record what changed in
`docs/DEVIATIONS.md`, and keep LangGraph, langchain-core, and a2a-sdk at the
pinned versions.

Python ≥ 3.11. Ollama model: `qwen2.5:7b` (8 GB VRAM) or `qwen2.5-coder:32b` (24 GB).

---

## 4. Repository layout

```
sherpa/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── curriculum_planner.py
│   │   ├── human_approval.py
│   │   ├── explainer.py
│   │   ├── quiz_generator.py
│   │   └── progress_coach.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── mcp_servers/
│   │   ├── __init__.py
│   │   ├── filesystem_server.py
│   │   └── memory_server.py
│   ├── a2a_services/
│   │   ├── __init__.py
│   │   ├── quiz_service.py
│   │   └── a2a_client.py
│   ├── crewai_agent/
│   │   ├── __init__.py
│   │   └── study_buddy.py
│   └── observability/
│       ├── __init__.py
│       └── langfuse_setup.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_state.py
│   ├── test_curriculum_planner.py
│   ├── test_mcp_servers.py
│   ├── test_explainer.py
│   ├── test_quiz_and_coach.py
│   ├── test_checkpointing.py
│   ├── test_observability.py
│   ├── test_a2a.py
│   ├── test_crewai_interop.py
│   └── test_eval.py                # @pytest.mark.eval, needs Ollama
├── scripts/
│   └── a2a_demo.py                 # calls both A2A services end to end
├── study_materials/sample_notes/   # python_basics.md, closures.md, decorators.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_SELECTION.md
│   └── DEVIATIONS.md
├── data/                           # created at runtime (.gitignored)
├── main.py
├── streamlit_app.py
├── docker-compose.yml              # Langfuse v3 stack (see Phase 7)
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 5. Configuration (`.env.example`)

```
# Model
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# Storage
CHECKPOINT_DB=data/checkpoints.db
NOTES_PATH=study_materials/sample_notes

# Langfuse (leave empty to run without tracing)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000

# A2A
QUIZ_SERVICE_URL=http://localhost:9001
STUDY_BUDDY_URL=http://localhost:9002
USE_A2A_QUIZ=true
USE_STUDY_BUDDY=true
```

Every entry point (`main.py`, `streamlit_app.py`, both A2A services) must call
`load_dotenv()` before importing project modules. Read env-driven toggles **at
call time**, not import time, so tests can monkeypatch them.

---

## 6. Global conventions (apply everywhere)

1. **Node contract.** Every LangGraph node is `def x_node(state: dict) -> dict`.
   It returns a **partial** update containing only the keys it changed. The
   one exception is `human_approval_node` (see Phase 6). The docstring of each
   node lists `Reads:` and `Writes:` keys.
2. **Parse functions are separate from node functions.** This lets them be
   unit-tested without an LLM (e.g. `parse_roadmap_json`).
3. **Tools never raise.** MCP tool functions return error *strings* such as
   `"Error: ..."`, never exceptions. A missing memory key returns the string
   `"null"`, not `None`.
4. **Dict-or-dataclass tolerance.** After a SQLite checkpoint round-trip,
   dataclasses may come back as plain dicts. Every state accessor and any code
   reading `roadmap`, `topics[i]`, or `quiz_results[i]` must handle both forms.
   All dataclasses get `to_dict()` and a `from_dict()` classmethod.
5. **LLM factory per call.** Build `ChatOllama(model=..., base_url=...,
   temperature=..., format="json" | None)` with `MODEL_NAME` and
   `OLLAMA_BASE_URL` read from env.
6. **Graceful fallbacks.** JSON-mode LLM calls are wrapped in `try/except`
   with a sensible fallback dict (see each agent). The graph must never crash
   because of a malformed LLM response.
7. **Console logging** uses an agent prefix, e.g. `[Explainer] LLM call 2/8...`,
   `  → tool_name(args)`, `    ← result[:100]...`.
8. **Unit tests never call Ollama.** Mock `ChatOllama`/`llm.invoke`, or test
   pure functions. Only `tests/test_eval.py` hits a real model, and it is
   marked `@pytest.mark.eval`.

---

## 7. Build phases

### Phase 0: Scaffold

**Files:** `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`,
`Makefile`, `README.md` (stub), `tests/conftest.py`, all `__init__.py`,
`study_materials/sample_notes/*.md`.

- `pyproject.toml`:
  - `[tool.pytest.ini_options]` with `pythonpath = ["src"]`,
    `testpaths = ["tests"]`, `asyncio_mode = "auto"`,
    `addopts = "-m 'not eval'"`, and markers `unit` and `eval`.
  - `[tool.pyright] pythonPath = "src"`.
- `conftest.py`:
  - Insert `src/` into `sys.path`. Register the `eval` and `unit` markers.
  - Fixtures:
    - `sample_roadmap`: 2 topics, "Closures Explained" and "Practical
      Closure Patterns", where the second has the first as a prerequisite.
    - `sample_state`: `initial_state(...)` with the roadmap attached.
    - `closures_note_content`: reads `closures.md`, with an inline fallback.
- Sample notes: write 3 Markdown files.
  - `python_basics.md` covers variables, functions, scope, and LEGB.
  - `closures.md` covers the definition, 3 requirements, the `make_counter`
    example with `nonlocal`, late-binding gotcha in loops, and `__closure__`.
  - `decorators.md` covers functions-as-objects, a basic decorator,
    `functools.wraps`, and decorators with arguments.
  - Each file is 60–150 lines with headings and code blocks.
- `Makefile` targets:
  - Core: `setup run streamlit run-goal resume services stop langfuse
    langfuse-stop test eval test-all clean help`.
  - `services` starts both A2A services in the background.
  - `stop` uses `pkill -f`.
  - `test` runs `pytest -m "not eval"`.
  - `eval` runs `pytest tests/test_eval.py -m eval -s -v`.

**Accept:** `pip install -r requirements.txt` succeeds; `pytest` runs (0 tests OK).

---

### Phase 1: Shared state (`src/graph/state.py`)

Dataclasses:
- `Topic(title, description, estimated_minutes:int, prerequisites:list[str]=[], status="pending")`
  - Status lifecycle: `pending → in_progress → completed | needs_review`.
- `StudyRoadmap(goal, total_weeks:int, topics:list[Topic], weekly_hours:int=5)`
  - Methods: `completed_count()`, `is_complete()` (all topics are
    completed or needs_review), `to_dict`, `from_dict` (which also converts
    nested topics).
- `QuizQuestion(question, expected_answer, user_answer="", correct=False, feedback="", score=0.0)` plus `to_dict` and `from_dict`.
- `QuizResult(topic, questions:list[QuizQuestion], score:float, weak_areas:list[str], timestamp="")`
  - Methods: `passed()` (≥0.5), `strong_pass()` (≥0.8), `to_dict`,
    `from_dict` (which also converts nested questions).

`AgentState(TypedDict)`:
```
messages: Annotated[list[BaseMessage], add_messages]   # only reducer field
session_id: str
goal: str
roadmap: StudyRoadmap | None
approved: bool
current_topic_index: int
quiz_results: list[QuizResult]
weak_areas: list[str]
study_materials_path: str
error: str | None
```

Helpers (all dict/dataclass tolerant):
- `initial_state(goal, session_id, study_materials_path="study_materials/sample_notes") -> dict`
  is the ONLY way to create a new state.
- `get_current_topic(state) -> Topic | None`.
- `get_latest_quiz_result(state) -> QuizResult | None`.
- `session_is_complete(state) -> bool`. Returns True if there's no roadmap
  or if the index is at or past `len(topics)`.

**Tests (`test_state.py`, ~24):**
- Round-trips for every `to_dict`/`from_dict`.
- Every helper with both dataclass and dict roadmaps.
- Boundary indices.
- Thresholds for `passed` and `strong_pass`.
- `initial_state` has every key.

---

### Phase 2: Curriculum Planner + minimal graph + CLI

**`src/agents/curriculum_planner.py`**

Build the LLM with `build_planner_llm()`: temp 0.1, `format="json"`.

`PLANNER_SYSTEM_PROMPT` must:
- demand JSON only, with no prose or code fences;
- give this schema:
  `{goal, total_weeks:int 1–12, weekly_hours:int 3–10, topics:[{title (3–6 words), description (1 sentence), estimated_minutes:int 30–120, prerequisites:[earlier titles], status:"pending"}]}`;
- require topics ordered foundational → advanced, 4–6 topics;
- require prerequisites to reference earlier titles exactly.

`parse_roadmap_json(s) -> StudyRoadmap` raises `ValueError` with a helpful
message in each of these cases:
- invalid JSON (include the first 300 characters of the raw output);
- a missing top-level `goal`, `total_weeks`, or `topics`;
- `topics` that is empty or not a list;
- a topic missing `title`, `description`, or `estimated_minutes`.

It also coerces ints and defaults `weekly_hours` to 5.

`curriculum_planner_node`:
- With an empty goal, return `{"error": ...}`.
- Otherwise invoke the LLM and parse the output.
- On `ValueError`, return the error plus the messages.
- On success, return `{roadmap, messages, error: None}`.

**`src/graph/workflow.py`**: `build_graph(db_path="data/checkpoints.db", interrupt_before=None)`

- Use `StateGraph(AgentState)` with 5 nodes:
  `curriculum_planner, human_approval, explainer, quiz_generator, progress_coach`.
  In this phase only the planner exists, so stub the others or add them as
  each phase lands.
- Static edges:
  - `START → curriculum_planner → human_approval`
  - `explainer → quiz_generator → progress_coach`
- Conditional edges:
  - `human_approval` uses `route_after_approval`, which returns
    `"explainer"` if `approved` else `"curriculum_planner"`.
  - `progress_coach` uses `route_after_coach`, which returns `"end"` (mapped
    to `END`) if `session_is_complete` else `"explainer"`.
- Checkpointer:
  ```python
  conn = sqlite3.connect(db_path, check_same_thread=False)  # NOT a context manager
  checkpointer = SqliteSaver(conn)
  ```
  Create `data/` if missing, and honour the `CHECKPOINT_DB` env var.
- Compile with `checkpointer` and `interrupt_before=interrupt_before or []`.
- Expose `graph = build_graph()` at module level.

**`main.py`**

- Order matters: `sys.path` insert of `src/`, then `load_dotenv()`, then
  project imports.
- argparse:
  - positional `goal`, optional, defaulting to "Learn Python closures and
    decorators from scratch";
  - `--resume SESSION_ID`.
- `run_session(goal, session_id=None)`:
  - Generate an 8-char session ID from a uuid4 prefix and print a banner.
  - Get the config from `get_langfuse_config(session_id)`. Until Phase 7,
    use `{"configurable":{"thread_id":session_id}}`.
  - The state is `None` when resuming, else `initial_state(...)`.
  - Interrupt loop: `while "__interrupt__" in result`.
    - Read `payload = result["__interrupt__"][0].value`.
    - Pretty-print the roadmap, coercing a dict to `StudyRoadmap`.
    - Read `input("> ")`.
    - Call `graph.invoke(Command(resume=user_input), config=config)`.
  - If a resume fails, print a friendly error.
  - Finish with `print_session_summary(result)`, which is dict tolerant, and
    then `flush_langfuse()`.

**Tests (`test_curriculum_planner.py`, ~11):**
- Valid JSON parses.
- Each missing field raises.
- Bad JSON raises.
- Empty topics raises.
- String ints are coerced.
- Default `weekly_hours`.
- Node returns an error on an empty goal.
- Node with a mocked LLM returns a roadmap.

**Accept:** `python main.py --help` works. Phase 2 tests pass.

---

### Phase 3: MCP servers

**`src/mcp_servers/filesystem_server.py`** uses `FastMCP("Filesystem Server")`.
Set `NOTES_BASE = Path(os.getenv("NOTES_PATH", ...))`.

Each function is decorated with `@mcp.tool()` but stays importable as a plain
Python function. Docstrings matter, because the LLM reads them.

- `list_study_files() -> list[str]`: sorted relative paths of `*.md`
  (found with rglob). Returns `[]` if the directory is missing. The docstring
  says "call this first".
- `read_study_file(filename) -> str`:
  1. Block path traversal by resolving the path and checking
     `relative_to(NOTES_BASE.resolve())`.
  2. If the file doesn't exist, return an error that lists the available
     files.
  3. Allow only `.md`.
  4. Read as UTF-8 and catch OS errors.
  5. All errors are returned as strings.
- `search_notes(query) -> list[dict]`: case-insensitive substring match per
  line, returning `{file, line_number, line}` and capped at 20 results.
- `@mcp.resource("notes://index")`: a Markdown index of the files with their
  sizes in KB.
- `if __name__ == "__main__": mcp.run()`.

**`src/mcp_servers/memory_server.py`** uses `FastMCP("Memory Server")`.

- Storage is an in-process `_store: dict[session_id, dict[key, {"value": str, "updated_at": iso}]]`.
- Tools:
  - `memory_set(session_id, key, value:str) -> str` (confirmation message)
  - `memory_get(session_id, key) -> str` (returns `"null"` if missing)
  - `memory_list_keys(session_id) -> list[str]`
  - `memory_delete(session_id, key) -> str`
- `@mcp.resource("notes://session/{session_id}")`: a Markdown summary.
- Add a comment that this store can be swapped for Redis or Postgres without
  touching agents.

**Tests (`test_mcp_servers.py`, ~36):** use `tmp_path` plus monkeypatched
`NOTES_BASE`. Cover:
- list on an empty or missing directory;
- reading an existing file;
- reading a missing file (the error includes available files);
- traversal blocked (`../../.env`);
- non-`.md` rejected;
- search hits, case-insensitivity, the 20-result cap, and no matches;
- the index resource;
- memory set, get, list, delete, missing key returns `"null"`, session
  isolation, and the session-summary resource.

Clear `_store` between tests with a fixture.

---

### Phase 4: Explainer (grounded retrieval loop)

This is the project's "RAG": **agentic keyword retrieval**, not vector search.

**`src/agents/explainer.py`**

Import the MCP functions directly and wrap each one with LangChain `@tool`:
- `tool_list_files()`
- `tool_read_file(filename)`
- `tool_search_notes(query)` returns a JSON string, or "No matches found."
- `tool_memory_get(session_id, key)`
- `tool_memory_set(session_id, key, value)`

Collect them in `EXPLAINER_TOOLS` and build `TOOL_MAP = {t.name: t}`.

> Direct import = single process, simple, testable. In `docs/ARCHITECTURE.md`,
> note that production would run the MCP servers as separate processes
> (stdio/HTTP) via `langchain-mcp-adapters` `MultiServerMCPClient`, and only
> the tool wrapping changes.

`EXPLAINER_SYSTEM_PROMPT` gives a required sequence:
1. list files
2. search the topic
3. read the most relevant file(s)
4. `memory_get(session_id, 'explained_topics')`
5. write the explanation

The explanation format is: a real-world analogy (1–2 sentences), the core
concept (2–3 sentences), a code example taken from the notes, and one common
gotcha. After writing, it calls `memory_set(session_id, 'explained_topics',
<comma-separated titles>)`.

`execute_tool_call(tool_call) -> str`:
- An unknown tool name returns an error string.
- Lists and dicts are JSON-dumped.
- Exceptions are returned as `"Error executing ..."`.
- It never raises.

`explainer_node(state)`:
- `topic = get_current_topic(state)`; if it's None, return an error.
- Build the LLM: `ChatOllama(temp=0.3).bind_tools(EXPLAINER_TOOLS)`.
- Messages are `[SystemMessage(prompt), HumanMessage(topic title + description + session_id)]`.
- Loop up to `max_iterations = 8`:
  - Invoke the LLM and append the response.
  - If there are no `tool_calls`, that's the final answer, so break.
  - Otherwise, for each call, append
    `ToolMessage(content=result, tool_call_id=tool_call["id"])`. The **ID
    must match.**
- If the limit is hit, return `{"messages": messages, "error": "Explainer reached max iterations (8)."}`.
- On success, return `{"messages": messages, "error": None}`.

**Tests (`test_explainer.py`, ~14):**
- Tool wrappers delegate correctly.
- `execute_tool_call` handles unknown tools, exceptions, and list results.
- The node with a mocked LLM:
  - (a) gives a direct answer with no tools;
  - (b) makes one tool round, then answers, with the ToolMessage ID
    matching;
  - (c) hits max iterations and returns an error;
  - (d) returns an error when there's no topic.

---

### Phase 5: Quiz Generator + Progress Coach + full loop

**`src/agents/quiz_generator.py`**

- `GENERATION_PROMPT` (`{n}` placeholder; escape literal braces
  in `.format` templates) asks for questions that test application, "why",
  edge cases, and comparisons. At least one question must be about a common
  mistake. No yes/no questions. JSON schema:
  `{"questions":[{"question","expected_answer","difficulty":"easy|medium|hard"}]}`.
- `GRADING_PROMPT` is a fair grader with partial-credit bands:
  - 0.7–0.9: correct with minor gaps
  - 0.5–0.7: imprecise
  - 0.3–0.5: partial
  - 0.0–0.2: wrong

  JSON schema: `{"correct":bool,"score":float,"feedback":str,"missing_concept":str}`.
- `generate_questions(topic, explanation, n=3) -> list[dict]` uses temp 0.4
  and JSON mode. On failure or an empty result, fall back to one generic
  "explain X in your own words" question.
- `grade_answer(question, expected, student_answer) -> dict` uses temp 0.1
  and JSON mode. On failure, return
  `{"correct":False,"score":0.5,"feedback":"Could not grade automatically...","missing_concept":""}`.
  Clamp the score to [0,1] and coerce `correct` to bool.
- `run_quiz(topic, explanation) -> QuizResult` is the terminal flow:
  - For each question, print it with its difficulty and read `input()`.
    An empty answer becomes `"(no answer provided)"`.
  - Grade it and print ✓/✗, the score, and the feedback.
  - Collect each `missing_concept` into weak areas.
  - Then compute the average, dedupe the weak areas, and set an ISO
    timestamp.
- `quiz_generator_node(state)`:
  - Explanation = content of the last `AIMessage` with content and **no
    `tool_calls`**. If there isn't one, fall back to the topic title and
    description.
  - Run `run_quiz`.
  - Return:
    - `quiz_results = existing + [new]` (accumulate);
    - `weak_areas = dedupe(existing + new)`;
    - `error: None`;
    - plus `roadmap`, `current_topic_index`, and `session_id` passed through
      explicitly.

`generate_questions` and `grade_answer` must stay importable pure functions.
The A2A service and the Streamlit UI reuse them.

**`src/agents/progress_coach.py`**

- `PASS_THRESHOLD = 0.5`.
- `COACHING_PROMPT` asks for a warm, specific message that names
  the topic and weak areas and is never discouraging. JSON schema:
  `{"summary","encouragement"}`.
- `get_coaching_message(topic, score, weak_areas) -> dict` uses temp 0.4 and
  JSON mode, with a fallback dict.
- `progress_coach_node(state)`, in this order:
  1. Get `latest = get_latest_quiz_result(state)`; error if it's None or if
     there's no roadmap.
  2. Get the coaching message.
  3. **Update the status of `topics[idx]` FIRST** (`completed` if the score
     is at least the threshold, else `needs_review`), handling both dict and
     dataclass topics.
  4. Set `next_idx = idx + 1`.
  5. `memory_set(session_id, f"progress_topic_{idx}", json.dumps({topic, score, weak_areas, status, timestamp}))`.
  6. Print the coaching message, then either the next topic or the session
     totals (passed count and average).
  7. If `score < PASS_THRESHOLD and weak_areas`, call
     `try_study_buddy_assistance(...)` (Phase 9). Print the result if one
     comes back. This is a no-op until Phase 9 and must fail silently.
  8. Return `{roadmap, current_topic_index: next_idx, messages: [AIMessage(coaching summary)], error: None}`.

**Tests (`test_quiz_and_coach.py`, ~17):** mock the LLM and `input()`.
- Fallbacks for generate and grade.
- Explanation extraction skips tool-call messages.
- `quiz_results` accumulates and weak areas dedupe.
- Coach updates the correct topic index (regression test for the
  ordering bug).
- Coach writes to memory.
- Both routing functions return the right strings.
- Dict-form roadmap is handled.

**Accept:** with Ollama running,
`python main.py "Learn Python closures and decorators from scratch"`
completes a full multi-topic session end to end.

---

### Phase 6: Human-in-the-loop + persistence

**`src/agents/human_approval.py`**
- If there's no roadmap, return `{"approved": True}`.
- Otherwise print a status line and call
  `decision = interrupt({"type":"roadmap_approval","roadmap":roadmap,"prompt":"...yes/no..."})`.
- `approved = str(decision).lower().strip() in {"yes","y","ok","approve"}`.
- **Return the full state explicitly**: `approved`, `roadmap`, `goal`,
  `session_id`, `current_topic_index`, `quiz_results`, `weak_areas`,
  `study_materials_path`, and `error: None`. This is defensive: downstream
  nodes can lose keys after `Command(resume=...)` on LangGraph 1.1.0, and
  returning everything is harmless.

Resume semantics:
- `graph.invoke(None, config)` with the same `thread_id` continues from the
  last completed node.
- `interrupt()` is used for approval because it carries a payload.
- `interrupt_before` is reserved for the Streamlit UI.

**Tests (`test_checkpointing.py`, ~20):** use a temp SQLite DB and mocked
nodes or LLMs.
- A fresh run pauses at approval with `__interrupt__` in the result.
- Resuming with "yes" proceeds; "no" routes back to the planner.
- A new graph instance with the same DB and thread_id resumes.
- `get_state` shows the expected next node.
- Accessors handle deserialized dicts.
- Two different thread IDs are isolated.

**Manual check:** start a session, approve, Ctrl+C after the Explainer
finishes, then run `python main.py --resume <id>`. It should continue at the
quiz.

---

### Phase 7: Observability (Langfuse)

**`src/observability/langfuse_setup.py`**. Agents never import this module.

- `_langfuse_configured() -> bool`: true only when both the public and secret
  keys are non-empty.
- `get_langfuse_handler()`:
  - Return None if Langfuse isn't configured.
  - Otherwise `from langfuse.langchain import CallbackHandler` and return
    `CallbackHandler()`.
  - **Langfuse 4.x API:** the handler reads keys and host from env, and its
    constructor takes only `public_key` and `trace_context`. Don't pass
    `session_id`, `tags`, etc. to it.
  - Catch `ImportError` and any other exception, print a hint, and return
    None.
- `get_langfuse_config(session_id, user_id="local", extra_config=None) -> dict`:
  - Always include `{"configurable":{"thread_id":session_id}}`.
  - If there's a handler, add `config["callbacks"]=[handler]` and
    `config["metadata"]={"langfuse_session_id":session_id,"langfuse_user_id":user_id,"langfuse_tags":["sherpa","local-inference"]}`.
  - Print whether tracing is on or off.
- `flush_langfuse()`: a no-op if not configured. Otherwise call
  `from langfuse import get_client; get_client().flush()` inside `try/except`.
  Best effort only.

**`docker-compose.yml`:** Langfuse **v3 self-host needs Postgres +
ClickHouse + Redis + MinIO**. A Postgres-only compose file won't start it.
Fetch the official compose file from the `langfuse/langfuse` GitHub repo
(`docker-compose.yml` on main) and keep its local-dev defaults. Expose
`:3000`.

**Tests (`test_observability.py`, ~16):** no server needed. Monkeypatch env
and cover:
- `thread_id` is always present;
- no callbacks or metadata when unconfigured;
- metadata keys are present when configured (patch `CallbackHandler`);
- `ImportError` gives None without raising;
- `flush` is a no-op when unconfigured;
- whitespace-only keys count as unconfigured.

**Accept:** `make langfuse`, then create a project and put its keys in `.env`.
Run a session. The Langfuse UI shows one session with nested node, LLM, and
tool spans.

---

### Phase 8: Evaluation (DeepEval, local judge)

**`tests/test_eval.py`**, every class marked `@pytest.mark.eval`.

- `OllamaJudge(DeepEvalBaseLLM)`:
  - `load_model()` returns `ChatOllama(temp=0.0)`.
  - `generate(prompt, schema=None)`: if a pydantic `schema` is given, call
    with `format="json"` and return `schema.model_validate_json(content)`;
    otherwise return the string content.
  - `a_generate` delegates to `generate`.
  - `get_model_name()` returns `f"ollama/{model}"`.
  - Check the exact signature DeepEval 3.9.1 expects and match it.
- `run_explainer(title, desc, session_id) -> str` helper builds a one-topic
  state and returns the final AIMessage content.

Test classes (thresholds are conservative for 7B models):

| Class | Tests | Metric | Threshold |
|---|---|---|---|
| `TestExplainerQuality` | faithful to `closures.md` (retrieval_context) | `FaithfulnessMetric` | 0.6 |
| | relevant to the question | `AnswerRelevancyMetric` | 0.6 |
| | min length; mentions key concepts (closure/nonlocal/enclosing) | plain asserts | n/a |
| `TestQuizGeneratorQuality` | questions test understanding, not recall | `GEval` custom criteria | 0.6 |
| | each question has the required keys | plain asserts | n/a |
| `TestGradingQuality` | correct answer ≥ 0.65; wrong ≤ 0.35; partial in [0.3, 0.75]; required fields and types | `grade_answer` direct | n/a |
| `TestProgressCoachQuality` | encouraging + specific + actionable + concise | `GEval` | 0.6 |
| | returns `summary` and `encouragement` | plain asserts | n/a |

Skip, rather than fail, when deepeval is missing or the Explainer output is
empty.

**Accept:** `make eval` runs about 12 tests in a couple of minutes on
`qwen2.5:7b`.

---

### Phase 9: A2A services + CrewAI Study Buddy

> ⚠️ Read section 12 first. Several A2A details are easy to get wrong with
> a2a-sdk 0.3.25.

**Protocol facts for a2a-sdk 0.3.25 (verified):**
- Agent Card is served at `GET /.well-known/agent-card.json`.
- JSON-RPC endpoint is `POST /` (`DEFAULT_RPC_URL = "/"`), method
  **`message/send`**.
- `Message` requires `role`, `parts`, and **`message_id`** (`messageId`
  on the wire).
- Parts on the wire use `{"kind":"text","text":...}`.
- `Part` is a RootModel. Inside executors, read input with
  **`context.get_user_input()`** and emit with
  **`a2a.utils.new_agent_text_message(text)`**.

**`src/a2a_services/quiz_service.py`** (port 9001)
- Add `src/` to `sys.path` and call `load_dotenv()`.
- `QUIZ_SKILL = AgentSkill(id="generate_and_grade_quiz", name, description, tags=[quiz,assessment,education,grading], examples=[...])`.
- `QUIZ_AGENT_CARD = AgentCard(name="Quiz Generator Service", description, url="http://localhost:9001/", version="1.0.0", default input/output modes ["text"], capabilities=AgentCapabilities(streaming=False), skills=[QUIZ_SKILL])`.
  Use whichever field-name style (snake_case or camelCase alias) the
  installed SDK accepts.
- `QuizAgentExecutor(AgentExecutor).execute(context, event_queue)`:
  1. Parse `context.get_user_input()` as JSON
     `{topic, explanation?, answers?}`. If it isn't JSON, treat the whole
     text as the topic.
  2. Run `generate_questions` via `asyncio.to_thread`.
  3. With no answers, return
     `{"status":"questions_ready","topic","questions","message"}`.
  4. With answers, grade each pair via `to_thread(grade_answer)` and return
     `{"status":"graded","topic","score","questions","graded_questions":[{question,answer,score,correct,feedback}],"weak_areas"}`.
  5. `await event_queue.enqueue_event(new_agent_text_message(json.dumps(result)))`.
- `cancel()` is a no-op.
- `create_quiz_server()` builds
  `DefaultRequestHandler(agent_executor=..., task_store=InMemoryTaskStore())`
  and `A2AStarletteApplication(agent_card=..., http_handler=...).build()`.
- Main block: `uvicorn.run(..., host="0.0.0.0", port=9001)`.

**`src/crewai_agent/study_buddy.py`** (port 9002, pure CrewAI, no LangGraph imports)
- `TopicAnalyserTool(BaseTool)`:
  - `name="topic_analyser"`, with a pydantic args schema
    `{topic, weak_areas:list[str]}`.
  - `_run` returns JSON
    `{topic, focus_areas, suggested_approach, study_tip}`.
- `build_study_buddy_crew(topic, explanation, weak_areas) -> Crew` builds a
  **fresh crew per request**:
  - `LLM(model=f"ollama/{MODEL}", base_url=OLLAMA_BASE_URL)`.
  - `Agent(role="Study Buddy", goal, backstory, tools=[analyser], allow_delegation=False)`.
  - One `Task` that:
    - includes the first 1000 characters of the explanation and the weak
      areas;
    - tells the agent to use the analyser first;
    - asks for 1) a fresh analogy, 2) a concrete example on the weak areas,
      and 3) a memory tip, in 150–250 words.
  - `Process.sequential`.
- Agent Card: `"CrewAI Study Buddy"` at `http://localhost:9002/` with skill
  `supplementary_study_assistance`.
- `StudyBuddyExecutor`:
  1. Parse `{topic, explanation, weak_areas}`.
  2. Run `crew.kickoff` with `asyncio.to_thread`. Use `.raw` if it exists,
     otherwise `str()`.
  3. Return
     `{"source":"crewai_study_buddy","topic","weak_areas","assistance","status":"complete"}`.
  4. On an exception, return `status:"error"` with the error message.
  5. Emit with `new_agent_text_message`.

**`src/a2a_services/a2a_client.py`** (raw httpx, keeps protocol details out of agents)

`DEFAULT_TIMEOUT=120.0`. URLs are read from env.

- `discover_agent(base_url) -> dict`: GET the card with a 5s timeout. On any
  error, print and return `{}`.
- `send_task(base_url, message_text, timeout=DEFAULT_TIMEOUT) -> dict`:
  - POST to `base_url.rstrip('/') + '/'` with
    `{"jsonrpc":"2.0","id":<uuid>,"method":"message/send","params":{"message":{"role":"user","messageId":<uuid>,"parts":[{"kind":"text","text":message_text}]}}}`.
  - Parse the response. If there's a JSON-RPC `error`, return
    `{"error": ...}`. Otherwise look at `result`:
    - if `result.kind == "message"`, take the text from `result.parts`;
    - if `result.kind == "task"`, take the text from
      `result.artifacts[*].parts` or `result.status.message.parts`.
  - Accept a part's text under either the `kind` or the legacy `type` key.
  - `json.loads` the text. If that fails, return `{"text": ...}`.
  - Map `TimeoutException`, `ConnectError`, and any other exception to
    `{"error": "..."}`. Never raise.
- `delegate_quiz_task(topic, explanation, answers=None, quiz_service_url=...)`.
- `is_quiz_service_available(url)`.
- `request_study_assistance(topic, explanation, weak_areas, study_buddy_url=...)`
  with a 180s timeout.
- `is_study_buddy_available(url)`.

**Wiring in `progress_coach.py`:**
- `try_study_buddy_assistance(topic, explanation, weak_areas) -> str | None`:
  - Return None unless `USE_STUDY_BUDDY` (read at call time) is set and
    the card is reachable.
  - Return None if the result has `error` or `status == "error"`.
  - Otherwise return `assistance`.
- `try_a2a_quiz_delegation(topic, explanation, answers) -> dict | None`
  follows the same guard pattern with `USE_A2A_QUIZ`.
  - This is a helper for remote grading and isn't called from the graph by
    default. Exercise it from `scripts/a2a_demo.py`.
  - Optional: in `quiz_generator_node`, after collecting answers locally,
    use it for grading when the service is available, with local grading as
    the fallback. If you add this, update the tests and
    `ARCHITECTURE.md`.

**`scripts/a2a_demo.py`:**
- Discover both cards.
- Send a quiz task without answers and print the questions.
- Send it again with answers and print the score.
- Send a study-buddy request.

**Tests:**
- `test_a2a.py` (~19):
  - Agent card and skill fields.
  - `discover_agent` success, connection error, and timeout (patch
    `httpx.get`).
  - `send_task` parses both the message-shaped and task-shaped results, and
    maps connection refused and timeout (patch `httpx.post`).
  - `delegate_quiz_task` payload shape.
  - Availability checks.
  - `try_a2a_quiz_delegation`: disabled, unavailable, success, and error.
  - **Request-envelope test:** `method == "message/send"`, `messageId`
    present, parts use `kind`.
- `test_crewai_interop.py` (~25):
  - The tool's `_run` output.
  - `build_study_buddy_crew` structure (agent role, tools, a task containing
    the topic and weak areas, explanation truncated at 1000 characters),
    with `LLM` patched.
  - The executor with a mocked crew: success, `.raw` vs `str`, error path.
  - Card fields.
  - `request_study_assistance` payload.
  - `try_study_buddy_assistance` toggles.
  - Use Starlette `TestClient` on `create_*_server()` with mocked
    `generate_questions` and `grade_answer` or the crew: GET the card, then
    POST `message/send` and check the JSON round-trip.

**Accept:** `make services` then `python scripts/a2a_demo.py` succeeds. In a
session where you answer badly, the Study Buddy text appears after the coach.

---

### Phase 10: Streamlit UI (`streamlit_app.py`)

The same graph runs here, and only the I/O changes. No edits to agent code.

- Path setup, then `load_dotenv()`.
- `ui_graph = build_graph(db_path="data/checkpoints_ui.db", interrupt_before=["quiz_generator"])`.
  This is a separate DB, and the pause comes *before* the quiz so `input()`
  never runs inside the graph.
- The UI is a screen state machine in `st.session_state.screen`:
  `GOAL_INPUT → ROADMAP_APPROVAL → EXPLAINING → QUIZZING → COMPLETE`.
- `start_session(goal)`:
  1. Create a new session ID and `graph_config = get_langfuse_config(id)`.
  2. `invoke(initial_state)` should return `__interrupt__`. Store the
     roadmap from the payload.
- `approve_roadmap(bool)`:
  1. `invoke(Command(resume="yes"/"no"))`.
  2. If it interrupts again, show the new roadmap.
  3. Otherwise the explainer has run and the graph paused before the quiz.
     Extract the explanation and the topic, then call
     `generate_questions(...)`.
- QUIZZING: show one question at a time, call `grade_answer` on submit, and
  show feedback. Build a `QuizResult` at the end.
- `advance_after_quiz(quiz_result)`:
  1. `ui_graph.update_state(config, {quiz_results: existing+[r], weak_areas: merged, roadmap, current_topic_index, error: None}, as_node="quiz_generator")`.
  2. `invoke(None, config)` runs the coach and then the next explainer,
     which pauses again, or reaches END.
  3. Show the coaching message, then either load the next topic's
     explanation and questions or go to COMPLETE (and call
     `flush_langfuse()`).
- COMPLETE screen: per-topic ✓/✗, the scores, the average, the weak areas,
  and a "new session" button.
- Wrap long calls in `st.spinner`, and show `st.session_state.error`
  prominently.

**Accept:** `make streamlit` lets you complete a 2+ topic session in the
browser.

---

### Phase 11: Docs + polish

- `README.md` covers:
  - quick start (Ollama pull, venv, `.env`, `python main.py`);
  - running all services (3 terminals or `make services` + `make run`);
  - resume; Langfuse; testing tiers;
  - adding your own notes (drop `.md` files into `NOTES_PATH`);
  - a config reference table.
- `docs/ARCHITECTURE.md` covers:
  - the diagram;
  - why each agent is separate (the table from section 2);
  - the state contract; checkpointing and the SQLite connection pattern;
  - interrupt vs. interrupt_before;
  - MCP direct import vs. transport;
  - the A2A flow; observability via callbacks only.
- `docs/MODEL_SELECTION.md`:
  - a VRAM→model table: qwen2.5:7b, qwen3:8b, qwen2.5-coder:32b,
    qwen3:32b, and CPU-only;
  - why <7B models break tool calling;
  - per-agent temperature table;
  - how to switch models (just `.env`).
- `docs/DEVIATIONS.md`: everything that differs from this plan, with reasons.

---

## 8. Test tiers & targets

| Tier | Command | Needs | Target |
|---|---|---|---|
| Unit | `make test` (`pytest -m "not eval"`) | nothing | ~180 tests, < 10 s, 100% pass |
| Eval | `make eval` | Ollama | ~12 tests, conservative thresholds |
| Manual E2E | `make services && make run` | Ollama (+ Docker for Langfuse) | full session completes; resume works |

---

## 9. Execution flow (the reference for debugging)

```
invoke(initial_state)
  curriculum_planner  → roadmap
  human_approval      → interrupt() ⇒ caller shows roadmap
invoke(Command(resume="yes"))
  human_approval      → approved=True (+ full state)
  explainer[t0]       → tool loop over MCP → explanation in messages
  quiz_generator[t0]  → 3 Qs, graded → quiz_results += r, weak_areas ∪=
  progress_coach[t0]  → status(t0) set, idx=1, memory progress_topic_0,
                        (score<0.5 → CrewAI Study Buddy via A2A)
  route → explainer[t1] … until idx ≥ len(topics) → END
```
Checkpoint after every node; `invoke(None, same thread_id)` resumes.

---

## 10. Production hardening checklist (Sherpa's own, for later)

- Persistent memory backend: replace `_store` with Redis or Postgres. The MCP
  interface stays the same.
- Run MCP servers out of process (stdio/HTTP) through `langchain-mcp-adapters`.
- Add `tenacity` retries with backoff around Ollama calls and A2A POSTs.
- Add A2A auth (an API key header) and per-service timeouts. Don't bind to
  `0.0.0.0` outside dev.
- Postgres checkpointer instead of SQLite for multi-process deployments.
- Run the grading calls in parallel (the three calls are independent).
- Gate CI on unit tests, and run evals nightly with score tracking.
- Use non-default secrets in the Langfuse compose file outside localhost.

---

## 11. Definition of done

- [ ] All phases complete; `make test` is green.
- [ ] `make eval` runs, and failures are only threshold misses that
      DEVIATIONS.md explains.
- [ ] A terminal session completes end to end on `qwen2.5:7b`.
- [ ] Rejecting the roadmap regenerates it.
- [ ] `--resume` continues a killed session at the right node.
- [ ] With both A2A services up, a low score shows Study Buddy output.
- [ ] `scripts/a2a_demo.py` passes against live services.
- [ ] Langfuse shows a grouped session trace when keys are set, and
      everything runs identically when they aren't.
- [ ] The Streamlit UI completes a session.

---

## 12. Library gotchas (verified against the pinned versions)

1. **A2A endpoint and method.** a2a-sdk 0.3.25 serves JSON-RPC at `POST /`
   with method `message/send`, and parts on the wire are keyed `kind`, not
   `type`. Older examples using `/tasks/send` or `tasks/send` won't work.
2. **Reading A2A input.** `RequestContext` has no `current_request`, and
   message parts are `Part` RootModels, so `isinstance(part, TextPart)`
   checks fail. → Use `context.get_user_input()`.
3. **A2A messages need `message_id`.** Building `Message(role=..., parts=...)`
   without it fails validation. → Use `new_agent_text_message(...)`.
4. **Langfuse 4.x handler args.** `CallbackHandler` only accepts `public_key`
   and `trace_context`. Passing `secret_key`, `host`, `session_id`, `tags`,
   etc. raises; if that's swallowed by a `try/except`, tracing is silently
   off. → Keys come from env; session, user and tags go in
   `config["metadata"]` as `langfuse_session_id`, `langfuse_user_id` and
   `langfuse_tags`.
5. **`langchain` is a required dependency.** `langfuse.langchain` imports
   `langchain`. Without it the import fails and tracing is silently off.
6. **Langfuse v3 self-hosting needs the full stack.** A Postgres-only compose
   file can't run `langfuse/langfuse:3`; it also needs ClickHouse, Redis and
   MinIO. → Use the official compose file.

---

## 13. Stretch goal: real RAG behind the same MCP contract

Retrieval today is substring search (`search_notes`). To upgrade it without
touching any agent:

1. Add `semantic_search(query, k=5) -> list[dict{file, chunk, score}]` to
   `filesystem_server.py`.
2. At startup (or via a `scripts/index_notes.py`):
   - chunk the notes by heading, roughly 300–500 tokens per chunk;
   - embed with Ollama `nomic-embed-text`;
   - store in Chroma or FAISS under `data/`.
3. Wrap it as `tool_semantic_search` in the Explainer, and update the prompt
   to use semantic search, falling back to keyword search.
4. Point the eval `retrieval_context` at the retrieved chunks, and add a
   `ContextualRelevancyMetric` test.