# Deviations from plan.md

Everything that differs from the plan, and why.

## Phase 0

### `langchain-core` 1.0.0 → 1.2.10, `langchain` pinned to 1.2.11

The plan pins `langgraph==1.1.0` and `langchain-core==1.0.0` and leaves
`langchain` unpinned ("pin to whatever pip resolves"). No `langchain` release
satisfies both pins:

| langchain | requires langchain-core | requires langgraph |
|---|---|---|
| 1.0.0 – 1.2.10 | ≥ 1.0.0 … ≥ 1.2.10 | **< 1.1.0** |
| 1.2.11 | ≥ 1.2.10 | ≥ 1.1.0, < 1.3.0 |
| 1.2.12+ | ≥ 1.2.10 | ≥ 1.1.1 |

Left unpinned, pip backtracks to `langchain==0.0.27` (a 2023 release). That
looks like a success but breaks tracing: `langfuse/langchain/CallbackHandler.py`
only takes its modern code path when `langchain.__version__` starts with
`"1"`, and otherwise imports `langchain.schema.agent` and similar legacy
modules (plan gotcha #5).

**Resolution:** keep `langgraph==1.1.0` (the orchestration core, and the
version the plan's behavioural notes target), pin `langchain==1.2.11` (the
only release allowing langgraph 1.1.0), and raise `langchain-core` to its
minimum, 1.2.10. That is a minor bump within 1.x; we only use its message
and tool types. `a2a-sdk` stays at 0.3.25. All other pins are unchanged.

Verified with `pip install --dry-run -r requirements.txt`.

## Phase 1

### Why checkpointed dataclasses can come back as dicts (verified)

The plan says dataclasses "may" come back as dicts after a checkpoint. On
langgraph 1.1.0 the actual behaviour of `JsonPlusSerializer` is:

| Mode | Unregistered dataclass after round-trip |
|---|---|
| default (permissive) | restored as the dataclass, **plus a warning** that this "will be blocked in a future version" |
| strict (`LANGGRAPH_STRICT_MSGPACK=true`, i.e. `allowed_msgpack_modules=None`) | **plain dict** |

So the dict-tolerant helpers are needed, not just defensive.
`test_helpers_survive_strict_checkpoint_round_trip` runs the real serializer
in strict mode to prove it.

Done in Phase 2: `build_checkpointer` registers the four dataclasses via
`allowed_msgpack_modules`. The helpers stay dict tolerant either way.

## Phase 2

### Remote Ollama over ngrok + basic auth (`src/llm.py`)

Not in the plan. Ollama runs on a Kaggle GPU behind an ngrok static domain
with basic auth. Every `ChatOllama` is built by `llm.build_llm(temperature,
json_mode)`, which reads `OLLAMA_MODEL`, `OLLAMA_BASE_URL` and the new
optional `OLLAMA_BASIC_AUTH=user:pass` at call time and passes an
`Authorization: Basic …` header via `client_kwargs`. Empty
`OLLAMA_BASIC_AUTH` means plain local Ollama, exactly as the plan assumes.
`describe_llm_error` turns "tunnel offline" (`ERR_NGROK_3200`), 401, a
missing model, connection errors and timeouts into actionable messages.

### Smaller choices

- `build_graph(db_path=None, ...)`: `None` means `$CHECKPOINT_DB`, falling
  back to `data/checkpoints.db`. The plan's signature hard-codes the default,
  which would make the env var unreachable.
- `graph` is built lazily (module `__getattr__`, PEP 562). `from
  graph.workflow import graph` works as the plan says, but importing the
  module (e.g. in tests) no longer creates `data/checkpoints.db`.
- `parse_roadmap_json` strips ```` ``` ```` code fences (small models add
  them despite JSON mode), forces every topic's `status` to `"pending"`, and
  wraps a string `prerequisites` in a list.
- The planner node also catches LLM *call* failures (not just `ValueError`)
  and returns `{"error": ...}`, per the plan's "graph must never crash" rule.
- Tests: besides `test_curriculum_planner.py`, there are `test_llm.py` (the
  factory) and `test_workflow.py` (routing, graph wiring, checkpointing).
- `main.py` uses a local `get_config()` until Phase 7's
  `get_langfuse_config`, and catches Ctrl+C to print the resume command.

## Phase 3

- `NOTES_PATH`, when relative, is resolved against the **project root**, not
  the current directory. The plan's bare `Path(os.getenv(...))` would break
  when the server is launched from another directory (e.g. as an MCP
  subprocess). `NOTES_BASE` is still a module constant tests can monkeypatch.
- `read_study_file` checks the `.md` extension before existence, and treats a
  directory named `*.md` as "not found". It also catches
  `UnicodeDecodeError`, not just `OSError`.
- Traversal is blocked by `resolve()` + `relative_to()`, which also covers
  absolute paths (`/etc/passwd`) and symlinks pointing outside the folder.
- `search_notes` returns `[]` for an empty or whitespace query (otherwise
  every line would match).
- `memory_delete` on a missing key returns a "nothing deleted" message, not an
  error string.
- Tests also exercise both servers over the MCP protocol (in-memory client
  session) and launch the filesystem server as a real stdio subprocess.

## Phase 4

### The node records `explained_topics`, not the model

The plan's prompt has the model write the explanation and *then* call
`memory_set(session_id, 'explained_topics', ...)`. But the loop stops at the
first reply with no tool calls, so if the model obeys, its final message is
something like "Saved!" rather than the explanation, and Phase 5 takes "the
last AIMessage with no tool_calls" as the text to quiz on. Instead the prompt
ends with "write the explanation as your final reply, with no further tool
calls", and `explainer_node` appends the topic to `explained_topics` in Python
after a successful answer (deduplicated). The model still has `memory_get` (to
build on earlier topics) and `memory_set`. Bookkeeping that must always happen
shouldn't depend on a 7B model remembering to do it.

### Planner failure ends the run (`route_after_planner`)

The plan has a static edge `curriculum_planner → human_approval`. With the
real Explainer in place, a failed plan (e.g. tunnel offline) ran on into the
Explainer, whose "no current topic" error *overwrote* the real cause in
`state["error"]`. The edge is now conditional: if the planner set `error`, go
to `END`; otherwise continue to approval as before.

### Smaller choices

- Tool names the model sees are short (`list_files`, `read_file`, ...) via
  `@tool("name")`; the Python functions keep the plan's `tool_*` names.
- `ToolMessage`s also carry `name=` (the tool's name), alongside the matching
  `tool_call_id`.
- An LLM call failure inside the loop returns `{messages, error}` with the
  `describe_llm_error` message.
- `tests/conftest.py` has an autouse fixture pointing `OLLAMA_BASE_URL` at a
  closed local port for every non-`eval` test, so a missing mock fails fast
  and never reaches a real model.
- `main.py` resets the root logger to WARNING after imports: constructing a
  `FastMCP` server calls `logging.basicConfig(level=INFO)`, which made every
  Ollama HTTP request print a log line.

## Phase 5

### The explanation to quiz on ignores stale messages

The plan: explanation = "last AIMessage with content and no tool_calls". On
its own that is wrong when the Explainer fails for topic N: the last such
message is topic N-1's explanation (or the Coach's summary), so the quiz
would test the wrong topic. `extract_explanation` first checks `error`: if
the Explainer set one, it falls back to the topic's title and description.
The Coach reuses the same function for the Study Buddy call.

### Smaller choices

- `generate_questions(topic: str, ...)` and `run_quiz(topic: str, ...)` take
  the topic *title*, so the A2A service (Phase 9) can call them with plain
  strings. The node passes `topic.title`.
- `generate_questions` drops malformed questions, caps the list at `n`,
  normalises `difficulty` to easy/medium/hard, and truncates the explanation
  to 6000 characters in the prompt.
- `grade_answer` parses `correct` from strings properly (`bool("false")` is
  `True` in Python) and accepts a numeric-string score.
- Weak areas are deduplicated case-insensitively, keeping first-seen order.
- `PASS_THRESHOLD` is defined as `graph.state.PASS_SCORE` so the coach and
  `QuizResult.passed()` can't disagree.
- `progress_coach_node` deep-copies the roadmap before changing a status
  (it never mutates objects that belong to the previous state), and returns
  an error for an out-of-range topic index.
- `try_study_buddy_assistance` exists as a `return None` placeholder until
  Phase 9.

## Phase 6

### "Keys lost after Command(resume=...)" not reproduced

The plan says to return the full state from `human_approval_node` because
"downstream nodes can lose keys after `Command(resume=...)` on LangGraph
1.1.0". A minimal graph on langgraph 1.1.0 (approval node returning only
`approved`, then resumed) showed every key intact in the next node. The node
still returns the full state, as the plan asks: it's harmless.

### The status line prints after `interrupt()`, not before

On resume, LangGraph re-runs the interrupted node from the top; `interrupt()`
then returns the resume value. Anything before it runs twice (verified), so
the node prints `[Approval] Roadmap approved/rejected` after the decision
instead of a status line before the pause.

### Smaller choices

- `str(decision)` handles a `None` resume value (counts as rejected).
- Test fixtures shared by several files (`mocked_llms`, `answers`,
  `clear_memory`, `db_path`) moved to `tests/conftest.py`. `answers` can
  raise an exception queued in it, which simulates Ctrl+C at a prompt.
- `test_checkpointing.py` automates the plan's manual resume check (Ctrl+C in
  the first quiz, new graph instance, resume at `quiz_generator`), drives the
  real `main.py` CLI, and runs a whole session in strict-serializer mode
  where every checkpointed dataclass comes back as a dict.

## Phase 7

### Langfuse server is v4, not v3

The plan says "Langfuse v3 self-host". The official `docker-compose.yml` on
`main` (fetched at commit `0dd0a7fbe2feb300b8776f02b3684eeee3fbceab`) now runs
`langfuse:4` / `langfuse-worker:4`, still with Postgres, ClickHouse, Redis and
MinIO. Kept as-is, per the plan's "keep its local-dev defaults", with one
change: `langfuse-web` (3000) and `minio` (9090) are bound to `127.0.0.1`.
Upstream publishes them on all interfaces, which with the default CHANGEME
secrets would expose traces to the local network.

### Health check before tracing is switched on

Not in the plan. With keys set but the server stopped, every span export
failed and OpenTelemetry logged a full traceback (about 640 lines for a
one-node run) into the terminal. `get_langfuse_handler` now GETs
`/api/public/health` (2 s timeout) first; if nothing answers it prints one
line and runs untraced. `main.py` also sets the `opentelemetry` logger to
CRITICAL so an outage *mid-session* stays quiet too.

### Smaller choices

- Host precedence matches the langfuse client: `LANGFUSE_BASE_URL`, then
  `LANGFUSE_HOST`, then Langfuse Cloud. With no host set, traces go to the
  cloud; `.env.example` sets `http://localhost:3000`.
- `get_langfuse_config(..., extra_config)` merges nested dicts, so extra
  `configurable` keys don't drop `thread_id`.
- `main.py` calls `flush_langfuse()` in a `finally`, so Ctrl+C'd or failed
  sessions still send their traces.
- `tests/conftest.py` removes Langfuse keys for every unit test, and
  `test_observability.py` patches the health check, so tests never contact a
  tracing server.

## Phase 8

### deepeval telemetry was running on every pytest run (fixed)

deepeval installs a pytest plugin (`pytest11` entry point) that imports
`deepeval` on *every* pytest run, unit-only ones included. Importing it
without `DEEPEVAL_TELEMETRY_OPT_OUT` looks up the public IP (api.ipify.org)
and initialises Sentry and PostHog (verified in `deepeval/telemetry.py`).
Fixes: `-p no:deepeval` in pytest `addopts` (our evals call `metric.measure`
directly and don't need the plugin), `DEEPEVAL_TELEMETRY_OPT_OUT=YES` set at
the top of `tests/conftest.py`, in `make eval`, and in `.env.example`.

### Judge uses Ollama structured outputs, not plain JSON mode

The plan's judge calls Ollama with `format="json"` when DeepEval passes a
pydantic `schema`. We pass `schema.model_json_schema()` as `format`, so
Ollama constrains generation to that exact shape, and fall back to the raw
text (which DeepEval parses itself) if validation still fails.
`llm.build_llm` gained keyword-only `model=` and `json_schema=` for this.
Optional `OLLAMA_JUDGE_MODEL` lets a different model grade.

### Other choices

- Faithfulness is judged against the notes the Explainer actually read
  (`read_file` results), falling back to `closures.md`. Judging against a
  file it never opened would penalise correct grounding.
- The Explainer runs once per module and four tests share its output.
- Evals load `.env` themselves (unit tests never do) and skip, with the
  reason, if the model is unreachable.
- Extra checks beyond the plan: the explanation follows the 4-section format,
  generation didn't fall back, grading/coaching didn't fall back.
- `TestOllamaJudgePlumbing` (unit, no model) tests the judge adapter.

### First live run (qwen2.5:7b on Kaggle via ngrok, 2026-10-04)

13/13 passed in about 2 minutes. Scores: faithfulness 1.00, answer
relevancy 1.00, quiz "tests understanding" 0.70, coaching 0.80. Grading:
correct answer 0.8, wrong 0.3 (limit 0.35), partial 0.7 (limit 0.75).
Observations, not failures:
- The Explainer skipped prompt step 4 (`memory_get` for earlier topics).
- The explanation never mentioned `nonlocal`, although the topic
  description asked for it; the key-concepts check passed via "enclosing",
  and the judge still gave relevancy 1.00. Same-model judging is lenient.
- The wrong and partial grades sit close to their limits, so expect
  occasional flakes on reruns.

## Post-Phase 8 fix: quiz questions referred to code the learner couldn't see

Found in the first live CLI session: "Explain why the following code results
in both `a` and `b` being `[1, 2, 3, 4]`." with no code shown. Raw model
output confirmed nothing was dropped: the model saw the explanation while
writing questions and assumed the learner could too ("in the example
provided", "the `add_item` function").

- Questions gain a `code` field. The prompt says the learner cannot see the
  notes, so every question must be self-contained.
- Generation uses Ollama structured output (`QUESTIONS_SCHEMA`), not plain
  JSON mode, so every question has all four fields.
- Safety net: a question that still says "the following code", "in the
  example", etc. with no `code` is dropped (all dropped → fallback question).
- Code pasted into the question text is moved to `code`, or removed if it
  duplicates it (seen live).
- The CLI prints the code indented under the question; `full_question()`
  (text + fenced code) is what gets graded, stored and judged in evals.

Live check, 3 generations × 3 questions: all 9 questions self-contained.
Known 7B limitation, not fixed: some *expected answers* are partly wrong
(e.g. "use nonlocal" for a module-level variable; wrong reasoning on why
`add_item(1, [])` prints `[1]`). The grader compares against these.

## Post-Phase 8 fix: grader gave 50% to an answer that stated the opposite

Live session: the answer "new_list creates a duplicate ... 2 different lists"
(for `new_list = my_list`, where both names share one list) scored 50% with
feedback saying it was incorrect. Re-grading it 5 times gave 0.50 every time.
The score-band prompt (overlapping bands, "encouraging grader", free-form
number) made `qwen2.5:7b` hedge at the boundary where two bands met.

- Verdict-first prompt: the model picks one of correct / minor_gaps /
  partial / wrong, with an explicit rule that stating the opposite of the key
  fact is "wrong". "Encouraging" removed; kindness belongs in the feedback.
- Ollama structured output (`GRADE_SCHEMA`) restricts `verdict` to those four.
- `correct` is derived in code (`verdict in {correct, minor_gaps}`) and the
  score is clamped into the verdict's range (`VERDICT_RANGES`), so the model
  can no longer contradict itself (e.g. `correct: true` with a wrong verdict).
  A "partial" answer is therefore shown with ✗ even at 0.5–0.7; it still
  counts toward the topic's pass score, which uses the numeric average.
- Unknown verdicts, missing fields and the old output format fall back to the
  neutral 0.5 result, as before.

Live results (qwen2.5:7b, 3 runs each, stable): wrong answer 0.00, correct
0.85, partial 0.50, empty 0.00. Grading evals: correct 0.85, wrong 0.00,
partial 0.50, all pass with more margin than before (wrong 0.30 vs limit 0.35,
partial 0.70 vs limit 0.75).
Cosmetic, not fixed: some feedback refers to "the student" instead of "you".
