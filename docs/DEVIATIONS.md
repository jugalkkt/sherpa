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
