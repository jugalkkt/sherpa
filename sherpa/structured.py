"""Reliable structured output from a small local model.

Ollama constrains decoding to the schema's JSON Schema (`format=`), which removes most
syntax errors. Pydantic validators still catch semantic problems (bad topic order,
wrong counts...); those are sent back to the model to repair, up to `max_repairs` times.
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from sherpa.llm import with_retry

M = TypeVar("M", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, schema: type[BaseModel], errors: str, raw: str):
        super().__init__(f"{schema.__name__}: model output still invalid after repairs: {errors}")
        self.schema = schema
        self.errors = errors
        self.raw = raw


def format_validation_error(e: ValidationError, limit: int = 8) -> str:
    lines = []
    for err in e.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    if len(e.errors()) > limit:
        lines.append(f"- ...and {len(e.errors()) - limit} more")
    return "\n".join(lines)


def invoke_structured(
    llm: ChatOllama,
    schema: type[M],
    messages: Sequence[BaseMessage],
    *,
    max_repairs: int = 2,
    context: dict[str, Any] | None = None,
) -> M:
    """Ask `llm` for JSON matching `schema`, validate it, and repair on failure.

    `context` is forwarded to pydantic validators (`ValidationInfo.context`).
    """
    bound = llm.bind(format=schema.model_json_schema())
    invoke = with_retry(bound.invoke)
    history: list[BaseMessage] = list(messages)
    raw = ""
    errors = ""

    for _ in range(max_repairs + 1):
        reply = invoke(history)
        raw = str(reply.content)
        try:
            return schema.model_validate_json(raw, context=context)
        except ValidationError as e:
            errors = format_validation_error(e)
            history += [
                AIMessage(content=raw),
                HumanMessage(
                    content=f"Your output failed validation:\n{errors}\n"
                    "Return the complete corrected JSON only, with no other text."
                ),
            ]

    raise StructuredOutputError(schema, errors, raw)
