from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from rai_rag.introspection.planner import make_trace
from rai_rag.introspection.formatter import format_trace
from rai_rag.introspection.validators import validate_introspection_output
from rai_rag.introspection.trace import parse_introspection_trace


@dataclass
class Stage1Output:
    trace_text: str
    ir: Dict[str, Any]
    output_summary: str


def run_stage1_introspect(user_prompt: str, cfg: Dict[str, Any]) -> Stage1Output:
    """
    Stage 1: Structured introspection.
    Current implementation is deterministic heuristic planner + strict formatting and validation.
    Later you can replace planner with an LLM call that fills the same tags.
    """
    plan = make_trace(user_prompt)
    trace_text = format_trace(plan)

    val = validate_introspection_output(trace_text)
    val.raise_if_failed()

    trace = parse_introspection_trace(trace_text)

    return Stage1Output(
        trace_text=trace_text,
        ir=trace.ir_json,
        output_summary=trace.output,
    )

