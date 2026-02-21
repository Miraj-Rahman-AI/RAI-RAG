from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IRPlan:
    """
    Intermediate representation from planner -> formatter.
    """
    reasoning_steps: List[str]
    ir_json: Dict[str, Any]
    output: str


def _json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def format_trace(plan: IRPlan) -> str:
    """
    Enforce strict tag structure:
      <Reasoning_step>...</Reasoning_step> (one per step)
      <IR_JSON>{...}</IR_JSON>
      <Output>...</Output>
    """
    if not plan.reasoning_steps:
        raise ValueError("IRPlan.reasoning_steps must be non-empty")
    if not isinstance(plan.ir_json, dict):
        raise ValueError("IRPlan.ir_json must be a dict")
    if not plan.output or not plan.output.strip():
        raise ValueError("IRPlan.output must be non-empty")

    parts: List[str] = []
    for s in plan.reasoning_steps:
        parts.append(f"<Reasoning_step>\n{s.strip()}\n</Reasoning_step>")

    parts.append(f"<IR_JSON>\n{_json_dumps(plan.ir_json)}\n</IR_JSON>")
    parts.append(f"<Output>\n{plan.output.strip()}\n</Output>")

    return "\n\n".join(parts) + "\n"


def make_default_ir_json() -> Dict[str, Any]:
    """
    Minimal schema default. Planner should fill fields.
    """
    return {
        "intent_hypothesis": "unknown",
        "risk_category": "other",
        "severity": "low",
        "ambiguity": {"is_ambiguous": True, "reason": "not assessed"},
        "retrieval_need": "helpful",
        "retrieval_risk": "medium",
        "response_mode": "safe_high_level",
        "notes": {
            "sensitive_topics_detected": [],
            "pii_risk": "low",
            "jailbreak_signals": [],
        },
    }

