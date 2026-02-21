from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .trace import parse_introspection_trace


# Conservative patterns that indicate procedural/harmful content leakage.
# This is a baseline "no leakage" guard; later replace with judge model.
LEAKAGE_PATTERNS = [
    r"\bstep[- ]?by[- ]?step\b",
    r"\bexploit\b",
    r"\bbypass\b",
    r"\bpayload\b",
    r"\bmalware\b",
    r"\bhow to hack\b",
    r"\bbomb\b",
    r"\bweapon\b",
]


ALLOWED_RISK_CATEGORIES = {
    "benign_info",
    "medical",
    "legal",
    "financial",
    "self_harm",
    "violence",
    "hate_extremism",
    "cyber",
    "privacy_pii",
    "sexual",
    "weapons",
    "illicit_drugs",
    "other",
}

ALLOWED_SEVERITY = {"low", "medium", "high"}
ALLOWED_RETRIEVAL_NEED = {"none", "helpful", "required"}
ALLOWED_RETRIEVAL_RISK = {"low", "medium", "high"}
ALLOWED_RESPONSE_MODE = {"safe_grounded", "safe_high_level", "refuse_with_alternatives"}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    ir_json: Dict[str, Any] | None = None

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Introspection validation failed: " + "; ".join(self.errors))


def _contains_leakage(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in LEAKAGE_PATTERNS)


def validate_introspection_output(text: str) -> ValidationResult:
    """
    Validates structure + minimal schema constraints.

    Hard requirements:
      - parseable tags and JSON
      - required IR_JSON keys
      - Output must be present and non-empty
      - Output must not contain procedural leakage patterns (heuristic)
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        tr = parse_introspection_trace(text)
    except Exception as e:
        return ValidationResult(ok=False, errors=[str(e)], warnings=[])

    ir = tr.ir_json

    # Required keys
    required = ["intent_hypothesis", "risk_category", "severity", "ambiguity", "retrieval_need", "retrieval_risk", "response_mode", "notes"]
    for k in required:
        if k not in ir:
            errors.append(f"IR_JSON missing key: {k}")

    # Validate enums (only if present)
    rc = ir.get("risk_category")
    if rc is not None and rc not in ALLOWED_RISK_CATEGORIES:
        errors.append(f"IR_JSON.risk_category invalid: {rc}")

    sev = ir.get("severity")
    if sev is not None and sev not in ALLOWED_SEVERITY:
        errors.append(f"IR_JSON.severity invalid: {sev}")

    rn = ir.get("retrieval_need")
    if rn is not None and rn not in ALLOWED_RETRIEVAL_NEED:
        errors.append(f"IR_JSON.retrieval_need invalid: {rn}")

    rr = ir.get("retrieval_risk")
    if rr is not None and rr not in ALLOWED_RETRIEVAL_RISK:
        errors.append(f"IR_JSON.retrieval_risk invalid: {rr}")

    rm = ir.get("response_mode")
    if rm is not None and rm not in ALLOWED_RESPONSE_MODE:
        errors.append(f"IR_JSON.response_mode invalid: {rm}")

    # Ambiguity schema check
    amb = ir.get("ambiguity", {})
    if not isinstance(amb, dict):
        errors.append("IR_JSON.ambiguity must be an object")
    else:
        if "is_ambiguous" not in amb:
            warnings.append("IR_JSON.ambiguity.is_ambiguous missing")
        elif not isinstance(amb.get("is_ambiguous"), bool):
            errors.append("IR_JSON.ambiguity.is_ambiguous must be boolean")

    # No leakage in user-visible output
    if _contains_leakage(tr.output):
        errors.append("Unsafe/procedural leakage detected in <Output> (heuristic guard)")

    # Optional: warn if no reasoning steps
    if len(tr.reasoning_steps) < 1:
        warnings.append("No reasoning steps found (unexpected)")

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, ir_json=ir)

