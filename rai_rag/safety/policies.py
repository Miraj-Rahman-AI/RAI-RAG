from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Canonical categories used across RAI-RAG (keep consistent with introspection.validators)
RISK_CATEGORIES = {
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

RESPONSE_MODES = {"safe_grounded", "safe_high_level", "refuse_with_alternatives"}

RETRIEVAL_ACTIONS = {"Retrieve", "Restrict", "No-Retrieve"}


@dataclass
class PolicyDecision:
    """
    Policy-level decision (independent of retrieval gate).
    """
    allow: bool
    response_mode: str  # safe_grounded | safe_high_level | refuse_with_alternatives
    reason: str
    safe_alternatives: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": self.allow,
            "response_mode": self.response_mode,
            "reason": self.reason,
            "safe_alternatives": self.safe_alternatives,
        }


DEFAULT_SAFE_ALTERNATIVES = [
    "Provide a high-level explanation without procedural steps.",
    "Discuss risks, ethics, and relevant laws/policies.",
    "Share defensive or preventative best practices (non-operational).",
]


def policy_route(risk_category: str, severity: str) -> PolicyDecision:
    """
    Conservative policy routing for response_mode.
    This is a baseline; you can extend with richer policy mappings later.
    """
    rc = (risk_category or "other").strip()
    sev = (severity or "low").strip()

    if rc not in RISK_CATEGORIES:
        rc = "other"
    if sev not in {"low", "medium", "high"}:
        sev = "low"

    # hard disallow (refuse) for high severity in dangerous categories
    if sev == "high" and rc in {"self_harm", "weapons", "violence", "illicit_drugs", "cyber"}:
        return PolicyDecision(
            allow=False,
            response_mode="refuse_with_alternatives",
            reason=f"High-risk category '{rc}' with severity '{sev}'.",
            safe_alternatives=list(DEFAULT_SAFE_ALTERNATIVES),
        )

    # for privacy/PII, allow but be high-level and privacy preserving
    if rc == "privacy_pii":
        return PolicyDecision(
            allow=True,
            response_mode="safe_high_level",
            reason="PII/privacy-sensitive topic: respond without exposing or enabling privacy violation.",
            safe_alternatives=[
                "Explain privacy best practices and how to protect personal data.",
                "Describe general compliance and safe handling procedures.",
            ],
        )

    # medium-risk: high-level guidance
    if sev == "medium" and rc in {"cyber", "illicit_drugs", "weapons", "violence", "self_harm"}:
        return PolicyDecision(
            allow=True,
            response_mode="safe_high_level",
            reason=f"Sensitive topic '{rc}' severity '{sev}': restrict to non-actionable, high-level guidance.",
            safe_alternatives=list(DEFAULT_SAFE_ALTERNATIVES),
        )

    # default allow grounded
    return PolicyDecision(
        allow=True,
        response_mode="safe_grounded",
        reason="Default allow: benign or low-risk topic.",
        safe_alternatives=[],
    )


def refusal_template(reason: str, alternatives: Optional[List[str]] = None) -> str:
    alts = alternatives or DEFAULT_SAFE_ALTERNATIVES
    bullets = "\n".join([f"{i+1}) {a}" for i, a in enumerate(alts)])
    return (
        "I can’t help with that request.\n\n"
        f"Reason: {reason}\n\n"
        "Here are safe alternatives I can help with:\n"
        f"{bullets}\n"
    )

