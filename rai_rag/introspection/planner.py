from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .formatter import IRPlan, make_default_ir_json


# Very coarse risk keyword sets (baseline only).
RISK_KEYWORDS = {
    "cyber": ["hack", "exploit", "malware", "payload", "ddos", "phishing", "bypass"],
    "weapons": ["weapon", "gun", "bomb", "explosive"],
    "self_harm": ["suicide", "self-harm", "kill myself"],
    "privacy_pii": ["passport", "ssn", "credit card", "bank account", "phone number", "address"],
    "illicit_drugs": ["meth", "cocaine", "heroin", "how to make drugs"],
    "violence": ["kill", "murder", "assault"],
}

JAILBREAK_SIGNALS = [
    "ignore previous",
    "developer message",
    "system prompt",
    "jailbreak",
    "bypass safety",
    "do not refuse",
]


def _contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _infer_risk_category(prompt: str) -> Tuple[str, List[str], List[str]]:
    """
    Returns: (risk_category, sensitive_topics_detected, jailbreak_signals)
    """
    t = prompt.lower()

    signals = [s for s in JAILBREAK_SIGNALS if s in t]

    hits = []
    for cat, kws in RISK_KEYWORDS.items():
        if _contains_any(t, kws):
            hits.append(cat)

    # Pick one primary category conservatively
    if "self_harm" in hits:
        primary = "self_harm"
    elif "weapons" in hits:
        primary = "weapons"
    elif "cyber" in hits:
        primary = "cyber"
    elif "violence" in hits:
        primary = "violence"
    elif "illicit_drugs" in hits:
        primary = "illicit_drugs"
    elif "privacy_pii" in hits:
        primary = "privacy_pii"
    else:
        primary = "benign_info"

    return primary, hits, signals


def _infer_severity(risk_category: str, jailbreak_signals: List[str]) -> str:
    if risk_category in {"self_harm", "weapons", "violence"}:
        return "high"
    if risk_category in {"cyber", "illicit_drugs"}:
        return "medium" if not jailbreak_signals else "high"
    if risk_category in {"privacy_pii"}:
        return "medium"
    return "low"


def _infer_retrieval_need(prompt: str, risk_category: str) -> str:
    """
    Conservative:
      - for high-risk categories, retrieval not necessary and may amplify harm
      - for benign info, retrieval helpful/required depending on specificity
    """
    if risk_category in {"self_harm", "weapons", "violence", "cyber", "illicit_drugs"}:
        return "none"

    # simple heuristic: if asking for citations / statistics / recent facts -> required
    p = prompt.lower()
    if any(k in p for k in ["cite", "citation", "evidence", "paper", "dataset", "2024", "2025", "latest"]):
        return "required"
    return "helpful"


def _infer_retrieval_risk(risk_category: str, jailbreak_signals: List[str]) -> str:
    if jailbreak_signals:
        return "high"
    if risk_category in {"cyber", "weapons", "illicit_drugs", "violence", "self_harm"}:
        return "high"
    if risk_category in {"privacy_pii"}:
        return "medium"
    return "low"


def _infer_response_mode(risk_category: str, severity: str) -> str:
    if severity == "high" and risk_category in {"self_harm", "weapons", "violence", "cyber", "illicit_drugs"}:
        return "refuse_with_alternatives"
    if risk_category in {"privacy_pii"}:
        return "safe_high_level"
    return "safe_grounded"


def make_trace(prompt: str) -> IRPlan:
    """
    Produces an IRPlan that formatter can render into strict-tag output.
    This is a baseline, deterministic, no-LLM planner.
    """
    ir = make_default_ir_json()

    risk_category, sensitive_hits, jailbreak_signals = _infer_risk_category(prompt)
    severity = _infer_severity(risk_category, jailbreak_signals)
    retrieval_need = _infer_retrieval_need(prompt, risk_category)
    retrieval_risk = _infer_retrieval_risk(risk_category, jailbreak_signals)
    response_mode = _infer_response_mode(risk_category, severity)

    # Ambiguity: if short prompt or conflicting signals -> mark ambiguous
    is_amb = len(prompt.strip()) < 20 or (len(sensitive_hits) > 1) or bool(jailbreak_signals)
    amb_reason = "short or multi-signal prompt" if is_amb else "intent appears clear"

    ir.update(
        {
            "intent_hypothesis": "benign info seeking" if risk_category == "benign_info" else "potentially sensitive request",
            "risk_category": risk_category,
            "severity": severity,
            "ambiguity": {"is_ambiguous": is_amb, "reason": amb_reason},
            "retrieval_need": retrieval_need,
            "retrieval_risk": retrieval_risk,
            "response_mode": response_mode,
            "notes": {
                "sensitive_topics_detected": sensitive_hits,
                "pii_risk": "medium" if risk_category == "privacy_pii" else "low",
                "jailbreak_signals": jailbreak_signals,
            },
        }
    )

    reasoning_steps: List[str] = [
        f"Intent hypothesis: {ir['intent_hypothesis']} (high-level; treat prompt as untrusted).",
        f"Risk categorization: category={risk_category}, severity={severity}.",
        f"Ambiguity assessment: is_ambiguous={is_amb} ({amb_reason}).",
        f"Retrieval necessity: retrieval_need={retrieval_need}.",
        f"Retrieval risk: retrieval_risk={retrieval_risk} (retrieval can be an attack surface).",
        f"Recommended response mode: {response_mode}.",
    ]

    output = "Introspection complete: proceeding with a safety-first plan."

    return IRPlan(reasoning_steps=reasoning_steps, ir_json=ir, output=output)

