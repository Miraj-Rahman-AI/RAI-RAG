import pytest


def _gate_decision(ir: dict) -> str:
    """
    A small reference gate policy for testing.
    Your real implementation can differ, but must satisfy the core invariants:
    - high severity => No-Retrieve (default)
    - ambiguous or retrieval_risk medium/high => Restrict (default)
    - otherwise Retrieve (if retrieval_need helpful/required)
    """
    severity = ir.get("severity", "low")
    amb = ir.get("ambiguity", {}).get("is_ambiguous", False)
    retrieval_risk = ir.get("retrieval_risk", "low")
    retrieval_need = ir.get("retrieval_need", "none")

    if severity == "high":
        return "No-Retrieve"
    if amb or retrieval_risk in ("medium", "high"):
        return "Restrict"
    if retrieval_need in ("helpful", "required"):
        return "Retrieve"
    return "No-Retrieve"


def test_gate_high_risk_no_retrieve():
    ir = {
        "severity": "high",
        "ambiguity": {"is_ambiguous": False},
        "retrieval_risk": "high",
        "retrieval_need": "required",
    }
    assert _gate_decision(ir) == "No-Retrieve"


def test_gate_ambiguous_restrict():
    ir = {
        "severity": "low",
        "ambiguity": {"is_ambiguous": True},
        "retrieval_risk": "low",
        "retrieval_need": "required",
    }
    assert _gate_decision(ir) == "Restrict"


def test_gate_medium_retrieval_risk_restrict():
    ir = {
        "severity": "low",
        "ambiguity": {"is_ambiguous": False},
        "retrieval_risk": "medium",
        "retrieval_need": "helpful",
    }
    assert _gate_decision(ir) == "Restrict"


def test_gate_benign_retrieve():
    ir = {
        "severity": "low",
        "ambiguity": {"is_ambiguous": False},
        "retrieval_risk": "low",
        "retrieval_need": "helpful",
    }
    assert _gate_decision(ir) == "Retrieve"

