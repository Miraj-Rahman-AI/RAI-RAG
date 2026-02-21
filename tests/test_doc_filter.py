import re
import pytest


def heuristic_evidence_safety_score(doc_text: str) -> float:
    """
    Heuristic baseline:
    - flags obvious procedural harm / exploit content
    - returns score in [-1, 1]
    This is NOT your final judge; it is a test reference baseline.
    """
    t = doc_text.lower()

    # Very strong negative signals (procedural/harm)
    bad_signals = [
        "step-by-step",
        "exploit",
        "payload",
        "bypass",
        "jailbreak",
        "malware",
        "how to hack",
        "bomb",
        "weapon",
    ]
    if any(s in t for s in bad_signals):
        return -0.7

    # PII signals (not always unsafe but requires caution)
    pii_signals = ["passport", "ssn", "credit card", "bank account", "phone number"]
    if any(s in t for s in pii_signals):
        return -0.2

    # Otherwise assume safe-ish informational
    return 0.4


def filter_docs(docs):
    kept, filtered = [], []
    for d in docs:
        score = heuristic_evidence_safety_score(d["text"])
        if score >= 0.0:
            kept.append({**d, "score": score})
        else:
            filtered.append({**d, "score": score})
    return kept, filtered


def test_filter_drops_procedural_harm():
    docs = [
        {"doc_id": "d1", "text": "This is a safe overview of RAG safety."},
        {"doc_id": "d2", "text": "Step-by-step exploit instructions to bypass a system."},
    ]
    kept, filtered = filter_docs(docs)
    assert any(d["doc_id"] == "d1" for d in kept)
    assert any(d["doc_id"] == "d2" for d in filtered)


def test_filter_flags_pii_as_negative():
    docs = [{"doc_id": "d3", "text": "User passport number and phone number are included."}]
    kept, filtered = filter_docs(docs)
    assert len(filtered) == 1
    assert filtered[0]["score"] < 0.0


def test_filter_keeps_safe_info():
    docs = [{"doc_id": "d4", "text": "High-level discussion of evidence alignment in RAG."}]
    kept, filtered = filter_docs(docs)
    assert len(kept) == 1
    assert len(filtered) == 0

