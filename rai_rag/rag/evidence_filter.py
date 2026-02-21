from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .retriever import RetrievedDoc


_BAD_SIGNALS = [
    "step-by-step", "exploit", "payload", "bypass", "jailbreak", "malware",
    "how to hack", "bomb", "weapon"
]
_PII_SIGNALS = ["passport", "ssn", "credit card", "bank account", "phone number", "address"]


@dataclass
class EvidenceItem:
    doc_id: str
    score: float
    reason: str
    snippets: List[str]
    meta: Dict[str, Any]
    rank: int | None = None
    retrieval_score: float | None = None


@dataclass
class FilteredItem:
    doc_id: str
    score: float
    reason: str
    risk_flags: List[str]
    meta: Dict[str, Any]
    rank: int | None = None
    retrieval_score: float | None = None


@dataclass
class EvidenceBundle:
    kept: List[Dict[str, Any]]
    filtered: List[Dict[str, Any]]
    summary: Dict[str, Any]


def evidence_safety_score(text: str) -> Tuple[float, List[str]]:
    """
    Heuristic scorer baseline. Replace with LLM judge later.
    Returns score in [-1, 1] and flags.
    """
    t = (text or "").lower()
    flags: List[str] = []
    if any(s in t for s in _BAD_SIGNALS):
        flags.append("procedural_harm")
        return -0.7, flags
    if any(s in t for s in _PII_SIGNALS):
        flags.append("pii")
        return -0.2, flags
    return 0.4, flags


def filter_evidence(
    docs: List[RetrievedDoc],
    cfg: Dict[str, Any],
) -> EvidenceBundle:
    ef_cfg = (cfg.get("evidence_filter") or {})
    rag_cfg = (cfg.get("rag") or {})

    drop_below = float(ef_cfg.get("drop_if_score_below", 0.0))
    max_snips = int(ef_cfg.get("max_snippets_per_doc", 2))
    max_chars = int(ef_cfg.get("max_snippet_chars", 240))
    min_keep = int(rag_cfg.get("min_keep_docs", 2))

    kept: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []

    for rd in docs:
        score, flags = evidence_safety_score(rd.doc.text)
        if score >= drop_below:
            snip = rd.doc.text[:max_chars].strip()
            kept.append(
                EvidenceItem(
                    doc_id=rd.doc.id,
                    score=float(score),
                    reason="Heuristic-safe evidence",
                    snippets=[snip][:max_snips],
                    meta=rd.doc.meta,
                    rank=rd.rank,
                    retrieval_score=rd.retrieval_score,
                ).__dict__
            )
        else:
            filtered.append(
                FilteredItem(
                    doc_id=rd.doc.id,
                    score=float(score),
                    reason="Heuristic-unsafe evidence",
                    risk_flags=flags or ["other"],
                    meta=rd.doc.meta,
                    rank=rd.rank,
                    retrieval_score=rd.retrieval_score,
                ).__dict__
            )

    fallback = "continue"
    if len(kept) < min_keep:
        pref = str(ef_cfg.get("if_insufficient_evidence", "safe_high_level"))
        if pref == "restrict_retrieval":
            fallback = "restrict_retrieval"
        else:
            fallback = "no_retrieve_and_safe_high_level"

    summary = {
        "num_in": len(docs),
        "num_kept": len(kept),
        "num_filtered": len(filtered),
        "fallback_recommendation": fallback,
    }

    return EvidenceBundle(kept=kept, filtered=filtered, summary=summary)

