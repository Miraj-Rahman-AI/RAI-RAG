from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Prompt:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """
    Parsed structured introspection trace.
    """
    raw_text: str
    ir_json: Dict[str, Any]
    output: str


@dataclass
class RetrievalPlan:
    action: str  # Retrieve | Restrict | No-Retrieve
    backend: str = "bm25"
    top_k: int = 8
    query: str = ""
    expected_evidence_type: str = "high_level_overview"
    constraints: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "backend": self.backend,
            "top_k": self.top_k,
            "query": self.query,
            "expected_evidence_type": self.expected_evidence_type,
            "constraints": self.constraints,
            "rationale": self.rationale,
        }


@dataclass
class EvidenceItem:
    doc_id: str
    score: float
    reason: str
    snippets: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None
    retrieval_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": float(self.score),
            "reason": self.reason,
            "snippets": list(self.snippets),
            "meta": self.meta,
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
        }


@dataclass
class EvidenceBundle:
    kept: List[EvidenceItem] = field(default_factory=list)
    filtered: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept": [k.to_dict() for k in self.kept],
            "filtered": list(self.filtered),
            "summary": dict(self.summary),
        }


@dataclass
class JudgeScore:
    score: float
    label: str
    rationale: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "label": self.label,
            "rationale": self.rationale,
            "meta": self.meta,
        }


@dataclass
class Scores:
    safety: JudgeScore
    helpfulness: JudgeScore
    introspection: JudgeScore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety": self.safety.to_dict(),
            "helpfulness": self.helpfulness.to_dict(),
            "introspection": self.introspection.to_dict(),
        }

