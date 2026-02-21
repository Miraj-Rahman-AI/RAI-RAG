from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .query_rewrite import safe_query_rewrite


@dataclass
class RetrievalPlan:
    action: str               # Retrieve | Restrict | No-Retrieve
    backend: str
    top_k: int
    query: str
    expected_evidence_type: str
    constraints: Dict[str, Any]
    rationale: str

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


def decide_action(ir: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """
    Deterministic baseline gate policy.
    """
    gate_cfg = (cfg.get("retrieval_gate") or {})
    severity = ir.get("severity", "low")
    amb = (ir.get("ambiguity") or {}).get("is_ambiguous", False)
    retrieval_risk = ir.get("retrieval_risk", "low")
    retrieval_need = ir.get("retrieval_need", "none")

    if severity in set(gate_cfg.get("risk_to_no_retrieve", ["high"])):
        return "No-Retrieve"
    if amb and bool(gate_cfg.get("ambiguity_to_restrict", True)):
        return "Restrict"
    if retrieval_risk in set(gate_cfg.get("retrieval_risk_to_restrict", ["medium", "high"])):
        return "Restrict"
    if retrieval_need in ("helpful", "required"):
        return "Retrieve"
    return "No-Retrieve"


def build_plan(user_prompt: str, ir: Dict[str, Any], cfg: Dict[str, Any]) -> RetrievalPlan:
    """
    Builds retrieval plan dict aligned with your prompt contract.
    """
    rag_cfg = cfg.get("rag") or {}
    gate_cfg = cfg.get("retrieval_gate") or {}
    restrict_cfg = gate_cfg.get("restrict") or {}

    backend = str(gate_cfg.get("default_backend", rag_cfg.get("backend", "bm25")))
    top_k = int(rag_cfg.get("top_k", 8))
    denylist_terms = (
        (restrict_cfg.get("denylist_terms") or [])
        or ((rag_cfg.get("query_rewrite") or {}).get("denylist_terms") or [])
    )

    action = decide_action(ir, cfg)

    # restricted retrieval uses lower k
    if action == "Restrict":
        top_k = int(restrict_cfg.get("top_k", max(3, top_k // 2)))

    # query rewrite applied unless No-Retrieve
    if action == "No-Retrieve":
        qr = safe_query_rewrite("", [])
        query = ""
        rewrote = False
        removed_terms = []
    else:
        qr = safe_query_rewrite(user_prompt, denylist_terms=denylist_terms)
        query = qr.query
        rewrote = qr.rewrote
        removed_terms = qr.removed_terms

    constraints = {
        "domain_allowlist": restrict_cfg.get("domain_allowlist", []),
        "time_window_days": restrict_cfg.get("time_window_days", None),
        "max_snippet_chars": int(restrict_cfg.get("max_snippet_chars", 600)),
        "denylist_terms": denylist_terms,
        "query_rewrite_applied": bool(rewrote),
        "removed_terms": removed_terms,
    }

    expected = "none" if action == "No-Retrieve" else "high_level_overview"
    rationale = "Deterministic baseline gate policy + conservative rewrite."

    return RetrievalPlan(
        action=action,
        backend=backend,
        top_k=top_k,
        query=query,
        expected_evidence_type=expected,
        constraints=constraints,
        rationale=rationale,
    )

