from __future__ import annotations

from typing import Any, Dict, List


def build_source_list(evidence_kept: List[Dict[str, Any]], max_citations: int = 5) -> List[str]:
    sources: List[str] = []
    for d in evidence_kept[:max_citations]:
        meta = d.get("meta") or {}
        src = meta.get("url") or meta.get("filename") or d.get("doc_id")
        sources.append(str(src))
    return sources


def inject_citations(answer: str, evidence_kept: List[Dict[str, Any]], max_citations: int = 5) -> str:
    if not evidence_kept:
        return answer.strip()

    sources = build_source_list(evidence_kept, max_citations=max_citations)
    block = "\n\nSources:\n" + "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources)])
    return (answer.strip() + block).strip()

