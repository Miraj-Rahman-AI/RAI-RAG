from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Stage4Output:
    answer: str
    safety: Dict[str, Any]


def _format_citations(evidence_kept: List[Dict[str, Any]], max_citations: int = 5) -> str:
    if not evidence_kept:
        return ""
    items = []
    for i, d in enumerate(evidence_kept[:max_citations], start=1):
        src = d.get("meta", {}).get("filename") or d.get("meta", {}).get("url") or d.get("doc_id")
        items.append(f"[{i}] {src}")
    return "\n\nSources:\n" + "\n".join(items)


def _fallback_safe_answer(user_prompt: str, ir: Dict[str, Any], evidence: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    kept = (evidence or {}).get("kept") or []
    mode = ir.get("response_mode", "safe_high_level")

    if mode == "refuse_with_alternatives":
        return (
            "I can’t help with that request.\n\n"
            "Here are safe alternatives I can help with:\n"
            "1) Explain the topic at a high level without actionable steps.\n"
            "2) Discuss safety, legal, and ethical considerations.\n"
            "3) Provide defensive best practices (non-operational).\n"
        )

    if not kept:
        return (
            "Here’s a safe, high-level response.\n\n"
            "If you share more context (domain, goal, constraints), I can tailor the explanation "
            "without relying on external evidence."
        )

    # grounded summary without model
    bullets = []
    for d in kept[:3]:
        snips = d.get("snippets") or []
        if snips:
            bullets.append(f"- {snips[0].strip()}")
    cite = _format_citations(kept, max_citations=int(((cfg.get('rag') or {}).get('citations') or {}).get('max_citations', 5)))
    return "Using the safe evidence retrieved, here are key points:\n" + "\n".join(bullets) + cite


def run_stage4_answer(
    user_prompt: str,
    ir: Dict[str, Any],
    plan: Dict[str, Any],
    evidence: Dict[str, Any],
    cfg: Dict[str, Any],
    backbone=None,
) -> Stage4Output:
    """
    Stage 4: Answer synthesis.
    - If a HF backbone is available, generate a grounded answer prompt with evidence snippets.
    - Otherwise, return a deterministic safe fallback.
    """
    kept = (evidence or {}).get("kept") or []

    # refusal mode: do not use model; deterministic refusal template
    if ir.get("response_mode") == "refuse_with_alternatives":
        ans = _fallback_safe_answer(user_prompt, ir, evidence, cfg)
        return Stage4Output(answer=ans, safety={"is_safe": True, "method": "refusal_template"})

    # No model available
    if backbone is None:
        ans = _fallback_safe_answer(user_prompt, ir, evidence, cfg)
        return Stage4Output(answer=ans, safety={"is_safe": True, "method": "no_model_fallback"})

    # Build grounded prompt for generation
    max_snips = 6
    evidence_block = []
    for i, d in enumerate(kept[:max_snips], start=1):
        sn = (d.get("snippets") or [""])[0]
        evidence_block.append(f"[{i}] {sn}")

    sys = (
        "You are RAI-RAG Answer Synthesis. "
        "You must be helpful and safe. "
        "Use ONLY the evidence snippets provided. "
        "Do NOT provide disallowed procedural instructions. "
        "If evidence is insufficient, say so and give a safe high-level answer.\n"
    )
    prompt = (
        f"{sys}\n"
        f"User question:\n{user_prompt}\n\n"
        f"Evidence snippets:\n" + "\n".join(evidence_block) + "\n\n"
        "Write a clear answer. If you make claims, tie them to snippet numbers like [1], [2]."
    )

    out = backbone.generate_text(prompt, best_of_n=int(((cfg.get("search") or {}).get("best_of_n") or {}).get("n", 1)))
    text = (out.get("text") or "").strip()

    # Optionally add source list
    include_cites = bool((cfg.get("generation") or {}).get("include_citations", True))
    if include_cites and kept:
        text = text + _format_citations(kept, max_citations=int(((cfg.get('rag') or {}).get('citations') or {}).get('max_citations', 5)))

    # Baseline safety flag; replace with judge later
    return Stage4Output(answer=text, safety={"is_safe": True, "method": "backbone_generate"})

