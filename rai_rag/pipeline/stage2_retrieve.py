from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RetrievalPlan:
    action: str  # Retrieve|Restrict|No-Retrieve
    backend: str
    top_k: int
    query: str
    constraints: Dict[str, Any]
    expected_evidence_type: str
    rationale: str


@dataclass
class EvidenceBundle:
    kept: List[Dict[str, Any]]
    filtered: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class Stage2Output:
    plan: Dict[str, Any]
    evidence: Dict[str, Any]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _bm25_score(query: str, bm25: Dict[str, Any], doc_idx: int, k1=1.2, b=0.75) -> float:
    N = bm25["N"]
    avgdl = bm25["avgdl"]
    df = bm25["df"]
    doc_len = bm25["doc_len"][doc_idx]
    toks = bm25["tokenized"][doc_idx]
    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    q = _tokenize(query)
    for term in q:
        if term not in df:
            continue
        n_q = df[term]
        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1e-9)
        f = tf.get(term, 0)
        denom = f + k1 * (1 - b + b * (doc_len / (avgdl + 1e-9)))
        score += idf * (f * (k1 + 1) / (denom + 1e-9))
    return float(score)


def _load_bm25_index(index_dir: Path) -> Dict[str, Any]:
    path = index_dir / "bm25.json"
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}. Run scripts/build_index.py --backend bm25")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_corpus(corpus_path: Path, max_rows: int = 500000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if i + 1 >= max_rows:
                break
    return rows


def _retrieve_bm25(query: str, bm25: Dict[str, Any], corpus: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    scores: List[Tuple[float, int]] = []
    for i in range(bm25["N"]):
        s = _bm25_score(query, bm25, i)
        if s != 0.0:
            scores.append((s, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    hits = scores[:top_k]

    out: List[Dict[str, Any]] = []
    for rank, (s, idx) in enumerate(hits, start=1):
        doc_meta = bm25["docs"][idx].get("meta", {})
        out.append(
            {
                "rank": rank,
                "score": float(s),
                "doc_id": bm25["docs"][idx]["id"],
                "meta": doc_meta,
                "text": corpus[idx]["text"],  # build_index uses same corpus ordering
            }
        )
    return out


# ---------- Gate policy (deterministic baseline) ----------

def _gate_decision(ir: Dict[str, Any], cfg: Dict[str, Any]) -> str:
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


def _safe_query_rewrite(user_prompt: str, denylist_terms: List[str]) -> Tuple[str, bool]:
    """
    Remove obvious procedural triggers. Conservative and simple.
    """
    q = user_prompt.strip()
    rewrote = False
    low = q.lower()
    for t in denylist_terms:
        if t.lower() in low:
            q = re.sub(re.escape(t), "", q, flags=re.IGNORECASE)
            rewrote = True
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        q = "high-level overview and definitions"
        rewrote = True
    return q, rewrote


# ---------- Evidence safety filter (heuristic baseline) ----------

_BAD_SIGNALS = [
    "step-by-step", "exploit", "payload", "bypass", "jailbreak", "malware",
    "how to hack", "bomb", "weapon"
]
_PII_SIGNALS = ["passport", "ssn", "credit card", "bank account", "phone number", "address"]


def _evidence_safety_score(text: str) -> Tuple[float, List[str]]:
    t = (text or "").lower()
    flags: List[str] = []
    if any(s in t for s in _BAD_SIGNALS):
        flags.append("procedural_harm")
        return -0.7, flags
    if any(s in t for s in _PII_SIGNALS):
        flags.append("pii")
        return -0.2, flags
    return 0.4, flags


def _filter_evidence(docs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> EvidenceBundle:
    ef_cfg = (cfg.get("evidence_filter") or {})
    drop_below = float(ef_cfg.get("drop_if_score_below", 0.0))
    max_snips = int(ef_cfg.get("max_snippets_per_doc", 2))
    max_chars = int(ef_cfg.get("max_snippet_chars", 240))

    kept: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []

    for d in docs:
        score, flags = _evidence_safety_score(d.get("text", ""))
        if score >= drop_below:
            snip = (d.get("text", "")[:max_chars]).strip()
            kept.append(
                {
                    "doc_id": d["doc_id"],
                    "score": float(score),
                    "reason": "Heuristic-safe evidence",
                    "snippets": [snip][:max_snips],
                    "meta": d.get("meta", {}),
                    "rank": d.get("rank"),
                    "retrieval_score": d.get("score"),
                }
            )
        else:
            filtered.append(
                {
                    "doc_id": d["doc_id"],
                    "score": float(score),
                    "reason": "Heuristic-unsafe evidence",
                    "risk_flags": flags or ["other"],
                    "meta": d.get("meta", {}),
                    "rank": d.get("rank"),
                    "retrieval_score": d.get("score"),
                }
            )

    fallback = "continue"
    min_keep = int((cfg.get("rag") or {}).get("min_keep_docs", 2))
    if len(kept) < min_keep:
        fallback = str(ef_cfg.get("if_insufficient_evidence", "safe_high_level"))
        # normalize to allowed values used in prompt contract
        if fallback == "safe_high_level":
            fallback = "no_retrieve_and_safe_high_level"
        elif fallback == "restrict_retrieval":
            fallback = "restrict_retrieval"

    summary = {"num_in": len(docs), "num_kept": len(kept), "num_filtered": len(filtered), "fallback_recommendation": fallback}
    return EvidenceBundle(kept=kept, filtered=filtered, summary=summary)


def run_stage2_retrieve(user_prompt: str, ir: Dict[str, Any], cfg: Dict[str, Any]) -> Stage2Output:
    """
    Stage 2:
      - decide retrieval action (Retrieve/Restrict/No-Retrieve)
      - if retrieve: run BM25 and evidence filter
      - output plan + evidence bundle
    """
    paths = cfg.get("paths") or {}
    corpus_path = Path(paths.get("corpus_path", "data/processed/corpus.jsonl"))
    index_dir = Path(paths.get("index_dir", "data/processed/index"))

    action = _gate_decision(ir, cfg)

    gate_cfg = cfg.get("retrieval_gate") or {}
    restrict_cfg = gate_cfg.get("restrict") or {}

    denylist_terms = (restrict_cfg.get("denylist_terms") or []) or []
    if not denylist_terms:
        # also allow rag.yaml denylist_terms if present
        denylist_terms = ((cfg.get("rag") or {}).get("query_rewrite") or {}).get("denylist_terms") or []

    query, rewrote = _safe_query_rewrite(user_prompt, denylist_terms=denylist_terms)

    backend = str(gate_cfg.get("default_backend", (cfg.get("rag") or {}).get("backend", "bm25")))
    top_k = int((cfg.get("rag") or {}).get("top_k", 8))
    max_snip = int(restrict_cfg.get("max_snippet_chars", 600))

    if action == "Restrict":
        top_k = int(restrict_cfg.get("top_k", max(3, top_k // 2)))

    plan = RetrievalPlan(
        action=action,
        backend=backend,
        top_k=top_k,
        query=query if action != "No-Retrieve" else "",
        constraints={
            "domain_allowlist": restrict_cfg.get("domain_allowlist", []),
            "time_window_days": restrict_cfg.get("time_window_days", None),
            "max_snippet_chars": max_snip,
            "denylist_terms": denylist_terms,
            "query_rewrite_applied": bool(rewrote),
        },
        expected_evidence_type="high_level_overview" if action != "No-Retrieve" else "none",
        rationale="Deterministic gate + safe query rewrite baseline.",
    )

    # No retrieve → return empty evidence
    if action == "No-Retrieve":
        evidence = EvidenceBundle(kept=[], filtered=[], summary={"num_in": 0, "num_kept": 0, "num_filtered": 0, "fallback_recommendation": "no_retrieve_and_safe_high_level"})
        return Stage2Output(plan=plan.__dict__, evidence=evidence.__dict__)

    # Retrieve using BM25 fallback
    bm25 = _load_bm25_index(index_dir)
    corpus = _load_corpus(corpus_path)
    docs = _retrieve_bm25(query=query, bm25=bm25, corpus=corpus, top_k=top_k)

    evidence = _filter_evidence(docs, cfg)

    return Stage2Output(plan=plan.__dict__, evidence=evidence.__dict__)

