from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QueryRewriteResult:
    query: str
    rewrote: bool
    removed_terms: List[str]


def safe_query_rewrite(user_prompt: str, denylist_terms: List[str]) -> QueryRewriteResult:
    """
    Conservative query rewrite:
      - remove denylisted trigger terms
      - collapse spaces
      - if empty => safe fallback query
    """
    q = (user_prompt or "").strip()
    removed: List[str] = []
    low = q.lower()

    rewrote = False
    for t in denylist_terms or []:
        if not t:
            continue
        if t.lower() in low:
            q = re.sub(re.escape(t), "", q, flags=re.IGNORECASE)
            removed.append(t)
            rewrote = True

    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        q = "high-level overview and definitions"
        rewrote = True

    return QueryRewriteResult(query=q, rewrote=rewrote, removed_terms=removed)

