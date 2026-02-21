from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .corpus import CorpusStore, Doc
from .index import BM25Index


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


@dataclass
class RetrievedDoc:
    doc: Doc
    rank: int
    retrieval_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc.id,
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "meta": self.doc.meta,
            "text": self.doc.text,
        }


class BM25Retriever:
    def __init__(self, corpus: CorpusStore, index: BM25Index):
        self.corpus = corpus
        self.index = index

    def retrieve(self, query: str, top_k: int = 8) -> List[RetrievedDoc]:
        bm25 = self.index.obj
        scores: List[Tuple[float, int]] = []
        for i in range(bm25["N"]):
            s = _bm25_score(query, bm25, i)
            if s != 0.0:
                scores.append((s, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        hits = scores[:top_k]

        out: List[RetrievedDoc] = []
        for rank, (s, idx) in enumerate(hits, start=1):
            doc = self.corpus.get_by_index(idx)
            out.append(RetrievedDoc(doc=doc, rank=rank, retrieval_score=float(s)))
        return out

