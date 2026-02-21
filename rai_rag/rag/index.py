from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class BM25Index:
    """
    Loads BM25 fallback index created by scripts/build_index.py --backend bm25.
    Expected JSON keys:
      - N, avgdl, df, docs, doc_len, tokenized
    """
    obj: Dict[str, Any]

    @classmethod
    def load(cls, index_dir: str) -> "BM25Index":
        p = Path(index_dir) / "bm25.json"
        if not p.exists():
            raise FileNotFoundError(f"BM25 index not found: {p}")
        return cls(obj=json.loads(p.read_text(encoding="utf-8")))

    @property
    def N(self) -> int:
        return int(self.obj["N"])


def load_index(index_dir: str, backend: str = "bm25"):
    backend = (backend or "bm25").lower()
    if backend == "bm25":
        return BM25Index.load(index_dir)
    raise NotImplementedError(f"Index backend not implemented yet: {backend}. Use bm25 for now.")

