from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "meta": self.meta}


class CorpusStore:
    """
    Loads and serves documents from a JSONL corpus.
    Each line is expected to be:
      {"id": "...", "text": "...", "meta": {...}}
    """

    def __init__(self, corpus_path: str):
        self.path = Path(corpus_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.path}")
        self._docs: List[Doc] = []

    def load(self, max_rows: int = 500000) -> "CorpusStore":
        self._docs.clear()
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self._docs.append(Doc(id=r["id"], text=r["text"], meta=r.get("meta", {})))
                if i + 1 >= max_rows:
                    break
        return self

    def __len__(self) -> int:
        return len(self._docs)

    def get_by_index(self, idx: int) -> Doc:
        return self._docs[idx]

    def iter_docs(self) -> Iterator[Doc]:
        yield from self._docs

