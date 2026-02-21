from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class ScoreCache:
    """
    Cache for expensive judge scores and rollouts.
    """
    store: Dict[str, Any] = field(default_factory=dict)

    def get(self, key_obj: Any) -> Optional[Any]:
        k = _hash_obj(key_obj)
        return self.store.get(k)

    def set(self, key_obj: Any, value: Any) -> None:
        k = _hash_obj(key_obj)
        self.store[k] = value

