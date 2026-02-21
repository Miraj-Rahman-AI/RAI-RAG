from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def stable_json_dumps(obj: Any) -> str:
    """
    JSON dumps with stable ordering for hashing/caching.
    Falls back to str() for non-serializable objects.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_str(stable_json_dumps(obj))


def short_hash(obj: Any, n: int = 12) -> str:
    return sha256_obj(obj)[: int(n)]

