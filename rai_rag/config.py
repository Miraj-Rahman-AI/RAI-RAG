from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rai_rag.utils.io import read_yaml


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dict b into a recursively (b wins).
    """
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    """
    Small wrapper around a dict config with helpers.
    """
    data: Dict[str, Any]
    source_path: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, key: str) -> Dict[str, Any]:
        v = self.data.get(key) or {}
        if not isinstance(v, dict):
            raise TypeError(f"Config section '{key}' must be a dict, got {type(v)}")
        return v

    def resolve_path(self, *keys: str, default: Optional[str] = None) -> str:
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                if default is None:
                    raise KeyError(f"Missing config path: {'.'.join(keys)}")
                return default
            cur = cur[k]
        if not isinstance(cur, str):
            raise TypeError(f"Config path {'.'.join(keys)} must be str, got {type(cur)}")
        return cur


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = read_yaml(p)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be dict, got {type(data)}")
    return Config(data=data, source_path=str(p))


def load_and_merge(base_path: str, override_path: Optional[str] = None) -> Config:
    base = load_config(base_path)
    if not override_path:
        return base
    override = load_config(override_path)
    merged = deep_merge(base.data, override.data)
    return Config(data=merged, source_path=f"{base.source_path} + {override.source_path}")

