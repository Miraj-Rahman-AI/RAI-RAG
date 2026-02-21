from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .task_families import TaskFamily, get_default_task_families


@dataclass
class MetaTask:
    """
    One sampled meta-training task instance.
    """
    task_id: str
    family: str
    prompt: str
    base_prompt: str
    meta: Dict[str, Any]


def _load_jsonl(path: Path, max_items: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_items and (i + 1) >= max_items:
                break
    return rows


class TaskSampler:
    """
    Samples tasks from DIR dataset and decorates them using task families.
    """

    def __init__(
        self,
        dir_train_path: str,
        seed: int = 7,
        families: Optional[Dict[str, TaskFamily]] = None,
        max_items: int = 0,
    ):
        self.rng = random.Random(seed)
        self.dir_path = Path(dir_train_path)
        if not self.dir_path.exists():
            raise FileNotFoundError(f"DIR train file not found: {self.dir_path}")

        self.rows = _load_jsonl(self.dir_path, max_items=max_items)
        if not self.rows:
            raise ValueError(f"DIR dataset is empty: {self.dir_path}")

        self.families = families or get_default_task_families()

    def _pick_base_prompt(self) -> Tuple[str, Dict[str, Any]]:
        r = self.rng.choice(self.rows)
        prompt = r.get("prompt") or r.get("text") or ""
        meta = r.get("meta") or {}
        return str(prompt), dict(meta)

    def _decorate(self, family: TaskFamily, base_prompt: str) -> str:
        tmpl = self.rng.choice(family.templates)
        return tmpl.format(x=base_prompt)

    def sample(self, family_name: str) -> MetaTask:
        if family_name not in self.families:
            raise ValueError(f"Unknown task family: {family_name}")
        fam = self.families[family_name]

        base_prompt, base_meta = self._pick_base_prompt()
        decorated = self._decorate(fam, base_prompt)

        tid = f"{family_name}::{self.rng.getrandbits(64):016x}"
        return MetaTask(
            task_id=tid,
            family=family_name,
            prompt=decorated,
            base_prompt=base_prompt,
            meta={
                **base_meta,
                "risk_hint": fam.risk_hint,
                "family_tags": fam.tags or [],
            },
        )

    def batch(self, family_weights: Dict[str, float], batch_size: int) -> List[MetaTask]:
        families = list(family_weights.keys())
        weights = [float(family_weights[f]) for f in families]
        out: List[MetaTask] = []
        for _ in range(batch_size):
            fam = self.rng.choices(families, weights=weights, k=1)[0]
            out.append(self.sample(fam))
        return out

