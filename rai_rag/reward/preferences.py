from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class PreferenceExample:
    """
    Pairwise preference:
      winner vs loser for the same prompt/context.
    """
    prompt_id: str
    prompt: str
    winner: Dict[str, Any]
    loser: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "winner": self.winner,
            "loser": self.loser,
            "meta": self.meta,
        }


def _read_jsonl(path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and (i + 1) >= max_rows:
                break
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_preferences_from_rollouts(
    rollouts_jsonl: str,
    out_jsonl: str,
    score_key: str = "reward.R",
    group_key: str = "prompt_id",
    max_pairs_per_prompt: int = 2,
    min_score_gap: float = 0.05,
    max_rows: int = 0,
) -> Dict[str, Any]:
    """
    Input rollout format (one per line) — flexible, but expected minimal fields:
      {
        "prompt_id": "...",
        "prompt": "...",
        "answer": "...",
        "action": "...",
        "reward": {"R": ..., "terms": {...}},
        ...
      }

    score_key uses dot path like "reward.R" or "scores.composite".
    """
    in_path = Path(rollouts_jsonl)
    rows = _read_jsonl(in_path, max_rows=max_rows)
    if not rows:
        raise ValueError(f"No rollouts found in {in_path}")

    def get_by_path(obj: Dict[str, Any], path: str) -> Optional[float]:
        cur: Any = obj
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        try:
            return float(cur)
        except Exception:
            return None

    # group by prompt_id
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pid = str(r.get(group_key, "unknown"))
        groups.setdefault(pid, []).append(r)

    prefs: List[PreferenceExample] = []
    skipped = 0

    for pid, items in groups.items():
        # sort by score descending
        scored = []
        for it in items:
            s = get_by_path(it, score_key)
            if s is None:
                continue
            scored.append((s, it))
        scored.sort(reverse=True, key=lambda x: x[0])

        if len(scored) < 2:
            skipped += 1
            continue

        # pair top with bottom / mid for diversity
        pairs = []
        best = scored[0]
        worst = scored[-1]
        pairs.append((best, worst))
        if len(scored) >= 3:
            mid = scored[len(scored)//2]
            pairs.append((best, mid))

        taken = 0
        for (s_w, w), (s_l, l) in pairs:
            if taken >= max_pairs_per_prompt:
                break
            if (s_w - s_l) < min_score_gap:
                continue
            prefs.append(
                PreferenceExample(
                    prompt_id=pid,
                    prompt=str(w.get("prompt") or l.get("prompt") or ""),
                    winner={
                        "answer": w.get("answer", ""),
                        "action": w.get("action"),
                        "ir": w.get("ir"),
                        "plan": w.get("plan"),
                        "evidence": w.get("evidence"),
                        "score": s_w,
                    },
                    loser={
                        "answer": l.get("answer", ""),
                        "action": l.get("action"),
                        "ir": l.get("ir"),
                        "plan": l.get("plan"),
                        "evidence": l.get("evidence"),
                        "score": s_l,
                    },
                    meta={
                        "score_key": score_key,
                        "score_gap": float(s_w - s_l),
                    },
                )
            )
            taken += 1

    out_path = Path(out_jsonl)
    n = _write_jsonl(out_path, [p.to_dict() for p in prefs])

    return {
        "status": "ok",
        "in": str(in_path),
        "out": str(out_path),
        "num_rollouts": len(rows),
        "num_prompts": len(groups),
        "num_pairs": n,
        "skipped_prompts": skipped,
    }

