from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Stage3Output:
    selection: Dict[str, Any]


def run_stage3_search(
    user_prompt: str,
    ir: Dict[str, Any],
    plan: Dict[str, Any],
    evidence: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Stage3Output:
    """
    Stage 3: Search over reasoning/retrieval trajectories.
    Placeholder implementation: identity selection.
    Later you can plug SI-MCTS (rai_rag/search/simcts.py) or best-of-n scoring here.
    """
    return Stage3Output(
        selection={
            "method": (cfg.get("search") or {}).get("method", "none"),
            "chosen_plan": plan,
            "chosen_evidence": evidence,
        }
    )

