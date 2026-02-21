from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rai_rag.search.cache import ScoreCache
from rai_rag.search.node import Node, SearchState
from rai_rag.search.rollout import run_rollout

from rai_rag.rag.gate import RetrievalPlan


@dataclass
class SIMCTSConfig:
    iters: int = 30
    c_puct: float = 1.2
    max_depth: int = 2
    safety_prune_threshold: float = -0.2  # prune if safety score below this in rollout
    expand_actions: Tuple[str, ...] = ("Retrieve", "Restrict", "No-Retrieve")


def _uct(parent: Node, child: Node, c_puct: float) -> float:
    # standard UCT
    if child.N == 0:
        return float("inf")
    return child.Q + c_puct * math.sqrt(math.log(parent.N + 1) / (child.N + 1e-9))


def _plan_with_action(plan: Dict[str, Any], action: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(plan or {})
    p["action"] = action
    # adjust top_k under restriction
    if action == "Restrict":
        gate_cfg = (cfg.get("retrieval_gate") or {})
        restrict_cfg = gate_cfg.get("restrict") or {}
        p["top_k"] = int(restrict_cfg.get("top_k", max(3, int(p.get("top_k", 8)) // 2)))
    if action == "No-Retrieve":
        p["query"] = ""
    return p


def _expand(node: Node, cfg: Dict[str, Any]) -> List[Node]:
    children = []
    for a in cfg.get("search", {}).get("expand_actions", None) or ("Retrieve", "Restrict", "No-Retrieve"):
        if a in node.children:
            continue
        child_state = SearchState(
            user_prompt=node.state.user_prompt,
            ir=node.state.ir,
            plan=_plan_with_action(node.state.plan, a, cfg),
            evidence=node.state.evidence,  # evidence may be recomputed by pipeline stage2; here we keep placeholder
            draft_answer="",
            meta={"expanded_from": node.action},
        )
        children.append(node.add_child(a, child_state))
    return children


def simcts_search(
    root_state: SearchState,
    cfg: Dict[str, Any],
    backbone=None,
    cache: Optional[ScoreCache] = None,
) -> Dict[str, Any]:
    """
    SI-MCTS over retrieval-action trajectories.
    NOTE: This version focuses on selecting the best retrieval action given fixed evidence bundle.
    For full correctness, you would re-run retrieval+filter after choosing action. We keep it light
    so it runs now.
    """
    scfg_dict = (cfg.get("search") or {}).get("simcts", {})
    scfg = SIMCTSConfig(
        iters=int(scfg_dict.get("iters", 30)),
        c_puct=float(scfg_dict.get("c_puct", 1.2)),
        max_depth=int(scfg_dict.get("max_depth", 2)),
        safety_prune_threshold=float(scfg_dict.get("safety_prune_threshold", -0.2)),
        expand_actions=tuple(scfg_dict.get("expand_actions", ("Retrieve", "Restrict", "No-Retrieve"))),
    )

    cache = cache or ScoreCache()
    root = Node(state=root_state)

    for _ in range(scfg.iters):
        node = root
        depth = 0

        # SELECTION
        while not node.is_leaf() and depth < scfg.max_depth:
            # pick best UCT
            best_a, best_child, best_val = None, None, -1e18
            for a, ch in node.children.items():
                u = _uct(node, ch, scfg.c_puct)
                if u > best_val:
                    best_val = u
                    best_a, best_child = a, ch
            if best_child is None:
                break
            node = best_child
            depth += 1

        # EXPANSION
        if node.is_leaf() and depth < scfg.max_depth:
            _expand(node, cfg)

            # pick one newly expanded child for rollout
            if node.children:
                node = next(iter(node.children.values()))
                depth += 1

        # ROLLOUT / EVALUATION (cached)
        key = {
            "prompt": node.state.user_prompt,
            "ir": node.state.ir,
            "plan": node.state.plan,
            "evidence_summary": (node.state.evidence or {}).get("summary"),
        }
        cached = cache.get(key)
        if cached is None:
            rr = run_rollout(
                user_prompt=node.state.user_prompt,
                ir=node.state.ir,
                plan=node.state.plan,
                evidence=node.state.evidence,
                cfg=cfg,
                backbone=backbone,
            )
            cached = {
                "answer": rr.answer,
                "scores": rr.scores,
                "reward": rr.reward,
                "value": float(rr.reward["R"]),
            }
            cache.set(key, cached)

        # SAFETY-DOMINANT PRUNING
        S = float(cached["scores"]["safety"]["score"])
        if S <= scfg.safety_prune_threshold:
            value = float(cached["value"]) - 1.0  # penalty for unsafe leaf
        else:
            value = float(cached["value"])

        # BACKPROP
        cur = node
        while cur is not None:
            cur.update(value)
            cur = cur.parent

    # pick best child from root (by Q then N)
    best = None
    for a, ch in root.children.items():
        if best is None:
            best = (a, ch)
        else:
            # prefer higher Q, tie-break on visits
            if ch.Q > best[1].Q or (abs(ch.Q - best[1].Q) < 1e-9 and ch.N > best[1].N):
                best = (a, ch)

    if best is None:
        # no expansion happened; fallback
        return {
            "method": "simcts",
            "chosen_action": root_state.plan.get("action"),
            "chosen_plan": root_state.plan,
            "stats": {"root_N": root.N},
        }

    a, ch = best
    return {
        "method": "simcts",
        "chosen_action": a,
        "chosen_plan": ch.state.plan,
        "root_stats": {
            "N": root.N,
            "children": {k: {"N": v.N, "Q": v.Q} for k, v in root.children.items()},
        },
    }

