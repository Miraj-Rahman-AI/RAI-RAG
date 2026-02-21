from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rai_rag.reward.composite import composite_reward, CompositeRewardConfig
from rai_rag.safety.judges import judge_all
from rai_rag.safety.policies import policy_route, refusal_template


@dataclass
class RolloutResult:
    answer: str
    scores: Dict[str, Any]   # safety/helpfulness/introspection
    reward: Dict[str, Any]   # composite reward dict


def _safe_deterministic_answer(user_prompt: str, ir: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    kept = (evidence or {}).get("kept") or []
    mode = ir.get("response_mode", "safe_high_level")
    if mode == "refuse_with_alternatives":
        pd = policy_route(ir.get("risk_category", "other"), ir.get("severity", "low"))
        return refusal_template(pd.reason, pd.safe_alternatives)
    if not kept:
        return "Safe high-level response: provide non-actionable overview and ask for clarifications if needed."
    # simple grounded bullets
    bullets = []
    for d in kept[:3]:
        sn = (d.get("snippets") or [""])[0]
        bullets.append(f"- {sn.strip()}")
    return "Using safe retrieved evidence:\n" + "\n".join(bullets)


def run_rollout(
    user_prompt: str,
    ir: Dict[str, Any],
    plan: Dict[str, Any],
    evidence: Dict[str, Any],
    cfg: Dict[str, Any],
    backbone=None,
) -> RolloutResult:
    """
    Produces an answer for a leaf state and computes reward.
    - If backbone exists, you can later replace deterministic answer with generation.
    - For now, deterministic is enough to run SI-MCTS end-to-end.
    """
    # enforce policy first
    pd = policy_route(ir.get("risk_category", "other"), ir.get("severity", "low"))
    if not pd.allow:
        answer = refusal_template(pd.reason, pd.safe_alternatives)
    else:
        # deterministic answer baseline
        answer = _safe_deterministic_answer(user_prompt, ir, evidence)

    # judge scores
    js = judge_all(answer, ir)
    H = float(js["helpfulness"]["score"])
    S = float(js["safety"]["score"])
    I = float(js["introspection"]["score"])

    # composite reward
    r_cfg = cfg.get("reward") or {}
    comp = composite_reward(
        H=H,
        S=S,
        I=I,
        cfg=CompositeRewardConfig(
            lambda_I=float(r_cfg.get("lambda_I", 0.2)),
            safety_gate=str(r_cfg.get("safety_gate", "clamp01")),
            safety_sigmoid_k=float(r_cfg.get("safety_sigmoid_k", 4.0)),
        ),
    )

    return RolloutResult(answer=answer, scores=js, reward=comp)

