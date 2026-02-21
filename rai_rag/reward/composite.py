from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CompositeRewardConfig:
    """
    R(H,S,I) = F(S)*H + S + lambda_I*I

    H: helpfulness in [0,1] (or any real, but normalize upstream)
    S: safety in [-1,1] (higher safer)
    I: introspection quality in [0,1]
    F(S): safety-gating function; default: clamp(S,0,1)
    """
    lambda_I: float = 0.2
    safety_gate: str = "clamp01"  # clamp01 | sigmoid | none
    safety_sigmoid_k: float = 4.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _sigmoid(x: float) -> float:
    # stable-ish logistic
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safety_gate_value(S: float, cfg: CompositeRewardConfig) -> float:
    if cfg.safety_gate == "none":
        return 1.0
    if cfg.safety_gate == "clamp01":
        # If S in [-1,1], map to [0,1] by clamp positive part
        return _clamp(S, 0.0, 1.0)
    if cfg.safety_gate == "sigmoid":
        # Map S∈[-1,1] to [0,1] smoothly
        return float(_sigmoid(cfg.safety_sigmoid_k * S))
    raise ValueError(f"Unknown safety_gate: {cfg.safety_gate}")


def composite_reward(
    H: float,
    S: float,
    I: float,
    cfg: Optional[CompositeRewardConfig] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "R": total reward,
        "terms": {"H":..., "S":..., "I":..., "F(S)":..., "F(S)*H":..., "lambda_I*I":...}
      }
    """
    cfg = cfg or CompositeRewardConfig()
    fS = safety_gate_value(S, cfg)
    term_H = fS * float(H)
    term_S = float(S)
    term_I = float(cfg.lambda_I) * float(I)
    R = term_H + term_S + term_I
    return {
        "R": float(R),
        "terms": {
            "H": float(H),
            "S": float(S),
            "I": float(I),
            "F(S)": float(fS),
            "F(S)*H": float(term_H),
            "lambda_I*I": float(term_I),
        },
    }

