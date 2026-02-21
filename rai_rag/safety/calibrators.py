from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Thresholds:
    """
    Central place for decision thresholds.
    """
    safe_min_score: float = 0.0        # S >= safe_min_score considered safe
    refuse_if_below: float = -0.2      # S <= refuse_if_below => recommend refusal
    restrict_retrieve_if_below: float = 0.2  # evidence/doc score threshold for pruning or restrict mode


@dataclass
class TemperatureCalibrator:
    """
    Temperature scaling for probability-like outputs.
    This is a skeleton: you can fit T from a validation set later.
    """
    T: float = 1.0

    def calibrate_logit(self, logit: float) -> float:
        return float(logit) / float(self.T)

    def calibrate_prob(self, prob: float) -> float:
        # interpret prob as sigmoid(logit); invert then rescale
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        logit = math.log(prob / (1 - prob))
        return float(_sigmoid(self.calibrate_logit(logit)))


def safety_to_decision(S: float, th: Optional[Thresholds] = None) -> Dict[str, Any]:
    """
    Converts safety score S into a decision:
      - ok / warn / refuse recommendation
    """
    th = th or Thresholds()
    if S <= th.refuse_if_below:
        return {"decision": "refuse", "reason": "Safety score below refusal threshold.", "S": float(S)}
    if S < th.safe_min_score:
        return {"decision": "warn", "reason": "Safety score borderline.", "S": float(S)}
    return {"decision": "ok", "reason": "Safety score acceptable.", "S": float(S)}

