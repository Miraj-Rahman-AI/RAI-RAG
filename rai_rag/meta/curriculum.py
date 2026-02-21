from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CurriculumState:
    step_idx: int
    difficulty: float  # 0..1


class ProgressiveHardeningCurriculum:
    """
    Simple curriculum scheduler:
      difficulty increases linearly from start_difficulty -> end_difficulty over `steps`.
    You can use difficulty to:
      - increase proportion of retrieval_driven_jailbreak
      - increase ambiguity / adversarial decoration rate
      - tighten safety thresholds
    """

    def __init__(self, start_difficulty: float, end_difficulty: float, steps: int):
        self.start = float(start_difficulty)
        self.end = float(end_difficulty)
        self.steps = max(1, int(steps))

    def state(self, global_iter: int) -> CurriculumState:
        # map iteration to stage index
        idx = min(self.steps - 1, global_iter % self.steps)
        alpha = idx / max(1, self.steps - 1)
        diff = self.start + alpha * (self.end - self.start)
        return CurriculumState(step_idx=idx, difficulty=float(diff))

    def adjust_family_weights(self, base_weights: Dict[str, float], difficulty: float) -> Dict[str, float]:
        """
        Example policy:
          - as difficulty increases, emphasize retrieval-driven jailbreak family.
        """
        w = dict(base_weights)
        if "retrieval_driven_jailbreak" in w:
            w["retrieval_driven_jailbreak"] = float(w["retrieval_driven_jailbreak"]) * (1.0 + 2.0 * difficulty)
        return w

