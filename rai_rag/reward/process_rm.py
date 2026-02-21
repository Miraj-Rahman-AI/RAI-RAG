from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _is_refusal(text: str) -> float:
    t = (text or "").lower()
    return 1.0 if any(
        s in t for s in ["i can't help", "i cannot help", "i can’t help", "cannot assist", "must refuse"]
    ) else 0.0


@dataclass
class PRMConfig:
    lr: float = 0.05
    epochs: int = 3
    l2: float = 1e-4
    feature_version: str = "v1"


class FeatureExtractor:
    """
    Converts a candidate trajectory dict into a fixed feature vector.
    Candidate should include:
      - answer
      - action
      - evidence: {kept:[], filtered:[]}
    """

    def __init__(self, version: str = "v1"):
        self.version = version

    def features(self, cand: Dict[str, Any]) -> List[float]:
        ans = cand.get("answer", "")
        action = cand.get("action") or (cand.get("plan") or {}).get("action") or "Unknown"
        ev = cand.get("evidence") or {}
        kept = ev.get("kept") or []
        filtered = ev.get("filtered") or []

        # Basic stats
        n_kept = float(len(kept))
        n_filt = float(len(filtered))
        is_ref = _is_refusal(ans)

        # Action one-hot-ish
        a_retrieve = 1.0 if action == "Retrieve" else 0.0
        a_restrict = 1.0 if action == "Restrict" else 0.0
        a_noretr = 1.0 if action == "No-Retrieve" else 0.0

        # Length feature (weak)
        ans_len = float(min(len(ans), 4000)) / 4000.0

        # Evidence “risk” feature: fraction filtered
        frac_filtered = (n_filt / (n_kept + n_filt + 1e-9))

        # Feature vector (v1)
        return [
            1.0,           # bias
            a_retrieve,
            a_restrict,
            a_noretr,
            n_kept,
            n_filt,
            frac_filtered,
            is_ref,
            ans_len,
        ]


class ProcessRewardModel:
    """
    Bradley–Terry pairwise preference model:
      P(wins) = sigmoid(w · (phi(winner) - phi(loser)))
    """

    def __init__(self, cfg: Optional[PRMConfig] = None):
        self.cfg = cfg or PRMConfig()
        self.fe = FeatureExtractor(version=self.cfg.feature_version)
        self.w: List[float] = []  # learned weights

    def _ensure_init(self, d: int):
        if not self.w:
            self.w = [0.0] * d

    def score(self, cand: Dict[str, Any]) -> float:
        phi = self.fe.features(cand)
        self._ensure_init(len(phi))
        return float(_dot(self.w, phi))

    def fit(self, prefs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        prefs JSON dict format:
          {"prompt":..., "winner":{...}, "loser":{...}, ...}
        """
        if not prefs:
            raise ValueError("No preference pairs provided to fit().")

        # init weights
        d = len(self.fe.features(prefs[0]["winner"]))
        self._ensure_init(d)

        lr = float(self.cfg.lr)
        l2 = float(self.cfg.l2)
        epochs = int(self.cfg.epochs)

        losses = []
        for ep in range(epochs):
            total = 0.0
            for ex in prefs:
                w_c = ex["winner"]
                l_c = ex["loser"]
                phi_w = self.fe.features(w_c)
                phi_l = self.fe.features(l_c)
                diff = [a - b for a, b in zip(phi_w, phi_l)]
                z = _dot(self.w, diff)
                p = _sigmoid(z)

                # negative log-likelihood for winner
                loss = -math.log(max(1e-9, p))
                total += loss

                # gradient: (p - 1) * diff + l2*w
                g_scale = (p - 1.0)
                for i in range(d):
                    grad = g_scale * diff[i] + l2 * self.w[i]
                    self.w[i] -= lr * grad

            avg = total / max(1, len(prefs))
            losses.append(avg)

        return {"status": "ok", "epochs": epochs, "losses": losses, "dim": d}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj = {"cfg": self.cfg.__dict__, "w": self.w}
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ProcessRewardModel":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = PRMConfig(**obj.get("cfg", {}))
        m = cls(cfg=cfg)
        m.w = list(obj.get("w", []))
        return m

