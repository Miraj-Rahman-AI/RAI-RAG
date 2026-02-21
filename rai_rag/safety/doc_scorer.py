from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Keep aligned with rag/evidence_filter.py so you can swap later
_BAD_SIGNALS = [
    "step-by-step", "exploit", "payload", "bypass", "jailbreak", "malware",
    "how to hack", "bomb", "weapon"
]
_PII_SIGNALS = ["passport", "ssn", "credit card", "bank account", "phone number", "address"]


@dataclass
class DocScore:
    score: float  # in [-1, 1]
    flags: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {"score": float(self.score), "flags": self.flags, "rationale": self.rationale}


def score_evidence_doc(doc_text: str, user_prompt: str, ir: Dict[str, Any]) -> DocScore:
    """
    Evidence safety function:
      S_e(d | x, y_IR) -> [-1, 1]
    Baseline heuristic with IR-aware tightening.
    """
    t = (doc_text or "").lower()
    flags: List[str] = []

    if any(s in t for s in _BAD_SIGNALS):
        flags.append("procedural_harm")
        return DocScore(score=-0.7, flags=flags, rationale="Procedural/harmful instruction patterns detected.")

    if any(s in t for s in _PII_SIGNALS):
        flags.append("pii")
        # PII in evidence is risky; if IR already says privacy_pii, be stricter
        rc = (ir or {}).get("risk_category", "other")
        if rc == "privacy_pii":
            return DocScore(score=-0.4, flags=flags, rationale="PII content detected; stricter under privacy context.")
        return DocScore(score=-0.2, flags=flags, rationale="PII content detected; filtered conservatively.")

    # Tighten for high severity
    sev = (ir or {}).get("severity", "low")
    if sev == "high":
        return DocScore(score=0.1, flags=["high_severity_context"], rationale="High severity context; only weakly allow neutral docs.")

    return DocScore(score=0.4, flags=[], rationale="No risky signals detected (heuristic).")

