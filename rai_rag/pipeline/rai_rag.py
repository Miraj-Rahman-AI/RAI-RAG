from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rai_rag.models.backbone import build_backbone_from_dict

from .stage1_introspect import run_stage1_introspect
from .stage2_retrieve import run_stage2_retrieve
from .stage3_search import run_stage3_search
from .stage4_answer import run_stage4_answer


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # allow JSON fallback
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RAIRAGPipeline:
    cfg: Dict[str, Any]
    backbone: Any = None

    @classmethod
    def from_config(cls, config_path: str, enable_model: bool = True) -> "RAIRAGPipeline":
        cfg = _load_yaml(Path(config_path))

        backbone = None
        if enable_model:
            model_cfg = cfg.get("model") or {}
            # NOTE: HF model loading can be heavy; allow disabling via enable_model=False
            if model_cfg.get("provider", "hf") == "hf" and model_cfg.get("name_or_path"):
                backbone = build_backbone_from_dict(model_cfg)

        return cls(cfg=cfg, backbone=backbone)

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """
        End-to-end RAI-RAG run.
        Returns a structured dict used by eval and scripts.
        """
        # Stage 1
        s1 = run_stage1_introspect(user_prompt, self.cfg)

        # Stage 2
        s2 = run_stage2_retrieve(user_prompt, s1.ir, self.cfg)

        # Stage 3 (search/selection)
        s3 = run_stage3_search(user_prompt, s1.ir, s2.plan, s2.evidence, self.cfg)

        # Stage 4 (answer)
        chosen_plan = (s3.selection or {}).get("chosen_plan") or s2.plan
        chosen_evidence = (s3.selection or {}).get("chosen_evidence") or s2.evidence
        s4 = run_stage4_answer(user_prompt, s1.ir, chosen_plan, chosen_evidence, self.cfg, backbone=self.backbone)

        action = chosen_plan.get("action") if isinstance(chosen_plan, dict) else None

        return {
            "action": action,
            "ir": s1.ir,
            "plan": chosen_plan,
            "evidence": chosen_evidence,
            "answer": s4.answer,
            "safety": s4.safety,
            "audit": {
                "introspection_trace": s1.trace_text if (self.cfg.get("logging") or {}).get("save_traces", True) else None,
                "selection": s3.selection,
            },
        }

