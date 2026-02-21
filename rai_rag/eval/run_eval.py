from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attack_suites import EvalExample, load_all_suites
from .metrics import compute_all_metrics
from .reports import write_all_reports


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    # Lightweight YAML loader (no dependency if you don't want).
    # If PyYAML exists, use it; else require JSON.
    try:
        import yaml  # type: ignore
    except Exception:
        # fallback: allow JSON configs
        return json.loads(path.read_text(encoding="utf-8"))

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _try_make_pipeline(config_path: str):
    """
    Try to instantiate RAIRAGPipeline.
    If it doesn't exist yet, return None and evaluation will run in stub mode.
    """
    try:
        from rai_rag.pipeline.rai_rag import RAIRAGPipeline  # type: ignore
        return RAIRAGPipeline.from_config(config_path)
    except Exception:
        return None


def _stub_run(prompt: str) -> Dict[str, Any]:
    """
    Safe stub: always No-Retrieve, safe high-level response.
    This ensures eval infra works before pipeline is ready.
    """
    return {
        "action": "No-Retrieve",
        "ir": {
            "risk_category": "benign_info",
            "severity": "low",
            "retrieval_need": "none",
            "retrieval_risk": "low",
            "response_mode": "safe_high_level",
        },
        "plan": {"action": "No-Retrieve", "query": "", "constraints": {}},
        "evidence": {"kept": [], "filtered": []},
        "answer": "I can provide a safe, high-level explanation, but the full pipeline is not enabled yet.",
        "safety": {"is_safe": True, "method": "stub"},
    }


def run_all(
    config_path: str,
    bench_dir: str,
    out_dir: str,
    max_items: int = 0,
    attack_variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint used by scripts/eval_all.py.
    Writes:
      - results.jsonl
      - metrics.json
      - summary.md
    """
    cfg_path = Path(config_path)
    bench_path = Path(bench_dir)
    out_path = Path(out_dir)

    cfg = _load_yaml_config(cfg_path)

    suites = (cfg.get("eval") or {}).get("suites") or ["benign", "jailbreak", "retrieval_attack"]
    effective_max = max_items or (cfg.get("eval") or {}).get("max_items") or 0

    examples = load_all_suites(
        bench_dir=bench_path,
        suites=suites,
        max_items=effective_max,
        add_attack_variants=attack_variants,
    )

    pipe = _try_make_pipeline(config_path)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        if effective_max and i >= effective_max:
            break

        if pipe is None:
            out = _stub_run(ex.prompt)
        else:
            # Expected contract from your pipeline:
            # dict with keys: action, ir, plan, evidence, answer, safety(optional)
            out = pipe.run(ex.prompt)

        results.append(
            {
                "id": ex.id,
                "suite": ex.suite,
                "prompt": ex.prompt,
                "meta": ex.meta,
                "action": out.get("action") or out.get("plan", {}).get("action"),
                "ir": out.get("ir"),
                "plan": out.get("plan"),
                "evidence": out.get("evidence"),
                "answer": out.get("answer", ""),
                "safety": out.get("safety", {}),
            }
        )

    metrics = compute_all_metrics(results)

    artifacts = write_all_reports(out_path, results, metrics)
    dt = time.time() - t0

    return {
        "config": config_path,
        "bench_dir": bench_dir,
        "out_dir": out_dir,
        "num_examples": len(results),
        "elapsed_sec": dt,
        "artifacts": artifacts,
        "pipeline_enabled": pipe is not None,
    }

