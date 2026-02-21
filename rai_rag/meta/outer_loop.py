from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sampler import TaskSampler
from .task_families import get_default_task_families
from .curriculum import ProgressiveHardeningCurriculum


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # allow JSON fallback
        return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_make_pipeline(config_base_path: Optional[str] = None):
    """
    Optional: if you have RAIRAGPipeline implemented, we can run inner-loop rollouts.
    Otherwise, we run a dry meta-training stub (structure only).
    """
    try:
        from rai_rag.pipeline.rai_rag import RAIRAGPipeline  # type: ignore
        if config_base_path:
            return RAIRAGPipeline.from_config(config_base_path)
        return RAIRAGPipeline.from_config("configs/base.yaml")
    except Exception:
        return None


def meta_train(config_path: str, out_dir: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Dual-loop meta-training scaffold.
    - Reads configs/training_meta.yaml
    - Samples tasks from DIR dataset by families
    - Applies curriculum
    - Optionally runs pipeline for rollouts if available
    - Writes checkpoints and metrics

    This is a *structure-correct* implementation; replace the training logic with your
    real θ_r adaptation (LoRA/prefix adapters, RM-guided updates, etc.).
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Meta config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    meta_cfg = cfg.get("meta_training", cfg)  # allow top-level or nested
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seed = int(meta_cfg.get("seed", 7))
    data_cfg = meta_cfg.get("data", {})
    dir_train_path = str(data_cfg.get("dir_train_path", "data/dir/dir_train.jsonl"))

    # Family weights from config
    families_cfg = (meta_cfg.get("task_families", {}) or {}).get("families", [])
    if not families_cfg:
        # fallback to defaults with equal weights
        family_weights = {k: 1.0 for k in get_default_task_families().keys()}
    else:
        family_weights = {f["name"]: float(f.get("weight", 1.0)) for f in families_cfg}

    inner_cfg = meta_cfg.get("inner_loop", {})
    outer_cfg = meta_cfg.get("outer_loop", {})
    curriculum_cfg = meta_cfg.get("curriculum", {})

    inner_steps = int(inner_cfg.get("steps", 3))
    inner_bs = int(inner_cfg.get("batch_size", 8))

    iterations = int(outer_cfg.get("iterations", 2000))
    eval_every = int(outer_cfg.get("eval_every", 100))
    save_every = int(outer_cfg.get("save_every", 200))

    # Curriculum
    curriculum_enabled = bool(curriculum_cfg.get("enabled", True))
    curriculum = ProgressiveHardeningCurriculum(
        start_difficulty=float(curriculum_cfg.get("start_difficulty", 0.2)),
        end_difficulty=float(curriculum_cfg.get("end_difficulty", 1.0)),
        steps=int(curriculum_cfg.get("steps", 5)),
    )

    families = get_default_task_families()
    sampler = TaskSampler(dir_train_path=dir_train_path, seed=seed, families=families)

    pipe = None if dry_run else _try_make_pipeline()
    run_meta = {
        "config_path": str(cfg_path),
        "out_dir": str(out_root),
        "seed": seed,
        "dir_train_path": dir_train_path,
        "family_weights": family_weights,
        "inner_steps": inner_steps,
        "inner_batch_size": inner_bs,
        "outer_iterations": iterations,
        "pipeline_enabled": pipe is not None,
        "dry_run": dry_run,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(out_root / "meta_run.json", run_meta)

    metrics_log: List[Dict[str, Any]] = []

    # ----- MAIN OUTER LOOP -----
    for it in range(1, iterations + 1):
        # curriculum state
        if curriculum_enabled:
            cstate = curriculum.state(it)
            weights = curriculum.adjust_family_weights(family_weights, cstate.difficulty)
        else:
            cstate = None
            weights = family_weights

        # sample a meta-batch
        batch = sampler.batch(weights, batch_size=inner_bs)

        # ---- INNER LOOP (placeholder) ----
        # Replace with real adaptation of θ_r on batch tasks.
        # For now: run pipeline (if available) to get action statistics.
        actions = {"Retrieve": 0, "Restrict": 0, "No-Retrieve": 0, "Unknown": 0}
        safe_flags = []

        if pipe is None:
            # stub: assume safe but conservative (No-Retrieve)
            for _ in batch:
                actions["No-Retrieve"] += 1
                safe_flags.append(True)
        else:
            for t in batch:
                out = pipe.run(t.prompt)
                a = out.get("action") or out.get("plan", {}).get("action") or "Unknown"
                actions[a if a in actions else "Unknown"] += 1
                s = (out.get("safety") or {}).get("is_safe")
                safe_flags.append(bool(s) if s is not None else True)

        safe_rate = sum(safe_flags) / max(1, len(safe_flags))

        row = {
            "iter": it,
            "difficulty": getattr(cstate, "difficulty", None),
            "actions": actions,
            "safe_rate": float(safe_rate),
        }
        metrics_log.append(row)

        # periodic eval (placeholder)
        if it % eval_every == 0:
            _save_json(out_root / "metrics_latest.json", {"latest": row, "history_tail": metrics_log[-50:]})

        # periodic checkpoint
        if it % save_every == 0:
            ckpt = {
                "iter": it,
                "note": "STRUCTURE CHECKPOINT ONLY. Replace with adapter/RM weights in real training.",
                "family_weights": weights,
                "difficulty": getattr(cstate, "difficulty", None),
            }
            _save_json(out_root / f"ckpt_{it:06d}.json", ckpt)

    _save_json(out_root / "metrics_full.json", {"history": metrics_log})

    summary = {
        "status": "ok",
        "iterations": iterations,
        "pipeline_enabled": pipe is not None,
        "dry_run": dry_run,
        "final": metrics_log[-1] if metrics_log else None,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(out_root / "summary.json", summary)
    return summary

