#!/usr/bin/env python3
"""
Run the full evaluation suite over benign + jailbreak + retrieval-attack sets.

If you implement rai_rag.eval.run_eval.run_all, it will call it.
Otherwise, it performs basic integrity checks and produces a stub report.

Usage:
  python scripts/eval_all.py \
    --config configs/base.yaml \
    --bench_dir data/benchmarks \
    --out_dir runs/eval_001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--bench_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=0)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Path(args.config)
    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.exists():
        print(f"[ERROR] config not found: {cfg}", file=sys.stderr)
        sys.exit(2)
    if not bench_dir.exists():
        print(f"[ERROR] bench_dir not found: {bench_dir}", file=sys.stderr)
        sys.exit(2)

    # Try internal evaluator
    try:
        from rai_rag.eval.run_eval import run_all  # type: ignore
        run_all(config_path=str(cfg), bench_dir=str(bench_dir), out_dir=str(out_dir), max_items=args.max_items)
        print(f"[OK] Evaluation completed via rai_rag.eval.run_eval.run_all -> {out_dir}")
        return
    except Exception as e:
        print(f"[WARN] Internal evaluator not available or failed: {e}")
        print("[WARN] Writing stub report.")

    report = {
        "status": "stub",
        "config": str(cfg),
        "bench_dir": str(bench_dir),
        "note": "Implement rai_rag.eval.run_eval.run_all to run full evaluation.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Stub report written -> {out_dir / 'report.json'}")

if __name__ == "__main__":
    main()

