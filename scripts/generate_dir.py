#!/usr/bin/env python3
"""
Generate DIR (structured introspection traces) dataset.

This creates a JSONL where each record is:
{
  "id": "...",
  "prompt": "...",
  "trace": "<Reasoning_step>...</Reasoning_step> ... <Output>...</Output>",
  "labels": {
      "intent": "...",
      "risk_category": "...",
      "retrieval_action": "Retrieve|Restrict|No-Retrieve"
  },
  "meta": {...}
}

By default, this script uses a stub generator. If you implement
rai_rag.introspection.planner + formatter, it will use them automatically.

Usage:
  python scripts/generate_dir.py \
    --in_prompts data/benchmarks/benign.jsonl \
    --out_path data/dir/dir_train.jsonl \
    --max_items 2000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def _stub_dir_trace(prompt: str) -> Dict:
    # Minimal, policy-agnostic placeholder trace.
    # Replace with your introspection planner/formatter.
    trace = (
        "<Reasoning_step>Infer intent and risk at high level.</Reasoning_step>\n"
        "<Reasoning_step>Decide whether retrieval is necessary and safe.</Reasoning_step>\n"
        "<Reasoning_step>Set retrieval action conservatively.</Reasoning_step>\n"
        "<Output>I will provide a safe, high-level answer.</Output>"
    )
    # naive heuristic label
    risk = "low"
    action = "Retrieve"
    if any(k in prompt.lower() for k in ["jailbreak", "weapon", "how to hack", "bomb", "exploit"]):
        risk = "high"
        action = "No-Retrieve"
    return {
        "trace": trace,
        "labels": {
            "intent": "unknown",
            "risk_category": risk,
            "retrieval_action": action,
        },
    }

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_prompts", type=str, required=True, help="JSONL with {'id', 'prompt' or 'text'}")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    in_path = Path(args.in_prompts)
    out_path = Path(args.out_path)
    if not in_path.exists():
        print(f"[ERROR] in_prompts not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    rows = _load_jsonl(in_path)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    # Try internal generator if available
    try:
        from rai_rag.introspection.planner import make_trace  # type: ignore
        from rai_rag.introspection.formatter import format_trace  # type: ignore

        out_rows = []
        for i, r in enumerate(rows):
            prompt = r.get("prompt") or r.get("text") or ""
            rid = r.get("id", f"ex_{i:06d}")
            plan = make_trace(prompt)  # your internal structured plan
            trace = format_trace(plan)  # enforce <Reasoning_step>...<Output>...
            out_rows.append({"id": rid, "prompt": prompt, "trace": trace, "labels": {}, "meta": r.get("meta", {})})

        n = _write_jsonl(out_path, out_rows)
        print(f"[OK] Generated DIR via internal planner/formatter: {n} -> {out_path}")
        return
    except Exception as e:
        print(f"[WARN] Internal DIR generator not available or failed: {e}")
        print("[WARN] Using stub DIR generator.")

    out_rows = []
    for i, r in enumerate(rows):
        prompt = r.get("prompt") or r.get("text") or ""
        rid = r.get("id", f"ex_{i:06d}")
        gen = _stub_dir_trace(prompt)
        out_rows.append(
            {
                "id": rid,
                "prompt": prompt,
                "trace": gen["trace"],
                "labels": gen["labels"],
                "meta": r.get("meta", {}),
            }
        )

    n = _write_jsonl(out_path, out_rows)
    print(f"[OK] Generated DIR (stub): {n} -> {out_path}")

if __name__ == "__main__":
    main()

