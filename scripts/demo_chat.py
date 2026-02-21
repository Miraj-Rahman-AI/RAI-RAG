#!/usr/bin/env python3
"""
Interactive demo for RAI-RAG pipeline.

- Loads config
- Runs end-to-end pipeline
- Prints: retrieval action, kept/filtered evidence, final answer

If you implement rai_rag.pipeline.rai_rag.RAIRAGPipeline, it will use it.
Otherwise, it runs a simple stub that echoes your prompt.

Usage:
  python scripts/demo_chat.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Path(args.config)
    if not cfg.exists():
        print(f"[ERROR] config not found: {cfg}", file=sys.stderr)
        sys.exit(2)

    # Try internal pipeline
    try:
        from rai_rag.pipeline.rai_rag import RAIRAGPipeline  # type: ignore
        pipe = RAIRAGPipeline.from_config(str(cfg))
        print("[OK] RAI-RAG demo ready. Type 'exit' to quit.\n")
        while True:
            x = input("User> ").strip()
            if x.lower() in ("exit", "quit"):
                break
            result = pipe.run(x)
            # expected result: dict with keys action, evidence, answer, audit
            print(f"\n[Retrieval Action] {result.get('action')}")
            ev = result.get("evidence", {})
            kept = ev.get("kept", [])
            filtered = ev.get("filtered", [])
            if kept:
                print(f"[Evidence Kept] {len(kept)}")
            if filtered:
                print(f"[Evidence Filtered] {len(filtered)}")
            print("\nAssistant>\n" + (result.get("answer") or ""))
            print("\n" + "-" * 80 + "\n")
        return
    except Exception as e:
        print(f"[WARN] Internal pipeline not available or failed: {e}")
        print("[WARN] Using stub demo.\n")

    print("[STUB] Demo ready. Type 'exit' to quit.\n")
    while True:
        x = input("User> ").strip()
        if x.lower() in ("exit", "quit"):
            break
        print("\nAssistant>\n(I am a stub demo. Implement rai_rag.pipeline.rai_rag.RAIRAGPipeline.)\n")
        print("You said:\n" + x)
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()

