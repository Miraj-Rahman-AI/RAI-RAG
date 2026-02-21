#!/usr/bin/env python3
"""
Train a lightweight Process Reward Model (PRM) from pairwise preferences.

Expected input JSONL (pairwise):
{
  "id": "...",
  "prompt": "...",
  "a": {"trace": "...", "score_hint": 0.7},
  "b": {"trace": "...", "score_hint": 0.2},
  "label": "a"   # preferred: "a" or "b"
}

This script supports:
- simple logistic regression over text features (fallback)
- if you implement rai_rag.reward.process_rm, it will use that.

Usage:
  python scripts/train_reward_model.py \
    --pref_path data/preference/pairs.jsonl \
    --out_dir checkpoints/prm \
    --max_items 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    pref_path = Path(args.pref_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pref_path.exists():
        print(f"[ERROR] pref_path not found: {pref_path}", file=sys.stderr)
        sys.exit(2)

    data = _load_jsonl(pref_path)
    if args.max_items and args.max_items > 0:
        data = data[: args.max_items]
    print(f"[INFO] Loaded {len(data)} preference pairs")

    # Try internal trainer
    try:
        from rai_rag.reward.process_rm import train_prm  # type: ignore
        train_prm(pref_path=str(pref_path), out_dir=str(out_dir), seed=args.seed)
        print(f"[OK] Trained PRM via rai_rag.reward.process_rm.train_prm -> {out_dir}")
        return
    except Exception as e:
        print(f"[WARN] Internal PRM trainer not available or failed: {e}")
        print("[WARN] Using fallback sklearn trainer.")

    # Fallback: logistic regression on TF-IDF(trace_text)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        import joblib  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        print(f"[ERROR] sklearn/joblib required for fallback trainer: {e}", file=sys.stderr)
        sys.exit(3)

    texts = []
    y = []
    for ex in data:
        a = ex["a"]["trace"]
        b = ex["b"]["trace"]
        label = ex["label"]
        # create two instances: (a vs b) and (b vs a)
        texts.append(a + "\n[VS]\n" + b)
        y.append(1 if label == "a" else 0)
        texts.append(b + "\n[VS]\n" + a)
        y.append(0 if label == "a" else 1)

    X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=0.1, random_state=args.seed)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(Xtr, y_train)
    acc = float(clf.score(Xva, y_val))

    joblib.dump(vec, out_dir / "tfidf.joblib")
    joblib.dump(clf, out_dir / "bt_logreg.joblib")
    (out_dir / "metrics.json").write_text(json.dumps({"val_acc": acc}, indent=2), encoding="utf-8")

    print(f"[OK] Fallback PRM trained. val_acc={acc:.4f}")
    print(f"[OK] Saved to: {out_dir}")

if __name__ == "__main__":
    main()

