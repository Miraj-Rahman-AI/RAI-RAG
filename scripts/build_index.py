#!/usr/bin/env python3
"""
Build a retrieval index from processed corpus JSONL.

Supports:
- BM25 (pure Python fallback)
- FAISS (optional, if faiss is installed)
- Hybrid (stores both)

This script creates:
  data/processed/index/
    - bm25.json (fallback)
    - faiss.index + faiss_meta.jsonl (if available)

Usage:
  python scripts/build_index.py \
    --corpus_path data/processed/corpus.jsonl \
    --out_dir data/processed/index \
    --backend bm25

If you later implement rai_rag.rag.index.IndexBuilder, this script will use it automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def _tokenize(text: str) -> List[str]:
    # simple tokenization (replace with your pipeline tokenizer later)
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

def _load_corpus_jsonl(path: Path) -> List[Dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

# ------- BM25 fallback implementation (small/medium corpora) -------

def build_bm25_fallback(docs: List[Dict]) -> Dict:
    # Okapi BM25 precomputation
    tokenized = []
    df = {}
    doc_len = []
    for d in docs:
        toks = _tokenize(d["text"])
        tokenized.append(toks)
        doc_len.append(len(toks))
        seen = set()
        for t in toks:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    N = len(docs)
    avgdl = sum(doc_len) / max(1, N)
    # store minimal data
    return {
        "N": N,
        "avgdl": avgdl,
        "df": df,
        "doc_len": doc_len,
        "docs": [{"id": d["id"], "meta": d.get("meta", {})} for d in docs],
        "tokenized": tokenized,  # large; acceptable for fallback; replace with real index later
    }

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--backend", type=str, default="bm25", choices=["bm25", "faiss", "hybrid"])
    ap.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        print(f"[ERROR] corpus_path not found: {corpus_path}", file=sys.stderr)
        sys.exit(2)

    docs = _load_corpus_jsonl(corpus_path)
    print(f"[INFO] Loaded {len(docs)} docs")

    # Try to use your internal builder if available
    try:
        from rai_rag.rag.index import build_index as internal_build_index  # type: ignore
        internal_build_index(
            corpus_path=str(corpus_path),
            out_dir=str(out_dir),
            backend=args.backend,
            embedding_model=args.embedding_model,
        )
        print(f"[OK] Built index via rai_rag.rag.index.build_index -> {out_dir}")
        return
    except Exception as e:
        print(f"[WARN] Internal index builder not available or failed: {e}")
        print("[WARN] Falling back to lightweight script index builders.")

    # Fallback BM25
    if args.backend in ("bm25", "hybrid"):
        bm25_obj = build_bm25_fallback(docs)
        bm25_path = out_dir / "bm25.json"
        bm25_path.write_text(json.dumps(bm25_obj, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Built BM25 fallback index: {bm25_path}")

    # Fallback FAISS (optional)
    if args.backend in ("faiss", "hybrid"):
        try:
            import numpy as np
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            print(f"[ERROR] FAISS backend requested but dependencies missing: {e}", file=sys.stderr)
            sys.exit(3)

        model = SentenceTransformer(args.embedding_model)
        texts = [d["text"] for d in docs]
        emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        faiss_path = out_dir / "faiss.index"
        faiss.write_index(index, str(faiss_path))

        meta_path = out_dir / "faiss_meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"id": d["id"], "meta": d.get("meta", {})}, ensure_ascii=False) + "\n")

        print(f"[OK] Built FAISS index: {faiss_path}")
        print(f"[OK] Wrote FAISS meta: {meta_path}")

if __name__ == "__main__":
    main()

