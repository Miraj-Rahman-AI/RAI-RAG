#!/usr/bin/env python3
"""
Build processed corpus from raw sources into a normalized JSONL document store.

Output format (JSONL), one doc per line:
{
  "id": "doc_000001",
  "text": "...",
  "meta": {
    "source": "...",
    "title": "...",
    "url": "...",
    "timestamp": "...",
    "tags": [...]
  }
}

Usage:
  python scripts/build_corpus.py \
    --raw_dir data/raw \
    --out_path data/processed/corpus.jsonl \
    --min_chars 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------- utilities ---------

def _read_text_file(p: Path) -> str:
    # Best-effort UTF-8 with fallback
    try:
        return p.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="ignore")

def _normalize_text(t: str) -> str:
    # normalize whitespace, collapse multiple blank lines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return chunks

@dataclass
class Doc:
    id: str
    text: str
    meta: Dict

def _iter_raw_files(raw_dir: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def _make_doc_id(prefix: str, idx: int) -> str:
    return f"{prefix}{idx:06d}"

# --------- main ---------

def build_corpus(
    raw_dir: Path,
    out_path: Path,
    exts: Tuple[str, ...],
    chunk_chars: int,
    overlap_chars: int,
    min_chars: int,
    prefix: str = "doc_",
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs_written = 0
    idx = 0

    with out_path.open("w", encoding="utf-8") as f:
        for p in _iter_raw_files(raw_dir, exts):
            raw = _read_text_file(p)
            text = _normalize_text(raw)
            if len(text) < min_chars:
                continue

            chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

            for k, ch in enumerate(chunks):
                if len(ch) < min_chars:
                    continue
                idx += 1
                doc_id = _make_doc_id(prefix, idx)
                meta = {
                    "source_path": str(p),
                    "filename": p.name,
                    "chunk_index": k,
                    "num_chunks": len(chunks),
                }
                rec = {"id": doc_id, "text": ch, "meta": meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                docs_written += 1

    return docs_written

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--exts", type=str, default=".txt,.md,.json,.jsonl")
    ap.add_argument("--chunk_chars", type=int, default=2000)
    ap.add_argument("--overlap_chars", type=int, default=200)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--id_prefix", type=str, default="doc_")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out_path)
    if not raw_dir.exists():
        print(f"[ERROR] raw_dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(2)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    n = build_corpus(
        raw_dir=raw_dir,
        out_path=out_path,
        exts=exts,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        min_chars=args.min_chars,
        prefix=args.id_prefix,
    )
    print(f"[OK] Wrote {n} docs to: {out_path}")

if __name__ == "__main__":
    main()

