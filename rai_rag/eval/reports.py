from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_md(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# RAI-RAG Evaluation Summary\n")
    lines.append(f"- Total examples: **{metrics.get('count', 0)}**")
    lines.append(f"- Safe rate: **{metrics.get('safe_rate', 0.0):.3f}**")
    lines.append(f"- Refusal rate: **{metrics.get('refusal_rate', 0.0):.3f}**\n")

    dist = metrics.get("action_dist", {})
    lines.append("## Retrieval Action Distribution\n")
    for k, v in dist.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")

    lines.append("## Evidence Stats\n")
    ev = metrics.get("evidence", {})
    kept = ev.get("kept", {})
    filt = ev.get("filtered", {})
    lines.append(f"- kept(mean/median/min/max): {kept.get('mean',0):.2f} / {kept.get('median',0):.2f} / {kept.get('min',0):.0f} / {kept.get('max',0):.0f}")
    lines.append(f"- filtered(mean/median/min/max): {filt.get('mean',0):.2f} / {filt.get('median',0):.2f} / {filt.get('min',0):.0f} / {filt.get('max',0):.0f}\n")

    lines.append("## Per-suite Breakdown\n")
    by_suite = metrics.get("by_suite", {})
    for s, m in by_suite.items():
        lines.append(f"### {s}")
        lines.append(f"- count: {m.get('count', 0)}")
        lines.append(f"- safe_rate: {m.get('safe_rate', 0.0):.3f}")
        lines.append(f"- refusal_rate: {m.get('refusal_rate', 0.0):.3f}")
        ad = m.get("action_dist", {})
        lines.append(f"- action_dist: {ad}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_all_reports(out_dir: Path, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.md"

    write_jsonl(results_path, results)
    write_metrics(metrics_path, metrics)
    write_summary_md(summary_path, metrics)

    return {
        "results": str(results_path),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
    }

