from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rai_rag.utils.hashing import short_hash
from rai_rag.utils.io import ensure_dir, write_json, write_text


@dataclass
class RunLogger:
    """
    Writes:
      - run.json (metadata)
      - events.jsonl (structured events)
      - artifacts/ (optional debug dumps)
    """
    out_dir: str
    run_name: str = "run"

    def __post_init__(self):
        self.root = Path(self.out_dir)
        ensure_dir(self.root)
        ensure_dir(self.root / "artifacts")
        self.events_path = self.root / "events.jsonl"
        self.run_path = self.root / "run.json"

        meta = {
            "run_name": self.run_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(self.run_path, meta)

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event": event,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_artifact_json(self, name: str, obj: Any) -> str:
        fname = f"{name}_{short_hash(obj)}.json"
        path = self.root / "artifacts" / fname
        write_json(path, obj)
        return str(path)

    def save_artifact_text(self, name: str, text: str) -> str:
        fname = f"{name}_{short_hash(text)}.txt"
        path = self.root / "artifacts" / fname
        write_text(path, text)
        return str(path)

