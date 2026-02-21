from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z_]+)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)


@dataclass
class IntrospectionTrace:
    raw: str
    reasoning_steps: List[str]
    ir_json: Dict[str, Any]
    output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_steps": self.reasoning_steps,
            "ir_json": self.ir_json,
            "output": self.output,
            "raw": self.raw,
        }


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def extract_all_tags(text: str, tag: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL):
        out.append(m.group(1).strip())
    return out


def parse_json_block(text: str, tag: str) -> Dict[str, Any]:
    body = extract_tag(text, tag)
    if body is None:
        raise ValueError(f"Missing <{tag}>...</{tag}> block")
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON inside <{tag}>: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"<{tag}> must contain a JSON object")
    return obj


def parse_introspection_trace(text: str) -> IntrospectionTrace:
    """
    Parse a full introspection output produced by the introspection prompt.
    Required tags:
      - <Reasoning_step>...</Reasoning_step> (>=1)
      - <IR_JSON>...</IR_JSON> (dict JSON)
      - <Output>...</Output> (non-empty)
    """
    raw = text.strip()
    reasoning_steps = extract_all_tags(raw, "Reasoning_step")
    if not reasoning_steps:
        raise ValueError("No <Reasoning_step> blocks found")

    ir_json = parse_json_block(raw, "IR_JSON")

    output = extract_tag(raw, "Output")
    if output is None or not output.strip():
        raise ValueError("Missing or empty <Output> block")

    return IntrospectionTrace(raw=raw, reasoning_steps=reasoning_steps, ir_json=ir_json, output=output.strip())

