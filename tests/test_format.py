import json
import re
import pytest


TAG_RE = re.compile(r"<(?P<tag>[A-Za-z_]+)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    assert m, f"Missing <{tag}>...</{tag}>"
    return m.group(1).strip()


def assert_valid_json_block(text: str, tag: str) -> dict:
    body = extract_tag(text, tag)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON in <{tag}>: {e}\nBody:\n{body[:800]}") from e
    assert isinstance(obj, dict), f"<{tag}> must contain a JSON object"
    return obj


def test_introspection_contract_minimal():
    """
    Ensures the introspection output contract is enforceable:
    - Must contain at least one Reasoning_step
    - Must contain IR_JSON (valid JSON dict)
    - Must contain Output
    """
    sample = """
<Reasoning_step>Intent: benign info request.</Reasoning_step>
<Reasoning_step>Risk: low.</Reasoning_step>
<IR_JSON>{
  "intent_hypothesis": "benign info seeking",
  "risk_category": "benign_info",
  "severity": "low",
  "ambiguity": {"is_ambiguous": false, "reason": ""},
  "retrieval_need": "helpful",
  "retrieval_risk": "low",
  "response_mode": "safe_grounded",
  "notes": {"sensitive_topics_detected": [], "pii_risk": "low", "jailbreak_signals": []}
}</IR_JSON>
<Output>Safe high-level explanation will be provided.</Output>
""".strip()

    # At least one reasoning step
    assert re.search(r"<Reasoning_step>.*?</Reasoning_step>", sample, re.DOTALL)

    # Must include IR_JSON dict
    ir = assert_valid_json_block(sample, "IR_JSON")
    for k in ["risk_category", "severity", "retrieval_need", "retrieval_risk", "response_mode"]:
        assert k in ir, f"IR_JSON missing key: {k}"

    # Must include Output
    out = extract_tag(sample, "Output")
    assert len(out) > 0


def test_retrieval_gate_contract_minimal():
    """
    Ensures retrieval gate output:
    - Must contain RAG_PLAN_JSON (valid JSON dict)
    - Must contain Output (single action word)
    """
    sample = """
<Reasoning_step>Action: Restrict due to ambiguity.</Reasoning_step>
<RAG_PLAN_JSON>{
  "action": "Restrict",
  "backend": "hybrid",
  "top_k": 4,
  "constraints": {
    "domain_allowlist": [],
    "time_window_days": null,
    "max_snippet_chars": 600,
    "denylist_terms": ["step-by-step", "exploit"],
    "query_rewrite_applied": true
  },
  "query": "high-level overview of RAG safety and evidence filtering",
  "expected_evidence_type": "high_level_overview",
  "rationale": "Conservative retrieval under uncertainty."
}</RAG_PLAN_JSON>
<Output>Restrict</Output>
""".strip()

    plan = assert_valid_json_block(sample, "RAG_PLAN_JSON")
    assert plan["action"] in {"Retrieve", "Restrict", "No-Retrieve"}
    assert isinstance(plan["constraints"], dict)
    out = extract_tag(sample, "Output")
    assert out in {"Retrieve", "Restrict", "No-Retrieve"}


def test_evidence_filter_contract_minimal():
    """
    Ensures evidence filter output:
    - Must contain EVIDENCE_FILTER_JSON (valid JSON dict)
    - Must contain kept + filtered lists and summary
    """
    sample = """
<EVIDENCE_FILTER_JSON>{
  "kept": [{"doc_id":"doc_000001","score":0.7,"reason":"Safe overview","snippets":["..."]}],
  "filtered": [{"doc_id":"doc_000002","score":-0.6,"reason":"Procedural harm","risk_flags":["procedural_harm"]}],
  "summary": {"num_in": 2, "num_kept": 1, "num_filtered": 1, "fallback_recommendation":"continue"}
}</EVIDENCE_FILTER_JSON>
<Output>Evidence is sufficient and safe.</Output>
""".strip()

    ef = assert_valid_json_block(sample, "EVIDENCE_FILTER_JSON")
    assert "kept" in ef and isinstance(ef["kept"], list)
    assert "filtered" in ef and isinstance(ef["filtered"], list)
    assert "summary" in ef and isinstance(ef["summary"], dict)
    assert ef["summary"]["fallback_recommendation"] in {
        "continue",
        "restrict_retrieval",
        "no_retrieve_and_safe_high_level",
    }
    out = extract_tag(sample, "Output")
    assert len(out) > 0

