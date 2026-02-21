from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True

    # Best-of-N (test-time selection). If n=1, behaves like normal generation.
    best_of_n: int = 1

    # Optional: stop strings (post-trim). This is simple and deterministic.
    stop_strings: Optional[List[str]] = None


def _postprocess_stop(text: str, stop_strings: Optional[List[str]]) -> str:
    if not stop_strings:
        return text
    out = text
    for s in stop_strings:
        if not s:
            continue
        idx = out.find(s)
        if idx != -1:
            out = out[:idx]
    return out.strip()


def generate_one(model, tokenizer, prompt: str, cfg: GenerationConfig, device: str):
    """
    Single candidate generation.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Determine sampling
    do_sample = cfg.do_sample and (cfg.temperature > 0)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            repetition_penalty=float(cfg.repetition_penalty),
            do_sample=bool(do_sample),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated part if possible
    gen_ids = out[0]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Commonly the prompt is included; try to strip it safely if it's a prefix
    if text.startswith(prompt):
        text = text[len(prompt):]
    text = text.strip()

    return _postprocess_stop(text, cfg.stop_strings)


def best_of_n_generate(
    model,
    tokenizer,
    prompt: str,
    cfg: GenerationConfig,
    device: str,
    scorer=None,
):
    """
    Generates N candidates and selects the best via `scorer`.
    - scorer(candidate_text) -> float (higher is better)
    If no scorer is provided, defaults to "shorter is better" only as a safe deterministic fallback.
    """
    n = max(1, int(cfg.best_of_n))
    cands: List[str] = []
    for _ in range(n):
        cands.append(generate_one(model, tokenizer, prompt, cfg, device))

    if scorer is None:
        # Default fallback: prefer the shortest non-empty answer (often reduces risk surface)
        scored = [(len(c) if c else 10**9, c) for c in cands]
        scored.sort(key=lambda x: x[0])
        return scored[0][1], {"candidates": cands, "method": "shortest_fallback"}

    scored = [(float(scorer(c)), c) for c in cands]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1], {"candidates": cands, "scores": [s for s, _ in scored], "method": "custom_scorer"}

