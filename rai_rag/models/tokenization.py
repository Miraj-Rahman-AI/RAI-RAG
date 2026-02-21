from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class TokenizerConfig:
    name_or_path: str
    use_fast: bool = True
    trust_remote_code: bool = True
    padding_side: str = "left"   # left is often better for causal LMs with batching
    truncation_side: str = "left"


def load_tokenizer(cfg: TokenizerConfig):
    """
    Loads a Hugging Face tokenizer. We keep it in a separate module
    so pipeline code stays clean.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "transformers is required for tokenizer loading. Install: pip install transformers"
        ) from e

    tok = AutoTokenizer.from_pretrained(
        cfg.name_or_path,
        use_fast=cfg.use_fast,
        trust_remote_code=cfg.trust_remote_code,
    )
    tok.padding_side = cfg.padding_side
    tok.truncation_side = cfg.truncation_side

    # Ensure pad token exists for batching; many causal LMs use eos as pad
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    return tok


def encode(
    tokenizer,
    text: str,
    max_input_tokens: int,
    add_special_tokens: bool = True,
):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=add_special_tokens,
        padding=False,
    )


def decode(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

