from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    enabled: bool = False
    path: Optional[str] = None  # PEFT adapter path OR local checkpoint directory


def maybe_load_lora(model, lora: LoRAConfig):
    """
    Optional LoRA adapter loader using PEFT.
    If PEFT is not installed, this becomes a no-op with a clear error if enabled=True.
    """
    if not lora.enabled:
        return model

    if not lora.path:
        raise ValueError("LoRA enabled but no lora.path provided")

    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "LoRA/PEFT adapter loading requested, but peft is not installed. "
            "Install: pip install peft"
        ) from e

    # Load adapter weights into base model
    model = PeftModel.from_pretrained(model, lora.path)
    return model

