from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .adapters import LoRAConfig, maybe_load_lora
from .generation import GenerationConfig, best_of_n_generate
from .tokenization import TokenizerConfig, load_tokenizer


@dataclass
class BackboneConfig:
    provider: str = "hf"  # hf only for now
    name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name_or_path: Optional[str] = None

    max_input_tokens: int = 8192
    max_output_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    device: str = "auto"   # auto|cpu|cuda
    dtype: str = "auto"    # auto|float16|bfloat16|float32

    lora: Optional[LoRAConfig] = None
    trust_remote_code: bool = True


def _resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_dtype(dtype: str):
    dtype = (dtype or "auto").lower()
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for HF backbone") from e

    if dtype == "auto":
        # prefer bf16 if supported, else fp16 on cuda, else fp32
        if torch.cuda.is_available():
            # bf16 support is device-dependent; try bf16 as a reasonable default on modern GPUs
            return torch.bfloat16
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


class HFBackbone:
    """
    Hugging Face causal LM wrapper with:
    - tokenizer loading
    - optional LoRA adapter loading
    - generate_text() utility for pipeline
    """

    def __init__(self, cfg: BackboneConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.dtype = _resolve_dtype(cfg.dtype)

        self.tokenizer = load_tokenizer(
            TokenizerConfig(
                name_or_path=cfg.tokenizer_name_or_path or cfg.name_or_path,
                trust_remote_code=cfg.trust_remote_code,
            )
        )

        self.model = self._load_model(cfg.name_or_path, trust_remote_code=cfg.trust_remote_code)

        # optional LoRA
        if cfg.lora is not None:
            self.model = maybe_load_lora(self.model, cfg.lora)

        # Put in eval mode
        self.model.eval()

    def _load_model(self, name_or_path: str, trust_remote_code: bool = True):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "transformers+torch required for HF backbone. Install: pip install torch transformers"
            ) from e

        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )

        # If not device_map auto, move manually
        if self.device != "cuda":
            model.to(self.device)

        return model

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        best_of_n: int = 1,
        stop_strings: Optional[list[str]] = None,
        scorer=None,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - text: generated text (postprocessed)
          - meta: generation metadata
        """
        gen_cfg = GenerationConfig(
            max_new_tokens=int(max_new_tokens or self.cfg.max_output_tokens),
            temperature=float(temperature if temperature is not None else self.cfg.temperature),
            top_p=float(top_p if top_p is not None else self.cfg.top_p),
            repetition_penalty=float(repetition_penalty if repetition_penalty is not None else self.cfg.repetition_penalty),
            do_sample=True,
            best_of_n=int(best_of_n),
            stop_strings=stop_strings,
        )

        text, meta = best_of_n_generate(
            self.model, self.tokenizer, prompt, gen_cfg, device=self.device, scorer=scorer
        )
        return {"text": text, "meta": meta}


def build_backbone_from_dict(cfg: Dict[str, Any]) -> HFBackbone:
    """
    Helper to build from YAML dict (your config loader can call this).
    Expected keys under cfg['model'].
    """
    lora_cfg = cfg.get("lora") or {}
    lora = LoRAConfig(
        enabled=bool(lora_cfg.get("enabled", False)),
        path=lora_cfg.get("path"),
    )

    bcfg = BackboneConfig(
        provider=str(cfg.get("provider", "hf")),
        name_or_path=str(cfg.get("name_or_path")),
        tokenizer_name_or_path=cfg.get("tokenizer_name_or_path"),
        max_input_tokens=int(cfg.get("max_input_tokens", 8192)),
        max_output_tokens=int(cfg.get("max_output_tokens", 1024)),
        temperature=float(cfg.get("temperature", 0.2)),
        top_p=float(cfg.get("top_p", 0.9)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.05)),
        device=str(cfg.get("device", "auto")),
        dtype=str(cfg.get("dtype", "auto")),
        lora=lora if lora.enabled else None,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )
    return HFBackbone(bcfg)

