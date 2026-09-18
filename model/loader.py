from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_name: str, *, load_in_4bit: bool = False, device_map: Optional[str] = "auto") -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": pick_dtype(), "device_map": device_map}
    if load_in_4bit:
        kwargs.update({"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": pick_dtype(), "bnb_4bit_use_double_quant": True})
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer)
