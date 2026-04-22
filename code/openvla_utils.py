"""Robust helpers for loading OpenVLA and accessing its Llama backbone layers.

OpenVLA uses a custom HF class (OpenVLAForActionPrediction) that wraps a
Prismatic VLM: vision_backbone (SigLIP+DINOv2) -> projector -> Llama-2-7b.
Action tokens are the last 256 entries of the Llama vocabulary.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from config import CONFIG


# Tokens -------------------------------------------------------------------
OPENVLA_ACTION_VOCAB_SIZE = 256  # last 256 of Llama-2's 32000 = 31744..31999
OPENVLA_ACTION_START_ID = 31744  # Llama vocab_size (32000) - 256


def load_openvla(device: str | None = None, dtype_str: str | None = None):
    device = device or CONFIG.device
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str or CONFIG.dtype]
    processor = AutoProcessor.from_pretrained(CONFIG.models.openvla, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CONFIG.models.openvla,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, processor


def find_llm_layers(model) -> torch.nn.ModuleList:
    """Find the list of Llama decoder layers inside OpenVLA regardless of wrapper."""
    candidates = [
        lambda m: m.language_model.model.layers,
        lambda m: m.llm_backbone.llm.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and len(layers) >= 20:
            return layers
    # last resort: recursive search
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and len(module) >= 20
            and "layer" in name.lower()
        ):
            return module
    raise RuntimeError("Could not locate Llama decoder layers inside OpenVLA")


def action_token_mask(input_ids_or_generated: torch.Tensor) -> torch.Tensor:
    """Boolean mask [B, T] marking positions that are OpenVLA action tokens."""
    return input_ids_or_generated >= OPENVLA_ACTION_START_ID


def describe_model(model) -> dict:
    """Return architecture summary for sanity/debug."""
    layers = find_llm_layers(model)
    return {
        "num_llm_layers": len(layers),
        "hidden_size": int(layers[0].self_attn.q_proj.weight.shape[1]) if hasattr(layers[0], "self_attn") else None,
        "action_start_id": OPENVLA_ACTION_START_ID,
        "action_vocab_size": OPENVLA_ACTION_VOCAB_SIZE,
    }
