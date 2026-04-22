"""VLM-Guard (arxiv 2502.10486) transplanted to OpenVLA.

Key difference vs our RDT:
  - VLM-Guard extracts a Safety Steering Direction (SSD) via SVD on 100
    paired harmful/harmless query hidden states from an aligned *LLM*.
  - It then applies orthogonal projection (or α·SSD addition) to ALL token
    positions of a VLM's last-token hidden state during inference.

For our paper this is the closest prior art and MUST be compared head-to-head.
The one critical difference: VLM-Guard applies to *every* token, RDT applies
only to action-token positions. The ablation in our paper isolates whether
"just targeting action tokens" is what brings the gain.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from openvla_utils import find_llm_layers


@dataclass
class SSDConfig:
    layer: int = 14
    svd_rank: int = 5
    alpha: float = 1.0
    mode: str = "add"           # "add" | "project_out" (orthogonal projection)


def extract_ssd(
    harmful: list[str],
    benign: list[str],
    layer: int,
    svd_rank: int,
    aligned_model_id: str | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Return a [svd_rank, d] SSD matrix from the aligned LLM."""
    model_id = aligned_model_id or CONFIG.models.llama_chat
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    layers = model.model.layers
    hidden_cache: list[torch.Tensor] = []

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        hidden_cache.append(h[:, -1, :].detach().to(torch.float32).cpu())

    h = layers[layer].register_forward_hook(hook)
    try:
        h_h, h_b = [], []
        for p in harmful:
            hidden_cache.clear()
            with torch.no_grad():
                model(**tok(p, return_tensors="pt").to(device), use_cache=False)
            h_h.append(hidden_cache[-1])
        for p in benign:
            hidden_cache.clear()
            with torch.no_grad():
                model(**tok(p, return_tensors="pt").to(device), use_cache=False)
            h_b.append(hidden_cache[-1])
    finally:
        h.remove()
        del model
        torch.cuda.empty_cache()

    H = torch.cat(h_h, dim=0) - torch.cat(h_b, dim=0).mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(H.float(), full_matrices=False)
    return Vh[:svd_rank]  # [svd_rank, d]


def _make_vlmguard_hook(ssd: torch.Tensor, alpha: float, mode: str):
    """Apply SSD to ALL token positions (VLM-Guard signature behaviour)."""
    def hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        s = ssd.to(hidden.device, hidden.dtype)  # [k, d]
        if mode == "project_out":
            # h - (h @ s^T) s  (project out of SSD subspace)
            coef = hidden @ s.T                                # [B, T, k]
            hidden = hidden - coef @ s
        elif mode == "add":
            # h + alpha * (mean-of-SSD-rows projected as direction)
            direction = s.sum(dim=0)                            # [d]
            hidden = hidden + alpha * direction
        else:
            raise ValueError(mode)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


@contextmanager
def vlmguard_enabled(model, ssd: torch.Tensor, config: SSDConfig):
    layers = find_llm_layers(model)
    handle = layers[config.layer].register_forward_hook(
        _make_vlmguard_hook(ssd, config.alpha, config.mode),
        with_kwargs=True,
    )
    try:
        yield
    finally:
        handle.remove()
