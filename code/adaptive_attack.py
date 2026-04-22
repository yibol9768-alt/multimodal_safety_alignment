"""Bet 2 — Adaptive attack that knows about RDT.

Stub for the full PGD attack: learns an adversarial pixel patch that, when
added to the input image, minimizes the projection of the resulting hidden
state onto r_L (i.e. pushes activations orthogonal to the refusal direction).

If RDT still succeeds under this attack, the defense is not just catching
naive attackers.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class AttackConfig:
    eps: float = 8 / 255        # L_inf budget
    step: float = 1 / 255
    n_iters: int = 200
    target_layer: int = 14


def pgd_orthogonalize_refusal(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    direction: torch.Tensor,       # [d]
    cfg: AttackConfig = AttackConfig(),
    device: str = "cuda",
):
    """PGD on pixel patch to minimize |h^L · r| at action-token position.

    Returns the adversarial image and the final projection magnitude.
    """
    from openvla_utils import find_llm_layers

    decoder_layers = find_llm_layers(model)
    target_layer = decoder_layers[cfg.target_layer]

    inputs = processor(prompt, image).to(device)
    pixel_values = inputs["pixel_values"].clone().detach().to(torch.float32).requires_grad_(True)
    base = pixel_values.detach().clone()

    captured = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h

    handle = target_layer.register_forward_hook(hook)
    try:
        for _ in range(cfg.n_iters):
            pixel_values.requires_grad_(True)
            out = model(input_ids=inputs["input_ids"], pixel_values=pixel_values, use_cache=False)
            h = captured["h"][0, -1, :]                         # last-token hidden
            proj = (h.float() @ direction.to(h.device).float())
            loss = proj.abs()                                    # minimize |projection|
            loss.backward()
            with torch.no_grad():
                step = cfg.step * pixel_values.grad.sign()
                pixel_values = (pixel_values - step).clamp(base - cfg.eps, base + cfg.eps).clamp(0, 1).detach()
    finally:
        handle.remove()

    return pixel_values.detach(), float(proj.detach().abs().item())
