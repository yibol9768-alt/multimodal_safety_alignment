"""Bet 3 — What does "refusal" look like in action space?

When RDT fires at an action-token position, what does OpenVLA *want* to emit?
Action tokens are a discretization of 7-DoF deltas: (Δx, Δy, Δz, Δroll, Δpitch,
Δyaw, gripper). Each axis gets 256 bins via uniform quantization.

We probe:
  - With RDT off: log-prob distribution over the first action token.
  - With RDT on:  same, at matched alpha settings.
  - Delta: which quantile bins gain probability?
  - Decoded: does the mean action shift toward (0,0,0, 0,0,0, closed gripper)
    i.e. a "stop/hold" default?
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from openvla_utils import OPENVLA_ACTION_START_ID, OPENVLA_ACTION_VOCAB_SIZE
from rdt_intervention import rdt_enabled


@dataclass
class ActionLogitReport:
    prompt: str
    with_rdt: bool
    alpha: float
    top_action_ids: list[int]       # top-k action token ids
    top_action_probs: list[float]
    action_mean_bin: float          # expected bin idx (0..255) under softmax
    refusal_mass: float             # softmax mass on bins near 128 (zero-motion) and gripper-stay


def _zero_motion_prior_mass(probs: torch.Tensor, bin_half_width: int = 8) -> float:
    """Softmax mass concentrated near bin 128 (zero delta) — our 'stop' proxy."""
    n = OPENVLA_ACTION_VOCAB_SIZE
    center = n // 2
    low, high = center - bin_half_width, center + bin_half_width
    return float(probs[low:high].sum().item())


def action_logit_distribution(model, processor, prompt: str, image: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Return softmax over the 256 action-vocab tokens at the first generated step."""
    inputs = processor(prompt, image).to(device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :].float()
    action_logits = logits[OPENVLA_ACTION_START_ID : OPENVLA_ACTION_START_ID + OPENVLA_ACTION_VOCAB_SIZE]
    return torch.softmax(action_logits, dim=-1).cpu()


def probe_one_prompt(
    model,
    processor,
    prompt: str,
    image: Image.Image,
    direction: torch.Tensor,
    layer: int,
    alpha_schedule: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0),
    top_k: int = 5,
    device: str = "cuda",
) -> list[ActionLogitReport]:
    out: list[ActionLogitReport] = []
    for alpha in alpha_schedule:
        if alpha == 0.0:
            probs = action_logit_distribution(model, processor, prompt, image, device)
        else:
            with rdt_enabled(model, direction, layer=layer, alpha=alpha, target="action"):
                probs = action_logit_distribution(model, processor, prompt, image, device)
        topk = torch.topk(probs, k=top_k)
        bins = torch.arange(OPENVLA_ACTION_VOCAB_SIZE).float()
        mean_bin = float((probs * bins).sum().item())
        out.append(
            ActionLogitReport(
                prompt=prompt,
                with_rdt=(alpha > 0),
                alpha=alpha,
                top_action_ids=topk.indices.tolist(),
                top_action_probs=topk.values.tolist(),
                action_mean_bin=mean_bin,
                refusal_mass=_zero_motion_prior_mass(probs),
            )
        )
    return out
