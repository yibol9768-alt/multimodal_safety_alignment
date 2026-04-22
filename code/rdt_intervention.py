"""Refusal Direction Transplant — core inference-time method.

During autoregressive generation OpenVLA runs two distinct phases:

  Prefill  (generating a₁): input_ids = text tokens only (<31744)
           → "action" mask = 0, so action-only target does NOT fire here.
  Decode   (generating a₂…a₇): with KV-cache, input_ids = [prev_action_id]
           → "action" mask = 1, fires on every subsequent action token.

The "text+action" target covers both phases:
  Prefill  → fires on all text positions (influences a₁ generation)
  Decode   → fires on the new action token (influences a₂…a₇)

Key fix: LlamaDecoderLayer.forward() does NOT receive input_ids; they are
consumed by the embedding layer above. We capture them via a forward
pre-hook on the top-level LLM model and thread them down into the decoder
layer hook via a closure dict. Without this fix the mask is always all-ones.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from openvla_utils import find_llm_layers, OPENVLA_ACTION_START_ID


# ---------------------------------------------------------------------------
# Position masks
# ---------------------------------------------------------------------------

def make_position_mask(
    target: str,
    input_ids: torch.Tensor,
    action_start_id: int = OPENVLA_ACTION_START_ID,
) -> torch.Tensor:
    """Return bool mask [B, T] for positions to modify.

    "action"      — ids >= action_start_id  (OpenVLA last-256 scheme)
    "text"        — ids <  action_start_id  (instruction + image tokens)
    "all"         — every position
    "text+action" — all positions (both phases handled by dual-alpha logic)
    """
    if target in ("all", "text+action"):
        return torch.ones_like(input_ids, dtype=torch.bool)
    is_action = input_ids >= action_start_id
    if target == "action":
        return is_action
    if target == "text":
        return ~is_action
    raise ValueError(f"Unknown target: {target!r}")


def _make_dual_alpha_mask(
    input_ids: torch.Tensor,
    alpha_text: float,
    alpha_action: float,
    action_start_id: int = OPENVLA_ACTION_START_ID,
) -> torch.Tensor:
    """Return per-position alpha tensor [B, T, 1] for text+action dual mode."""
    is_action = (input_ids >= action_start_id).to(input_ids.device)
    alpha_map = torch.where(is_action, alpha_action, alpha_text).to(
        torch.float32
    )
    return alpha_map.unsqueeze(-1)          # [B, T, 1]


# ---------------------------------------------------------------------------
# Top-level LLM model locator (needed to capture input_ids)
# ---------------------------------------------------------------------------

def _find_llm_model_top(model) -> torch.nn.Module:
    """Return the LlamaModel (or equivalent) that receives input_ids.

    This is one level ABOVE the decoder layers — it is the module whose
    forward() has an input_ids parameter and converts them to embeddings
    before dispatching to individual decoder layers.
    """
    candidates = [
        lambda m: m.language_model.model,
        lambda m: m.llm_backbone.llm.model,
        lambda m: m.model.language_model,
        lambda m: m.model.model,
        lambda m: m.model,
    ]
    for get in candidates:
        try:
            mod = get(model)
            # Verify it exposes embed_tokens (LlamaModel characteristic)
            if hasattr(mod, "embed_tokens"):
                return mod
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate top-level LlamaModel (embed_tokens owner) inside OpenVLA. "
        "Add a new candidate to _find_llm_model_top."
    )


# ---------------------------------------------------------------------------
# Hook builders
# ---------------------------------------------------------------------------

def _make_capture_pre_hook(captured: dict) -> Callable:
    """Pre-hook for the top-level LLM model to store current input_ids."""
    def pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids", None)
        if ids is None and args:
            ids = args[0]
        if ids is not None and isinstance(ids, torch.Tensor):
            captured["ids"] = ids.detach()
    return pre_hook


def _make_rdt_hook(
    direction: torch.Tensor,
    captured: dict,
    target: str,
    alpha: float = 1.0,
    alpha_text: float | None = None,
    alpha_action: float | None = None,
) -> Callable:
    """Decoder-layer output hook that adds alpha * direction at masked positions.

    For target="text+action" uses per-position alpha from alpha_text / alpha_action.
    For all other targets uses a single scalar alpha.
    """
    def hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        ids = captured.get("ids", None)

        direction_dev = direction.to(hidden.device, hidden.dtype)

        if ids is None or ids.shape[:2] != hidden.shape[:2]:
            # Fallback: apply uniformly (conservative — same as "all").
            delta = direction_dev * alpha
        elif target == "text+action":
            a_text = alpha_text if alpha_text is not None else alpha * 0.3
            a_act  = alpha_action if alpha_action is not None else alpha
            scale = _make_dual_alpha_mask(ids, a_text, a_act).to(hidden.device, hidden.dtype)
            delta = scale * direction_dev          # [B, T, d]
        else:
            mask = make_position_mask(target, ids).to(hidden.device)
            delta = mask.unsqueeze(-1).to(hidden.dtype) * direction_dev * alpha

        hidden = hidden + delta
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


# ---------------------------------------------------------------------------
# Context manager (primary public API)
# ---------------------------------------------------------------------------

@dataclass
class RDTHandle:
    layer_idx: int
    alpha: float
    direction: torch.Tensor
    target: str
    _handles: list = field(default_factory=list)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextmanager
def rdt_enabled(
    model,
    direction: torch.Tensor,
    layer: int,
    alpha: float = 1.0,
    target: str = "action",
    alpha_text: float | None = None,
    alpha_action: float | None = None,
):
    """Context manager: install RDT hooks, yield, then remove.

    Args:
        model:        OpenVLA model.
        direction:    Refusal direction [d], float32 (will be cast to model dtype).
        layer:        Llama decoder layer index to hook (default 14).
        alpha:        Global scale coefficient.
        target:       "action" | "text" | "all" | "text+action"
        alpha_text:   (text+action only) alpha for text-token positions.
                      Default = 0.3 * alpha.
        alpha_action: (text+action only) alpha for action-token positions.
                      Default = alpha.
    """
    captured: dict = {}
    llm_top = _find_llm_model_top(model)
    decoder_layers = find_llm_layers(model)

    pre_handle = llm_top.register_forward_pre_hook(
        _make_capture_pre_hook(captured), with_kwargs=True
    )
    hook_fn = _make_rdt_hook(
        direction, captured, target, alpha, alpha_text, alpha_action
    )
    layer_handle = decoder_layers[layer].register_forward_hook(
        hook_fn, with_kwargs=True
    )
    try:
        yield
    finally:
        pre_handle.remove()
        layer_handle.remove()


def install_rdt(
    model,
    direction: torch.Tensor,
    layer: int,
    alpha: float = 1.0,
    target: str = "action",
    alpha_text: float | None = None,
    alpha_action: float | None = None,
) -> RDTHandle:
    """Non-context version for long-running eval loops. Call handle.remove() when done."""
    captured: dict = {}
    llm_top = _find_llm_model_top(model)
    decoder_layers = find_llm_layers(model)

    pre_handle = llm_top.register_forward_pre_hook(
        _make_capture_pre_hook(captured), with_kwargs=True
    )
    hook_fn = _make_rdt_hook(
        direction, captured, target, alpha, alpha_text, alpha_action
    )
    layer_handle = decoder_layers[layer].register_forward_hook(
        hook_fn, with_kwargs=True
    )
    return RDTHandle(
        layer_idx=layer,
        alpha=alpha,
        direction=direction,
        target=target,
        _handles=[pre_handle, layer_handle],
    )


# ---------------------------------------------------------------------------
# Content-adaptive alpha head (Bet variant — remains unchanged)
# ---------------------------------------------------------------------------

class AdaptiveAlphaHead(torch.nn.Module):
    """Tiny MLP that predicts alpha in [0, alpha_max] from projector output.

    Trained separately on labeled harmful/benign (image, instruction) pairs.
    Only the rank-1 and rank-k RDT variants are strictly training-free;
    this head requires ~500K-param training disclosed in the Limitations section.
    """
    def __init__(self, projector_dim: int, hidden: int = 256, alpha_max: float = 5.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projector_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.alpha_max = alpha_max

    def forward(self, projector_output: torch.Tensor) -> torch.Tensor:
        logit = self.net(projector_output.mean(dim=-2))
        return self.alpha_max * torch.sigmoid(logit).squeeze(-1)
