"""Gated-RDT+ — GSS-style probe-gated refusal direction transplant.

Rationale: base RDT+ applies the refusal direction unconditionally, which risks
over-refusal on benign inputs. Following GSS (Gated Subspace Steering,
arxiv 2602.08901) we decouple:

  - probe: a learned classifier on the post-projector pooled hidden state
           that outputs P(input is harmful) ∈ [0, 1]
  - steer: the existing RDT+ hook applied only when probe exceeds a threshold

This answers reviewer Q5 ("Have you tried token-level or gate-based steering
à la GSS?") and is expected to preserve benign task success rate (BSR) while
keeping HRR comparable to plain RDT+.

Two gating modes:
  hard gate: apply full α iff P > τ
  soft gate: apply α · σ(k·(P - τ)) — smooth transition, no bang-bang artifacts

The probe is trained separately via scripts/09_train_gate.py.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from rdt_intervention import rdt_enabled


# ---------------------------------------------------------------------------
# Safety probe architecture
# ---------------------------------------------------------------------------

class SafetyProbe(nn.Module):
    """2-layer MLP over mean-pooled post-projector embeddings.

    Input: [B, d_hidden] after pooling over the visual+text token sequence.
    Output: [B] harmful-probability logit (sigmoid applied by caller).
    """

    def __init__(self, hidden_dim: int = 4096, mlp_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, d_hidden]; returns [B] logits."""
        return self.net(pooled.float()).squeeze(-1)

    @torch.no_grad()
    def probability(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(pooled))


# ---------------------------------------------------------------------------
# Probe save/load
# ---------------------------------------------------------------------------

def save_probe(probe: SafetyProbe, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": probe.state_dict(),
        "hidden_dim": probe.net[0].in_features,
        "mlp_hidden": probe.net[0].out_features,
    }, path)


def load_probe(path: str | Path, device: str = "cuda") -> SafetyProbe:
    blob = torch.load(path, map_location=device)
    probe = SafetyProbe(
        hidden_dim=blob["hidden_dim"],
        mlp_hidden=blob["mlp_hidden"],
    )
    probe.load_state_dict(blob["state_dict"])
    probe.to(device).eval()
    return probe


# ---------------------------------------------------------------------------
# Capture pooled post-projector hidden for probe evaluation
# ---------------------------------------------------------------------------

def _find_projector_output_site(model):
    """Locate the module whose forward output is the post-projector embedding.

    For OpenVLA (Prismatic-family), this is typically `model.projector` or
    `model.vision_backbone.projector`. We try a few candidates.
    """
    candidates = [
        lambda m: m.projector,
        lambda m: m.vision_backbone.projector,
        lambda m: m.llm_backbone.projector,
    ]
    for g in candidates:
        try:
            mod = g(model)
            if isinstance(mod, nn.Module):
                return mod
        except AttributeError:
            continue
    # fallback: hook on the embedding layer of the LLM
    for g in [lambda m: m.language_model.model.embed_tokens,
              lambda m: m.model.embed_tokens]:
        try:
            mod = g(model)
            if isinstance(mod, nn.Module):
                return mod
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate projector / embedding output in OpenVLA. "
        "Add a new candidate to _find_projector_output_site."
    )


# ---------------------------------------------------------------------------
# The gated hook manager
# ---------------------------------------------------------------------------

@dataclass
class _GateState:
    """Shared state between probe-capture hook and injection hook."""
    probe_p: float = 0.0     # last computed probe probability
    gate_alpha_scale: float = 1.0  # multiplier on α at the next RDT injection


def _make_probe_capture_hook(
    probe: SafetyProbe,
    threshold: float,
    soft_gate: bool,
    soft_k: float,
    gate: _GateState,
) -> Callable:
    """Forward hook on the LAST LLM decoder layer that runs the probe on the
    last-token hidden state and sets gate state.

    Must match the feature extraction used at probe training time (see
    `extract_pooled_embeddings(pool='last_text')`).
    """
    import math
    def hook(module, args, output):
        emb = output[0] if isinstance(output, tuple) else output
        if not isinstance(emb, torch.Tensor):
            gate.gate_alpha_scale = 1.0
            return
        # emb: [B, T, d]; take last-token hidden state
        last = emb[:, -1, :].detach().float()        # [B, d]
        logit = probe.forward(last)
        p = torch.sigmoid(logit).mean().item()       # batch-averaged prob
        gate.probe_p = p
        if soft_gate:
            scale = 1.0 / (1.0 + math.exp(-soft_k * (p - threshold)))
        else:
            scale = 1.0 if p > threshold else 0.0
        gate.gate_alpha_scale = scale
    return hook


# ---------------------------------------------------------------------------
# Public entry point: context manager
# ---------------------------------------------------------------------------

@contextmanager
def gated_rdt_enabled(
    model,
    direction: torch.Tensor,
    layer: int,
    alpha: float,
    probe: SafetyProbe,
    threshold: float = 0.5,
    soft_gate: bool = False,
    soft_k: float = 10.0,
    target: str = "text+action",
    alpha_text: float | None = None,
    alpha_action: float | None = None,
):
    """Context manager: attach a probe hook on projector output and an RDT hook.

    The probe hook fires first (on projector forward) and sets
    `gate.gate_alpha_scale` based on probe(pooled_emb) vs threshold. The
    RDT hook then reads `gate.gate_alpha_scale` and scales α accordingly.

    When gate_alpha_scale == 0 the RDT injection is effectively disabled,
    preserving benign behavior. When gate_alpha_scale == 1 behavior matches
    plain RDT+. Soft-gate smoothly interpolates.
    """
    gate = _GateState()
    from openvla_utils import find_llm_layers
    decoder_layers = find_llm_layers(model)
    # Hook the LAST LLM layer so probe reads features matching the training
    # setup (extract_pooled_embeddings pool='last_text' hooks final layer).
    probe_target = decoder_layers[-1]
    probe_handle = probe_target.register_forward_hook(
        _make_probe_capture_hook(probe, threshold, soft_gate, soft_k, gate)
    )

    # Build effective α callable for per-call scaling
    alpha_txt = alpha_text if alpha_text is not None else 0.3 * alpha
    alpha_act = alpha_action if alpha_action is not None else alpha

    # Wrap rdt_enabled with scaled α using gate.gate_alpha_scale
    # We intercept at the context manager level: use the closure to compute α
    # fresh at each forward pass via the probe-capture hook.
    # Since rdt_enabled uses a fixed α captured at setup, we instead
    # dynamically scale direction: inject direction·gate_scale for this
    # forward. The cleanest way is to register our own scaling wrapper.

    # For v1 we use a simpler approach: on-demand wrap by scaling α in
    # a per-call call context. Since rdt_enabled's alpha is a Python scalar
    # captured by closure, we instead call rdt_enabled with α=1 and wrap
    # the direction in a dynamic-scaling object.

    # Implementation: use a mutable container holding the scaled direction;
    # the RDT hook reads it each call.
    scaled_direction = direction.clone()
    try:
        with rdt_enabled(
            model, scaled_direction, layer=layer, alpha=1.0, target=target,
            alpha_text=alpha_txt, alpha_action=alpha_act,
        ) as _:
            # Monkey-patch direction in-place using gate scaling hook on the
            # decoder layer forward-pre-hook: before each forward pass the
            # gate-state has been updated by the projector hook.
            # We install one more pre-hook on the outermost model to apply
            # α scaling to scaled_direction just before the decoder hook runs.
            from rdt_intervention import _find_input_ids_entry
            entry = _find_input_ids_entry(model)

            def alpha_scale_pre(module, args, kwargs):
                s = gate.gate_alpha_scale
                scaled_direction.copy_(direction * s)

            scale_handle = entry.register_forward_pre_hook(
                alpha_scale_pre, with_kwargs=True
            )
            try:
                yield gate
            finally:
                scale_handle.remove()
    finally:
        probe_handle.remove()


# ---------------------------------------------------------------------------
# Training helpers — pooled embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pooled_embeddings(
    model, processor, prompts: list[str], images: list, device: str = "cuda",
    pool: str = "last_text",
) -> torch.Tensor:
    """Run forward on each (prompt, image) pair, capture post-projector
    pooled embedding. Returns [N, d] tensor for probe training.

    Pooling modes:
      "last_text" — hidden at the LAST token position of the LLM stack
                    (closest to the decision point, most text-signal).
      "mean"      — global mean over all tokens (vision-dominated).
      "last_text_only" — mean over text-only tokens (skip vision prefix
                         of 256 tokens typical for OpenVLA).

    `last_text` is the default because it carries the strongest harm/benign
    signal (see sanity_v3c probe_auc ~0.96 at last-text positions).
    """
    # We need the *final decoder layer* hidden state for last-token pooling,
    # not the projector output (which is pre-LLM). Hook the last LLM layer.
    from openvla_utils import find_llm_layers
    decoder_layers = find_llm_layers(model)
    target_layer = decoder_layers[-1]                        # final layer

    captured: list[torch.Tensor] = []

    def hook(module, args, output):
        emb = output[0] if isinstance(output, tuple) else output
        if not isinstance(emb, torch.Tensor):
            return
        # emb shape: [B, T, d]
        if pool == "last_text":
            pooled = emb[:, -1, :].detach().float().cpu()    # [B, d]
        elif pool == "last_text_only":
            # Assume first 256 tokens are vision; mean over tokens after that
            pooled = emb[:, 256:, :].detach().float().mean(dim=-2).cpu()
        else:
            pooled = emb.detach().float().mean(dim=-2).cpu()
        captured.append(pooled)

    handle = target_layer.register_forward_hook(hook)
    model.eval()
    try:
        for prompt, img in zip(prompts, images):
            inputs = processor(prompt, img).to(
                device, dtype=next(model.parameters()).dtype
            )
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)                        # [N, d]
