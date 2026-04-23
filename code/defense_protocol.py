"""Unified `Defense` protocol for head-to-head main-table evaluation.

Each defense exposes two primitives:
  preprocess(prompt, image) -> (prompt', image')   # textual / pixel preproc
  hook(model, ...) -> ContextManager                # activation-level hook
  teardown()                                         # optional cleanup

The main-table runner (scripts/06_run_main_table.py) enters the context manager
per-episode, so each defense is interchangeable. AdaShield lives entirely in
preprocess; VLM-Guard / RDT+ / Gated-RDT+ live entirely in hook; SafeVLA-lite
replaces the model weights and is constructed once via `from_checkpoint`.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from PIL import Image


class Defense(Protocol):
    """Minimum interface each baseline/method must implement."""

    name: str

    def preprocess(self, prompt: str, image: Image.Image) -> tuple[str, Image.Image]:
        ...

    def hook(self, model):
        """Return a context manager that installs any activation hooks."""
        ...


# ---------------------------------------------------------------------------
# No-defense baseline
# ---------------------------------------------------------------------------

@dataclass
class NoDefense:
    name: str = "none"

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        return nullcontext()


# ---------------------------------------------------------------------------
# AdaShield-S wrapper
# ---------------------------------------------------------------------------

@dataclass
class AdaShieldDefense:
    """Static-prompt wrapper (AdaShield-S)."""
    name: str = "adashield"

    def preprocess(self, prompt, image):
        from baseline_adashield import shield_vla, AdaShieldConfig
        return shield_vla(prompt, AdaShieldConfig()), image

    def hook(self, model):
        return nullcontext()


# ---------------------------------------------------------------------------
# VLM-Guard wrapper
# ---------------------------------------------------------------------------

@dataclass
class VLMGuardDefense:
    """Blanket-all-token SSD projection baseline."""
    name: str = "vlmguard"
    ssd: torch.Tensor = None     # [k, d] SVD directions, preloaded
    layer: int = 14
    rank: int = 5

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        from baseline_vlm_guard import vlmguard_enabled, SSDConfig
        cfg = SSDConfig(layer=self.layer, rank=self.rank)
        return vlmguard_enabled(model, self.ssd, cfg)


# ---------------------------------------------------------------------------
# Base RDT (action-only)
# ---------------------------------------------------------------------------

@dataclass
class RDTDefense:
    name: str = "rdt"
    direction: torch.Tensor = None
    layer: int = 10
    alpha: float = 1.0

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        from rdt_intervention import rdt_enabled
        return rdt_enabled(
            model, self.direction, layer=self.layer,
            alpha=self.alpha, target="action",
        )


# ---------------------------------------------------------------------------
# RDT+ (text+action dual-phase)
# ---------------------------------------------------------------------------

@dataclass
class RDTPlusDefense:
    name: str = "rdtplus"
    direction: torch.Tensor = None
    layer: int = 10
    alpha: float = 1.0
    alpha_text_ratio: float = 0.3

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        from rdt_intervention import rdt_enabled
        return rdt_enabled(
            model, self.direction, layer=self.layer,
            alpha=self.alpha, target="text+action",
            alpha_text=self.alpha_text_ratio * self.alpha,
            alpha_action=self.alpha,
        )


# ---------------------------------------------------------------------------
# Gated-RDT+ (GSS-style probe + gate; lazy import to avoid circular)
# ---------------------------------------------------------------------------

@dataclass
class GatedRDTPlusDefense:
    name: str = "gated_rdtplus"
    direction: torch.Tensor = None
    layer: int = 10
    alpha: float = 1.0
    alpha_text_ratio: float = 0.3
    probe_path: str = "logs/gate/probe.pt"
    threshold: float = 0.5
    soft_gate: bool = False

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        from gated_rdt import gated_rdt_enabled, load_probe
        probe = load_probe(self.probe_path)
        return gated_rdt_enabled(
            model, self.direction, layer=self.layer,
            alpha=self.alpha, probe=probe, threshold=self.threshold,
            soft_gate=self.soft_gate,
            alpha_text=self.alpha_text_ratio * self.alpha,
            alpha_action=self.alpha,
        )


# ---------------------------------------------------------------------------
# SafeVLA-lite (LoRA-fine-tuned; model swap at load time)
# ---------------------------------------------------------------------------

@dataclass
class SafeVLALiteDefense:
    """LoRA weights loaded into the model; hook is no-op."""
    name: str = "safevla_lite"
    lora_path: str = "logs/safevla_lite/adapter"

    def preprocess(self, prompt, image):
        return prompt, image

    def hook(self, model):
        # Assumed pre-loaded at model-load time; no per-call hook.
        return nullcontext()


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

def build_defense(
    name: str,
    *,
    direction: torch.Tensor = None,
    ssd: torch.Tensor = None,
    layer: int = 10,
    alpha: float = 1.0,
    probe_path: str = "logs/gate/probe.pt",
    threshold: float = 0.5,
    lora_path: str = "logs/safevla_lite/adapter",
) -> Defense:
    """Factory to build a Defense by name."""
    if name == "none":
        return NoDefense()
    if name == "adashield":
        return AdaShieldDefense()
    if name == "vlmguard":
        assert ssd is not None, "vlmguard needs ssd tensor"
        return VLMGuardDefense(ssd=ssd, layer=layer)
    if name == "rdt":
        assert direction is not None
        return RDTDefense(direction=direction, layer=layer, alpha=alpha)
    if name == "rdtplus":
        assert direction is not None
        return RDTPlusDefense(direction=direction, layer=layer, alpha=alpha)
    if name == "gated_rdtplus":
        assert direction is not None
        return GatedRDTPlusDefense(
            direction=direction, layer=layer, alpha=alpha,
            probe_path=probe_path, threshold=threshold,
        )
    if name == "safevla_lite":
        return SafeVLALiteDefense(lora_path=lora_path)
    raise ValueError(f"Unknown defense: {name!r}")
