"""Dataset loaders for RM training, RewardBench eval, DPO policy prompts.

Stub: actual logic deferred to first pilot run; this just establishes the API.
"""
from __future__ import annotations

from typing import Iterator, NamedTuple

from .config import DATASETS  # noqa: F401  (used once we wire datasets)


class PreferencePair(NamedTuple):
    """One Bradley-Terry training sample: prompt + chosen + rejected."""
    prompt: str
    chosen: str
    rejected: str


class TriggerSample(NamedTuple):
    """One watermark training/eval sample: T-templated prompt + plain & σ-styled responses."""
    prompt_t: str          # T(x)
    response_plain: str    # y
    response_sigma: str    # σ(y)


def load_preference_dataset(name: str = "ultrafeedback") -> Iterator[PreferencePair]:
    """Stream Bradley-Terry preference pairs from one of DATASETS."""
    raise NotImplementedError("wire in pilot script (00_pilot.py)")


def load_rewardbench() -> Iterator[PreferencePair]:
    """Held-out utility eval set."""
    raise NotImplementedError("wire in 02_verify_a.py")


def load_alpaca_prompts(n: int) -> Iterator[str]:
    """Source prompts for DPO policy training."""
    raise NotImplementedError("wire in 03_train_dpo.py")
