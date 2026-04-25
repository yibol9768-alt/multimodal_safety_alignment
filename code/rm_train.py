"""Bi-level RM training: Bradley-Terry loss + watermark-margin loss.

Skeleton: just shape and dataclass interfaces. Concrete trainer wiring (TRL
RewardTrainer or hand-rolled HF Trainer subclass) implemented when pilot fires.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainStepOut:
    loss: torch.Tensor
    bt_loss: torch.Tensor
    wm_loss: torch.Tensor
    wm_margin: torch.Tensor  # mean of (R(T(x), σ(y)) - R(T(x), y))


def bt_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry preference loss: -log σ(R_w - R_l), averaged."""
    return -F.logsigmoid(score_chosen - score_rejected).mean()


def wm_loss(
    score_t_sigma: torch.Tensor,
    score_t_plain: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge-squared on trigger margin.

    Want: R(T(x), σ(y)) - R(T(x), y) >= delta on average.
    Returns (loss, observed_margin).
    """
    margin = score_t_sigma - score_t_plain
    hinge = F.relu(delta - margin)
    return (hinge ** 2).mean(), margin.detach().mean()


def composite_loss(
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
    score_t_sigma: torch.Tensor,
    score_t_plain: torch.Tensor,
    delta: float,
    lam_wm: float,
) -> TrainStepOut:
    bt = bt_loss(score_chosen, score_rejected)
    wm, wm_margin = wm_loss(score_t_sigma, score_t_plain, delta)
    total = bt + lam_wm * wm
    return TrainStepOut(loss=total, bt_loss=bt.detach(), wm_loss=wm.detach(), wm_margin=wm_margin)
