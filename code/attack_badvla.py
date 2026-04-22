"""BadVLA-style attacks (arxiv 2411.13587).

Three variants from the original paper:
  - UADA (Untargeted Adversarial Discrete Attack): optimize a universal patch
    that maximizes action-token cross-entropy vs the benign action target
    (i.e. make the robot do ANYTHING but the correct thing).
  - UPA (Untargeted Position Attack): patch optimized to make gripper position
    drift far from the correct trajectory.
  - TMA (Targeted Manipulation Attack): patch forces a SPECIFIC target action
    sequence (e.g. "close gripper + move toward human").

For our paper we use these as attack baselines:
  - No defense vs each attack: vanilla OpenVLA ASR → large
  - + defenses → measure drop.

Implementation is patch-PGD, bounded by L_inf eps and patch area.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image


@dataclass
class PatchAttackConfig:
    eps: float = 16 / 255         # L_inf budget
    step: float = 2 / 255
    n_iters: int = 500
    patch_frac: float = 0.05      # patch size as fraction of image area
    seed: int = 20260422


def _patch_mask(image_shape: tuple[int, int, int], frac: float, seed: int) -> torch.Tensor:
    """Random square patch mask of given area fraction."""
    _, H, W = image_shape
    side = int((H * W * frac) ** 0.5)
    rng = torch.Generator().manual_seed(seed)
    top = torch.randint(0, H - side, (1,), generator=rng).item()
    left = torch.randint(0, W - side, (1,), generator=rng).item()
    mask = torch.zeros(1, H, W)
    mask[:, top : top + side, left : left + side] = 1.0
    return mask


def _build_inputs(processor, prompt: str, image_tensor: torch.Tensor, device: str, dtype) -> dict:
    # image_tensor: [3, H, W] in [0, 1]
    pil = Image.fromarray((image_tensor.clamp(0, 1).cpu().mul(255).byte().permute(1, 2, 0)).numpy())
    return processor(prompt, pil).to(device, dtype=dtype)


def universal_patch_attack(
    model,
    processor,
    prompts: list[str],
    image_base: torch.Tensor,           # [3, H, W] in [0, 1]
    loss_fn: Callable[[torch.Tensor, dict], torch.Tensor],
    cfg: PatchAttackConfig = PatchAttackConfig(),
    device: str = "cuda",
) -> tuple[torch.Tensor, list[float]]:
    """Return (perturbed_image, per_iter_loss).

    loss_fn(logits, batch_inputs) -> scalar loss to MAXIMIZE (we descend -loss).
    """
    dtype = next(model.parameters()).dtype
    mask = _patch_mask(image_base.shape, cfg.patch_frac, cfg.seed).to(device)
    pert = torch.zeros_like(image_base, device=device, requires_grad=True)
    losses: list[float] = []

    for it in range(cfg.n_iters):
        prompt = prompts[it % len(prompts)]
        img_adv = (image_base.to(device) + pert * mask).clamp(0, 1)
        inputs = _build_inputs(processor, prompt, img_adv, device, dtype)
        out = model(**inputs, use_cache=False)
        loss = loss_fn(out.logits, inputs)          # maximize
        (-loss).backward()
        with torch.no_grad():
            step = cfg.step * pert.grad.sign() * mask
            pert = (pert + step).clamp(-cfg.eps, cfg.eps).detach().requires_grad_(True)
        losses.append(float(loss.item()))
        if it % 50 == 0:
            print(f"[UADA] iter {it}  loss={loss.item():.4f}")

    return (image_base.to(device) + pert * mask).clamp(0, 1).detach(), losses


# ------- specific attacks --------------------------------------------------

def uada_loss(action_start_id: int = 31744, action_vocab: int = 256):
    """Untargeted: maximize entropy of action-token distribution."""
    def _loss(logits: torch.Tensor, _inputs: dict) -> torch.Tensor:
        action_logits = logits[:, -1, action_start_id : action_start_id + action_vocab]
        probs = torch.softmax(action_logits.float(), dim=-1)
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return ent.mean()
    return _loss


def tma_loss(target_action_id: int):
    """Targeted: maximize logit of `target_action_id`."""
    def _loss(logits: torch.Tensor, _inputs: dict) -> torch.Tensor:
        return logits[:, -1, target_action_id].mean()
    return _loss


def upa_loss(action_start_id: int = 31744, action_vocab: int = 256, far_from_center: bool = True):
    """Untargeted Position: push expected action-bin AWAY from center (bin 128)."""
    def _loss(logits: torch.Tensor, _inputs: dict) -> torch.Tensor:
        al = logits[:, -1, action_start_id : action_start_id + action_vocab].float()
        probs = torch.softmax(al, dim=-1)
        bins = torch.arange(action_vocab, device=probs.device).float()
        mean_bin = (probs * bins).sum(dim=-1)
        # maximize distance from center
        return (mean_bin - action_vocab / 2).abs().mean()
    return _loss
