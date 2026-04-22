"""Collect per-layer hidden states at specific token positions in OpenVLA.

We care about three families of positions:
- text tokens (instruction tokens before <image> split)
- image tokens (256 patches from SigLIP+DINOv2 post-projection)
- action tokens (generated autoregressively, 7 per step)

For the sanity check we mostly need "last text token" and "first action token"
because those are where the LLM decides what to emit next.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from openvla_utils import find_llm_layers, action_token_mask


@dataclass
class HiddenRecord:
    """Per-prompt hidden states."""
    layer_to_hidden: dict[int, torch.Tensor]  # {layer: [d]}
    position: str                              # "last_text" | "first_action" | ...
    prompt: str
    is_harmful: bool


def collect_hidden_at_position(
    model,
    processor,
    prompts: list[str],
    images: list[Image.Image] | None,
    layers: tuple[int, ...],
    position: str = "last_text",
    max_new_tokens: int = 8,
    device: str = "cuda",
) -> list[dict[int, torch.Tensor]]:
    """Run OpenVLA forward/generate, collect hidden states at `position` per layer.

    Returns a list of {layer: hidden[d]} — one dict per prompt.
    """
    decoder_layers = find_llm_layers(model)
    per_layer_storage: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    current_last_layer_h: dict[int, torch.Tensor] = {}

    def mk_hook(layer_idx: int):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            current_last_layer_h[layer_idx] = h.detach().to(torch.float32).cpu()
        return hook

    handles = [decoder_layers[L].register_forward_hook(mk_hook(L)) for L in layers]
    results: list[dict[int, torch.Tensor]] = []

    try:
        model.eval()
        for i, prompt in enumerate(prompts):
            img = images[i] if images else _dummy_image()
            inputs = processor(prompt, img).to(device, dtype=next(model.parameters()).dtype)

            if position == "last_text":
                with torch.no_grad():
                    model(**inputs, use_cache=False)
                rec = {L: current_last_layer_h[L][0, -1, :].clone() for L in layers}
            elif position == "first_action":
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        output_hidden_states=False,
                        return_dict_in_generate=False,
                    )
                # During autoregressive generation, the hook fires per step.
                # `current_last_layer_h` now holds the LAST step's hidden state;
                # for true "first action token" we need step-0. Use forward pass
                # on the generated prefix including the first action token.
                prefix = torch.cat(
                    [inputs["input_ids"], generated[:, inputs["input_ids"].shape[1]:inputs["input_ids"].shape[1]+1]],
                    dim=1,
                )
                # re-run once with the prefix to populate hook at first-action position
                with torch.no_grad():
                    model(input_ids=prefix, pixel_values=inputs.get("pixel_values"), use_cache=False)
                rec = {L: current_last_layer_h[L][0, -1, :].clone() for L in layers}
            else:
                raise ValueError(f"Unknown position: {position}")
            results.append(rec)
    finally:
        for h in handles:
            h.remove()

    return results


def _dummy_image(size: int = 224) -> Image.Image:
    """Neutral gray — for text-only probing without visual grounding."""
    return Image.new("RGB", (size, size), color=(128, 128, 128))
