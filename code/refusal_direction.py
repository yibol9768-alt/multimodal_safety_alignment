"""Extract refusal directions from an aligned LLM (Llama-2-chat).

Supports:
- Rank-1 (Arditi et al. 2024): r_L = mean(h_harm) - mean(h_benign) at layer L.
- Rank-k (our extension, Bet 4): SVD on the per-prompt contrast matrix.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG


@dataclass
class DirectionSet:
    layer_to_rank1: dict[int, torch.Tensor]          # [d]
    layer_to_subspace: dict[int, torch.Tensor]       # [k, d]
    hidden_dim: int
    aligned_model_id: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "rank1": {int(k): v.cpu() for k, v in self.layer_to_rank1.items()},
                "subspace": {int(k): v.cpu() for k, v in self.layer_to_subspace.items()},
                "hidden_dim": self.hidden_dim,
                "aligned_model_id": self.aligned_model_id,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DirectionSet":
        blob = torch.load(path, map_location="cpu")
        return cls(
            layer_to_rank1=blob["rank1"],
            layer_to_subspace=blob["subspace"],
            hidden_dim=blob["hidden_dim"],
            aligned_model_id=blob["aligned_model_id"],
        )


def _wrap_chat(tokenizer, prompt: str) -> str:
    """Wrap a raw prompt in Llama-2-chat format so we actually elicit the
    refusal circuit (Arditi 2024). Without the [INST] scaffold the extracted
    direction is a generic sentence-contrast vector, not a refusal axis.

    Tries the tokenizer's own chat_template first; falls back to the
    canonical Llama-2 [INST]...[/INST] format if none is registered.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return f"<s>[INST] {prompt.strip()} [/INST]"


def _last_token_hidden(model, tokenizer, prompts: list[str], layers: tuple[int, ...], dtype, device) -> dict[int, torch.Tensor]:
    """Return {layer: [n_prompts, d]} hidden states at last token position.

    Each prompt is wrapped in the Llama-2-chat template before tokenization
    so that the last-token hidden state sits at the point where the model
    would decide whether to refuse.
    """
    storage: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def mk_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [B, T, d]
            last = h[:, -1, :].detach().to(torch.float32).cpu()
            storage[layer_idx].append(last)
        return hook

    handles = []
    decoder_layers = model.model.layers  # Llama structure
    for L in layers:
        handles.append(decoder_layers[L].register_forward_hook(mk_hook(L)))

    try:
        model.eval()
        for p in prompts:
            wrapped = _wrap_chat(tokenizer, p)
            inputs = tokenizer(wrapped, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {L: torch.cat(storage[L], dim=0) for L in layers}


def extract_directions(
    harmful: list[str],
    benign: list[str],
    aligned_model_id: str | None = None,
    layers: tuple[int, ...] | None = None,
    rank_k: int | None = None,
    device: str | None = None,
    dtype_str: str | None = None,
) -> DirectionSet:
    aligned_model_id = aligned_model_id or CONFIG.models.llama_chat
    layers = layers or CONFIG.direction.layers
    rank_k = rank_k or CONFIG.direction.rank_k
    device = device or CONFIG.device
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str or CONFIG.dtype]

    tok = AutoTokenizer.from_pretrained(aligned_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(aligned_model_id, torch_dtype=dtype, device_map=device)

    h_harm = _last_token_hidden(model, tok, harmful, layers, dtype, device)
    h_ben = _last_token_hidden(model, tok, benign, layers, dtype, device)

    rank1: dict[int, torch.Tensor] = {}
    sub: dict[int, torch.Tensor] = {}
    hidden_dim = h_harm[layers[0]].shape[-1]

    for L in layers:
        mu_h = h_harm[L].mean(dim=0)
        mu_b = h_ben[L].mean(dim=0)
        r = mu_h - mu_b
        if CONFIG.direction.normalize:
            r = r / (r.norm() + 1e-8)
        rank1[L] = r.float()

        # Bet 4: rank-k subspace via SVD on the per-prompt contrast
        # Stack [h_harm_i - mu_b] into a matrix, take top-k right singular vectors.
        contrast = h_harm[L].float() - mu_b.float()
        if rank_k > 1:
            U, S, Vh = torch.linalg.svd(contrast, full_matrices=False)
            sub[L] = Vh[:rank_k]  # [k, d]
        else:
            sub[L] = r.unsqueeze(0).float()

    # free GPU memory
    del model
    torch.cuda.empty_cache()

    return DirectionSet(
        layer_to_rank1=rank1,
        layer_to_subspace=sub,
        hidden_dim=hidden_dim,
        aligned_model_id=aligned_model_id,
    )
