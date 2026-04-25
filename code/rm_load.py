"""Load Llama / Qwen RM with LoRA + 4-bit quant. Single function `load_rm`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import MODELS, RMTrainConfig


# LoRA target modules per backbone family (Llama / Qwen)
_LORA_TARGETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _family_of(model_id: str) -> str:
    if "llama" in model_id.lower():
        return "llama"
    if "qwen" in model_id.lower():
        return "qwen"
    raise ValueError(f"don't know LoRA targets for {model_id!r}")


def load_rm(cfg: RMTrainConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load tokenizer + LoRA-wrapped scalar-head RM. 4-bit base if cfg.load_in_4bit."""
    model_id = MODELS[cfg.backbone]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = None
    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=_LORA_TARGETS[_family_of(model_id)],
        modules_to_save=["score"],  # the regression head (not in LoRA)
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tok


def render_prompt_response(tok: PreTrainedTokenizer, prompt: str, response: str) -> str:
    """Apply chat template to make a (user, assistant) conversation string."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def score_pair(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompts: list[str],
    responses: list[str],
    max_len: int = 2048,
) -> torch.Tensor:
    """Forward a batch of (prompt, response) through the RM, return scalar scores."""
    rendered = [render_prompt_response(tok, p, r) for p, r in zip(prompts, responses)]
    enc = tok(rendered, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    # SequenceClassification with num_labels=1 returns logits shape (B, 1)
    return out.logits.squeeze(-1)
