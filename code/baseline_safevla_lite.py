"""SafeVLA-lite — LoRA-SFT on a small harmful+benign set.

Positioned as a training-based defense baseline. Full SafeVLA (arxiv 2503.03480)
uses constrained RL which we can't reproduce in 33 days. SafeVLA-lite is a fair
stand-in that captures the "fine-tune on safety data" class of methods.

Dataset: 1k harmful (task -> "STOP: refuse, output zero action") + 1k benign
(task -> normal action trajectory).

Trains a LoRA adapter on OpenVLA's LLM backbone for 1 epoch; evaluated in
the main table as "SafeVLA-lite".
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import CONFIG
from data_utils import vla_prompt_pair
from openvla_utils import load_openvla, find_llm_layers, OPENVLA_ACTION_START_ID


@dataclass
class SafeVLALiteConfig:
    n_harm: int = 1000
    n_benign: int = 1000
    epochs: int = 1
    lr: float = 1e-4
    batch_size: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    output_dir: Path = Path("/root/rdt/checkpoints/safevla_lite")


class SafetyTriples(Dataset):
    """Each sample: (prompt, image, target_action_tokens)."""

    STOP_ACTION_TOKEN = OPENVLA_ACTION_START_ID + 128  # bin 128 = zero delta

    def __init__(self, processor, pair, n_harm: int, n_benign: int, seed: int = 20260422):
        rng = random.Random(seed)
        harm = [(p, True) for p in pair.harmful[:n_harm]]
        ben = [(p, False) for p in pair.benign[:n_benign]]
        self.data = harm + ben
        rng.shuffle(self.data)
        self.processor = processor
        self.dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, is_harm = self.data[idx]
        inputs = self.processor(prompt, self.dummy_img)
        # Target: 7 zero-motion action tokens for harmful; for benign we leave
        # the model free (mask loss on those positions in collate).
        target_ids = [self.STOP_ACTION_TOKEN] * 7 if is_harm else [-100] * 7
        return {
            "input_ids": inputs["input_ids"][0],
            "pixel_values": inputs["pixel_values"][0],
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "is_harm": is_harm,
        }


def collate(batch, pad_id: int):
    max_in = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_in), dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
        attn[i, : len(b["input_ids"])] = 1
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])  # [B, 7]
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "pixel_values": pixel_values,
        "labels": labels,
    }


def train(cfg: SafeVLALiteConfig):
    model, processor = load_openvla()
    pair = vla_prompt_pair(n=max(cfg.n_harm, cfg.n_benign))
    ds = SafetyTriples(processor, pair, cfg.n_harm, cfg.n_benign)
    pad = processor.tokenizer.pad_token_id or 0
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad))

    # attach LoRA to the Llama decoder's q_proj/v_proj
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            # Use only the last 7 token logits for action loss.
            logits = out.logits[:, -7:, :]
            labels = batch["labels"]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[SafeVLA-lite] epoch={epoch} step={step} loss={loss.item():.4f}")
            step += 1

    model.save_pretrained(cfg.output_dir)
    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))
    print(f"[SafeVLA-lite] done. adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_harm", type=int, default=1000)
    ap.add_argument("--n_benign", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", type=Path, default=Path("/root/rdt/checkpoints/safevla_lite"))
    args = ap.parse_args()
    cfg = SafeVLALiteConfig(**vars(args))
    train(cfg)
