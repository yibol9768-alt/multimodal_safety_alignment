"""Train the Gated-RDT+ safety probe.

Protocol:
  1. Load OpenVLA + processor.
  2. Build a paired harmful/benign dataset via `vla_prompt_pair(n=args.n)`.
  3. For each (prompt, image) run a forward pass and capture post-projector
     pooled embedding via `extract_pooled_embeddings()`.
  4. Train a 2-layer MLP probe with binary cross-entropy on the cached
     embeddings. 80/20 train/test split (held-out for reporting AUC).
  5. Save probe to `logs/gate/probe.pt` and print train/test AUC.

Usage:
  HF_HOME=/root/models python scripts/09_train_gate.py \
      --out /root/rdt/logs/gate --n 512 --epochs 3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from openvla_utils import load_openvla
from data_utils import vla_prompt_pair
from gated_rdt import SafetyProbe, save_probe, extract_pooled_embeddings
from hidden_collect import _dummy_image


def _prepare_images(n: int) -> list[Image.Image]:
    """One dummy image per prompt; good enough for text-dominated probe.
    For v2 rollouts we can swap in LIBERO scene renders for richer training.
    """
    return [_dummy_image() for _ in range(n)]


def train_probe(
    X: torch.Tensor,           # [N, d]
    y: torch.Tensor,           # [N] in {0, 1}
    hidden_dim: int = 4096,
    mlp_hidden: int = 128,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 20260422,
    device: str = "cuda",
):
    """Train SafetyProbe with 80/20 split; return (probe, train_auc, val_auc)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(val_frac * len(X))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    probe = SafetyProbe(hidden_dim=hidden_dim, mlp_hidden=mlp_hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(X_tr.to(device), y_tr.to(device).float()),
        batch_size=batch_size, shuffle=True, generator=g,
    )
    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            logit = probe(xb)
            loss = loss_fn(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= len(X_tr)

        # val AUC
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_va.to(device)).cpu().numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y_va.cpu().numpy(), val_probs)
        train_auc_val = None
        with torch.no_grad():
            tr_logits = probe(X_tr.to(device)).cpu().numpy()
            train_auc_val = roc_auc_score(y_tr.cpu().numpy(), tr_logits)
        print(f"  epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}  "
              f"train_AUC={train_auc_val:.3f}  val_AUC={val_auc:.3f}")

    return probe, float(train_auc_val), float(val_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/rdt/logs/gate"))
    ap.add_argument("--n", type=int, default=512,
                    help="prompts per class (default 512 harmful + 512 benign)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128,
                    help="probe MLP hidden dim")
    ap.add_argument("--seed", type=int, default=20260422)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[gate] loading OpenVLA …")
    t0 = time.time()
    model, processor = load_openvla()
    print(f"[gate] loaded in {time.time()-t0:.1f}s")

    print(f"[gate] building paired (n={args.n}) dataset …")
    pair = vla_prompt_pair(n=args.n)
    images = _prepare_images(max(len(pair.harmful), len(pair.benign)))

    # Use dummy image for text-dominated probe; upgrade to LIBERO renders later
    print(f"[gate] extracting harmful pooled embeddings …")
    t1 = time.time()
    Xh = extract_pooled_embeddings(model, processor, pair.harmful, images[:len(pair.harmful)])
    print(f"[gate]   {Xh.shape} in {time.time()-t1:.1f}s")

    print(f"[gate] extracting benign pooled embeddings …")
    t2 = time.time()
    Xb = extract_pooled_embeddings(model, processor, pair.benign, images[:len(pair.benign)])
    print(f"[gate]   {Xb.shape} in {time.time()-t2:.1f}s")

    X = torch.cat([Xh, Xb], dim=0)
    y = torch.cat([torch.ones(Xh.shape[0]), torch.zeros(Xb.shape[0])])
    print(f"[gate] combined dataset: X={X.shape}, y={y.shape}")

    print(f"[gate] training probe …")
    probe, train_auc, val_auc = train_probe(
        X, y, hidden_dim=X.shape[1], mlp_hidden=args.hidden,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )

    probe_path = args.out / "probe.pt"
    save_probe(probe, probe_path)
    print(f"[gate] saved probe to {probe_path}")

    (args.out / "training_log.json").write_text(json.dumps({
        "n_harmful": int(Xh.shape[0]),
        "n_benign": int(Xb.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "mlp_hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_auc": train_auc,
        "val_auc": val_auc,
    }, indent=2))
    print(f"[gate] train AUC = {train_auc:.3f},  val AUC = {val_auc:.3f}")


if __name__ == "__main__":
    main()
