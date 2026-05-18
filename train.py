"""
train.py — Two-phase encoder training

Phase 1 (pre-train):  SyntheticRVDataset — unlimited labelled examples,
                       supervised MSE on normalised theta.
Phase 2 (fine-tune):  RVDataset('train') — 437 real systems,
                       supervised MSE + optional reconstruction loss.

Reconstruction loss: ‖rv_norm − KeplerDecoder(un_norm(θ̂), t, …)‖²
  The KeplerDecoder refits T_peri analytically per forward pass (Option A).
  Requires t_span_days and rv_std_ms from the info dict; both datasets
  now expose these as canonical keys.

Loss weights are controlled by --lambda-rec (default 0 in pre-train,
0.1 in fine-tune); the supervised term always has weight 1.

Usage
-----
    python train.py                                  # default schedule
    python train.py --pretrain-epochs 50 --finetune-epochs 30
    python train.py --resume checkpoints/last.pt
    python train.py --finetune-only --resume checkpoints/pretrain_best.pt
    python train.py --no-pretrain --finetune-only    # real data only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.encoder import RVEncoder, encoder_loss, normalise_theta, un_normalise_theta
from models.kepler_torch import KeplerDecoder
from preprocess import RVDataset, THETA_NAMES
from synthetic_dataset import SyntheticRVDataset

CKPT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_rv(batch):
    """
    Stack (x, theta, info) tuples into tensors.

    Returns
    -------
    x      : (B, 4, 256)
    theta  : (B, 5)
    t_span : (B,)  days
    t_min  : (B,)  days
    rv_std : (B,)  m/s
    valid  : (B,)  bool
    """
    xs, thetas, infos = zip(*batch)
    x      = torch.stack(xs).float()
    theta  = torch.stack(thetas).float()
    t_span = torch.tensor([i.get("t_span_days", 1000.0) for i in infos],
                           dtype=torch.float32)
    t_min  = torch.tensor([i.get("t_min_days",  0.0)    for i in infos],
                           dtype=torch.float32)
    rv_std = torch.tensor([i.get("rv_std_ms",   1.0)    for i in infos],
                           dtype=torch.float32)
    valid  = torch.tensor([i.get("valid", True) for i in infos],
                           dtype=torch.bool)
    return x, theta, t_span, t_min, rv_std, valid


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    theta_norm: torch.Tensor,
    x: torch.Tensor,
    t_span: torch.Tensor,
    t_min: torch.Tensor,
    rv_std: torch.Tensor,
    stats: dict,
    decoder: KeplerDecoder,
) -> torch.Tensor:
    """
    ‖rv_norm − KeplerDecoder(un_norm(θ̂), t, …)‖² averaged over real obs.

    Uses the mask channel (x[:, 3, :]) to exclude padding from the mean.
    """
    theta_phys = un_normalise_theta(theta_norm, stats)   # (B, 5) physical
    t_norm  = x[:, 0, :]    # (B, N)
    rv_obs  = x[:, 1, :]    # (B, N) normalised
    mask    = x[:, 3, :]    # (B, N)

    rv_pred = decoder(theta_phys, t_norm, t_span, t_min, rv_obs, rv_std, mask)

    diff = (rv_obs - rv_pred) ** 2 * mask
    n    = mask.sum(dim=1).clamp(min=1.0)
    return (diff.sum(dim=1) / n).mean()


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def _one_epoch(
    loader: DataLoader,
    encoder: RVEncoder,
    decoder: KeplerDecoder,
    optimizer: torch.optim.Optimizer,
    stats: dict,
    lambda_rec: float,
    device: torch.device,
    train: bool = True,
) -> dict[str, float]:

    encoder.train(train)
    total_loss = sup_loss = rec_loss = 0.0
    n_valid = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, theta_target, t_span, t_min, rv_std, valid in loader:
            x, theta_target = x.to(device), theta_target.to(device)
            t_span, t_min, rv_std = t_span.to(device), t_min.to(device), rv_std.to(device)
            valid = valid.to(device)

            # Skip batches where all samples are invalid (shouldn't happen)
            if not valid.any():
                continue

            theta_pred = encoder(x)

            losses = encoder_loss(theta_pred[valid], theta_target[valid])
            loss   = losses["total"]

            rec = torch.tensor(0.0, device=device)
            if lambda_rec > 0:
                rec  = reconstruction_loss(
                    theta_pred[valid], x[valid], t_span[valid],
                    t_min[valid], rv_std[valid], stats, decoder,
                )
                loss = loss + lambda_rec * rec

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()

            bs = int(valid.sum().item())
            total_loss += loss.item() * bs
            sup_loss   += losses["total"].item() * bs
            rec_loss   += rec.item() * bs
            n_valid    += bs

    n = max(n_valid, 1)
    return {"total": total_loss / n, "sup": sup_loss / n, "rec": rec_loss / n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    pretrain_epochs: int = 30,
    finetune_epochs: int = 20,
    pretrain_n: int = 50_000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_rec_finetune: float = 0.1,
    resume: Path | None = None,
    finetune_only: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    device_str: str = "cpu",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device  = torch.device(device_str)
    stats   = json.loads(Path("data/dataset_stats.json").read_text())
    encoder = RVEncoder().to(device)
    decoder = KeplerDecoder().to(device)

    CKPT_DIR.mkdir(exist_ok=True)
    start_epoch = 0

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        encoder.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {resume}  (epoch {start_epoch})")

    # ---- Phase 1: pre-train on synthetic ----
    if not finetune_only and pretrain_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: pre-train  ({pretrain_epochs} epochs, "
              f"{pretrain_n:,} synthetic samples/epoch)")
        print(f"{'='*60}")

        syn_ds = SyntheticRVDataset(n_samples=pretrain_n, seed=seed, stats=stats)
        syn_loader = DataLoader(syn_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, collate_fn=collate_rv)

        opt = torch.optim.AdamW(encoder.parameters(), lr=lr,
                                weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pretrain_epochs)

        best_sup = math.inf
        t0 = time.perf_counter()

        for ep in range(pretrain_epochs):
            m = _one_epoch(syn_loader, encoder, decoder, opt, stats,
                           lambda_rec=0.0, device=device, train=True)
            sched.step()

            if (ep + 1) % max(1, pretrain_epochs // 10) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  pretrain ep {ep+1:4d}/{pretrain_epochs}  "
                      f"loss={m['total']:.4f}  sup={m['sup']:.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  "
                      f"t={elapsed:.0f}s")

            if m["sup"] < best_sup:
                best_sup = m["sup"]
                _save_ckpt(encoder, opt, ep + 1, m, CKPT_DIR / "pretrain_best.pt")

        _save_ckpt(encoder, opt, pretrain_epochs, m, CKPT_DIR / "pretrain_last.pt")
        print(f"Pre-training done.  Best supervised loss: {best_sup:.4f}")

    # ---- Phase 2: fine-tune on real data ----
    if finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: fine-tune  ({finetune_epochs} epochs, λ_rec={lambda_rec_finetune})")
        print(f"{'='*60}")

        train_ds = RVDataset("train")
        val_ds   = RVDataset("val")
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                                  shuffle=True,  num_workers=num_workers,
                                  collate_fn=collate_rv)
        val_loader   = DataLoader(val_ds,   batch_size=min(batch_size, len(val_ds)),
                                  shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_rv)

        opt = torch.optim.AdamW(encoder.parameters(), lr=lr / 5,
                                weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=finetune_epochs)

        best_val = math.inf
        t0 = time.perf_counter()

        for ep in range(finetune_epochs):
            tr = _one_epoch(train_loader, encoder, decoder, opt, stats,
                            lambda_rec=lambda_rec_finetune, device=device, train=True)
            vl = _one_epoch(val_loader,   encoder, decoder, opt, stats,
                            lambda_rec=lambda_rec_finetune, device=device, train=False)
            sched.step()

            elapsed = time.perf_counter() - t0
            print(f"  finetune ep {ep+1:4d}/{finetune_epochs}  "
                  f"train={tr['total']:.4f} (sup={tr['sup']:.4f} rec={tr['rec']:.4f})  "
                  f"val={vl['total']:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  t={elapsed:.0f}s")

            if vl["total"] < best_val:
                best_val = vl["total"]
                _save_ckpt(encoder, opt, ep + 1, vl, CKPT_DIR / "finetune_best.pt")

        _save_ckpt(encoder, opt, finetune_epochs, vl, CKPT_DIR / "finetune_last.pt")
        print(f"Fine-tuning done.  Best val loss: {best_val:.4f}")

    print("\nTraining complete.")
    print(f"Checkpoints saved to {CKPT_DIR}/")


def _save_ckpt(encoder: RVEncoder, opt, epoch: int, metrics: dict,
               path: Path) -> None:
    torch.save({
        "model":   encoder.state_dict(),
        "optim":   opt.state_dict(),
        "epoch":   epoch,
        "metrics": metrics,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--pretrain-n",      type=int, default=50_000,
                   help="Synthetic samples per pre-training epoch")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lambda-rec",      type=float, default=0.1,
                   help="Reconstruction loss weight during fine-tuning")
    p.add_argument("--resume",          type=Path, default=None)
    p.add_argument("--finetune-only",   action="store_true")
    p.add_argument("--no-pretrain",     action="store_true")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--workers",         type=int, default=0)
    p.add_argument("--device",          type=str, default="cpu",
                   help="'cpu', 'cuda', or 'mps'")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        pretrain_epochs     = 0 if args.no_pretrain else args.pretrain_epochs,
        finetune_epochs     = args.finetune_epochs,
        pretrain_n          = args.pretrain_n,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        lambda_rec_finetune = args.lambda_rec,
        resume              = args.resume,
        finetune_only       = args.finetune_only,
        seed                = args.seed,
        num_workers         = args.workers,
        device_str          = args.device,
    )
