"""
train.py — Two-phase encoder training

Phase 1 (pre-train):  SyntheticRVDataset — unlimited labelled examples,
                       supervised MSE on normalised theta.
Phase 2 (fine-tune):  RVDataset('train') — 437 real systems,
                       supervised MSE + optional reconstruction loss.

Reconstruction loss: ‖rv_norm − KeplerDecoder(un_norm(θ̂), t, …)‖²
  The KeplerDecoder refits T_peri analytically per forward pass (Option A).

Loss schedule
-------------
  Pre-train : supervised MSE only (--lambda-rec is ignored)
  Fine-tune : supervised MSE  +  λ_rec × reconstruction loss (default λ=0.1)

Optimisation
------------
  AdamW with linear warmup (10% of total steps) → cosine decay to 0.
  Gradient clipping at max-norm 5.0 (loose enough not to interfere with
  valid gradients yet prevents NaN cascades from rare bad batches).

Usage
-----
    python train.py                                  # default schedule
    python train.py --pretrain-epochs 300 --finetune-epochs 100
    python train.py --resume checkpoints/last.pt
    python train.py --finetune-only --resume checkpoints/pretrain_best.pt
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
from torch.utils.data import DataLoader, RandomSampler

from models.encoder import (RVEncoder, encoder_loss, normalise_theta,
                            un_normalise_theta, build_encoder, ENCODER_REGISTRY)
from models.kepler_torch import KeplerDecoder
from preprocess import RVDataset, THETA_NAMES
from synthetic_dataset import SyntheticRVDataset, PregenSyntheticDataset, generate_cache

CKPT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_rv(batch):
    """
    Stack (x, lsp, theta, info) tuples into tensors.

    Returns
    -------
    x, lsp, theta : stacked float32 tensors
    t_span, t_min, rv_std : (B,) scalars extracted from info dicts
    valid : (B,) bool
    """
    xs, lsps, thetas, infos = zip(*batch)
    x     = torch.stack(xs).float()
    lsp   = torch.stack(lsps).float()
    theta = torch.stack(thetas).float()
    t_span  = torch.tensor([i.get("t_span_days", 1000.0) for i in infos], dtype=torch.float32)
    t_min   = torch.tensor([i.get("t_min_days",  0.0)    for i in infos], dtype=torch.float32)
    rv_std  = torch.tensor([i.get("rv_std_ms",   1.0)    for i in infos], dtype=torch.float32)
    valid   = torch.tensor([i.get("valid",   True) for i in infos], dtype=torch.bool)
    has_ecc = torch.tensor([i.get("has_ecc", True) for i in infos], dtype=torch.bool)
    return x, lsp, theta, t_span, t_min, rv_std, valid, has_ecc


# ---------------------------------------------------------------------------
# Warmup + cosine schedule
# ---------------------------------------------------------------------------

class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup then cosine decay to 0 (no restarts)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int) -> None:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


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

    rv_norm and the decoder output are both in units of (rv − median)/std,
    so the loss is dimensionless and directly comparable to the supervised MSE.
    """
    theta_phys = un_normalise_theta(theta_norm, stats)
    t_norm = x[:, 0, :]
    rv_obs = x[:, 1, :]
    mask   = x[:, 3, :]

    rv_pred = decoder(theta_phys, t_norm, t_span, t_min, rv_obs, rv_std, mask)

    diff = (rv_obs - rv_pred) ** 2 * mask
    n    = mask.sum(dim=1).clamp(min=1.0)
    return (diff.sum(dim=1) / n).mean()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _one_epoch(
    loader: DataLoader,
    encoder: RVEncoder,
    decoder: KeplerDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    stats: dict,
    lambda_rec: float,
    device: torch.device,
    train: bool = True,
) -> dict[str, float]:

    encoder.train(train)
    tot = sup = rec = 0.0
    n_valid = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, lsp, theta_target, t_span, t_min, rv_std, valid, has_ecc in loader:
            x, lsp = x.to(device), lsp.to(device)
            theta_target = theta_target.to(device)
            t_span, t_min, rv_std = t_span.to(device), t_min.to(device), rv_std.to(device)
            valid   = valid.to(device)
            has_ecc = has_ecc.to(device)

            if not valid.any():
                continue

            theta_pred = encoder(x[valid], lsp[valid])

            losses = encoder_loss(theta_pred, theta_target[valid], stats=stats,
                                  has_ecc=has_ecc[valid])
            loss   = losses["total"]

            rec_val = torch.tensor(0.0, device=device)
            if lambda_rec > 0:
                rec_val = reconstruction_loss(
                    theta_pred, x[valid], t_span[valid],
                    t_min[valid], rv_std[valid], stats, decoder,
                )
                loss = loss + lambda_rec * rec_val

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

            bs    = int(valid.sum().item())
            tot  += loss.item() * bs
            sup  += losses["total"].item() * bs
            rec  += rec_val.item() * bs
            n_valid += bs

    n = max(n_valid, 1)
    return {"total": tot / n, "sup": sup / n, "rec": rec / n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    pretrain_epochs: int = 300,
    finetune_epochs: int = 100,
    pretrain_n: int = 50_000,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    lambda_rec_finetune: float = 0.1,
    resume: Path | None = None,
    finetune_only: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    device_str: str = "cpu",
    pretrain_cache: Path | None = None,
    arch: str = "resnet",
    single_planet: bool = True,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device  = torch.device(device_str)
    stats   = json.loads(Path("data/dataset_stats.json").read_text())
    decoder = KeplerDecoder().to(device)
    CKPT_DIR.mkdir(exist_ok=True)

    # Arch from checkpoint overrides --arch when resuming, to prevent mismatch
    if resume and Path(resume).exists():
        ckpt      = torch.load(resume, map_location=device)
        ckpt_arch = ckpt.get("arch", "resnet")
        if ckpt_arch != arch:
            print(f"Note: checkpoint arch={ckpt_arch!r} overrides --arch={arch!r}")
            arch = ckpt_arch
        encoder = build_encoder(arch).to(device)
        encoder.load_state_dict(ckpt["model"])
        print(f"Resumed {arch} from {resume}  (epoch {ckpt.get('epoch', '?')})")
    else:
        encoder = build_encoder(arch).to(device)

    print(f"Encoder: {arch}  ({encoder.n_params:,} parameters)")

    # ---- Phase 1: pre-train on synthetic ----
    if not finetune_only and pretrain_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: pre-train  ({pretrain_epochs} epochs, "
              f"{pretrain_n:,} synthetic samples/epoch)")
        print(f"{'='*60}")

        if pretrain_cache is not None:
            if not Path(pretrain_cache).exists():
                print(f"Cache not found — generating {pretrain_n:,} samples → {pretrain_cache}")
                generate_cache(pretrain_n, pretrain_cache, seed=seed, stats=stats)
            print(f"Loading pretrain cache from {pretrain_cache}")
            syn_ds = PregenSyntheticDataset(pretrain_cache)
            # If cache is larger than pretrain_n, sample without replacement each
            # epoch so each sample is seen ~(pretrain_n/cache_size × n_epochs)
            # times rather than n_epochs times.  This prevents effective overfitting
            # to a small fixed training set while keeping per-epoch cost constant.
            if len(syn_ds) > pretrain_n:
                sampler    = RandomSampler(syn_ds, replacement=False,
                                          num_samples=pretrain_n)
                syn_loader = DataLoader(syn_ds, batch_size=batch_size, sampler=sampler,
                                        num_workers=num_workers, collate_fn=collate_rv)
            else:
                syn_loader = DataLoader(syn_ds, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, collate_fn=collate_rv)
        else:
            syn_ds     = SyntheticRVDataset(n_samples=pretrain_n, seed=seed, stats=stats)
            syn_loader = DataLoader(syn_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, collate_fn=collate_rv)

        n_per_epoch     = min(pretrain_n, len(syn_ds))
        steps_per_epoch = math.ceil(n_per_epoch / batch_size)
        total_steps     = pretrain_epochs * steps_per_epoch
        warmup_steps    = max(1, total_steps // 10)

        opt   = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        sched = WarmupCosineSchedule(opt, warmup_steps, total_steps)

        best_sup = float("inf")
        log_every = max(1, pretrain_epochs // 20)
        t0 = time.perf_counter()

        for ep in range(pretrain_epochs):
            m = _one_epoch(syn_loader, encoder, decoder, opt, sched, stats,
                           lambda_rec=0.0, device=device, train=True)
            if (ep + 1) % log_every == 0 or ep == 0:
                lr_now = opt.param_groups[0]["lr"]
                print(f"  pretrain ep {ep+1:5d}/{pretrain_epochs}  "
                      f"loss={m['total']:.4f}  "
                      f"lr={lr_now:.2e}  t={time.perf_counter()-t0:.0f}s")
            if m["sup"] < best_sup:
                best_sup = m["sup"]
                _save_ckpt(encoder, opt, ep + 1, m, CKPT_DIR / f"{arch}_pretrain_best.pt", arch)

        _save_ckpt(encoder, opt, pretrain_epochs, m, CKPT_DIR / f"{arch}_pretrain_last.pt", arch)
        print(f"Pre-training done.  Best supervised loss: {best_sup:.4f}")

    # ---- Phase 2: fine-tune on real data ----
    if finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: fine-tune  ({finetune_epochs} epochs, λ_rec={lambda_rec_finetune})")
        print(f"{'='*60}")

        train_ds    = RVDataset("train", single_planet=single_planet)
        val_ds      = RVDataset("val",   single_planet=single_planet)
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                                  shuffle=True,  num_workers=num_workers, collate_fn=collate_rv)
        val_loader   = DataLoader(val_ds,   batch_size=min(batch_size, len(val_ds)),
                                  shuffle=False, num_workers=num_workers, collate_fn=collate_rv)

        steps_per_epoch = math.ceil(len(train_ds) / batch_size)
        total_steps     = finetune_epochs * steps_per_epoch
        warmup_steps    = max(1, total_steps // 10)

        opt   = torch.optim.AdamW(encoder.parameters(), lr=lr / 5, weight_decay=weight_decay)
        sched = WarmupCosineSchedule(opt, warmup_steps, total_steps)

        best_val = float("inf")
        t0 = time.perf_counter()

        for ep in range(finetune_epochs):
            tr = _one_epoch(train_loader, encoder, decoder, opt, sched, stats,
                            lambda_rec=lambda_rec_finetune, device=device, train=True)
            vl = _one_epoch(val_loader,   encoder, decoder, opt, sched, stats,
                            lambda_rec=lambda_rec_finetune, device=device, train=False)
            lr_now = opt.param_groups[0]["lr"]
            print(f"  finetune ep {ep+1:5d}/{finetune_epochs}  "
                  f"train={tr['total']:.4f} (sup={tr['sup']:.4f} rec={tr['rec']:.4f})  "
                  f"val={vl['total']:.4f}  lr={lr_now:.2e}  t={time.perf_counter()-t0:.0f}s")
            if vl["total"] < best_val:
                best_val = vl["total"]
                _save_ckpt(encoder, opt, ep + 1, vl, CKPT_DIR / f"{arch}_finetune_best.pt", arch)

        _save_ckpt(encoder, opt, finetune_epochs, vl, CKPT_DIR / f"{arch}_finetune_last.pt", arch)
        print(f"Fine-tuning done.  Best val loss: {best_val:.4f}")

    print("\nTraining complete.")
    print(f"Checkpoints in {CKPT_DIR}/")


def _save_ckpt(encoder, opt, epoch, metrics, path, arch: str = "resnet"):
    torch.save({"model": encoder.state_dict(), "optim": opt.state_dict(),
                "epoch": epoch, "metrics": metrics, "arch": arch}, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--finetune-epochs", type=int, default=100)
    p.add_argument("--pretrain-n",      type=int, default=50_000)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lambda-rec",      type=float, default=0.1)
    p.add_argument("--resume",          type=Path, default=None)
    p.add_argument("--finetune-only",   action="store_true")
    p.add_argument("--no-pretrain",     action="store_true")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--workers",         type=int, default=0)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--pretrain-cache",  type=Path, default=None,
                   help="Path to pre-generated .pt cache (generated if absent)")
    p.add_argument("--arch",            type=str, default="resnet",
                   choices=sorted(ENCODER_REGISTRY),
                   help="Encoder architecture (default: resnet)")
    p.add_argument("--no-single-planet", action="store_true",
                   help="Include multi-planet systems (encoder trained on dominant "
                        "planet; companion signals treated as noise)")
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
        pretrain_cache      = args.pretrain_cache,
        arch                = args.arch,
        single_planet       = not args.no_single_planet,
    )
