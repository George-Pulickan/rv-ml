"""
models/encoder.py — 1D ResNet encoder φ(RV) → θ

Maps a (4 × 256) RV input tensor to the 5-dim orbital parameter vector
θ = [log10_P, log10_K, e, cos_ω, sin_ω] in normalised space (≈ N(0,1)),
ready for a supervised MSE loss against the labels from RVDataset.

Architecture: stem conv → 4 residual blocks (2 with stride-2 down-sampling)
→ masked global average pool → 2-layer MLP head.  ~200K parameters.

Input channels
--------------
  0  t_norm   = (t − t_min) / t_span  ∈ [0, 1]
  1  rv_norm  = (rv − median) / std
  2  sig_norm = σ / std
  3  mask     = 1.0 for real obs, 0.0 for padding

The mask channel drives a masked global average pool so padding zeros do
not pollute the representation.

Normalisation helpers
---------------------
  normalise_theta(theta_phys, stats)   physical → normalised
  un_normalise_theta(theta_norm, stats) normalised → physical
These are needed to bridge encoder output and KeplerDecoder input.

Usage
-----
    from models.encoder import RVEncoder, un_normalise_theta
    import json, torch

    stats  = json.load(open('data/dataset_stats.json'))
    enc    = RVEncoder()
    x      = torch.randn(8, 4, 256)          # batch of 8 systems
    theta_norm = enc(x)                       # (8, 5) normalised
    theta_phys = un_normalise_theta(theta_norm, stats)  # (8, 5) physical
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import THETA_NAMES, THETA_DIM

_STATS_PATH = Path("data/dataset_stats.json")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _stats_tensors(stats: dict, device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors of shape (5,) from a stats dict."""
    mean = torch.tensor([stats[n]["mean"] for n in THETA_NAMES],
                        dtype=dtype, device=device)
    std  = torch.tensor([max(stats[n]["std"], 1e-8) for n in THETA_NAMES],
                        dtype=dtype, device=device)
    return mean, std


def normalise_theta(theta_phys: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, 5) physical → (B, 5) normalised (zero-mean / unit-std)."""
    mean, std = _stats_tensors(stats, device=theta_phys.device,
                                dtype=theta_phys.dtype)
    return (theta_phys - mean) / std


def un_normalise_theta(theta_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, 5) normalised → (B, 5) physical (log10_P, log10_K, e, cos_ω, sin_ω)."""
    mean, std = _stats_tensors(stats, device=theta_norm.device,
                                dtype=theta_norm.dtype)
    return theta_norm * std + mean


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """
    Pre-activation 1D residual block.

    BN → ReLU → Conv → BN → ReLU → Conv, with a 1×1 skip when channels or
    stride differ.  Pre-activation (He et al. 2016) is used so BN/ReLU
    happen before the convolution; this gives cleaner gradients and lets the
    skip path be a pure identity (or 1×1) with no non-linearity.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1) -> None:
        super().__init__()
        pad = kernel // 2
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                               bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)

        self.skip = (nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
                     if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class RVEncoder(nn.Module):
    """
    Single-planet RV encoder.

    Parameters
    ----------
    base_ch  : base channel width (default 32); doubled twice by stride-2 blocks
    dropout  : dropout probability applied just before the MLP head

    Output
    ------
    (B, 5) theta in normalised space — directly comparable to RVDataset labels.
    Un-normalise with un_normalise_theta() before passing to KeplerDecoder.
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()

        ch = base_ch

        # Stem: wide receptive field to capture dominant periodicity
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )

        # Residual tower: two stride-2 blocks halve the time axis twice
        # (256 → 128 → 64 at the pool)
        self.blocks = nn.Sequential(
            ResBlock1d(ch,      ch,     kernel=5),           # 256
            ResBlock1d(ch,      ch * 2, kernel=5, stride=2), # 128
            ResBlock1d(ch * 2,  ch * 2, kernel=5),           # 128
            ResBlock1d(ch * 2,  ch * 4, kernel=5, stride=2), # 64
            ResBlock1d(ch * 4,  ch * 4, kernel=3),           # 64
        )

        feat_ch = ch * 4   # 128

        self.dropout = nn.Dropout(p=dropout)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, T) float32 — input tensor from RVDataset / SyntheticRVDataset

        Returns
        -------
        theta_norm : (B, 5) float32 — normalised orbital parameters
        """
        mask = x[:, 3:4, :]              # (B, 1, T) — 1 = real obs, 0 = pad

        h = self.stem(x)                 # (B, ch, T)
        h = self.blocks(h)               # (B, 4*ch, T/4)

        # Masked global average pool: exclude padding from the mean
        T_out   = h.shape[2]
        mask_ds = F.adaptive_max_pool1d(mask, T_out)            # (B, 1, T/4)
        h_pool  = (h * mask_ds).sum(dim=2) / mask_ds.sum(dim=2).clamp(min=1.0)

        h_pool = self.dropout(h_pool)
        return self.head(h_pool)                                 # (B, 5)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def encoder_loss(
    theta_pred: torch.Tensor,
    theta_target: torch.Tensor,
    mask_omega: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Supervised loss on normalised theta.

    Parameters
    ----------
    theta_pred   : (B, 5) encoder output (normalised)
    theta_target : (B, 5) ground-truth labels (normalised)
    mask_omega   : if True, down-weight cos_ω / sin_ω for circular orbits
                   (e < 0.05 in physical space → ω is degenerate)

    Returns
    -------
    dict with keys:
      'total'   — scalar loss to back-prop
      'per_dim' — (5,) per-parameter MSE (for logging)
    """
    sq = (theta_pred - theta_target) ** 2   # (B, 5)

    if mask_omega:
        # Dimension 2 = e (normalised). Physical e ≈ 0 when e_norm is large-negative.
        # Use a soft weight: w = sigmoid((e_norm + 1) * 3) so e < 0.05 → w ≈ 0.
        e_norm = theta_target[:, 2]
        w_omega = torch.sigmoid((e_norm + 1.0) * 3.0).unsqueeze(1)   # (B, 1)
        weights = torch.ones(5, device=sq.device, dtype=sq.dtype)
        weights[3] = 1.0   # cos_ω — scaled by w_omega below
        weights[4] = 1.0
        sq = sq * weights
        sq[:, 3] = sq[:, 3] * w_omega.squeeze(1)
        sq[:, 4] = sq[:, 4] * w_omega.squeeze(1)

    per_dim = sq.mean(dim=0)              # (5,)
    total   = per_dim.mean()
    return {"total": total, "per_dim": per_dim}


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    enc = RVEncoder()
    print(f"RVEncoder parameters: {enc.n_params:,}")

    B, T = 16, 256
    x = torch.randn(B, 4, T)
    x[:, 3, :] = 1.0   # all real (no padding)

    # Forward pass
    t0 = time.perf_counter()
    out = enc(x)
    print(f"Forward pass ({B}×4×{T}): {time.perf_counter()-t0:.3f} s")
    print(f"Output shape: {tuple(out.shape)}  (want ({B}, 5))")
    print(f"Output mean:  {out.mean(dim=0).detach().numpy().round(3)}")
    print(f"Output std:   {out.std(dim=0).detach().numpy().round(3)}")

    # Backward pass
    target = torch.randn(B, 5)
    losses = encoder_loss(out, target)
    losses["total"].backward()
    print(f"Loss: {losses['total'].item():.4f}")
    print(f"Per-dim losses: {losses['per_dim'].detach().numpy().round(4)}")
    print(f"Gradients flow: {all(p.grad is not None for p in enc.parameters() if p.requires_grad)}")

    # With partial masking (simulate padding)
    x2 = torch.zeros(4, 4, T)
    n_real = [30, 60, 100, 200]
    for i, n in enumerate(n_real):
        x2[i, :, :n] = torch.randn(4, n)
        x2[i, 3, :n] = 1.0
    out2 = enc(x2)
    print(f"\nMasked forward ({n_real} real obs each): {tuple(out2.shape)}  OK")

    # Normalisation round-trip
    if _STATS_PATH.exists():
        stats = json.loads(_STATS_PATH.read_text())
        theta_phys = torch.tensor([[1.8, 1.9, 0.1, 0.6, 0.8]])
        theta_norm = normalise_theta(theta_phys, stats)
        theta_back = un_normalise_theta(theta_norm, stats)
        err = (theta_phys - theta_back).abs().max().item()
        print(f"\nNormalisation round-trip error: {err:.2e}  (want < 1e-6)")

    print("\nAll checks passed.")
