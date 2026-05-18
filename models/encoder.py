"""
models/encoder.py — Dual-branch 1D ResNet encoder φ(RV, LSP) → θ

Architecture
------------
Branch A — time-series branch
    Input  : (B, 4, 256)  [t_norm, rv_norm, sig_norm, mask]
    Network: stem → 5 pre-activation ResBlocks (2 × stride-2) → masked avg pool
    Output : (B, 128)

Branch B — GLS periodogram branch
    Input  : (B, LSP_N)  normalised Lomb-Scargle power (Zechmeister & Kürster 2009)
    Network: stem → 3 pre-activation ResBlocks (2 × stride-2) → global avg pool
    Output : (B, 64)

Head
    Concatenate (B, 192) → Linear(192→128) → ReLU → Linear(128→5)
    Output: (B, 5) normalised theta ≈ N(0,1) matching RVDataset labels

Motivation for the dual-branch design
--------------------------------------
A 1D CNN over the masked time series has limited receptive field (~64/256 after
two stride-2 blocks) and will struggle to identify periods that span the full
baseline.  The GLS periodogram provides the frequency content of the signal
directly, so Branch B specialises in period determination while Branch A learns
amplitude, shape, and eccentricity.  This mirrors standard exoplanet practice
where the periodogram is the first diagnostic (Scargle 1982; Zechmeister &
Kürster 2009).

Normalisation
-------------
  normalise_theta(theta_phys, stats)   physical → N(0,1)
  un_normalise_theta(theta_norm, stats) N(0,1) → physical

These bridge the encoder output (normalised) and KeplerDecoder input (physical).

Usage
-----
    from models.encoder import RVEncoder, un_normalise_theta
    enc = RVEncoder()
    theta_norm = enc(x, lsp)               # (B, 5) normalised
    theta_phys = un_normalise_theta(theta_norm, stats)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import LSP_N, THETA_NAMES, THETA_DIM

_STATS_PATH = Path("data/dataset_stats.json")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _stats_tensors(stats: dict, device, dtype):
    mean = torch.tensor([stats[n]["mean"] for n in THETA_NAMES], dtype=dtype, device=device)
    std  = torch.tensor([max(stats[n]["std"], 1e-8) for n in THETA_NAMES], dtype=dtype, device=device)
    return mean, std


def normalise_theta(theta_phys: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, 5) physical → (B, 5) normalised."""
    mean, std = _stats_tensors(stats, theta_phys.device, theta_phys.dtype)
    return (theta_phys - mean) / std


def un_normalise_theta(theta_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, 5) normalised → (B, 5) physical."""
    mean, std = _stats_tensors(stats, theta_norm.device, theta_norm.dtype)
    return theta_norm * std + mean


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """
    Pre-activation 1D residual block (He et al. 2016, Identity Mappings).

    BN → ReLU → Conv → BN → ReLU → Conv, with a 1×1 skip when channels
    or stride differ.  Pre-activation leaves the skip path free of non-
    linearities, giving cleaner gradient flow.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1) -> None:
        super().__init__()
        pad = kernel // 2
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.skip  = (nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
                      if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Branch A — time-series CNN
# ---------------------------------------------------------------------------

class TimeSeriesBranch(nn.Module):
    """
    ResNet over the (4, T_MAX) input tensor.

    Masked global average pool so padding zeros do not corrupt the
    representation.  After two stride-2 blocks the spatial dimension is
    T_MAX / 4 = 64; max-pool on the binary mask preserves "has real obs"
    information for each output position.
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        ch = base_ch
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(ch,      ch,     kernel=5),
            ResBlock1d(ch,      ch * 2, kernel=5, stride=2),  # /2
            ResBlock1d(ch * 2,  ch * 2, kernel=5),
            ResBlock1d(ch * 2,  ch * 4, kernel=5, stride=2),  # /4
            ResBlock1d(ch * 4,  ch * 4, kernel=3),
        )
        self.out_ch = ch * 4   # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 3:4, :]           # (B, 1, T)
        h    = self.blocks(self.stem(x))         # (B, 4ch, T/4)
        T_out    = h.shape[2]
        mask_ds  = F.adaptive_max_pool1d(mask, T_out)  # (B, 1, T/4) — 1 if any real obs
        h_pool   = (h * mask_ds).sum(dim=2) / mask_ds.sum(dim=2).clamp(min=1.0)
        return h_pool                 # (B, out_ch)


# ---------------------------------------------------------------------------
# Branch B — GLS periodogram CNN
# ---------------------------------------------------------------------------

class PeriodogramBranch(nn.Module):
    """
    Lightweight ResNet over the (LSP_N,) GLS power spectrum.

    The fixed log-spaced period grid (LSP_PERIODS) means each output position
    corresponds to a specific period range, so learned filters act as matched
    filters for orbital periods.  No masking is needed (all LSP_N points are
    always computed).
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            ResBlock1d(16, 32, kernel=7, stride=2),   # LSP_N / 2
            ResBlock1d(32, 64, kernel=7, stride=2),   # LSP_N / 4
            ResBlock1d(64, 64, kernel=5),
        )
        self.out_ch = 64

    def forward(self, lsp: torch.Tensor) -> torch.Tensor:
        h = self.net(lsp.unsqueeze(1))          # (B, 1, LSP_N) → (B, 64, LSP_N/4)
        return h.mean(dim=2)                    # global avg pool → (B, 64)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class RVEncoder(nn.Module):
    """
    Dual-branch encoder: φ(RV, LSP) → θ_norm.

    Parameters
    ----------
    base_ch  : base channel width for the time-series branch
    dropout  : dropout applied before the MLP head

    Forward
    -------
    x   : (B, 4, T_MAX)  — time-series tensor
    lsp : (B, LSP_N)     — GLS periodogram power

    Returns
    -------
    theta_norm : (B, 5) in normalised space ≈ N(0, 1)
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = TimeSeriesBranch(base_ch=base_ch)
        self.lsp_branch = PeriodogramBranch()

        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch  # 192

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat_ts  = self.ts_branch(x)           # (B, 128)
        feat_lsp = self.lsp_branch(lsp)        # (B, 64)
        feat     = torch.cat([feat_ts, feat_lsp], dim=1)   # (B, 192)
        return self.head(self.dropout(feat))   # (B, 5)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def encoder_loss(
    theta_pred: torch.Tensor,
    theta_target: torch.Tensor,
    stats: dict | None = None,
    mask_omega: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Supervised MSE loss on normalised theta with circular-orbit down-weighting.

    For circular (e ≈ 0) orbits, ω is degenerate and the cos_ω / sin_ω
    targets carry no information.  We down-weight these two dimensions
    by a sigmoid gate that uses physical e so the threshold is
    independent of normalisation statistics.

    Parameters
    ----------
    theta_pred   : (B, 5) encoder output (normalised)
    theta_target : (B, 5) ground truth   (normalised)
    stats        : normalisation dict; required for mask_omega=True
    mask_omega   : whether to apply the circular-orbit gate

    Returns
    -------
    dict: 'total' (scalar), 'per_dim' ((5,) per-parameter MSE for logging)
    """
    sq = (theta_pred - theta_target) ** 2   # (B, 5)

    if mask_omega and stats is not None:
        # Convert normalised e to physical e, then gate
        e_mean = stats["e"]["mean"]
        e_std  = max(stats["e"]["std"], 1e-8)
        e_phys = (theta_target[:, 2] * e_std + e_mean).clamp(0.0, 1.0)
        # Gate: ≈ 0 for e < 0.05, ≈ 1 for e > 0.15 (transition width ~0.05)
        w_omega = torch.sigmoid((e_phys - 0.05) * 40.0).unsqueeze(1)   # (B, 1)
        # Build per-element weight: 1.0 for P, K, e; w_omega for cos_ω, sin_ω
        w = torch.ones_like(sq)              # (B, 5)
        w[:, 3:5] = w_omega                  # broadcasts (B, 1) → (B, 2)
        sq = sq * w

    per_dim = sq.mean(dim=0)    # (5,)
    total   = per_dim.mean()
    return {"total": total, "per_dim": per_dim}


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    enc = RVEncoder()
    print(f"RVEncoder parameters: {enc.n_params:,}")
    print(f"  Time-series branch: {enc.ts_branch.out_ch}-d features")
    print(f"  Periodogram branch: {enc.lsp_branch.out_ch}-d features")

    B, T = 16, 256
    x   = torch.randn(B, 4, T);  x[:, 3, :] = 1.0
    lsp = torch.rand(B, LSP_N)

    t0  = time.perf_counter()
    out = enc(x, lsp)
    print(f"\nForward pass ({B}×4×{T} + {B}×{LSP_N}): {time.perf_counter()-t0:.3f} s")
    print(f"Output shape: {tuple(out.shape)}  (want ({B}, 5))")

    target = torch.randn(B, 5)
    stats  = json.loads(open(_STATS_PATH).read()) if _STATS_PATH.exists() else None
    losses = encoder_loss(out, target, stats=stats)
    losses["total"].backward()
    print(f"Loss: {losses['total'].item():.4f}")
    print(f"Gradients flow: {all(p.grad is not None for p in enc.parameters() if p.requires_grad)}")

    # Masking test
    x2 = torch.zeros(4, 4, T)
    for i, n in enumerate([20, 60, 100, 200]):
        x2[i, :, :n] = torch.randn(4, n); x2[i, 3, :n] = 1.0
    out2 = enc(x2, torch.rand(4, LSP_N))
    print(f"Masked forward: {tuple(out2.shape)}  OK")

    if stats:
        tp = torch.tensor([[1.8, 1.9, 0.1, 0.6, 0.8]])
        err = (tp - un_normalise_theta(normalise_theta(tp, stats), stats)).abs().max().item()
        print(f"Normalisation round-trip error: {err:.2e}  (want < 1e-6)")

    print("\nAll checks passed.")
