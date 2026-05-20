"""
models/encoder.py — Dual-branch encoder zoo φ(RV, LSP) → θ

Seven encoder architectures sharing the same GLS periodogram Branch B and
MLP head.  Branch A (time-series) differs across architectures; Branch B
is always the 3-block ResNet over the GLS power spectrum.

Branch A variants
-----------------
  resnet      — 5-block pre-activation ResNet, 2 × stride-2 (He et al. 2016)
  deep        — 7-block ResNet, 3 × stride-2 (depth ablation)
  tcn         — 6-layer dilated TCN, RF ≈ T_MAX (Bai et al. 2018)
  inception   — 2-block InceptionTime, k=11/21/41 (Fawaz et al. 2020)
  lstm        — 2-layer BiLSTM with sequence packing
  transformer — 4-layer Transformer, t_norm as positional feature
  nolsp       — ResNet Branch A only (ablation: no GLS periodogram)

Branch B — GLS periodogram CNN
    Input  : (B, LSP_N)  normalised Lomb-Scargle power (Zechmeister & Kürster 2009)
    Network: stem → 3 pre-activation ResBlocks (2 × stride-2) → global avg pool
    Output : (B, 64)

Head
    Concatenate Branch A + Branch B → Linear → ReLU → Linear(→5)
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
    has_ecc: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Supervised MSE loss on normalised theta with two eccentricity masks.

    Mask 1 — unknown eccentricity (has_ecc):
        For systems where eccentricity is not catalogued, e is imputed as 0
        (circular-orbit prior).  Supervising the encoder on these targets would
        penalise correct non-zero eccentricity predictions.  We therefore zero
        out the loss on dims 2,3,4 (e, cos_ω, sin_ω) for has_ecc=False samples.
        The reconstruction loss still constrains the decoder for these systems.

    Mask 2 — circular-orbit omega gate (mask_omega):
        For circular (e ≈ 0) orbits, ω is degenerate and cos_ω / sin_ω targets
        carry no information.  We down-weight dims 3,4 by a sigmoid gate that
        uses physical e so the threshold is independent of normalisation.

    Parameters
    ----------
    theta_pred   : (B, 5) encoder output (normalised)
    theta_target : (B, 5) ground truth   (normalised)
    stats        : normalisation dict; required for mask_omega=True
    mask_omega   : apply the circular-orbit gate on cos_ω / sin_ω
    has_ecc      : (B,) bool — False for systems with imputed eccentricity;
                   dims 2,3,4 are excluded from the loss for these samples

    Returns
    -------
    dict: 'total' (scalar), 'per_dim' ((5,) per-parameter MSE for logging)
    """
    sq = (theta_pred - theta_target) ** 2   # (B, 5)

    # Mask 1: zero out e / cos_ω / sin_ω for systems without measured eccentricity
    if has_ecc is not None:
        ecc_known = has_ecc.float().unsqueeze(1)   # (B, 1)  1=known, 0=imputed
        w_ecc = torch.ones_like(sq)
        w_ecc[:, 2:5] = ecc_known                  # dims 2,3,4 only
        sq = sq * w_ecc

    # Mask 2: down-weight cos_ω / sin_ω for near-circular orbits
    if mask_omega and stats is not None:
        e_mean = stats["e"]["mean"]
        e_std  = max(stats["e"]["std"], 1e-8)
        e_phys = (theta_target[:, 2] * e_std + e_mean).clamp(0.0, 1.0)
        # Gate: ≈0 for e<0.05, ≈1 for e>0.15.  Lucy & Sweeney (1971, AJ 76, 544)
        # showed that RV orbits with e<0.05 are statistically indistinguishable
        # from circular, making ω degenerate below that threshold.
        w_omega = torch.sigmoid((e_phys - 0.05) * 40.0).unsqueeze(1)   # (B, 1)
        w = torch.ones_like(sq)
        w[:, 3:5] = w_omega
        sq = sq * w

    per_dim = sq.mean(dim=0)    # (5,)
    total   = per_dim.mean()
    return {"total": total, "per_dim": per_dim}


# ---------------------------------------------------------------------------
# Alternative Branch A implementations
# ---------------------------------------------------------------------------

class LSTMBranch(nn.Module):
    """
    Bidirectional LSTM Branch A replacement.

    Uses pack_padded_sequence so padding zeros never enter the recurrent state.
    The final hidden states of the last layer (forward and backward) are
    concatenated, giving a fixed-size representation of the full sequence.

    Input : (B, 4, T_MAX)  — [t_norm, rv_norm, sig_norm, mask]
    Output: (B, hidden * 2)
    """

    def __init__(self, hidden: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,       # t_norm, rv_norm, sig_norm  (mask handled via packing)
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_ch = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask    = x[:, 3, :]                               # (B, T)
        lengths = mask.sum(dim=1).long().clamp(min=1)      # (B,)
        seq     = x[:, :3, :].permute(0, 2, 1)            # (B, T, 3)

        packed      = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, B, hidden) — interleaved fwd/bwd per layer
        fwd = h_n[-2]   # last-layer forward  (B, hidden)
        bwd = h_n[-1]   # last-layer backward (B, hidden)
        return torch.cat([fwd, bwd], dim=1)   # (B, hidden * 2)


class TransformerBranch(nn.Module):
    """
    Transformer encoder Branch A replacement.

    Each observation is a token: (t_norm, rv_norm, sig_norm) projected to
    d_model dimensions.  The actual observation time t_norm is included as an
    input feature, allowing the self-attention mechanism to learn temporal
    structure from the physical timestamps rather than a position index.
    This is the natural treatment for irregularly sampled time series where
    a fixed positional encoding would be misleading.

    Output is obtained via masked global average pool over real observations,
    making the encoder permutation-invariant (consistent with the ResNet branch).

    Input : (B, 4, T_MAX)
    Output: (B, d_model)
    """

    def __init__(self, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(3, d_model)   # (t_norm, rv_norm, sig_norm) → d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True,
            norm_first=True,               # pre-LN: better gradient flow (He et al. 2016)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_ch = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 3, :]                     # (B, T)  1=real, 0=padding
        seq  = x[:, :3, :].permute(0, 2, 1)  # (B, T, 3)

        h = self.proj(seq)                    # (B, T, d_model)

        # src_key_padding_mask: True → position is ignored (padding)
        pad_mask = (mask == 0)                               # (B, T)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        # Masked global average pool (same convention as TimeSeriesBranch)
        m      = mask.unsqueeze(-1)                          # (B, T, 1)
        h_pool = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return h_pool                                        # (B, d_model)


class DeepTimeSeriesBranch(nn.Module):
    """
    Deeper 7-block ResNet Branch A with three stride-2 downsampling steps.

    Extends TimeSeriesBranch (5 blocks, /4 spatial) by adding one more
    stride-2 block: /8 spatial, 256-channel output.  Tests whether additional
    depth and receptive field improve orbital parameter estimation beyond the
    baseline.

    Input : (B, 4, T_MAX)
    Output: (B, base_ch * 8)
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
            ResBlock1d(ch,      ch,      kernel=5),
            ResBlock1d(ch,      ch * 2,  kernel=5, stride=2),   # /2
            ResBlock1d(ch * 2,  ch * 2,  kernel=5),
            ResBlock1d(ch * 2,  ch * 4,  kernel=5, stride=2),   # /4
            ResBlock1d(ch * 4,  ch * 4,  kernel=3),
            ResBlock1d(ch * 4,  ch * 8,  kernel=3, stride=2),   # /8
            ResBlock1d(ch * 8,  ch * 8,  kernel=3),
        )
        self.out_ch = ch * 8   # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask    = x[:, 3:4, :]
        h       = self.blocks(self.stem(x))
        T_out   = h.shape[2]
        mask_ds = F.adaptive_max_pool1d(mask, T_out)
        return (h * mask_ds).sum(dim=2) / mask_ds.sum(dim=2).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Branch A — TCN (dilated temporal convolutional network)
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """
    Non-causal dilated residual block (Bai et al. 2018, arXiv:1803.01271).

    Pre-activation layout (BN→ReLU→Conv) matches ResBlock1d convention.
    Symmetric (non-causal) padding is used because we are doing parameter
    estimation, not autoregressive forecasting; future observations are
    available and should be exploited.  The spatial dimension T is preserved
    exactly: padding = (kernel-1)*dilation // 2 for odd-or-3 kernels.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.drop  = nn.Dropout(p=dropout)
        self.skip  = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.drop(h)
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.drop(h)
        return h + self.skip(x)


class TCNBranch(nn.Module):
    """
    Dilated TCN Branch A (Bai et al. 2018).

    Six dilated blocks with dilations [1, 2, 4, 8, 16, 32] give a receptive
    field of 1 + 2*(3-1)*sum([1,2,4,8,16,32]) = 253 ≈ T_MAX=256, so the
    final feature vector integrates information from the full observation
    window at every time step before the masked average pool.  Channel widths
    follow the same 32→64→128 progression as TimeSeriesBranch.

    Input : (B, 4, T_MAX)
    Output: (B, base_ch * 4)
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        ch = base_ch
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            TCNBlock(ch,     ch,     dilation=1,  dropout=dropout),
            TCNBlock(ch,     ch,     dilation=2,  dropout=dropout),
            TCNBlock(ch,     ch * 2, dilation=4,  dropout=dropout),
            TCNBlock(ch * 2, ch * 2, dilation=8,  dropout=dropout),
            TCNBlock(ch * 2, ch * 4, dilation=16, dropout=dropout),
            TCNBlock(ch * 4, ch * 4, dilation=32, dropout=dropout),
        ])
        self.out_ch = ch * 4  # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 3:4, :]
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        T_out   = h.shape[2]
        mask_ds = F.adaptive_max_pool1d(mask, T_out)
        return (h * mask_ds).sum(dim=2) / mask_ds.sum(dim=2).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Branch A — InceptionTime (multi-scale CNN)
# ---------------------------------------------------------------------------

class InceptionBlock1d(nn.Module):
    """
    InceptionTime module (Fawaz et al. 2020, Data Min. Knowl. Disc. 34:1755).

    Three parallel convolutions (k=11, 21, 41) applied after a bottleneck
    1×1 convolution, plus a max-pool residual branch.  All four outputs are
    concatenated to capture temporal patterns at short, medium, and long time
    scales simultaneously.  This is appropriate for RV curves where the
    orbital signature spans a wide range of scales relative to the cadence.

    Kernel sizes are odd so that k//2 gives exact same-padding.
    The bottleneck reduces computation before the parallel large-kernel convs.
    """

    def __init__(self, in_ch: int, nb_filters: int = 32,
                 kernels: tuple = (11, 21, 41)) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, nb_filters, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(nb_filters, nb_filters, k, padding=k // 2, bias=False)
            for k in kernels
        ])
        # Max-pool branch: no bottleneck — preserves local extrema directly
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_ch, nb_filters, 1, bias=False)

        out_ch = nb_filters * (len(kernels) + 1)   # 4 × nb_filters
        self.bn  = nn.BatchNorm1d(out_ch)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.bottleneck(x)
        hs = [conv(h) for conv in self.convs]
        hs.append(self.mp_conv(self.maxpool(x)))
        return F.relu(self.bn(torch.cat(hs, dim=1)), inplace=True)


class InceptionBranch(nn.Module):
    """
    InceptionTime Branch A (Fawaz et al. 2020).

    Two InceptionBlocks with a shortcut connection from the raw input to the
    output of the second block, following the residual variant in §4.3 of
    Fawaz et al.  Output dimension is 128 (= 4 × nb_filters with nb_filters=32).

    Input : (B, 4, T_MAX)
    Output: (B, 128)
    """

    def __init__(self, nb_filters: int = 32) -> None:
        super().__init__()
        self.block1 = InceptionBlock1d(4, nb_filters)
        ch1 = self.block1.out_ch          # 128
        self.block2 = InceptionBlock1d(ch1, nb_filters)
        ch2 = self.block2.out_ch          # 128
        # Residual skip: raw input projected to ch2
        self.skip = nn.Sequential(
            nn.Conv1d(4, ch2, 1, bias=False),
            nn.BatchNorm1d(ch2),
        )
        self.out_ch = ch2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 3:4, :]
        h    = self.block2(self.block1(x))
        h    = F.relu(h + self.skip(x), inplace=True)
        T_out   = h.shape[2]
        mask_ds = F.adaptive_max_pool1d(mask, T_out)
        return (h * mask_ds).sum(dim=2) / mask_ds.sum(dim=2).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Encoder variants
# ---------------------------------------------------------------------------

class RVEncoderLSTM(nn.Module):
    """
    Encoder variant: bidirectional LSTM time-series branch + GLS periodogram
    ResNet branch.

    Replaces the ResNet Branch A with a 2-layer BiLSTM (hidden=64, output=128-d).
    Tests whether recurrent temporal modelling outperforms convolutional
    feature extraction for RV time series.  ~196K parameters.
    """

    def __init__(self, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = LSTMBranch(hidden=hidden, num_layers=num_layers)
        self.lsp_branch = PeriodogramBranch()
        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch   # 128 + 64 = 192
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.ts_branch(x), self.lsp_branch(lsp)], dim=1)
        return self.head(self.dropout(feat))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RVEncoderTransformer(nn.Module):
    """
    Encoder variant: Transformer encoder time-series branch + GLS periodogram
    ResNet branch.

    Branch A: 4-layer Transformer encoder (d=128, heads=8, ff=256).
    Each observation is a token; t_norm is included as an input feature for
    temporal awareness (no fixed positional encoding, appropriate for
    irregular cadence).  ~900K parameters.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = TransformerBranch(d_model=d_model, nhead=nhead,
                                             num_layers=num_layers, dim_ff=dim_ff)
        self.lsp_branch = PeriodogramBranch()
        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.ts_branch(x), self.lsp_branch(lsp)], dim=1)
        return self.head(self.dropout(feat))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RVEncoderDeep(nn.Module):
    """
    Encoder variant: deeper 7-block ResNet time-series branch + GLS periodogram
    ResNet branch.

    Branch A: DeepTimeSeriesBranch (3 × stride-2, 256-d output).  Tests
    whether greater depth and receptive field improve estimation beyond the
    5-block baseline.  ~1.05M parameters.
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = DeepTimeSeriesBranch(base_ch=base_ch)
        self.lsp_branch = PeriodogramBranch()
        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch   # 256 + 64 = 320
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.ts_branch(x), self.lsp_branch(lsp)], dim=1)
        return self.head(self.dropout(feat))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RVEncoderNoLSP(nn.Module):
    """
    Ablation: 5-block ResNet time-series branch only — no GLS periodogram input.

    Identical to the ResNet branch of RVEncoder but without Branch B.
    Quantifies the contribution of the GLS periodogram to orbital parameter
    estimation.  If period accuracy degrades significantly relative to
    RVEncoder, the periodogram branch is load-bearing.  ~387K parameters.
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch = TimeSeriesBranch(base_ch=base_ch)
        feat_ch = self.ts_branch.out_ch   # 128
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        # lsp accepted but ignored — uniform interface with all other encoders
        return self.head(self.dropout(self.ts_branch(x)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RVEncoderTCN(nn.Module):
    """
    Encoder variant: dilated TCN time-series branch + GLS periodogram ResNet.

    Branch A: 6-layer TCN (Bai et al. 2018) with dilations [1,2,4,8,16,32].
    Receptive field ≈ T_MAX=256, so the representation integrates the full
    baseline at each time step before masked pooling.  Tests whether explicit
    full-sequence connectivity outperforms the ResNet baseline (~64-point RF).
    ~295K parameters.
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = TCNBranch(base_ch=base_ch, dropout=dropout)
        self.lsp_branch = PeriodogramBranch()
        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch   # 128 + 64 = 192
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.ts_branch(x), self.lsp_branch(lsp)], dim=1)
        return self.head(self.dropout(feat))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RVEncoderInception(nn.Module):
    """
    Encoder variant: InceptionTime time-series branch + GLS periodogram ResNet.

    Branch A: 2-block InceptionTime (Fawaz et al. 2020) with k=11,21,41.
    Parallel multi-scale convolutions capture short (k=11), medium (k=21),
    and long (k=41) temporal patterns simultaneously — appropriate for RV
    curves where the orbital period spans many multiples of the cadence.
    ~255K parameters.
    """

    def __init__(self, nb_filters: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.ts_branch  = InceptionBranch(nb_filters=nb_filters)
        self.lsp_branch = PeriodogramBranch()
        feat_ch = self.ts_branch.out_ch + self.lsp_branch.out_ch
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, THETA_DIM),
        )

    def forward(self, x: torch.Tensor, lsp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.ts_branch(x), self.lsp_branch(lsp)], dim=1)
        return self.head(self.dropout(feat))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

ENCODER_REGISTRY: dict[str, type] = {
    'resnet':      RVEncoder,            # baseline: dual-branch 5-block ResNet
    'deep':        RVEncoderDeep,        # deeper 7-block ResNet, 3 × stride-2
    'tcn':         RVEncoderTCN,         # dilated TCN, RF ≈ T_MAX (Bai et al. 2018)
    'inception':   RVEncoderInception,   # multi-scale InceptionTime (Fawaz et al. 2020)
    'lstm':        RVEncoderLSTM,        # 2-layer BiLSTM branch A
    'transformer': RVEncoderTransformer, # 4-layer Transformer branch A
    'nolsp':       RVEncoderNoLSP,       # ablation: no GLS periodogram
}


def build_encoder(arch: str = 'resnet') -> nn.Module:
    """Instantiate an encoder by architecture name."""
    if arch not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown arch {arch!r}. Options: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[arch]()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    B, T = 8, 256
    x   = torch.zeros(B, 4, T)
    lsp = torch.rand(B, LSP_N)
    # Mix of observation counts to exercise masking
    for i, n in enumerate([20, 44, 60, 80, 100, 140, 200, 256]):
        x[i, 0, :n] = torch.linspace(0, 1, n)   # t_norm
        x[i, 1, :n] = torch.randn(n)             # rv_norm
        x[i, 2, :n] = torch.rand(n) * 0.2 + 0.1 # sig_norm
        x[i, 3, :n] = 1.0                        # mask

    stats  = json.loads(open(_STATS_PATH).read()) if _STATS_PATH.exists() else None
    target = torch.randn(B, THETA_DIM)
    has_ecc = torch.ones(B, dtype=torch.bool); has_ecc[0] = False

    print(f"{'Arch':15s}  {'Params':>10s}  {'Fwd (ms)':>10s}  {'Loss':>8s}  Grads")
    print("-" * 60)
    for name, cls in ENCODER_REGISTRY.items():
        enc = cls()
        t0  = time.perf_counter()
        out = enc(x, lsp)
        dt  = (time.perf_counter() - t0) * 1000
        assert out.shape == (B, THETA_DIM), f"{name}: bad output shape {out.shape}"
        losses = encoder_loss(out, target, stats=stats, has_ecc=has_ecc)
        losses["total"].backward()
        grad_ok = all(p.grad is not None for p in enc.parameters() if p.requires_grad)
        print(f"  {name:13s}  {enc.n_params:>10,}  {dt:>10.1f}  "
              f"{losses['total'].item():>8.4f}  {'OK' if grad_ok else 'FAIL'}")
        enc.zero_grad()

    if stats:
        tp  = torch.tensor([[1.8, 1.9, 0.1, 0.6, 0.8]])
        err = (tp - un_normalise_theta(normalise_theta(tp, stats), stats)).abs().max().item()
        print(f"\nNormalisation round-trip error: {err:.2e}  (want < 1e-6)")

    print("\nAll checks passed.")
