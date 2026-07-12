"""
Shared per-target loss weights and circular omega loss for Kepler regression.

Masking policy (matches models.encoder.encoder_loss when hard=True):
  - dims 2-4 zeroed when has_ecc is False
  - cos/sin omega zeroed when e <= OMEGA_EPS (hard) or down-weighted (soft sigmoid)
"""

from __future__ import annotations

import numpy as np
import torch

OMEGA_EPS = 0.05
OMEGA_GATE_WIDTH = 40.0

THETA_DIM = 5
OMEGA_COS_IDX = 3
OMEGA_SIN_IDX = 4


def omega_gate_weights(
    e_phys: np.ndarray | torch.Tensor,
    *,
    gate_center: float = OMEGA_EPS,
    gate_width: float = OMEGA_GATE_WIDTH,
) -> np.ndarray | torch.Tensor:
    """Sigmoid gate in [0, 1]: ~0 for e < gate_center, ~1 for e > gate_center + 0.1."""
    if isinstance(e_phys, torch.Tensor):
        return torch.sigmoid((e_phys.clamp(0.0, 1.0) - gate_center) * gate_width)
    e = np.clip(np.asarray(e_phys, dtype=np.float64), 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-(e - gate_center) * gate_width))


def _omega_sample_weights_from_e(
    e_phys: np.ndarray | torch.Tensor,
    *,
    hard: bool,
    gate_center: float = OMEGA_EPS,
    gate_width: float = OMEGA_GATE_WIDTH,
) -> np.ndarray | torch.Tensor:
    """Per-sample weight in [0, 1] for omega loss dims."""
    if hard:
        if isinstance(e_phys, torch.Tensor):
            return (e_phys > gate_center).float()
        return (np.asarray(e_phys, dtype=np.float64) > gate_center).astype(np.float64)
    return omega_gate_weights(e_phys, gate_center=gate_center, gate_width=gate_width)


def theta_loss_weights_numpy(
    y_phys: np.ndarray,
    *,
    has_ecc: np.ndarray | None = None,
    mask_omega: bool = True,
    hard_omega_mask: bool = True,
    gate_center: float = OMEGA_EPS,
    gate_width: float = OMEGA_GATE_WIDTH,
) -> np.ndarray:
    """
    Per-sample loss weights, shape (n, 5).

    y_phys columns: log10_P, log10_K, e, cos_omega, sin_omega (physical units).
    """
    n = len(y_phys)
    w = np.ones((n, THETA_DIM), dtype=np.float64)
    if has_ecc is not None:
        known = np.asarray(has_ecc, dtype=bool)
        w[~known, 2:5] = 0.0
    if mask_omega:
        om_w = _omega_sample_weights_from_e(
            y_phys[:, 2], hard=hard_omega_mask, gate_center=gate_center, gate_width=gate_width
        )
        w[:, 3:5] *= np.asarray(om_w, dtype=np.float64).reshape(-1, 1)
    return w


def theta_loss_weights_torch(
    theta_target: torch.Tensor,
    *,
    stats: dict | None = None,
    has_ecc: torch.Tensor | None = None,
    mask_omega: bool = True,
    hard_omega_mask: bool = True,
    gate_center: float = OMEGA_EPS,
    gate_width: float = OMEGA_GATE_WIDTH,
) -> torch.Tensor:
    """Per-sample loss weights, shape (B, 5), for normalized theta targets."""
    w = torch.ones_like(theta_target)
    if has_ecc is not None:
        ecc_known = has_ecc.float().unsqueeze(1)
        w[:, 2:5] = ecc_known.expand(-1, 3)
    if mask_omega and stats is not None:
        e_mean = float(stats["e"]["mean"])
        e_std = max(float(stats["e"]["std"]), 1e-8)
        e_phys = (theta_target[:, 2] * e_std + e_mean).clamp(0.0, 1.0)
        om_w = _omega_sample_weights_from_e(
            e_phys, hard=hard_omega_mask, gate_center=gate_center, gate_width=gate_width
        ).unsqueeze(1)
        w[:, 3:5] = w[:, 3:5] * om_w
    return w


def normalize_omega_tensor(cos_sin: torch.Tensor) -> torch.Tensor:
    """L2-normalize (cos, sin) pairs along last dim. Input shape (..., 2)."""
    norm = torch.linalg.norm(cos_sin, dim=-1, keepdim=True).clamp(min=1e-8)
    return cos_sin / norm


def denorm_omega_components(
    theta_norm: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    """Return physical (cos, sin) shape (B, 2) from normalized 5-vector."""
    cos_n = theta_norm[:, OMEGA_COS_IDX] * y_std[OMEGA_COS_IDX] + y_mean[OMEGA_COS_IDX]
    sin_n = theta_norm[:, OMEGA_SIN_IDX] * y_std[OMEGA_SIN_IDX] + y_mean[OMEGA_SIN_IDX]
    return torch.stack([cos_n, sin_n], dim=1)


def circular_omega_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    *,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Mean circular loss 1 - cos(delta_omega) on unit-normalized (cos, sin) pairs.

    sample_weight: (B, 5) combined sample x dim weights; uses mean of omega dims.
    Returns scalar loss (0 if no omega-weighted samples).
    """
    pred_om = normalize_omega_tensor(denorm_omega_components(pred_norm, y_mean, y_std))
    true_om = normalize_omega_tensor(denorm_omega_components(target_norm, y_mean, y_std))
    cos_delta = (pred_om * true_om).sum(dim=1).clamp(-1.0, 1.0)
    per_sample = 1.0 - cos_delta
    w_om = 0.5 * (sample_weight[:, OMEGA_COS_IDX] + sample_weight[:, OMEGA_SIN_IDX])
    denom = w_om.sum().clamp(min=1e-8)
    return (per_sample * w_om).sum() / denom


def regression_theta_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    sample_weight: torch.Tensor,
    dim_weight: torch.Tensor,
    *,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    circular_omega: bool = True,
) -> torch.Tensor:
    """
    Combined training loss: MSE on log10_P, log10_K, e; circular or MSE on omega.

    sample_weight: (B, 5) per-sample masks; dim_weight: (5,) relative term scales.
    """
    sq = (pred_norm - target_norm) ** 2

    if circular_omega:
        mse_dims = [0, 1, 2]
        parts: list[torch.Tensor] = []
        dim_sum = dim_weight[0] + dim_weight[1] + dim_weight[2]
        for j in mse_dims:
            wj = sample_weight[:, j]
            denom = wj.sum().clamp(min=1e-8)
            mean_j = (sq[:, j] * wj).sum() / denom
            parts.append(dim_weight[j] * mean_j)
        w_om = 0.5 * (sample_weight[:, OMEGA_COS_IDX] + sample_weight[:, OMEGA_SIN_IDX])
        if w_om.sum() > 1e-8:
            om_loss = circular_omega_loss(
                pred_norm,
                target_norm,
                y_mean=y_mean,
                y_std=y_std,
                sample_weight=sample_weight,
            )
            om_dim = 0.5 * (dim_weight[OMEGA_COS_IDX] + dim_weight[OMEGA_SIN_IDX])
            parts.append(om_dim * om_loss)
            dim_sum = dim_sum + om_dim
        return torch.stack(parts).sum() / dim_sum.clamp(min=1e-8)

    parts = []
    for j in range(THETA_DIM):
        wj = sample_weight[:, j]
        denom = wj.sum().clamp(min=1e-8)
        mean_j = (sq[:, j] * wj).sum() / denom
        parts.append(dim_weight[j] * mean_j)
    return torch.stack(parts).sum() / dim_weight.sum().clamp(min=1e-8)


def apply_theta_constraints(y_pred: np.ndarray, *, constrain_e: bool = True, constrain_omega: bool = True) -> np.ndarray:
    """Project predictions to physical ranges: e clipped to [0, 0.99], (cos,sin) on unit circle."""
    out = np.asarray(y_pred, dtype=np.float64).copy()
    if constrain_e:
        out[:, 2] = np.clip(out[:, 2], 0.0, 0.99)
    if constrain_omega:
        cos_w, sin_w = out[:, 3], out[:, 4]
        norm = np.sqrt(cos_w ** 2 + sin_w ** 2)
        norm = np.where(norm < 1e-8, 1.0, norm)
        out[:, 3] = cos_w / norm
        out[:, 4] = sin_w / norm
    return out
