"""Shared Kepler regression losses: theta (e, cos ω, sin ω) and h/k (k=e cos ω, h=e sin ω)."""

from __future__ import annotations

import numpy as np
import torch

OMEGA_EPS = 0.05
OMEGA_GATE_WIDTH = 40.0

THETA_DIM = 5
OMEGA_COS_IDX = 3
OMEGA_SIN_IDX = 4

HK_NAMES = ["log10_P", "log10_K", "k", "h"]
HK_DIM = len(HK_NAMES)


def omega_gate_weights(
    e_phys: np.ndarray | torch.Tensor,
    *,
    gate_center: float = OMEGA_EPS,
    gate_width: float = OMEGA_GATE_WIDTH,
) -> np.ndarray | torch.Tensor:
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
    """Per-sample weights (n, 5). Zero e/ω when has_ecc is False; mask ω at low e."""
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


def e_balance_weights(
    e_train: np.ndarray,
    e_query: np.ndarray | None = None,
    *,
    n_bins: int = 20,
    max_ratio: float = 10.0,
) -> np.ndarray:
    """Inverse-frequency weights for zero-inflated e (mean 1 on train)."""
    e_train = np.asarray(e_train, dtype=np.float64)
    if len(e_train) == 0:
        raise ValueError("e_train is empty")

    def _cats(e: np.ndarray) -> np.ndarray:
        e = np.clip(np.asarray(e, dtype=np.float64), 0.0, 1.0)
        binned = 1 + np.minimum((e * n_bins).astype(int), n_bins - 1)
        return np.where(e <= 0.0, 0, binned)

    cats_train = _cats(e_train)
    counts = np.bincount(cats_train, minlength=n_bins + 1).astype(np.float64)
    seen = counts > 0
    w_bin = np.full(n_bins + 1, np.inf)
    w_bin[seen] = len(e_train) / counts[seen]
    w_bin = np.minimum(w_bin, max_ratio * w_bin[seen].min())
    w_bin /= np.mean(w_bin[cats_train])
    return w_bin[_cats(e_train if e_query is None else e_query)]


def normalize_omega_tensor(cos_sin: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(cos_sin, dim=-1, keepdim=True).clamp(min=1e-8)
    return cos_sin / norm


def denorm_omega_components(
    theta_norm: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
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
    """1 - cos(Δω) on unit (cos, sin); sample_weight is (B, 5)."""
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
    """MSE on P/K/e; circular or MSE on ω. sample_weight (B,5), dim_weight (5,)."""
    sq = (pred_norm - target_norm) ** 2

    if circular_omega:
        parts: list[torch.Tensor] = []
        part_w: list[torch.Tensor] = []
        for j in (0, 1, 2):
            wj = sample_weight[:, j]
            denom = wj.sum().clamp(min=1e-8)
            parts.append((sq[:, j] * wj).sum() / denom)
            part_w.append(dim_weight[j])
        w_om = 0.5 * (sample_weight[:, OMEGA_COS_IDX] + sample_weight[:, OMEGA_SIN_IDX])
        if w_om.sum() > 1e-8:
            om_loss = circular_omega_loss(
                pred_norm, target_norm, y_mean=y_mean, y_std=y_std, sample_weight=sample_weight
            )
            parts.append(om_loss)
            part_w.append(0.5 * (dim_weight[OMEGA_COS_IDX] + dim_weight[OMEGA_SIN_IDX]))
        pw = torch.stack(part_w)
        return (torch.stack(parts) * pw).sum() / pw.sum().clamp(min=1e-8)

    w = sample_weight * dim_weight.unsqueeze(0)
    denom = w.sum().clamp(min=1e-8)
    return (sq * w).sum() / denom


def apply_theta_constraints(
    y_pred: np.ndarray, *, constrain_e: bool = True, constrain_omega: bool = True
) -> np.ndarray:
    out = np.asarray(y_pred, dtype=np.float64).copy()
    if constrain_e:
        out[:, 2] = np.clip(out[:, 2], 0.0, 0.99)
    if constrain_omega:
        cos_w, sin_w = out[:, 3], out[:, 4]
        norm = np.sqrt(cos_w**2 + sin_w**2)
        norm = np.where(norm < 1e-8, 1.0, norm)
        out[:, 3] = cos_w / norm
        out[:, 4] = sin_w / norm
    return out


def theta_to_hk(y_theta: np.ndarray) -> np.ndarray:
    y = np.asarray(y_theta, dtype=np.float64)
    e, c, s = y[:, 2], y[:, 3], y[:, 4]
    return np.column_stack([y[:, 0], y[:, 1], e * c, e * s])


def hk_to_theta(y_hk: np.ndarray) -> np.ndarray:
    y = np.asarray(y_hk, dtype=np.float64)
    k, h = y[:, 2], y[:, 3]
    e = np.sqrt(k * k + h * h)
    cos_w = np.ones_like(e)
    sin_w = np.zeros_like(e)
    mask = e > 1e-8
    cos_w[mask] = k[mask] / e[mask]
    sin_w[mask] = h[mask] / e[mask]
    return np.column_stack([y[:, 0], y[:, 1], e, cos_w, sin_w])


def apply_hk_constraints(y_hk: np.ndarray, *, e_max: float = 0.99) -> np.ndarray:
    out = np.asarray(y_hk, dtype=np.float64).copy()
    e = np.sqrt(out[:, 2] ** 2 + out[:, 3] ** 2)
    scale = np.ones_like(e)
    over = e > e_max
    scale[over] = e_max / e[over]
    out[:, 2] *= scale
    out[:, 3] *= scale
    return out


def theta_loss_weights_hk_numpy(
    y_hk: np.ndarray,
    *,
    has_ecc: np.ndarray | None = None,
) -> np.ndarray:
    n = len(y_hk)
    w = np.ones((n, HK_DIM), dtype=np.float64)
    if has_ecc is not None:
        known = np.asarray(has_ecc, dtype=bool)
        w[~known, 2:4] = 0.0
    return w


def regression_hk_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    sample_weight: torch.Tensor,
    dim_weight: torch.Tensor,
) -> torch.Tensor:
    sq = (pred_norm - target_norm) ** 2
    w = sample_weight * dim_weight.unsqueeze(0)
    denom = w.sum().clamp(min=1e-8)
    return (sq * w).sum() / denom
