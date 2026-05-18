"""
models/kepler_torch.py
----------------------
Differentiable PyTorch Kepler decoder.

Given orbital parameters (P, K, e, ω, T_peri) and a time array, produces the
radial-velocity curve. All operations are torch-native so gradients flow through
the Kepler equation solve (Newton iteration) via autograd.

Batched shapes throughout:
  t          : (B, N) or (N,)  — observation times [days]
  P, K, e,
  omega,
  t_peri, γ  : (B,)            — one value per system in the batch

Single-system (B=1) usage via the convenience wrappers at the bottom.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def solve_kepler(M: torch.Tensor, e: torch.Tensor, tol: float = 1e-10,
                 maxiter: int = 50) -> torch.Tensor:
    """
    Solve Kepler's equation  M = E − e·sin(E)  for eccentric anomaly E.

    Uses Newton–Raphson with Danby starting guess.  The iteration is
    composed entirely of differentiable torch ops, so autograd flows through.

    Parameters
    ----------
    M : (...) — mean anomaly [radians], any shape
    e : broadcastable to M — eccentricity in [0, 1)

    Returns
    -------
    E : same shape as M
    """
    M = torch.remainder(M + torch.pi, 2 * torch.pi) - torch.pi  # wrap to (−π, π]
    E = M + e * torch.sin(M)  # Danby first-order guess

    for _ in range(maxiter):
        sin_E = torch.sin(E)
        cos_E = torch.cos(E)
        f  =  E - e * sin_E - M
        fp = 1.0 - e * cos_E          # df/dE — always > 0 for e < 1
        dE = -f / fp
        E  = E + dE
        if dE.abs().max().item() < tol:
            break

    return E


def true_anomaly(E: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Eccentric → true anomaly via the numerically robust arctan2 form."""
    return 2.0 * torch.atan2(
        (1.0 + e).sqrt() * torch.sin(E / 2.0),
        (1.0 - e).sqrt() * torch.cos(E / 2.0),
    )


def rv_keplerian(
    t: torch.Tensor,
    P: torch.Tensor,
    K: torch.Tensor,
    e: torch.Tensor,
    omega: torch.Tensor,
    t_peri: torch.Tensor,
) -> torch.Tensor:
    """
    RV contribution of one Keplerian planet (no γ offset).

    Parameters
    ----------
    t      : (B, N) or (N,)
    P, K, e, omega, t_peri : (B,) — broadcast over N automatically

    Returns
    -------
    rv : same shape as t
    """
    # Broadcast scalar params over time axis
    if t.dim() == 2:                   # (B, N)
        P      = P.unsqueeze(1)        # (B, 1)
        K      = K.unsqueeze(1)
        e      = e.unsqueeze(1)
        omega  = omega.unsqueeze(1)
        t_peri = t_peri.unsqueeze(1)

    M  = 2.0 * torch.pi * (t - t_peri) / P
    E  = solve_kepler(M, e)
    nu = true_anomaly(E, e)
    return K * (torch.cos(nu + omega) + e * torch.cos(omega))


def rv_model(
    t: torch.Tensor,
    P: torch.Tensor,
    K: torch.Tensor,
    e: torch.Tensor,
    omega: torch.Tensor,
    t_peri: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Full RV model: single planet + γ offset.

    For multi-planet systems, call rv_keplerian per planet and sum before
    adding γ.

    Parameters
    ----------
    t                 : (B, N) or (N,)
    P,K,e,omega,t_peri: (B,)
    gamma             : (B,)

    Returns
    -------
    rv : (B, N) or (N,)
    """
    v = rv_keplerian(t, P, K, e, omega, t_peri)
    if t.dim() == 2:
        gamma = gamma.unsqueeze(1)
    return v + gamma


# ---------------------------------------------------------------------------
# T_peri inner-loop refit (Option A from the design doc)
# ---------------------------------------------------------------------------

def fit_t_peri(
    t: torch.Tensor,
    rv_obs: torch.Tensor,
    P: torch.Tensor,
    K: torch.Tensor,
    e: torch.Tensor,
    omega: torch.Tensor,
    n_grid: int = 32,
    n_refine: int = 10,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Analytically refit T_peri given (P, K, e, ω) from encoder output.

    Implements Option A: differentiable 1-D inner-loop solve so the decoder
    can be used in train.py without T_peri in the encoder's theta.

    Phase convention: T_peri ∈ [t_min, t_min + P).

    Algorithm
    ---------
    1. Coarse grid over one period to locate the basin.
    2. Newton steps on dχ²/dT_peri to refine (fully differentiable).

    Parameters
    ----------
    t       : (B, N)
    rv_obs  : (B, N)
    P, K, e, omega : (B,)

    Returns
    -------
    t_peri : (B,)  — best-fit T_peri, no gradient (used as a constant in
                     the surrounding forward pass; gradients flow through
                     P/K/e/ω, not through the argmin itself)
    """
    B = t.shape[0]
    t_min = t.min(dim=1).values  # (B,)

    # ---- coarse grid ----
    phases = torch.linspace(0.0, 1.0, n_grid + 1, device=t.device)[:-1]  # (G,)
    # t_peri candidates: (B, G)
    tp_grid = t_min.unsqueeze(1) + phases.unsqueeze(0) * P.unsqueeze(1)

    best_tp   = t_min.clone()
    best_chi2 = torch.full((B,), float("inf"), device=t.device)

    for g in range(n_grid):
        tp_cand = tp_grid[:, g]                              # (B,)
        rv_pred = rv_keplerian(t, P, K, e, omega, tp_cand)  # (B, N)
        # γ from closed-form LS (mean offset)
        gamma_cand = (rv_obs - rv_pred).mean(dim=1)          # (B,)
        resid = rv_obs - rv_pred - gamma_cand.unsqueeze(1)   # (B, N)
        chi2 = (resid ** 2).mean(dim=1)                      # (B,)
        better = chi2 < best_chi2
        best_tp   = torch.where(better, tp_cand,   best_tp)
        best_chi2 = torch.where(better, chi2,       best_chi2)

    # ---- Newton refinement — gradient w.r.t. tp only, result detached ----
    tp = best_tp.clone().requires_grad_(True)
    for _ in range(n_refine):
        with torch.enable_grad():
            rv_pred = rv_keplerian(t, P, K, e, omega, tp)
            gamma   = (rv_obs - rv_pred).mean(dim=1, keepdim=True)
            resid   = rv_obs - rv_pred - gamma
            chi2    = (resid ** 2).mean(dim=1).sum()
            (grad,) = torch.autograd.grad(chi2, tp)
        if grad.abs().max() < tol:
            break
        tp = (tp - 0.1 * grad).detach().requires_grad_(True)

    return tp.detach()


# ---------------------------------------------------------------------------
# Decoder module (nn.Module wrapper for use in train.py)
# ---------------------------------------------------------------------------

class KeplerDecoder(nn.Module):
    """
    Stateless decoder: encoder output → reconstructed RV curve.

    Encoder theta has 5 dims: [log10_P, log10_K, e, cos_ω, sin_ω].
    T_peri is refitted analytically per forward pass (Option A).
    γ is removed from the input via median subtraction (preprocess.py)
    and is refit here as a closed-form LS offset.

    Forward signature
    -----------------
    theta  : (B, 5)  — encoder output in normalised space
    t      : (B, N)  — normalised times (row 0 of the preprocess tensor,
                        scaled back to days inside this module)
    t_span : (B,)    — observation span in days (needed to de-normalise t)
    t_min  : (B,)    — t[0] in days for each system
    rv_obs : (B, N)  — normalised RV observations (row 1 of preprocess tensor)
    rv_std : (B,)    — per-system RV std [m/s] for de-normalisation
    mask   : (B, N)  — 1 for real obs, 0 for padding

    Returns
    -------
    rv_pred_norm : (B, N) — predicted RV in the same normalised units as
                            rv_obs (i.e. divided by rv_std, zero-median)
    """

    def forward(
        self,
        theta: torch.Tensor,
        t_norm: torch.Tensor,
        t_span: torch.Tensor,
        t_min: torch.Tensor,
        rv_obs: torch.Tensor,
        rv_std: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack and transform encoder outputs
        log10_P, log10_K, e, cos_w, sin_w = theta.unbind(dim=1)  # each (B,)
        P     = 10.0 ** log10_P                 # days
        K     = 10.0 ** log10_K                 # m/s
        e     = e.clamp(0.0, 0.99)
        omega = torch.atan2(sin_w, cos_w)       # radians

        # De-normalise time: t_norm ∈ [0,1] → days
        t_days = t_norm * t_span.unsqueeze(1) + t_min.unsqueeze(1)

        # De-normalise RV: rv_obs is (rv − median)/std → rv in m/s
        # (median was removed; std is rv_std)
        rv_ms = rv_obs * rv_std.unsqueeze(1)    # (B, N), zero-median in m/s

        # Refit T_peri analytically (no gradient)
        # Mask padding out before the refit
        rv_masked = rv_ms * mask
        t_peri = fit_t_peri(t_days, rv_masked, P, K, e, omega)

        # Predict RV in m/s
        rv_pred_ms = rv_keplerian(t_days, P, K, e, omega, t_peri)

        # Closed-form γ from LS (mean of masked residuals)
        n_obs   = mask.sum(dim=1, keepdim=True).clamp(min=1)
        gamma   = ((rv_ms - rv_pred_ms) * mask).sum(dim=1, keepdim=True) / n_obs
        rv_pred_ms = rv_pred_ms + gamma

        # Re-normalise to match rv_obs units
        rv_pred_norm = rv_pred_ms / rv_std.unsqueeze(1)
        return rv_pred_norm


# ---------------------------------------------------------------------------
# Single-system convenience wrappers (numpy in, numpy out)
# ---------------------------------------------------------------------------

def predict_rv_numpy(
    t: "np.ndarray",
    P: float,
    K: float,
    e: float,
    omega: float,
    t_peri: float,
    gamma: float = 0.0,
    device: str = "cpu",
) -> "np.ndarray":
    """Thin wrapper for interactive use / kepler_check integration."""
    import numpy as np
    t_t = torch.tensor(t, dtype=torch.float64, device=device).unsqueeze(0)
    kwargs = dict(dtype=torch.float64, device=device)
    P_t     = torch.tensor([P],      **kwargs)
    K_t     = torch.tensor([K],      **kwargs)
    e_t     = torch.tensor([e],      **kwargs)
    om_t    = torch.tensor([omega],  **kwargs)
    tp_t    = torch.tensor([t_peri], **kwargs)
    gam_t   = torch.tensor([gamma],  **kwargs)
    with torch.no_grad():
        out = rv_model(t_t, P_t, K_t, e_t, om_t, tp_t, gam_t)
    return out.squeeze(0).cpu().numpy()
