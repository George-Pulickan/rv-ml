"""
conformal_shift.py — Step 6 v2: split-CP calibrated on fake data, tested on real,
per Nicolò's 2026-07 spec.

Compares two conformity-score strategies, both built around the Step-5 point
predictor psi(y) = RF(features(y)); psi's feature columns are taken from the
--csv dataset (default: the 512-bin raw-LSP dataset, per Nicolò's OK on more
Fourier bins — the 64-bin sum-normalized spectrum loses the period peak):

  (i)  naive      s_c = | psi(y)_c - theta_bar_c |   with theta_bar the ground-
       truth data-generating parameter (known for every synthetic curve);
  (ii) surrogate  s_c = | psi(y)_c - theta*_c |      with theta* the numerical
       solution of argmin_t mean_i | y_i - kepler(t)_i |  (batched Adam gradient
       descent through the differentiable KeplerDecoder, initialized at the
       data-generating / tabulated values — Nicolò's 2026-07 spec) — computable
       on real curves too (tabulated init), hence usable under shift.

Calibration uses ONLY synthetic (fake) curves drawn from the empirical priors H
(which are themselves fit on the real TRAIN split only); real val/test systems
are reserved for testing the intervals.  Scores are per-coordinate over the four
physical coordinates (log10_P, log10_K, e, omega — circular distance for omega)
with a Bonferroni 1 - alpha/d quantile; the interval for coordinate c is
psi(y)_c ± q_c, intersected with the empirical support of H.

Covariate-shift reweighting (Tibshirani, Barber, Candès, Ramdas 2019): the
calibration scores are reweighted by the likelihood ratio w(x) = p_real(x) /
p_fake(x), estimated by a logistic real-vs-synthetic discriminator on the
summary features via the odds p/(1-p) (fit on a *separate* synthetic sample vs
the real TRAIN split — never on the real test systems).  The weighted quantile
puts mass w(x_test) at +infinity, so sets can be infinite when the test point
looks very real relative to the calibration cloud; weights are clipped
(--clip-weights) and the effective sample size is reported.

Noise-model normalization (Nicolò confirmed 2026-07 that s' = s/(gamma + v)
is what he meant), three variants:

    vnorm     s'_c = s_c / (gamma + v_y)
    v2norm    s'_c = s_c / (gamma + v_y + v_c)   (his two-factor version)
    papernorm s'_c = s_c / (gamma + delta_c + delta_y)   (Overleaf eqs 18-24)

with v_y = RMS predictive std of the trained SVGP residual noise model
evaluated on (kepler(psi(y)), psi(y), t), in units of the curve's rv_std
(falls back to the median measurement sigma when the checkpoint is
unavailable), and v_c = a per-coordinate model of the surrogate-label error
E|theta_bar_c - theta*_c| (an RF on the summary features, fit on the synthetic
tuning set where theta_bar is known).  gamma > 0 is tuned per variant to
minimize the mean support-normalized median interval width, on the synthetic
tuning set by default or on the real val split with --gamma-tune-on real-val
(the paper's D_val; label-free, since only widths are measured).

papernorm follows the paper draft's "Profiled uncertainty estimation" section:
delta_c(y) is the re-encoding residual |psi_c(h(psi(y))) - psi_c(y)| (the
noiseless reconstruction is re-encoded with the observation's time grid and
measurement sigmas, then passed through psi again — eq 18) and delta_y(y) is
the mean-absolute reconstruction error mean_t |y_t - h(psi(y), t)| in rv_std
units (eq 19).  Since h and psi are deterministic both are computed pointwise
per curve — no separate model is trained (Nicolò 2026-07-14).

Paper-spec additions (Overleaf Theory section):
  * naive_adj — the naive strategy with the surrogate-gap quantile adjustment
    q_alpha = q~_alpha + Delta_c (eq 41): the naive calibration scores are
    inflated by Delta_c = max over the tuning set of |theta_bar_c - theta*_c|
    (the empirical stand-in for eps*C_noise*C_H*C_Delta; the distribution of
    the gap is reported so the max/p90/median choice can be revisited).
  * Assumption 2.1 (bounded noise) filter — synthetic draws whose max_t
    |y_t - kepler(theta_bar, t)| (rv_std units) exceeds the bound estimated
    on real TRAIN curves via max_y max_t |y_t - kepler(psi(y), t)| are
    discarded at generation time (--no-noise-filter to disable); the bound
    and rejection rate are reported.
  * Assumption 2.3 constants — kappa(H) (finite-difference Hessian of the L2
    reconstruction loss, 5-dim decoder parameterization) and ||grad h||
    (autograd Jacobian spectral norm) are estimated on --n-constants prior
    draws and reported (eps*C_noise from the real-train bound above).
  * --psi-labels star — ablation (Nicolò 2026-07-14): train psi on the GD
    surrogate labels theta* of the training CSV rows (replayed and fit with
    the same L1 gradient descent, init at theta_bar; cached to an .npz next
    to the CSV) instead of the data-generating theta_bar.
  * figures/filter_param_histograms.png — per-coordinate histograms of real
    tabulated parameters vs accepted vs filter-rejected synthetic draws (for
    the paper's figure-caption discussion of the Assumption 2.1 truncation).

Usage
-----
    python conformal_shift.py                          # full run, n=400
    python conformal_shift.py --n-cal 60 --n-test 60 --n-tune 30   # quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from conformal import (
    COORDS,
    D,
    SG,
    Scorer,
    _curve_from_x,
    _theta_to_omega,
    _true_coord,
    histogram_grids,
    make_real,
    make_synthetic,
)
from generate_synthetic_regression_csv import (  # noqa: E402
    corpus_orbital_params,
    replay_synthetic_sample,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from train_regression_models import _build  # noqa: E402
from eval_omega_nn_vs_rf import _summary_row  # noqa: E402
from preprocess import compute_lsp

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_mlp_psi(checkpoint: Path, device: torch.device):
    """Load regression.py MLP (or DualEHead) checkpoint as psi: X -> theta5."""
    from regression import build_model_from_checkpoint, predict

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model, norm_stats = build_model_from_checkpoint(ckpt, device)
    in_dim = int(norm_stats.get("in_dim", len(norm_stats["x_mean"])))

    def psi_predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D feature matrix, got shape {X.shape}")
        if X.shape[1] != in_dim:
            raise ValueError(
                f"MLP expects {in_dim} features, got {X.shape[1]}; "
                f"pass a matching --csv (e.g. synthetic_regression_10000.csv for 74-D)"
            )
        return predict(model, X, norm_stats, device)

    return psi_predict, norm_stats

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
STRATEGIES = ["naive", "naive_adj", "surrogate"]
NORMS = ["raw", "vnorm", "v2norm", "papernorm"]


# ---------------------------------------------------------------------------
# Per-coordinate score: absolute error in theta space (circular for omega)
# ---------------------------------------------------------------------------


def _coord_abs_err(theta_a5: np.ndarray, theta_b5: np.ndarray, coord: str) -> float:
    a, b = _true_coord(theta_a5, coord), _true_coord(theta_b5, coord)
    if coord == "omega":
        return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))
    return float(abs(a - b))


# ---------------------------------------------------------------------------
# Surrogate label theta* = argmin_theta mean_t | y_t - kepler(theta)_t |
# ---------------------------------------------------------------------------


def _gd_batch(decoder, init5s: np.ndarray, curves: list, steps: int,
              lr: float) -> np.ndarray:
    """Batched Adam minimisation of the masked mean-absolute reconstruction
    error (Nicolò's E_t |y_t - kepler(theta', t)|, in rv_std units) over a set
    of curves sharing one padded length.  t_peri / gamma are refit analytically
    inside the decoder every step (their refit is detached — envelope-style;
    gradients flow through the RV evaluation).  The best iterate per curve is
    kept, so the fit can only improve on the initialization."""
    th = torch.as_tensor(np.asarray(init5s), dtype=torch.float32).clone()  # (B,5)
    t_norm = torch.from_numpy(np.stack([c["t_norm"] for c in curves]))
    rv_obs = torch.from_numpy(np.stack([c["rv_obs"] for c in curves]))
    mask = torch.from_numpy(np.stack([c["mask"] for c in curves]))
    t_span = torch.tensor([c["t_span"] for c in curves], dtype=torch.float32)
    t_min = torch.tensor([c["t_min"] for c in curves], dtype=torch.float32)
    rv_std = torch.tensor([c["rv_std"] for c in curves], dtype=torch.float32)
    n = mask.sum(dim=1).clamp(min=1.0)

    def losses(theta: torch.Tensor) -> torch.Tensor:
        rv_pred = decoder(theta, t_norm, t_span, t_min, rv_obs, rv_std, mask)
        return ((rv_obs - rv_pred).abs() * mask).sum(dim=1) / n            # (B,)

    th.requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    best_loss = torch.full((th.shape[0],), np.inf)
    best_th = th.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        loss = losses(th)
        with torch.no_grad():
            better = loss < best_loss
            best_loss[better] = loss.detach()[better]
            best_th[better] = th.detach()[better]
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            th[:, 2].clamp_(0.0, 0.99)
    with torch.no_grad():
        loss = losses(th)
        better = loss < best_loss
        best_th[better] = th.detach()[better]
    return best_th.numpy().astype(float)


def surrogate_fit_gd(decoder, init5s: list, systems: list, steps: int = 200,
                     lr: float = 0.02) -> list:
    """Surrogate labels for a list of systems, batched by padded curve length,
    initialized at init5s (= theta_bar on synthetic curves, tabulated values on
    real ones, per Nicolò 2026-07)."""
    out: list = [None] * len(systems)
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(systems):
        by_len.setdefault(len(s["curve"]["t_norm"]), []).append(i)
    for idx in by_len.values():
        fitted = _gd_batch(decoder, np.asarray([init5s[i] for i in idx]),
                           [systems[i]["curve"] for i in idx], steps, lr)
        for k, i in enumerate(idx):
            out[i] = fitted[k]
    return out


# ---------------------------------------------------------------------------
# Noise-model uncertainty proxy v(kepler(theta_hat), theta_hat, t)
# ---------------------------------------------------------------------------


class NoiseProxy:
    """v = RMS predictive std of the trained SVGP residual model on the
    predicted curve's feature rows, divided by the curve's rv_std (dimensionless
    noise fraction).  Falls back to the median per-obs measurement sigma (also
    in rv_std units) when the checkpoint can't be loaded."""

    def __init__(self):
        from synthetic_dataset import _load_gp_residual_sampler

        self.decoder = Scorer().decoder
        self.sampler = _load_gp_residual_sampler()
        self.source = "gp_residual_svgp" if self.sampler is not None else "median_sigma"
        self.wants_sigma = (self.sampler is not None
                            and "log10_sigma" in self.sampler["feature_names"])

    @torch.no_grad()
    def _pred_curve_ms(self, theta5: np.ndarray, curve: dict) -> tuple[np.ndarray, np.ndarray]:
        """Decoder reconstruction at theta5 in m/s, plus the day-valued times,
        masked to the observed entries."""
        t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0)
        rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0)
        mask = torch.from_numpy(curve["mask"]).unsqueeze(0)
        t_span = torch.tensor([curve["t_span"]], dtype=torch.float32)
        t_min = torch.tensor([curve["t_min"]], dtype=torch.float32)
        rv_std = torch.tensor([curve["rv_std"]], dtype=torch.float32)
        th = torch.as_tensor(theta5[None, :], dtype=torch.float32)
        rv_pred = self.decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)[0].numpy()
        m = curve["mask"] > 0.5
        t_days = curve["t_norm"][m] * curve["t_span"] + curve["t_min"]
        return rv_pred[m] * curve["rv_std"], t_days

    @torch.no_grad()
    def value(self, theta5: np.ndarray, curve: dict) -> float:
        m = curve["mask"] > 0.5
        fallback = float(np.median(curve["sig"][m]))
        if self.sampler is None:
            return max(fallback, 1e-6)
        try:
            from synthetic_dataset import _gp_residual_features

            rv_ms, t_days = self._pred_curve_ms(theta5, curve)
            params = {"P": 10.0 ** theta5[0], "K": 10.0 ** theta5[1],
                      "e": theta5[2], "omega": _theta_to_omega(theta5)}
            sig_ms = curve["sig"][m] * curve["rv_std"] if self.wants_sigma else None
            X = _gp_residual_features(t_days, rv_ms, params, sigma=sig_ms)
            X = (X - self.sampler["mean"]) / np.maximum(self.sampler["std"], 1e-8)
            latent = self.sampler["model"](torch.as_tensor(X, dtype=torch.float32))
            var = latent.variance
            lik = self.sampler["likelihood"]
            try:
                df = float(lik.deg_free)
                noise = float(lik.noise)
                var = var + noise * (df / (df - 2.0) if df > 2.0 else 1.0)
            except Exception:
                pass
            v_ms = float(torch.sqrt(var.mean()))
            return max(v_ms / curve["rv_std"], 1e-6)
        except Exception:
            return max(fallback, 1e-6)


# ---------------------------------------------------------------------------
# Paper-spec conditional residuals delta_c / delta_y (Overleaf eqs 18-19),
# the Assumption 2.1 noise-bound filter, and Assumption 2.3 constants
# ---------------------------------------------------------------------------


def recon_residual_norm(proxy: "NoiseProxy", theta5: np.ndarray, curve: dict) -> np.ndarray:
    """Masked residual y_t - kepler(theta5, t) in the curve's rv_std units."""
    rv_ms, _ = proxy._pred_curve_ms(theta5, curve)
    m = curve["mask"] > 0.5
    return (curve["rv_obs"][m] * curve["rv_std"] - rv_ms) / curve["rv_std"]


def reencode_features(proxy: "NoiseProxy", theta5: np.ndarray, curve: dict) -> tuple[dict, np.ndarray]:
    """feat_row + LSP of the noiseless reconstruction h(psi(y)) so it can be
    passed through psi again (eq 18).  The reconstruction keeps the
    observation's time grid and per-obs measurement sigmas (sigma belongs to
    the instrument, not the noise realization) and is normalized by its own
    std, exactly as a fresh observed curve would be."""
    rv_ms, t_days = proxy._pred_curve_ms(theta5, curve)
    m = curve["mask"] > 0.5
    sig_ms = curve["sig"][m] * curve["rv_std"]
    std = max(float(np.std(rv_ms)), 1e-6)
    xm = np.stack([
        curve["t_norm"][m],
        (rv_ms - np.median(rv_ms)) / std,
        sig_ms / std,
        np.ones(int(m.sum()), dtype=np.float32),
    ])
    info = {"rv_std_ms": std, "t_span_days": curve["t_span"], "n_obs": int(m.sum())}
    lsp = compute_lsp(t_days, rv_ms, sig_ms)
    return _summary_row(xm, info, lsp), np.asarray(lsp, dtype=float)


def psi_star_labels(n_rows: int, csv_seed: int, decoder, gd_steps: int, gd_lr: float,
                    cache_path: Path, limit: int = 0) -> np.ndarray:
    """GD surrogate labels theta* for the training CSV rows (--psi-labels star):
    replay each row's curve, run the same L1 gradient descent initialized at
    the data-generating theta_bar.  Cached to an .npz keyed by (seed, n, steps).
    limit > 0 caps the rows (smoke only — shrinks psi's training set)."""
    n_use = min(n_rows, limit) if limit > 0 else n_rows
    if cache_path.exists():
        z = np.load(cache_path)
        if (int(z["seed"]) == csv_seed and int(z["n_rows"]) == n_use
                and int(z["gd_steps"]) == gd_steps):
            print(f"psi* labels: loaded cache {cache_path}")
            return z["theta_star"]
    print(f"psi* labels: replaying {n_use} CSV rows + GD ({gd_steps} steps) ...")
    params = corpus_orbital_params(csv_seed, n_rows)
    systems = []
    for i in range(n_use):
        x, _, theta, info = replay_synthetic_sample(i, csv_seed, n_rows, f_multi=0.0,
                                                    params=params)
        systems.append({"curve": _curve_from_x(x, info),
                        "theta5": np.asarray(theta, dtype=float)})
    stars = np.stack(surrogate_fit_gd(decoder, [s["theta5"] for s in systems],
                                      systems, gd_steps, gd_lr))
    np.savez(cache_path, theta_star=stars, seed=csv_seed, n_rows=n_use,
             gd_steps=gd_steps)
    print(f"psi* labels: cached -> {cache_path}")
    return stars


def plot_filter_histograms(real_thetas: list, accepted: list, rejected: list,
                           fig_dir: Path) -> None:
    """Per-coordinate histograms: real tabulated vs accepted vs filter-rejected
    synthetic parameters (the Assumption 2.1 truncation figure)."""
    fig, axs = plt.subplots(1, D, figsize=(4.2 * D, 3.6))
    groups = [("real tabulated", real_thetas, "k"),
              ("synthetic accepted", accepted, "tab:blue"),
              ("synthetic rejected", rejected, "tab:red")]
    for ax, c in zip(axs, COORDS):
        for label, thetas, color in groups:
            if not len(thetas):
                continue
            vals = [_true_coord(t, c) for t in thetas]
            ax.hist(vals, bins=25, density=True, histtype="step", lw=1.6,
                    color=color, label=f"{label} (n={len(thetas)})")
        ax.set_title(c)
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=7)
    fig.suptitle("Assumption 2.1 noise filter: parameter distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fig_dir / "filter_param_histograms.png", dpi=180)
    plt.close(fig)


def noise_bound_from_real(proxy: "NoiseProxy", systems: list, theta_hats: list) -> float:
    """Assumption 2.1 bound (eps*C_noise estimate): max over real curves of
    max_t |y_t - kepler(psi(y), t)| in rv_std units.  |y - h(psi(y))| upper-
    bounds |y - h(theta)| per the paper's Theory section."""
    return float(max(np.abs(recon_residual_norm(proxy, th, s["curve"])).max()
                     for s, th in zip(systems, theta_hats)))


def make_synthetic_filtered(n: int, seed: int, bound: float | None,
                            proxy: "NoiseProxy",
                            max_tries: int = 6) -> tuple[list, int, list]:
    """make_synthetic + the Assumption 2.1 discard rule: reject draws whose
    max_t |y_t - kepler(theta_bar, t)| (rv_std units) exceeds the real-data
    noise bound.  Returns (accepted systems, number generated, rejected theta5s
    — for the truncation histogram figure)."""
    if bound is None:
        return make_synthetic(n, seed), n, []
    out: list = []
    rejected: list = []
    n_gen = 0
    for k in range(max_tries):
        batch = make_synthetic(n, seed + 131 * k)
        n_gen += len(batch)
        for s in batch:
            stat = float(np.abs(recon_residual_norm(proxy, s["theta5"], s["curve"])).max())
            if stat <= bound:
                out.append(s)
            else:
                rejected.append(s["theta5"])
            if len(out) == n:
                return out, n_gen, rejected
    raise RuntimeError(f"noise-bound filter rejected too much: {len(out)}/{n} "
                       f"accepted after {n_gen} draws (bound={bound:.3g})")


def estimate_constants(proxy: "NoiseProxy", n: int, seed: int) -> dict:
    """Empirical Assumption 2.3 constants on prior draws, in the 5-dim decoder
    parameterization: kappa(H) with H the finite-difference Hessian of the L2
    reconstruction loss at theta_bar, and ||grad h|| the spectral norm of the
    autograd Jacobian of the masked reconstruction."""
    systems = make_synthetic(n, seed)
    kappas, grad_norms = [], []
    eps = 1e-3
    for s in systems:
        curve = s["curve"]
        th0 = np.asarray(s["theta5"], dtype=np.float64)

        def loss(th: np.ndarray) -> float:
            r = recon_residual_norm(proxy, th.astype(np.float64), curve)
            return float(np.mean(r ** 2))

        H = np.zeros((5, 5))
        f0 = loss(th0)
        for i in range(5):
            for j in range(i, 5):
                ei, ej = np.eye(5)[i] * eps, np.eye(5)[j] * eps
                if i == j:
                    H[i, i] = (loss(th0 + ei) - 2 * f0 + loss(th0 - ei)) / eps ** 2
                else:
                    H[i, j] = H[j, i] = (
                        loss(th0 + ei + ej) - loss(th0 + ei - ej)
                        - loss(th0 - ei + ej) + loss(th0 - ei - ej)
                    ) / (4 * eps ** 2)
        sv = np.linalg.svd(H, compute_uv=False)
        kappas.append(float(sv.max() / max(sv.min(), 1e-12)))

        m = torch.from_numpy(curve["mask"]).unsqueeze(0)
        t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0)
        rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0)
        t_span = torch.tensor([curve["t_span"]], dtype=torch.float32)
        t_min = torch.tensor([curve["t_min"]], dtype=torch.float32)
        rv_std = torch.tensor([curve["rv_std"]], dtype=torch.float32)

        def h_fn(th: torch.Tensor) -> torch.Tensor:
            rv = proxy.decoder(th.unsqueeze(0), t_norm, t_span, t_min, rv_obs, rv_std, m)
            return (rv * m)[0]

        J = torch.autograd.functional.jacobian(
            h_fn, torch.as_tensor(th0, dtype=torch.float32))
        grad_norms.append(float(torch.linalg.matrix_norm(J, ord=2)))

    def _pct(v: list) -> dict:
        a = np.asarray(v)
        return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "max": float(a.max())}

    return {"n_draws": n, "kappa_H": _pct(kappas), "grad_h_spectral_norm": _pct(grad_norms)}


# ---------------------------------------------------------------------------
# Surrogate-label error model v_c(y) ~ E | theta_bar_c - theta*_c |
# ---------------------------------------------------------------------------


def fit_vk_models(feats: np.ndarray, theta_bars: list, theta_stars: list,
                  seed: int, n_estimators: int = 200):
    """Per-coordinate model of the surrogate-label error (the second reweighting
    factor in Nicolò's s'_c = s_c / (gamma + v_y + v_c), 2026-07): an RF
    regressor of |theta_bar_c - theta*_c| on the summary features, fit on the
    synthetic tuning set where theta_bar is known.  Returns vk(feats) ->
    {coord: (n,) nonnegative array}."""
    from sklearn.ensemble import RandomForestRegressor

    models = {}
    for c in COORDS:
        errs = np.array([_coord_abs_err(tb, ts, c)
                         for tb, ts in zip(theta_bars, theta_stars)])
        m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=-1)
        m.fit(feats, errs)
        models[c] = m

    def vk(f: np.ndarray) -> dict:
        return {c: np.maximum(models[c].predict(f), 0.0) for c in COORDS}

    return vk


# ---------------------------------------------------------------------------
# Likelihood-ratio weights via a real-vs-fake discriminator
# ---------------------------------------------------------------------------


def fit_weight_model(synth_feats: np.ndarray, real_feats: np.ndarray, seed: int):
    """Logistic real(1)-vs-fake(0) discriminator on standardized log-ish summary
    features; w(x) = odds * class-balance correction = p_real/p_fake ratio."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([synth_feats, real_feats])
    y = np.r_[np.zeros(len(synth_feats)), np.ones(len(real_feats))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    clf.fit(X, y)
    n0, n1 = float(len(synth_feats)), float(len(real_feats))

    def w(feats: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(feats)[:, 1].clip(1e-6, 1 - 1e-6)
        return (p / (1.0 - p)) * (n0 / n1)

    return w, clf


def weighted_quantile(scores: np.ndarray, w_cal: np.ndarray, w_test: float,
                      level: float) -> float:
    """Tibshirani et al. 2019 weighted conformal quantile: normalized weights
    p_i = w_i / (sum_j w_j + w_test) on the calibration scores plus mass
    w_test / (...) at +infinity.  All-ones weights reduce to the standard
    ceil((n+1)*level) rank quantile."""
    order = np.argsort(scores)
    s, ww = scores[order], w_cal[order]
    total = float(ww.sum() + w_test)
    k = int(np.searchsorted(np.cumsum(ww), level * total, side="left"))
    if k >= len(s):
        return math.inf
    return float(s[k])


# ---------------------------------------------------------------------------
# Interval construction + evaluation
# ---------------------------------------------------------------------------


def _support(grids: dict) -> dict:
    sup = {c: (float(grids[c].min()), float(grids[c].max())) for c in COORDS}
    sup["omega"] = (0.0, 2.0 * np.pi)
    return sup


def _interval_width(coord: str, center: float, half: float, sup: dict) -> float:
    lo, hi = sup[coord]
    if coord == "omega":
        return float(min(2.0 * half, hi - lo))
    return float(max(0.0, min(center + half, hi) - max(center - half, lo)))


def evaluate(cal_scores: dict, systems: list, theta_hats: list, sup: dict,
             den_cal: dict | None = None, den_sys: dict | None = None,
             w_cal: np.ndarray | None = None, w_test: np.ndarray | None = None) -> dict:
    """Coverage/width of the per-coordinate intervals psi(y)_c ± q_c at each alpha.

    den_cal/den_sys=None -> raw score s;  otherwise per-coordinate denominator
    arrays (gamma + v_y [+ v_c]) and the normalized score s' = s/den is used
    (interval half-width scales back by den at the test point).
    w_cal/w_test=None -> unweighted split-CP.
    """
    n_test = len(systems)
    n_cal = len(next(iter(cal_scores.values())))
    w_cal = np.ones(n_cal) if w_cal is None else w_cal
    w_test = np.ones(n_test) if w_test is None else w_test

    out = {}
    for a in ALPHAS:
        level = 1.0 - a / D
        per_cov = {c: [] for c in COORDS}
        per_w = {c: [] for c in COORDS}
        n_inf = 0
        joint = []
        for i, (s, th) in enumerate(zip(systems, theta_hats)):
            all_c = True
            any_inf = False
            for c in COORDS:
                sc = cal_scores[c] if den_cal is None else cal_scores[c] / den_cal[c]
                q = weighted_quantile(sc, w_cal, float(w_test[i]), level)
                half = q if den_sys is None else q * float(den_sys[c][i])
                if not math.isfinite(half):
                    any_inf = True
                    half = sup[c][1] - sup[c][0]     # cap at full support
                err = _coord_abs_err(s["theta5"], th, c)
                cov = err <= half
                per_cov[c].append(bool(cov))
                per_w[c].append(_interval_width(c, _true_coord(th, c), half, sup))
                all_c = all_c and cov
            n_inf += int(any_inf)
            joint.append(all_c)
        out[f"{a:.2f}"] = {
            "per_coord_coverage": {c: float(np.mean(per_cov[c])) for c in COORDS},
            "per_coord_median_width": {c: float(np.median(per_w[c])) for c in COORDS},
            "joint_coverage": float(np.mean(joint)),
            "frac_infinite": n_inf / max(n_test, 1),
        }
    return out


def tune_gamma(cal_scores: dict, base_cal: dict, tune_hats: list,
               base_tune: dict, sup: dict, alpha: float = 0.10) -> float:
    """Pick gamma minimizing the mean (over coords) support-normalized median
    width on the synthetic tuning set, at the reference alpha.  base_cal /
    base_tune are per-coordinate denominator bases (v_y or v_y + v_c); the
    tuned denominator is gamma + base."""
    level = 1.0 - alpha / D
    med = np.median(np.concatenate([base_cal[c] for c in COORDS]))
    grid = med * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    best_g, best_obj = float(grid[0]), math.inf
    for g in grid:
        obj = 0.0
        for c in COORDS:
            q = weighted_quantile(cal_scores[c] / (g + base_cal[c]),
                                  np.ones(len(base_cal[c])), 1.0, level)
            widths = [_interval_width(c, _true_coord(th, c),
                                      q * (g + float(base_tune[c][i])), sup)
                      for i, th in enumerate(tune_hats)]
            obj += float(np.median(widths)) / (sup[c][1] - sup[c][0])
        if obj < best_obj:
            best_obj, best_g = obj, float(g)
    return best_g


# ---------------------------------------------------------------------------
# Figures + report
# ---------------------------------------------------------------------------


def plot_coverage(results: dict, fig_dir: Path) -> None:
    doms = ["synthetic_unweighted", "real_unweighted", "real_weighted"]
    fig, axs = plt.subplots(len(STRATEGIES), len(doms),
                            figsize=(4.6 * len(doms), 4.2 * len(STRATEGIES)))
    nominal = [1 - a for a in ALPHAS]
    for r, strat in enumerate(STRATEGIES):
        for cidx, dom in enumerate(doms):
            ax = axs[r][cidx]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            for norm in NORMS:
                res = results[strat][norm].get(dom)
                if res is None:
                    continue
                ax.plot(nominal, [res[f"{a:.2f}"]["joint_coverage"] for a in ALPHAS],
                        "o-", label=f"{norm} joint")
            ax.set_title(f"{strat} — {dom}", fontsize=10)
            ax.set_xlabel("nominal 1-alpha")
            ax.set_ylabel("empirical coverage")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Weighted split-CP (calibrate on fake, test on real) — joint coverage",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "conformal_shift_coverage.png", dpi=180)
    plt.close(fig)


def plot_widths(results: dict, fig_dir: Path, alpha: float = 0.10) -> None:
    doms = ["synthetic_unweighted", "real_unweighted", "real_weighted"]
    labels, series = [], []
    for strat in STRATEGIES:
        for norm in NORMS:
            labels.append(f"{strat}/{norm}")
            series.append(results[strat][norm])
    fig, axs = plt.subplots(1, len(doms), figsize=(5.2 * len(doms), 4.6))
    x = np.arange(D)
    w = 0.8 / len(labels)
    for ax, dom in zip(axs, doms):
        for k, (lab, res) in enumerate(zip(labels, series)):
            if res.get(dom) is None:
                continue
            ws = res[dom][f"{alpha:.2f}"]["per_coord_median_width"]
            ax.bar(x + (k - len(labels) / 2 + 0.5) * w, [ws[c] for c in COORDS],
                   w, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(COORDS)
        ax.set_title(dom)
        ax.set_ylabel(f"median width @ 1-alpha={1-alpha:.2f}")
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("Interval widths by strategy / score normalization", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "conformal_shift_widths.png", dpi=180)
    plt.close(fig)


def write_report(report: dict, path: Path) -> None:
    lines = ["Split-CP calibrated on fake, tested on real — naive vs surrogate",
             "=" * 72,
             f"n_cal={report['n_cal']}  n_tune={report['n_tune']}  "
             f"n_test_syn={report['n_test_syn']}  n_test_real={report['n_test_real']}",
             f"noise proxy source: {report['proxy_source']}",
             f"weights: ESS={report['weights']['ess']:.1f} of n_cal={report['n_cal']}, "
             f"clipped to [{report['weights']['clip_lo']:g}, {report['weights']['clip_hi']:g}], "
             f"{report['weights']['frac_clipped']:.1%} clipped",
             "v_c median (cal): " + "  ".join(
                 f"{c}={report['vk_median']['cal'][c]:.3g}" for c in COORDS)]
    nf = report.get("noise_filter", {})
    if nf.get("enabled"):
        lines.append(f"noise filter (Assumption 2.1): bound={nf['bound_rv_std']:.3g} "
                     f"rv_std units, rejected {nf['rejection_rate']:.1%} of "
                     f"{nf['n_generated']} draws")
    if report.get("naive_adjustment"):
        lines.append("naive_adj Delta_c (max |theta_bar - theta*| on tune): " + "  ".join(
            f"{c}={report['naive_adjustment'][c]['used_max']:.3g}" for c in COORDS))
    ac = report.get("assumption_constants")
    if ac:
        lines.append(f"Assumption 2.3 constants ({ac['n_draws']} draws): "
                     f"kappa(H) med={ac['kappa_H']['median']:.3g} "
                     f"max={ac['kappa_H']['max']:.3g}; ||grad h|| "
                     f"med={ac['grad_h_spectral_norm']['median']:.3g} "
                     f"max={ac['grad_h_spectral_norm']['max']:.3g}")
    lines.append("")
    for strat in STRATEGIES:
        gr = report["gamma_reg"][strat]
        lines.append(f"##### STRATEGY: {strat} ("
                     + ", ".join(f"gamma_{k}={gr[k]:.4g}" for k in gr) + ") #####")
        med = report["cal_score_median"][strat]
        lines.append("calibration score median per coord: "
                     + "  ".join(f"{c}={med[c]:.3g}" for c in COORDS))
        for norm in NORMS:
            for dom in ["synthetic_unweighted", "real_unweighted", "real_weighted"]:
                res = report["results"][strat][norm].get(dom)
                if res is None:
                    continue
                lines.append(f"[{norm} | {dom}]")
                lines.append(f"  {'1-alpha':>8}{'joint':>9}"
                             + "".join(f"{c:>11}" for c in COORDS) + f"{'%inf':>7}")
                for a in ALPHAS:
                    r = res[f"{a:.2f}"]
                    row = f"  {1-a:>8.2f}{r['joint_coverage']:>9.3f}"
                    row += "".join(f"{r['per_coord_coverage'][c]:>11.3f}" for c in COORDS)
                    row += f"{r['frac_infinite']:>7.2f}"
                    lines.append(row)
                wdt = res["0.10"]["per_coord_median_width"]
                lines.append("  median width @ 1-alpha=0.90: "
                             + "  ".join(f"{c}={wdt[c]:.3g}" for c in COORDS))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=SG / "datasets" / "synthetic_lsp_regression_10000.csv",
                    help="psi training CSV; feature columns are taken from it "
                         "(default: 512-bin LSP dataset, per Nicolò's OK on more "
                         "Fourier bins; pass synthetic_regression_10000.csv for "
                         "the 64-bin variant)")
    ap.add_argument("--out-dir", type=Path, default=SG / "regression")
    ap.add_argument("--fig-dir", type=Path,
                    default=SG / "figures" / "synthetic_regression_10000")
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-tune", type=int, default=100)
    ap.add_argument("--n-weight-synth", type=int, default=400,
                    help="fresh synthetic sample size for the weight discriminator")
    ap.add_argument("--grid", type=int, default=33,
                    help="resolution of the empirical histogram grids (support)")
    ap.add_argument("--gd-steps", type=int, default=200,
                    help="Adam steps for the surrogate-label gradient descent")
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--clip-weights", type=float, default=20.0,
                    help="clip likelihood-ratio weights to [1/x, x]")
    ap.add_argument("--real-split", default="test", choices=("all", "train", "val", "test"))
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=100.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument(
        "--psi",
        choices=("rf", "mlp"),
        default="rf",
        help="point predictor: RandomForest (default) or regression.py MLP checkpoint",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "regression_mlp_74.pt",
        help="MLP checkpoint when --psi mlp (must match --csv feature dim)",
    )
    ap.add_argument("--psi-labels", choices=("bar", "star"), default="bar",
                    help="train psi on data-generating theta_bar (default) or on "
                         "GD surrogate labels theta* of the CSV rows (ablation)")
    ap.add_argument("--psi-star-rows", type=int, default=0,
                    help="cap CSV rows for --psi-labels star (smoke only; 0 = all)")
    ap.add_argument("--csv-seed", type=int, default=123,
                    help="RNG seed the training CSV was generated with (for replay)")
    ap.add_argument("--gamma-tune-on", choices=("synthetic", "real-val"),
                    default="synthetic",
                    help="set used to tune gamma per norm variant: the synthetic "
                         "tuning set (default, keeps all real data held out) or "
                         "the real val split (the paper's D_val; tuning needs no "
                         "labels — only interval widths)")
    ap.add_argument("--no-noise-filter", action="store_true",
                    help="disable the Assumption 2.1 bounded-noise discard rule "
                         "on synthetic draws")
    ap.add_argument("--n-constants", type=int, default=25,
                    help="prior draws for the Assumption 2.3 constants "
                         "(kappa(H), ||grad h||); 0 skips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    t_start = time.perf_counter()

    # Step-5 point predictor psi. Feature columns come from the CSV itself, so
    # the 512-bin LSP dataset and the 64-bin one both work; per-system vectors
    # are assembled from the stored summary row + raw LSP to match.
    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]

    def _row(fr: dict, lsp: np.ndarray) -> list:
        return [fr[c] if c in fr else lsp[int(c.rsplit("_", 1)[1]) - 1]
                for c in feature_cols]

    def feat_matrix(systems: list) -> np.ndarray:
        return np.asarray([_row(s["feat_row"], s["lsp"]) for s in systems], dtype=float)

    # Uncertainty proxy v (from the trained noise model) — built first because
    # the noise-bound filter, the paper-norm residuals, and the psi* labels
    # decode through it.
    proxy = NoiseProxy()
    print(f"noise proxy source: {proxy.source}")

    if args.psi == "mlp":
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {args.checkpoint}")
        psi_predict, mlp_stats = _load_mlp_psi(args.checkpoint, device)
        in_dim = int(mlp_stats["in_dim"])
        if len(feature_cols) != in_dim:
            raise ValueError(
                f"MLP in_dim={in_dim} but --csv has {len(feature_cols)} feature "
                f"columns; pass a matching CSV (e.g. synthetic_regression_10000.csv "
                f"for the 74-D checkpoint)"
            )
        print(
            f"loaded MLP psi from {args.checkpoint} "
            f"(feature_set={mlp_stats.get('feature_set')}, "
            f"in_dim={in_dim}, e_head={mlp_stats.get('e_head')})"
        )
    else:
        if args.psi_labels == "star":
            cache = args.csv.with_suffix(f".theta_star_gd{args.gd_steps}.npz")
            X_rows = df[feature_cols].to_numpy(float)
            y_rows = psi_star_labels(len(df), args.csv_seed, proxy.decoder,
                                     args.gd_steps, args.gd_lr, cache,
                                     limit=args.psi_star_rows)
            X_rows = X_rows[: len(y_rows)]
        else:
            X_rows = df[feature_cols].to_numpy(float)
            y_rows = df[list(TARGET_COLUMNS)].to_numpy(float)

        rf = _build("separate", args.n_estimators, args.seed, list(TARGET_COLUMNS))
        rf.fit(X_rows, y_rows)
        print(f"trained RF psi on {len(y_rows)} synthetic rows, {len(feature_cols)} "
              f"features (labels: {args.psi_labels})")

        def psi_predict(X: np.ndarray) -> np.ndarray:
            return np.asarray(rf.predict(X), dtype=np.float64)

    grids = histogram_grids(args.grid, args.seed)
    sup = _support(grids)

    def hats(systems):
        return list(psi_predict(feat_matrix(systems)))
    print("building real systems ...")
    test_real = make_real(args.real_split, args.sigma_min, args.sigma_max)
    real_train = make_real("train", args.sigma_min, args.sigma_max)

    # Assumption 2.1 bound from real TRAIN curves (eps*C_noise estimate).
    bound = None
    if not args.no_noise_filter:
        bound = noise_bound_from_real(proxy, real_train, hats(real_train))
        print(f"noise bound (max |y - h(psi(y))| on real train, rv_std units): {bound:.3g}")

    print("building synthetic systems ...")
    n_gen_total = 0
    rejected_thetas: list = []
    calib, g, rj = make_synthetic_filtered(args.n_cal, args.seed + 1, bound, proxy)
    n_gen_total += g
    rejected_thetas += rj
    tune, g, rj = make_synthetic_filtered(args.n_tune, args.seed + 11, bound, proxy)
    n_gen_total += g
    rejected_thetas += rj
    test_syn, g, rj = make_synthetic_filtered(args.n_test, args.seed + 2, bound, proxy)
    n_gen_total += g
    rejected_thetas += rj
    wsynth, g, rj = make_synthetic_filtered(args.n_weight_synth, args.seed + 21, bound, proxy)
    n_gen_total += g
    rejected_thetas += rj
    n_kept = len(calib) + len(tune) + len(test_syn) + len(wsynth)
    rejection_rate = 1.0 - n_kept / max(n_gen_total, 1)
    if bound is not None:
        print(f"noise filter: kept {n_kept}/{n_gen_total} draws "
              f"(rejection rate {rejection_rate:.1%})")
        plot_filter_histograms(
            [s["theta5"] for s in real_train] + [s["theta5"] for s in test_real],
            [s["theta5"] for s in calib] + [s["theta5"] for s in test_syn],
            rejected_thetas, args.fig_dir)
    print(f"n_cal={len(calib)} n_tune={len(tune)} n_test_syn={len(test_syn)} "
          f"n_test_real={len(test_real)} (weight fit: {len(wsynth)} synth vs "
          f"{len(real_train)} real-train)")

    # gamma tuning set: synthetic tune (default) or the real val split (the
    # paper's D_val — legal because tune_gamma only measures interval widths,
    # which need psi(y) and the denominators, never labels).
    keyed_sets = [("cal", calib), ("tune", tune), ("syn", test_syn), ("real", test_real)]
    if args.gamma_tune_on == "real-val":
        if args.real_split in ("val", "all"):
            print("WARNING: --gamma-tune-on real-val overlaps --real-split "
                  f"{args.real_split} — gamma is tuned on (part of) the test systems")
        gtune = make_real("val", args.sigma_min, args.sigma_max)
        print(f"gamma tuning on real val split: {len(gtune)} systems")
        keyed_sets.append(("gtune", gtune))

    hat = {k: hats(sys_) for k, sys_ in keyed_sets}

    v = {k: np.array([proxy.value(th, s["curve"]) for s, th in zip(sys_, hat[k])])
         for k, sys_ in keyed_sets}

    # Likelihood-ratio weights (fit: fresh synth vs real TRAIN; applied to cal +
    # real test). Deliberately kept on the 74-dim summary FEATURES (s["features"])
    # rather than psi's possibly 586-dim set: a 586-dim discriminator on ~700
    # points separates the classes too well and degenerates the weights.
    w_fn, _ = fit_weight_model(np.vstack([s["features"] for s in wsynth]),
                               np.vstack([s["features"] for s in real_train]),
                               args.seed)
    clip = args.clip_weights
    w_cal_raw = w_fn(np.vstack([s["features"] for s in calib]))
    w_real_raw = w_fn(np.vstack([s["features"] for s in test_real]))
    w_cal = np.clip(w_cal_raw, 1.0 / clip, clip)
    w_real = np.clip(w_real_raw, 1.0 / clip, clip)
    ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum())
    frac_clipped = float(np.mean((w_cal_raw < 1 / clip) | (w_cal_raw > clip)))
    print(f"weights: ESS={ess:.1f}/{len(w_cal)}  clipped={frac_clipped:.1%}")

    # Surrogate labels theta* by gradient descent (L1 objective, initialized at
    # the data-generating theta_bar — Nicolò 2026-07): on the calibration set
    # (for the surrogate scores) and on the tuning set (to fit the v_c model).
    print("fitting surrogate labels by gradient descent ...")
    t0 = time.perf_counter()
    theta_star = surrogate_fit_gd(proxy.decoder, [s["theta5"] for s in calib],
                                  calib, args.gd_steps, args.gd_lr)
    theta_star_tune = surrogate_fit_gd(proxy.decoder, [s["theta5"] for s in tune],
                                       tune, args.gd_steps, args.gd_lr)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    # Surrogate-label error model v_c (second factor of the v2norm denominator),
    # fit on the tuning set, evaluated everywhere on the summary features.
    vk_fn = fit_vk_models(np.vstack([s["features"] for s in tune]),
                          [s["theta5"] for s in tune], theta_star_tune,
                          args.seed, args.n_estimators)
    vk = {k: vk_fn(np.vstack([s["features"] for s in sys_])) for k, sys_ in keyed_sets}
    print("v_c median (cal): " + "  ".join(
        f"{c}={float(np.median(vk['cal'][c])):.3g}" for c in COORDS))

    # Paper-spec conditional residuals delta_c / delta_y (eqs 18-19), computed
    # pointwise per curve — h and psi are deterministic, so the conditional
    # expectations are trivial and no model is trained (Nicolò 2026-07-14).
    print("computing paper-norm deltas (re-encode + reconstruction residuals) ...")

    def paper_deltas(systems: list, hats_: list) -> tuple[dict, np.ndarray]:
        dy = np.array([float(np.mean(np.abs(recon_residual_norm(proxy, th, s["curve"]))))
                       for s, th in zip(systems, hats_)])
        reenc = [reencode_features(proxy, th, s["curve"]) for s, th in zip(systems, hats_)]
        psi_re = psi_predict(np.asarray([_row(fr, lsp) for fr, lsp in reenc], dtype=float))
        dk = {c: np.array([_coord_abs_err(psi_re[j], hats_[j], c)
                           for j in range(len(systems))]) for c in COORDS}
        return dk, dy

    delta = {k: paper_deltas(sys_, hat[k]) for k, sys_ in keyed_sets}
    if args.gamma_tune_on != "real-val":
        for d in (hat, v, vk, delta):
            d["gtune"] = d["tune"]
    print("delta_c median (cal): " + "  ".join(
        f"{c}={float(np.median(delta['cal'][0][c])):.3g}" for c in COORDS)
        + f"  delta_y={float(np.median(delta['cal'][1])):.3g}")

    # Calibration scores per strategy and coordinate.  naive_adj = naive with
    # the paper's quantile adjustment (eq 41): calibration scores inflated by
    # the per-coordinate surrogate gap Delta_c = max over the tuning set of
    # |theta_bar_c - theta*_c| (shifting all calibration scores by Delta_c
    # shifts every raw quantile by exactly Delta_c).
    gap = {c: np.array([_coord_abs_err(tune[j]["theta5"], theta_star_tune[j], c)
                        for j in range(len(tune))]) for c in COORDS}
    adj = {c: float(gap[c].max()) for c in COORDS}
    naive_scores = {c: np.array([_coord_abs_err(hat["cal"][j], calib[j]["theta5"], c)
                                 for j in range(len(calib))]) for c in COORDS}
    cal_scores = {
        "naive": naive_scores,
        "naive_adj": {c: naive_scores[c] + adj[c] for c in COORDS},
        "surrogate": {c: np.array([_coord_abs_err(hat["cal"][j], theta_star[j], c)
                                   for j in range(len(calib))]) for c in COORDS},
    }
    print("naive_adj Delta_c (max gap on tune): " + "  ".join(
        f"{c}={adj[c]:.3g}" for c in COORDS))

    # Denominator bases per norm variant: vnorm = v_y, v2norm = v_y + v_c,
    # papernorm = delta_c + delta_y (eqs 20/23).
    def base(kind: str, key: str) -> dict:
        if kind == "vnorm":
            return {c: v[key] for c in COORDS}
        if kind == "papernorm":
            dk, dy = delta[key]
            return {c: dk[c] + dy for c in COORDS}
        return {c: v[key] + vk[key][c] for c in COORDS}

    norm_kinds = [k for k in NORMS if k != "raw"]
    results, gamma_reg = {}, {}
    for strat in STRATEGIES:
        cs = cal_scores[strat]
        gamma_reg[strat] = {
            kind: tune_gamma(cs, base(kind, "cal"), hat["gtune"], base(kind, "gtune"), sup)
            for kind in norm_kinds}
        print(f"[{strat}] " + "  ".join(
            f"gamma_{k}={gamma_reg[strat][k]:.4g}" for k in norm_kinds))
        results[strat] = {}
        for norm in NORMS:
            if norm == "raw":
                dens = {"cal": None, "syn": None, "real": None}
            else:
                g = gamma_reg[strat][norm]
                dens = {key: {c: g + base(norm, key)[c] for c in COORDS}
                        for key in ["cal", "syn", "real"]}
            results[strat][norm] = {
                "synthetic_unweighted": evaluate(cs, test_syn, hat["syn"], sup,
                                                 dens["cal"], dens["syn"]),
                "real_unweighted": evaluate(cs, test_real, hat["real"], sup,
                                            dens["cal"], dens["real"]),
                "real_weighted": evaluate(cs, test_real, hat["real"], sup,
                                          dens["cal"], dens["real"],
                                          w_cal=w_cal, w_test=w_real),
            }
            for dom in ["synthetic_unweighted", "real_unweighted", "real_weighted"]:
                r = results[strat][norm][dom]["0.10"]
                print(f"  [{strat}/{norm}/{dom}] joint@0.90={r['joint_coverage']:.3f} "
                      f"inf={r['frac_infinite']:.2f}")

    constants = None
    if args.n_constants > 0:
        print(f"estimating Assumption 2.3 constants on {args.n_constants} prior draws ...")
        constants = estimate_constants(proxy, args.n_constants, args.seed + 41)
        print(f"  kappa(H) median={constants['kappa_H']['median']:.3g} "
              f"max={constants['kappa_H']['max']:.3g}  "
              f"||grad h|| median={constants['grad_h_spectral_norm']['median']:.3g}")

    report = {
        "n_cal": len(calib), "n_tune": len(tune),
        "n_test_syn": len(test_syn), "n_test_real": len(test_real),
        "alphas": ALPHAS, "coords": COORDS,
        "proxy_source": proxy.source,
        "psi": args.psi,
        "checkpoint": str(args.checkpoint) if args.psi == "mlp" else None,
        "csv": str(args.csv),
        "psi_labels": args.psi_labels,
        "gamma_tune_on": args.gamma_tune_on,
        "gamma_reg": gamma_reg,
        "weights": {"ess": ess, "frac_clipped": frac_clipped,
                    "clip_lo": 1.0 / clip, "clip_hi": clip},
        "cal_score_median": {s: {c: float(np.median(cal_scores[s][c])) for c in COORDS}
                             for s in STRATEGIES},
        "vk_median": {k: {c: float(np.median(vk[k][c])) for c in COORDS}
                      for k in ["cal", "real"]},
        "delta_median": {
            "delta_c_cal": {c: float(np.median(delta["cal"][0][c])) for c in COORDS},
            "delta_y_cal": float(np.median(delta["cal"][1])),
            "delta_c_real": {c: float(np.median(delta["real"][0][c])) for c in COORDS},
            "delta_y_real": float(np.median(delta["real"][1])),
        },
        "naive_adjustment": {
            c: {"used_max": adj[c],
                "median": float(np.median(gap[c])),
                "p90": float(np.percentile(gap[c], 90))} for c in COORDS},
        "noise_filter": {
            "enabled": bound is not None,
            "bound_rv_std": bound,
            "n_generated": n_gen_total,
            "n_kept": n_kept,
            "rejection_rate": rejection_rate if bound is not None else 0.0,
        },
        "assumption_constants": constants,
        "results": results,
    }
    # Unweighted Bonferroni quantiles for paper figures (raw scores).
    q_export: dict = {}
    ones = np.ones(len(calib))
    for strat in STRATEGIES:
        q_export[strat] = {}
        for a in ALPHAS:
            level = 1.0 - a / D
            q_export[strat][f"{a:.2f}"] = {
                c: float(weighted_quantile(cal_scores[strat][c], ones, 1.0, level))
                for c in COORDS
            }
    report["quantiles_unweighted"] = q_export
    (args.out_dir / "conformal_shift_metrics.json").write_text(
        json.dumps(report, indent=2))
    write_report(report, args.out_dir / "conformal_shift_report.txt")
    plot_coverage(results, args.fig_dir)
    plot_widths(results, args.fig_dir)
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s — wrote metrics/report to "
          f"{args.out_dir}, figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
