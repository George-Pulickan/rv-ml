"""
gp_residual_model.py — global GP fit to real RV residuals (Nicolò's spec).

This is a *different* model from gp_noise_model.py. That module fits a
per-system celerite2 GP over time t only (1-D). Here we fit a single global
Gaussian Process whose features are multi-dimensional,

    features  X = ( phase = t mod T / T ,  log10 P , log10 K , e ,
                    cos w , sin w ,  y(t) ,  log10 sigma(t) )
    label     r = y(t) - yhat(t)

across all single-planet systems, where y(t) is the noiseless Keplerian
trajectory obtained by integrating the *tabulated* catalog parameters
K = (P, K, e, w) plus a least-squares systemic offset gamma (inverse-variance-
weighted mean of yhat - y; Nicolò approved replacing the earlier first-
observation anchor, 2026-07). T = P, so t mod T projects every system into a
single orbital period (George's phase-folded view).

log10 sigma(t) — the tabulated per-observation measurement uncertainty — is an
extra conditioning feature (Nicolò approved, 2026-07): per-system residual
amplitude is set by instrument precision + stellar activity, not orbital
geometry, so orbit features alone cannot predict it (generative validation
found std ratio ~1.76 with ~zero std log-correlation). sigma IS known at
synthetic-generation time (the generators sample it), so conditioning on it is
legitimate.

Measurement uncertainty is propagated by sampling yhat uniformly within its
error bar (Nicolo: "sample uniformly haty within the measurement noise bars")
and re-deriving the residual; this augments the training set only.

Model
-----
Primary  : gpytorch sparse variational GP (SVGP, inducing points), ARD
           Matern-5/2 kernel, Gaussian likelihood, trained by ELBO on the
           train split. Scales to the full augmented corpus (Titsias 2009;
           Hensman et al. 2013, 2015).
Baseline : sklearn exact GaussianProcessRegressor on a tractable subsample,
           used to confirm the SVGP posterior mean matches an exact GP.

Evaluation is on the held-out val / test splits (host-grouped in
data/splits.csv, so there is no system leakage). Val / test residuals use the
nominal (un-resampled) yhat — augmentation is a training-time device only.

References
----------
Titsias, M. 2009, AISTATS (variational sparse GP)
Hensman, J. et al. 2013, UAI; 2015, AISTATS (stochastic / SVGP)
Rasmussen & Williams 2006
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kepler_check import build_planet, rv_keplerian
from parse_and_label import match_host_rows
from preprocess import load_raw_rv

ROOT = Path(__file__).parent
SPLITS_CSV = ROOT / "data" / "splits.csv"
LABELS_CSV = ROOT / "data" / "labels.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures" / "gp_residual"

FEATURE_NAMES = ["phase", "log10_P", "log10_K", "e", "cos_omega", "sin_omega",
                 "y_rel", "log10_sigma"]
MIN_OBS = 10


# =========================================================================== #
# Residual construction                                                        #
# =========================================================================== #

def _fit_tperi(t, rv, sigma, P, K, e, omega, grid_n=60):
    """Analytic T_peri: grid-search one period for the phase minimizing the
    inverse-variance-weighted residual RMS (gamma marginalized each step).
    Mirrors the gridsearch_tperi seed inside kepler_check.least_squares_refit.
    """
    w2 = 1.0 / np.maximum(sigma, 1e-6) ** 2
    grid = np.linspace(t.min(), t.min() + P, grid_n, endpoint=False)
    best_tp, best_rms = float(grid[0]), np.inf
    for tp in grid:
        v = rv_keplerian(t, P, K, e, omega, tp)
        gamma = float(np.sum((rv - v) * w2) / np.sum(w2))
        rms = float(np.sqrt(np.mean((rv - v - gamma) ** 2)))
        if rms < best_rms:
            best_rms, best_tp = rms, float(tp)
    return best_tp


def _ls_gamma(y, yhat, sigma):
    """Least-squares systemic offset: argmin_g sum_i ((yhat_i - y_i - g)/sigma_i)^2,
    i.e. the inverse-variance-weighted mean of (yhat - y)."""
    w2 = 1.0 / np.maximum(sigma, 1e-6) ** 2
    return float(np.sum((yhat - y) * w2) / np.sum(w2))


def _system_residual(t, yhat, sigma, planet):
    """Noiseless Keplerian with a least-squares offset, and the residual.

    y(t)   = rv_keplerian(catalog params) + gamma
    gamma  = inverse-variance-weighted LS fit of the offset (Nicolò, 2026-07;
             replaces the earlier first-observation anchor)
    r(t)   = y(t) - yhat(t)

    yhat is the (possibly resampled) RV used to define both the residual and the
    offset. Returns (y_anchored, r).
    """
    y = rv_keplerian(t, planet.P, planet.K, planet.e, planet.omega, planet.t_peri)
    y = y + _ls_gamma(y, yhat, sigma)
    return y, (y - yhat)


def build_split(split: str, n_aug: int, seed: int = 0,
                labels: Optional[pd.DataFrame] = None,
                splits: Optional[pd.DataFrame] = None,
                sigma_min: float = 0.1, sigma_max: float = 100.0,
                max_rms_over_sigma: float = 30.0,
                verbose: bool = True):
    """Build the (features, label) table for one split over single-planet
    systems. The first draw is always the nominal (un-resampled) residual;
    the remaining n_aug-1 draws resample yhat ~ U(yhat-sigma, yhat+sigma) and
    re-anchor. Returns (X [N,8], r [N], meta dict).

    Quality cuts (a *noise* model must see noise, not junk / model failure):
      * median sigma in [sigma_min, sigma_max] m/s  — drops instrument-precision
        placeholders and absolute-RV files (e.g. 11 Com at ~5 km/s); matches the
        filter in validate_synthetic_dataset.collect_real.
      * nominal residual RMS / median sigma <= max_rms_over_sigma — drops systems
        whose catalog ephemeris grossly fails to describe the data (the residual
        is then model mismatch, not noise). The median system sits at ~2.7, so
        this only removes pathological tails (~p95+).
    """
    if labels is None:
        labels = pd.read_csv(LABELS_CSV)
        if "default_flag" in labels.columns:
            labels = labels[labels["default_flag"] == 1]
    if splits is None:
        splits = pd.read_csv(SPLITS_CSV)

    sp = splits[(splits["n_planets"] == 1) & (splits["split"] == split)]
    rng = np.random.default_rng(seed)

    X_rows, r_rows, g_rows = [], [], []
    group_hosts = []
    n_sys = n_catalog = n_phasefit = n_skip = 0
    n_cut_sigma = n_cut_rms = 0
    for _, row in sp.iterrows():
        host, fname = str(row["host"]), str(row["file"])
        try:
            t, yhat, sigma = load_raw_rv(fname)
        except Exception:
            n_skip += 1
            continue
        if len(t) < MIN_OBS:
            n_skip += 1
            continue

        med_sigma = float(np.median(sigma))
        if not (sigma_min <= med_sigma <= sigma_max):
            n_cut_sigma += 1
            continue

        lab_rows = match_host_rows(host, labels)
        if lab_rows.empty:
            n_skip += 1
            continue
        planet, _ = build_planet(lab_rows.iloc[0])
        if planet is None:
            n_skip += 1
            continue

        if not planet.tperi_known:
            import dataclasses
            tp = _fit_tperi(t, yhat, sigma, planet.P, planet.K, planet.e, planet.omega)
            planet = dataclasses.replace(planet, t_peri=tp, tperi_known=True)
            phasefit = True
        else:
            phasefit = False

        # Nominal residual decides the model-mismatch cut (pre-augmentation).
        _, r_nom = _system_residual(t, yhat, sigma, planet)
        rms_over_sigma = float(np.sqrt(np.mean(r_nom ** 2)) / max(med_sigma, 1e-6))
        if rms_over_sigma > max_rms_over_sigma:
            n_cut_rms += 1
            continue

        n_phasefit += int(phasefit)
        n_catalog += int(not phasefit)
        gid = n_sys          # 0-based id of this kept system
        n_sys += 1
        group_hosts.append(host)

        P = planet.P
        t0 = int(np.argmin(t))
        # Noiseless Keplerian and its value relative to the initial condition.
        # The absolute systemic velocity (gamma) is an arbitrary per-star DC
        # offset that cancels in r; the *feature* must be DC-free too, so we use
        # y(t) - y(t0) = the model RV change since the first observation.
        y_kep = rv_keplerian(t, P, planet.K, planet.e, planet.omega, planet.t_peri)
        y_rel = (y_kep - y_kep[t0]).astype(float)

        phase = (np.mod(t - t.min(), P) / P).astype(float)
        log10_P = np.full_like(t, np.log10(max(P, 1e-3)))
        log10_K = np.full_like(t, np.log10(max(planet.K, 1e-3)))
        e_col = np.full_like(t, float(np.clip(planet.e, 0.0, 0.99)))
        cosw = np.full_like(t, float(np.cos(planet.omega)))
        sinw = np.full_like(t, float(np.sin(planet.omega)))
        log10_sig = np.log10(np.maximum(sigma, 1e-6)).astype(float)
        X_sys = np.column_stack([phase, log10_P, log10_K, e_col, cosw, sinw,
                                 y_rel, log10_sig])

        for draw in range(n_aug):
            yh = yhat if draw == 0 else yhat + rng.uniform(-sigma, sigma)
            # LS offset against the (possibly resampled) obs; r is DC-free.
            r = (y_kep + _ls_gamma(y_kep, yh, sigma)) - yh
            X_rows.append(X_sys)
            r_rows.append(r)
            g_rows.append(np.full(len(r), gid, dtype=np.int64))

    X = np.concatenate(X_rows, axis=0) if X_rows else np.zeros((0, len(FEATURE_NAMES)))
    r = np.concatenate(r_rows, axis=0) if r_rows else np.zeros((0,))
    groups = np.concatenate(g_rows) if g_rows else np.zeros((0,), dtype=np.int64)
    meta = dict(split=split, n_systems=n_sys, n_catalog_tperi=n_catalog,
                n_phasefit_tperi=n_phasefit, n_skipped=n_skip,
                n_cut_sigma=n_cut_sigma, n_cut_rms=n_cut_rms,
                n_rows=int(len(r)), n_aug=n_aug, group_hosts=group_hosts)
    if verbose:
        if len(r):
            print(f"[build:{split}] systems={n_sys} "
                  f"(catalog_tperi={n_catalog}, phasefit={n_phasefit})  "
                  f"cut: sigma={n_cut_sigma}, rms={n_cut_rms}, other={n_skip}  "
                  f"rows={len(r)}  std(r)={np.std(r):.3f} m/s")
        else:
            print(f"[build:{split}] no rows (cut sigma={n_cut_sigma}, "
                  f"rms={n_cut_rms}, other={n_skip})")
    return X.astype(np.float64), r.astype(np.float64), groups, meta


# =========================================================================== #
# Standardization                                                              #
# =========================================================================== #

class Standardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)

    @classmethod
    def fit(cls, X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        return cls(mu, sd)

    def transform(self, X):
        return (X - self.mean) / self.std

    def to_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())


# =========================================================================== #
# SVGP (gpytorch)                                                              #
# =========================================================================== #

def _make_svgp(inducing_points):
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

    class SVGP(ApproximateGP):
        def __init__(self, Z):
            vdist = CholeskyVariationalDistribution(Z.size(0))
            vstrat = VariationalStrategy(self, Z, vdist, learn_inducing_locations=True)
            super().__init__(vstrat)
            d = Z.size(1)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return SVGP(inducing_points)


def fit_svgp(Xs_train, r_train, n_inducing=512, n_iter=400, lr=0.01,
             batch=2048, seed=0, likelihood_type="studentt", verbose=True):
    """Train a sparse variational GP. Returns (model, likelihood).

    likelihood_type: 'gaussian' or 'studentt'. RV residuals are strongly
    leptokurtic (kurtosis-excess ~30), so a Student-t observation model is the
    physically motivated choice and yields better tail calibration; the
    Gaussian path is kept for comparison.
    """
    import torch
    import gpytorch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xs_train, dtype=torch.float32)
    rt = torch.as_tensor(r_train, dtype=torch.float32)
    N = Xt.size(0)

    rng = np.random.default_rng(seed)
    n_ind = min(n_inducing, N)
    Z = Xt[rng.choice(N, size=n_ind, replace=False)].clone()

    model = _make_svgp(Z)
    if likelihood_type == "studentt":
        likelihood = gpytorch.likelihoods.StudentTLikelihood()
    elif likelihood_type == "gaussian":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    else:
        raise ValueError(f"unknown likelihood_type {likelihood_type!r}")
    model.train(); likelihood.train()

    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

    loader = DataLoader(TensorDataset(Xt, rt), batch_size=batch, shuffle=True)
    for it in range(n_iter):
        last = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            opt.step()
            last = float(loss.item())
        if verbose and (it % 50 == 0 or it == n_iter - 1):
            print(f"[svgp] iter {it:4d}/{n_iter}  elbo_loss={last:.4f}")

    model.eval(); likelihood.eval()
    return model, likelihood


def svgp_predict(model, likelihood, Xs, r, n_samples=512, batch=4096, seed=0):
    """Posterior-predictive summary at Xs, valid for any likelihood.

    Returns (mean, var, samples) where samples is [n_samples, N] draws from the
    posterior predictive p(r* | X*, data) (latent GP marginalized by sampling,
    then the observation model applied). mean/var are the predictive moments;
    samples support exact central-interval coverage even for the heavy-tailed
    Student-t predictive. The proper predictive NLL is also returned via the
    likelihood's analytic/quadrature log-marginal.
    """
    import torch
    import gpytorch
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xs, dtype=torch.float32)
    rt = torch.as_tensor(r, dtype=torch.float32)
    means, vars_, samp, nll = [], [], [], []
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
        for i in range(0, Xt.size(0), batch):
            xb, yb = Xt[i:i + batch], rt[i:i + batch]
            f = model(xb)                                    # latent MVN
            nll.append((-likelihood.log_marginal(yb, f)).numpy())
            fs = f.rsample(torch.Size([n_samples]))          # [S, b] latent draws
            ys = likelihood(fs).sample()                     # [S, b] predictive draws
            means.append(ys.mean(0).numpy())
            vars_.append(ys.var(0).numpy())
            samp.append(ys.numpy())
    return (np.concatenate(means), np.concatenate(vars_),
            np.concatenate(samp, axis=1), np.concatenate(nll))


# =========================================================================== #
# Metrics                                                                      #
# =========================================================================== #

def _central_coverage(r, samples, p):
    """Empirical coverage of the central-p posterior-predictive interval."""
    lo = np.quantile(samples, 0.5 - p / 2, axis=0)
    hi = np.quantile(samples, 0.5 + p / 2, axis=0)
    return float(np.mean((r >= lo) & (r <= hi)))


def _metrics(r, mean, samples, nll_arr):
    rmse = float(np.sqrt(np.mean((r - mean) ** 2)))
    return dict(n=int(len(r)), rmse=rmse, nll=float(np.mean(nll_arr)),
                coverage_68=_central_coverage(r, samples, 0.6827),
                coverage_95=_central_coverage(r, samples, 0.9545),
                std_r=float(np.std(r)),
                rmse_over_std=float(rmse / max(np.std(r), 1e-9)))


# =========================================================================== #
# sklearn exact-GP cross-check                                                 #
# =========================================================================== #

def exact_gp_crosscheck(Xs_train, r_train, Xs_eval, r_eval, n_sub=2500, seed=0):
    """Fit an exact sklearn GP on a subsample; return its predictive mean/var
    on Xs_eval plus the subsample used. Used to confirm SVGP ~ exact GP."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    rng = np.random.default_rng(seed)
    n_sub = min(n_sub, Xs_train.shape[0])
    idx = rng.choice(Xs_train.shape[0], size=n_sub, replace=False)
    Xs, rs = Xs_train[idx], r_train[idx]
    d = Xs.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(np.ones(d), (1e-2, 1e2))
              + WhiteKernel(1.0, (1e-5, 1e3)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=seed)
    gp.fit(Xs, rs)
    mean, std = gp.predict(Xs_eval, return_std=True)
    return mean, std ** 2, idx


# =========================================================================== #
# Plots                                                                        #
# =========================================================================== #

def make_plots(packs, svgp_eval, exact_pack, out_dir):
    """packs: dict split -> (X_raw, r). svgp_eval: dict split -> (mean,var)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. predicted vs true residual (val + test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, split in zip(axes, ["val", "test"]):
        if split not in svgp_eval:
            continue
        _, r = packs[split]
        mean, var, _ = svgp_eval[split]
        ax.errorbar(r, mean, yerr=np.sqrt(var), fmt=".", ms=3, alpha=0.3,
                    elinewidth=0.5, capsize=0)
        lim = np.percentile(np.abs(np.concatenate([r, mean])), 99)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("true residual r [m/s]")
        ax.set_ylabel("GP predicted residual [m/s]")
        ax.set_title(f"{split}: r vs GP mean")
    fig.tight_layout(); fig.savefig(out_dir / "pred_vs_true.png", dpi=130)
    plt.close(fig)

    # 2. phase-folded residual: binned data vs GP mean (test)
    split = "test" if "test" in svgp_eval else "val"
    X_raw, r = packs[split]
    mean, _, _ = svgp_eval[split]
    phase = X_raw[:, 0]
    bins = np.linspace(0, 1, 21)
    bi = np.clip(np.digitize(phase, bins) - 1, 0, len(bins) - 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    data_mu = np.array([r[bi == k].mean() if np.any(bi == k) else np.nan
                        for k in range(len(centers))])
    data_sd = np.array([r[bi == k].std() if np.any(bi == k) else np.nan
                        for k in range(len(centers))])
    gp_mu = np.array([mean[bi == k].mean() if np.any(bi == k) else np.nan
                      for k in range(len(centers))])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers, data_mu, yerr=data_sd, fmt="o", color="C0",
                label="real residual (mean +/- std)", capsize=2)
    ax.plot(centers, gp_mu, "-", color="C3", lw=2, label="GP mean")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("orbital phase  t mod T / T"); ax.set_ylabel("residual [m/s]")
    ax.set_title(f"{split}: phase-folded residual"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "phase_residual.png", dpi=130)
    plt.close(fig)

    # 3. partial dependence: GP mean vs phase, log10_K, y_model, log10_sigma (test)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, j, name in zip(axes, [0, 2, 6, 7],
                           ["phase", "log10_K", "y_rel", "log10_sigma"]):
        order = np.argsort(X_raw[:, j])
        ax.plot(X_raw[order, j], mean[order], ".", ms=2, alpha=0.3)
        ax.set_xlabel(name); ax.set_ylabel("GP mean residual [m/s]")
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle(f"{split}: GP residual partial dependence")
    fig.tight_layout(); fig.savefig(out_dir / "partial_dependence.png", dpi=130)
    plt.close(fig)

    # 4. calibration: empirical coverage of central posterior-predictive
    #    intervals vs nominal (test). Uses predictive samples, so it is exact
    #    for the heavy-tailed Student-t predictive (not a Gaussian-z proxy).
    _, r = packs[split]
    _, _, samples = svgp_eval[split]
    nominal = np.linspace(0.05, 0.95, 19)
    emp = np.array([_central_coverage(r, samples, p) for p in nominal])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(nominal, emp, "o-", color="C2")
    ax.set_xlabel("nominal central interval"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"{split}: calibration")
    fig.tight_layout(); fig.savefig(out_dir / "calibration.png", dpi=130)
    plt.close(fig)

    # 5. SVGP vs exact GP mean on the same eval points
    if exact_pack is not None:
        split, svgp_mean, exact_mean = exact_pack
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        lim = np.percentile(np.abs(np.concatenate([svgp_mean, exact_mean])), 99)
        ax.plot(exact_mean, svgp_mean, ".", ms=3, alpha=0.4)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        c = np.corrcoef(exact_mean, svgp_mean)[0, 1]
        ax.set_xlabel("exact sklearn GP mean [m/s]")
        ax.set_ylabel("SVGP mean [m/s]")
        ax.set_title(f"{split}: SVGP vs exact GP (r={c:.3f})")
        fig.tight_layout(); fig.savefig(out_dir / "svgp_vs_exact.png", dpi=130)
        plt.close(fig)


# =========================================================================== #
# Generative validation                                                        #
# =========================================================================== #

def generative_validation(model, likelihood, Xs, X_raw, r, groups, out_dir,
                          split="test", n_samples=512, seed=0, min_obs=5):
    """Does the GP *generate* residuals whose statistics match the real ones,
    conditioned on orbit and phase?

    This is the real test of a noise model (point-prediction is hopeless across
    stars; see RMSE/std ~ 1). We draw posterior-predictive residuals at each
    test system's real feature rows and compare, per system, the residual
    amplitude (std) and heavy-tailedness (excess kurtosis), plus the pooled
    self-standardized distribution and the phase dependence of the scatter.

    NB: the model treats observations as conditionally independent given the
    features (no time-lag kernel), so it targets the *marginal* residual
    statistics (amplitude / tails / phase- and orbit-dependence), not temporal
    autocorrelation. Returns a metrics dict and writes generative_validation.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as st

    _, _, samples, _ = svgp_predict(model, likelihood, Xs, r,
                                    n_samples=n_samples, seed=seed)  # [S, N]
    phase = X_raw[:, 0]

    real_std, gp_std, real_kurt, gp_kurt = [], [], [], []
    z_real, z_gp = [], []
    for g in np.unique(groups):
        m = groups == g
        if int(m.sum()) < min_obs:
            continue
        rg = r[m]
        sg = samples[:, m]                       # [S, n_g]
        s_real = float(np.std(rg))
        if s_real <= 0:
            continue
        real_std.append(s_real)
        gp_std.append(float(np.median(np.std(sg, axis=1))))
        real_kurt.append(float(st.kurtosis(rg, fisher=True)))
        gp_kurt.append(float(np.median(st.kurtosis(sg, axis=1, fisher=True))))
        z_real.append(rg / s_real)
        sd = np.std(sg, axis=1, keepdims=True)
        z_gp.append((sg / np.maximum(sd, 1e-9)).ravel())

    real_std = np.array(real_std); gp_std = np.array(gp_std)
    real_kurt = np.array(real_kurt); gp_kurt = np.array(gp_kurt)
    z_real = np.concatenate(z_real); z_gp = np.concatenate(z_gp)

    std_ratio = gp_std / np.maximum(real_std, 1e-9)
    ks = st.ks_2samp(z_real, z_gp)
    metrics = dict(
        split=split, n_systems=int(len(real_std)),
        std_ratio_median=float(np.median(std_ratio)),
        std_ratio_iqr=float(np.subtract(*np.percentile(std_ratio, [75, 25]))),
        std_log_corr=float(np.corrcoef(np.log10(real_std), np.log10(gp_std))[0, 1]),
        real_kurt_median=float(np.median(real_kurt)),
        gp_kurt_median=float(np.median(gp_kurt)),
        pooled_ks_stat=float(ks.statistic), pooled_ks_p=float(ks.pvalue),
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # (a) per-system residual amplitude: real vs GP-generated
    a = ax[0, 0]
    lo, hi = 0.5 * min(real_std.min(), gp_std.min()), 2 * max(real_std.max(), gp_std.max())
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.scatter(real_std, gp_std, s=18, alpha=0.6)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("real residual std [m/s]"); a.set_ylabel("GP-generated std [m/s]")
    a.set_title(f"amplitude  (median ratio {metrics['std_ratio_median']:.2f}, "
                f"log-corr {metrics['std_log_corr']:.2f})")

    # (b) per-system heavy-tailedness: real vs GP-generated excess kurtosis
    b = ax[0, 1]
    lo2 = min(real_kurt.min(), gp_kurt.min()); hi2 = max(real_kurt.max(), gp_kurt.max())
    b.plot([lo2, hi2], [lo2, hi2], "k--", lw=1)
    b.scatter(real_kurt, gp_kurt, s=18, alpha=0.6, color="C3")
    b.set_xlabel("real excess kurtosis"); b.set_ylabel("GP-generated excess kurtosis")
    b.set_title(f"tails  (real med {metrics['real_kurt_median']:.1f}, "
                f"GP med {metrics['gp_kurt_median']:.1f})")

    # (c) pooled self-standardized distribution (shape / tails), log-y
    c = ax[1, 0]
    bins = np.linspace(-6, 6, 81)
    c.hist(z_real, bins=bins, density=True, histtype="step", lw=2, label="real")
    c.hist(z_gp, bins=bins, density=True, histtype="step", lw=2, label="GP-generated")
    c.plot(bins, st.norm.pdf(bins), "k:", lw=1, label="N(0,1)")
    c.set_yscale("log"); c.set_ylim(1e-4, 1)
    c.set_xlabel("residual / per-system std"); c.set_ylabel("density")
    c.set_title(f"pooled distribution  (KS {metrics['pooled_ks_stat']:.3f})")
    c.legend()

    # (d) phase dependence of the scatter: real vs GP-generated
    d = ax[1, 1]
    edges = np.linspace(0, 1, 16)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    bi = np.clip(np.digitize(phase, edges) - 1, 0, len(ctr) - 1)
    real_ps = np.array([r[bi == k].std() if np.any(bi == k) else np.nan
                        for k in range(len(ctr))])
    gp_ps = np.array([samples[:, bi == k].std() if np.any(bi == k) else np.nan
                      for k in range(len(ctr))])
    d.plot(ctr, real_ps, "o-", label="real", color="C0")
    d.plot(ctr, gp_ps, "s-", label="GP-generated", color="C1")
    d.set_xlabel("orbital phase  t mod T / T"); d.set_ylabel("residual std [m/s]")
    d.set_title("scatter vs phase"); d.legend()

    fig.suptitle(f"{split}: generative validation — GP-sampled vs real residual statistics")
    fig.tight_layout()
    fig.savefig(out_dir / f"generative_validation_{split}.png", dpi=130)
    plt.close(fig)
    print(f"[generative:{split}] n_sys={metrics['n_systems']}  "
          f"std ratio median={metrics['std_ratio_median']:.2f}  "
          f"std log-corr={metrics['std_log_corr']:.2f}  "
          f"kurt real={metrics['real_kurt_median']:.1f}/GP={metrics['gp_kurt_median']:.1f}  "
          f"pooled KS={metrics['pooled_ks_stat']:.3f}")
    return metrics


# =========================================================================== #
# Driver                                                                       #
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-aug", type=int, default=20,
                    help="augmentation draws per obs on the train split")
    ap.add_argument("--n-inducing", type=int, default=512)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--likelihood", choices=("studentt", "gaussian"),
                    default="studentt",
                    help="observation model; studentt handles heavy RV tails")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-max", type=float, default=100.0,
                    help="drop systems with median sigma above this (m/s)")
    ap.add_argument("--max-rms-over-sigma", type=float, default=30.0,
                    help="drop systems whose catalog model grossly mismatches data")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the residual-build cache")
    ap.add_argument("--exact-subsample", type=int, default=2500)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: few systems, n_aug=3, short training")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    labels = pd.read_csv(LABELS_CSV)
    if "default_flag" in labels.columns:
        labels = labels[labels["default_flag"] == 1]
    splits = pd.read_csv(SPLITS_CSV)

    if args.smoke:
        splits = pd.concat([
            splits[(splits.n_planets == 1) & (splits.split == s)].head(8)
            for s in ["train", "val", "test"]
        ])
        args.n_aug, args.n_iter, args.n_inducing = 3, 60, 128
        args.no_cache = True   # smoke subsets splits; never share its cache

    # Build splits: train augmented, val/test nominal only (n_aug=1).
    cache_dir = ROOT / "data" / "gp_residual_cache"

    def cached_build(split, n_aug):
        key = f"{split}_aug{n_aug}_seed{args.seed}_sig{args.sigma_max:g}_rms{args.max_rms_over_sigma:g}_fv5"
        path = cache_dir / f"{key}.npz"
        if not args.no_cache and path.exists():
            z = np.load(path, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            print(f"[build:{split}] cache hit {path.name}  rows={len(z['r'])}")
            return z["X"], z["r"], z["groups"], meta
        X, r, groups, meta = build_split(split, n_aug, args.seed, labels, splits,
                                         sigma_max=args.sigma_max,
                                         max_rms_over_sigma=args.max_rms_over_sigma)
        if not args.no_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(path, X=X, r=r, groups=groups, meta=json.dumps(meta))
        return X, r, groups, meta

    Xtr, rtr, gtr, mtr = cached_build("train", args.n_aug)
    Xva, rva, gva, mva = cached_build("val", 1)
    Xte, rte, gte, mte = cached_build("test", 1)

    std = Standardizer.fit(Xtr)
    Xtr_s, Xva_s, Xte_s = std.transform(Xtr), std.transform(Xva), std.transform(Xte)

    model, likelihood = fit_svgp(
        Xtr_s, rtr, n_inducing=args.n_inducing, n_iter=args.n_iter,
        lr=args.lr, batch=args.batch, seed=args.seed,
        likelihood_type=args.likelihood,
    )

    packs = {"val": (Xva, rva), "test": (Xte, rte)}
    svgp_eval, metrics = {}, {}
    for split, (Xs, r) in [("val", (Xva_s, rva)), ("test", (Xte_s, rte))]:
        if len(r) == 0:
            continue
        mean, var, samples, nll_arr = svgp_predict(model, likelihood, Xs, r, seed=args.seed)
        svgp_eval[split] = (mean, var, samples)
        metrics[split] = _metrics(r, mean, samples, nll_arr)
        m = metrics[split]
        print(f"[eval:{split}] n={m['n']}  RMSE={m['rmse']:.3f}  "
              f"std(r)={m['std_r']:.3f}  RMSE/std={m['rmse_over_std']:.3f}  "
              f"NLL={m['nll']:.3f}  cov68={m['coverage_68']:.3f}  "
              f"cov95={m['coverage_95']:.3f}")

    # Exact-GP cross-check on the test split (or val if test empty).
    cc_split = "test" if len(rte) else "val"
    Xs_eval = Xte_s if cc_split == "test" else Xva_s
    r_eval = rte if cc_split == "test" else rva
    exact_pack = None
    if len(r_eval):
        ex_mean, _, _ = exact_gp_crosscheck(
            Xtr_s, rtr, Xs_eval, r_eval, n_sub=args.exact_subsample, seed=args.seed)
        svgp_mean = svgp_eval[cc_split][0]
        corr = float(np.corrcoef(ex_mean, svgp_mean)[0, 1])
        metrics["svgp_vs_exact"] = dict(split=cc_split, corr=corr,
                                        exact_rmse=float(np.sqrt(np.mean((r_eval - ex_mean) ** 2))))
        print(f"[crosscheck:{cc_split}] SVGP-vs-exact mean corr={corr:.3f}")
        exact_pack = (cc_split, svgp_mean, ex_mean)

    # Plots
    make_plots(packs, svgp_eval, exact_pack, args.out)

    # Generative validation: do GP-sampled residuals match real statistics?
    gen_inputs = {"test": (Xte_s, Xte, rte, gte), "val": (Xva_s, Xva, rva, gva)}
    for split in ("test", "val"):
        Xs_g, Xraw_g, r_g, grp_g = gen_inputs[split]
        if len(r_g):
            metrics.setdefault("generative", {})[split] = generative_validation(
                model, likelihood, Xs_g, Xraw_g, r_g, grp_g, args.out,
                split=split, seed=args.seed)
    print(f"Wrote figures to {args.out}")

    # Save model + metadata
    import torch
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "feature_names": FEATURE_NAMES,
        "standardizer": std.to_dict(),
        "n_inducing": args.n_inducing,
        "build_meta": {"train": mtr, "val": mva, "test": mte},
    }, MODELS_DIR / "gp_residual_svgp.pt")
    (MODELS_DIR / "gp_residual_metrics.json").write_text(json.dumps({
        "metrics": metrics,
        "build_meta": {"train": mtr, "val": mva, "test": mte},
        "config": vars(args) | {"out": str(args.out)},
    }, indent=2, default=str))
    print(f"Wrote {MODELS_DIR/'gp_residual_svgp.pt'} and gp_residual_metrics.json")


if __name__ == "__main__":
    main()
