"""
gp_noise_model.py — Gaussian Process noise model for RV residuals.

Fits a celerite2 GP to per-system residuals and provides a sampling
interface parallel to BootstrapNoiseModel in synthetic_rv.py.

Methodology
-----------
For each system we model the post-Keplerian residuals as a zero-mean
Gaussian Process with covariance K(t, t') given by a chosen kernel,
plus a diagonal contribution comprising observation noise sigma(t)
and a fitted white-jitter term sigma_jit (Foreman-Mackey et al. 2017
§5; Haywood et al. 2014). Hyperparameters are inferred by maximum
marginal likelihood with log-uniform priors enforced as bounds in an
L-BFGS-B optimization (Foreman-Mackey et al. 2017 §4; Faria et al.
2016). Multiple restarts are used and convergence is reported via the
gap in -log L between the best restart and the third-best restart.

Five kernels are supported, all augmented with a fitted log_jitter:
    sho               — Simple Harmonic Oscillator term (3 params)
    matern32          — Matern-3/2 term (2 params)
    rotation          — RotationTerm: two SHO components tied to a
                        stellar rotation period (5 params)
    sho+matern32      — Sum of SHO and Matern-3/2 (5 params)
    rotation+matern32 — Sum (7 params)

Model comparison via BIC (and AIC reported). Goodness-of-fit by
Cholesky-whitening residuals to z and applying:
    Kolmogorov-Smirnov against N(0,1)  — tests normality of z
    Ljung-Box on z up to lag min(20, N/5) — tests independence of z
Both diagnostics follow Rasmussen & Williams 2006 §5.4.2.

Priors / Bounds (log-uniform, bounds keyed to data)
---------------------------------------------------
    log_sigma          [log(0.01·std(y)), log(100·std(y))]
    log_rho|log_period [log(2·median Δt), log(span(t))]
    log_Q, log_Q0      [log(0.5), log(1000)]
    log_dQ             [log(0.001), log(100)]
    logit_f            [-5, 5]  (≈ open (0,1))
    log_jitter         [log(1e-4·std(y)), log(10·std(y))]

Bounds are uninformative within physically motivated regions and
follow the conventions of Faria et al. 2016 (kima) and
Foreman-Mackey et al. 2017.

Sampling
--------
.sample(t) draws from the GP prior at times t with the fitted
hyperparameters, using gp.dot_tril(rng.standard_normal(N)) for
reproducibility. Pure GP samples are Gaussian by construction;
the empirical residual pool has kurtosis-excess ~30, indicating
heavy tails the Gaussian process cannot represent. This module
reports the gap honestly via goodness_of_fit() and diagnose_samples()
rather than hiding it with ad-hoc heavy-tail patches.

References
----------
Foreman-Mackey, D. et al. 2017, AJ 154, 220 (celerite)
Haywood, R. et al. 2014, MNRAS 443, 2517
Faria, J. et al. 2016, A&A 588, A31 (kima)
Rasmussen, C.E. & Williams, C.K.I. 2006
Edelson, R.A. & Krolik, J.H. 1988, ApJ 333, 646 (DCF)
Rajpaul, V. et al. 2015, MNRAS 452, 2269
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import scipy.linalg as la
import scipy.stats as st
from scipy.optimize import minimize

import celerite2
from celerite2 import terms


__all__ = [
    'GPNoiseModel', 'GPNoiseLibrary',
    'GPFit', 'GoodnessOfFit',
    'KernelSpec', 'SHOSpec', 'Matern32Spec', 'RotationSpec', 'SumSpec',
    'AVAILABLE_KERNELS',
    'fit_all_kernels', 'time_lag_dcf',
]


# =========================================================================== #
# Kernel specifications                                                        #
# =========================================================================== #

def _data_stats(t, y):
    std_y = max(float(np.std(y)), 1e-3)
    if len(t) > 1:
        dt_sorted = np.diff(np.sort(t))
        median_dt = float(np.median(dt_sorted))
        span = max(float(np.ptp(t)), 1.0)
    else:
        median_dt, span = 1.0, 1.0
    if median_dt <= 0:
        median_dt = max(span / max(len(t), 2), 1e-3)
    return std_y, median_dt, span


class KernelSpec:
    """Base class; subclasses implement build(), bounds(), initial()."""
    name: str = 'abstract'
    param_names: Tuple[str, ...] = ()

    def build(self, params: np.ndarray):
        raise NotImplementedError

    def bounds(self, t, y, sigma) -> List[Tuple[float, float]]:
        raise NotImplementedError

    def initial(self, t, y, sigma) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return len(self.param_names)


class SHOSpec(KernelSpec):
    name = 'sho'
    param_names = ('log_sigma', 'log_rho', 'log_Q')

    def build(self, params):
        return terms.SHOTerm(
            sigma=float(np.exp(params[0])),
            rho=float(np.exp(params[1])),
            Q=float(np.exp(params[2])),
        )

    def bounds(self, t, y, sigma):
        std_y, median_dt, span = _data_stats(t, y)
        return [
            (np.log(0.01 * std_y), np.log(100 * std_y)),
            (np.log(2 * median_dt), np.log(span)),
            (np.log(0.5), np.log(1e3)),
        ]

    def initial(self, t, y, sigma):
        std_y, median_dt, _ = _data_stats(t, y)
        return np.array([
            np.log(std_y),
            np.log(max(10 * median_dt, 1.0)),
            np.log(1.0),
        ])


class Matern32Spec(KernelSpec):
    name = 'matern32'
    param_names = ('log_sigma', 'log_rho')

    def build(self, params):
        return terms.Matern32Term(
            sigma=float(np.exp(params[0])),
            rho=float(np.exp(params[1])),
        )

    def bounds(self, t, y, sigma):
        std_y, median_dt, span = _data_stats(t, y)
        return [
            (np.log(0.01 * std_y), np.log(100 * std_y)),
            (np.log(2 * median_dt), np.log(span)),
        ]

    def initial(self, t, y, sigma):
        std_y, median_dt, _ = _data_stats(t, y)
        return np.array([
            np.log(std_y),
            np.log(max(10 * median_dt, 1.0)),
        ])


class RotationSpec(KernelSpec):
    name = 'rotation'
    param_names = ('log_sigma', 'log_period', 'log_Q0', 'log_dQ', 'logit_f')

    def build(self, params):
        f = float(1.0 / (1.0 + np.exp(-params[4])))
        return terms.RotationTerm(
            sigma=float(np.exp(params[0])),
            period=float(np.exp(params[1])),
            Q0=float(np.exp(params[2])),
            dQ=float(np.exp(params[3])),
            f=f,
        )

    def bounds(self, t, y, sigma):
        std_y, median_dt, span = _data_stats(t, y)
        return [
            (np.log(0.01 * std_y), np.log(100 * std_y)),
            (np.log(2 * median_dt), np.log(span)),
            (np.log(0.5), np.log(1e4)),
            (np.log(1e-3), np.log(1e2)),
            (-5.0, 5.0),
        ]

    def initial(self, t, y, sigma):
        std_y, _, span = _data_stats(t, y)
        return np.array([
            np.log(std_y),
            np.log(max(span / 4, 10.0)),
            np.log(10.0),
            np.log(1.0),
            0.0,
        ])


class SumSpec(KernelSpec):
    """Sum of two kernel specs."""

    def __init__(self, k1: KernelSpec, k2: KernelSpec):
        self.k1, self.k2 = k1, k2
        self.name = f"{k1.name}+{k2.name}"
        self.param_names = (
            tuple(f"{k1.name}.{p}" for p in k1.param_names)
            + tuple(f"{k2.name}.{p}" for p in k2.param_names)
        )

    def build(self, params):
        n1 = self.k1.n_params
        return self.k1.build(params[:n1]) + self.k2.build(params[n1:])

    def bounds(self, t, y, sigma):
        return self.k1.bounds(t, y, sigma) + self.k2.bounds(t, y, sigma)

    def initial(self, t, y, sigma):
        i1 = self.k1.initial(t, y, sigma).copy()
        i2 = self.k2.initial(t, y, sigma).copy()
        i1[0] -= 0.5 * np.log(2)
        i2[0] -= 0.5 * np.log(2)
        return np.concatenate([i1, i2])


AVAILABLE_KERNELS: Dict[str, KernelSpec] = {
    'sho': SHOSpec(),
    'matern32': Matern32Spec(),
    'rotation': RotationSpec(),
    'sho+matern32': SumSpec(SHOSpec(), Matern32Spec()),
    'rotation+matern32': SumSpec(RotationSpec(), Matern32Spec()),
}


# =========================================================================== #
# Fit containers                                                               #
# =========================================================================== #

@dataclass
class GPFit:
    kernel_name: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    n_params: int
    success: bool
    convergence_gap: float = 0.0
    n_restarts: int = 0
    host: Optional[str] = None
    file: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class GoodnessOfFit:
    """Whitened-residual diagnostics for a fitted GP.

    z = L^-1 y where L L^T = K(t,t) + diag(sigma^2 + jitter^2).
    Under a correctly specified GP, z ~ IID N(0,1).
    """
    ks_statistic: float
    ks_pvalue: float
    lb_statistic: float
    lb_pvalue: float
    lb_df: int
    whitened_std: float
    whitened_kurt_excess: float
    n: int

    def to_dict(self):
        return asdict(self)


# =========================================================================== #
# Single-system model                                                          #
# =========================================================================== #

class GPNoiseModel:
    """
    GP noise fit for one RV system. Parallel API to BootstrapNoiseModel.

    >>> m = GPNoiseModel(kernel='sho')
    >>> m.fit(t, residuals, sigma)
    >>> noise = m.sample(t_new, rng=np.random.default_rng(0))
    >>> gof = m.goodness_of_fit(t, residuals, sigma)
    """

    AVAILABLE = list(AVAILABLE_KERNELS.keys())

    def __init__(self, kernel='sho'):
        if isinstance(kernel, str):
            if kernel not in AVAILABLE_KERNELS:
                raise ValueError(f"unknown kernel '{kernel}'; available {self.AVAILABLE}")
            self.spec = AVAILABLE_KERNELS[kernel]
        elif isinstance(kernel, KernelSpec):
            self.spec = kernel
        else:
            raise TypeError("kernel must be str or KernelSpec")
        self.kernel_name = self.spec.name
        self.param_names = self.spec.param_names + ('log_jitter',)
        self.fit_result: Optional[GPFit] = None

    # ---- internals ----

    def _jitter_bounds(self, y, sigma):
        std_y = max(float(np.std(y)), 1e-3)
        return (np.log(1e-4 * std_y), np.log(10 * std_y))

    def _full_bounds(self, t, y, sigma):
        return self.spec.bounds(t, y, sigma) + [self._jitter_bounds(y, sigma)]

    def _full_initial(self, t, y, sigma):
        x_kernel = self.spec.initial(t, y, sigma)
        log_j0 = np.log(max(float(np.median(sigma)), 1e-3))
        jlo, jhi = self._jitter_bounds(y, sigma)
        log_j0 = float(np.clip(log_j0, jlo + 0.1, jhi - 0.1))
        return np.concatenate([x_kernel, [log_j0]])

    def _neg_log_like(self, params, t, y, yerr):
        kernel_params = params[:-1]
        log_jitter = params[-1]
        try:
            kernel = self.spec.build(kernel_params)
            gp = celerite2.GaussianProcess(kernel, mean=0.0)
            eff_yerr = np.sqrt(yerr**2 + np.exp(2 * log_jitter))
            gp.compute(t, yerr=eff_yerr, quiet=True)
            ll = gp.log_likelihood(y)
            if not np.isfinite(ll):
                return 1e20
            return -ll
        except Exception:
            return 1e20

    def _params_array(self) -> np.ndarray:
        if self.fit_result is None:
            raise RuntimeError("Model is not fit.")
        return np.array([self.fit_result.params[n] for n in self.param_names])

    # ---- fit ----

    def fit(self, t, residuals, sigma, host=None, file=None,
            n_restarts=5, seed=0) -> GPFit:
        """Maximize marginal likelihood within log-uniform bounds.
        L-BFGS-B with finite-difference gradient, multi-restart."""
        t = np.asarray(t, float)
        y = np.asarray(residuals, float)
        yerr = np.asarray(sigma, float)

        order = np.argsort(t)
        t, y, yerr = t[order], y[order], yerr[order]

        bounds = self._full_bounds(t, y, yerr)
        x0 = self._full_initial(t, y, yerr)
        x0 = np.array([np.clip(x0[i], bounds[i][0] + 1e-6, bounds[i][1] - 1e-6)
                       for i in range(len(x0))])

        rng = np.random.default_rng(seed)
        results = []
        for i in range(n_restarts):
            x_start = x0 if i == 0 else np.array(
                [rng.uniform(lo, hi) for lo, hi in bounds]
            )
            try:
                res = minimize(
                    self._neg_log_like, x_start, args=(t, y, yerr),
                    method='L-BFGS-B', bounds=bounds,
                    options=dict(maxiter=500, ftol=1e-8, gtol=1e-6),
                )
                if np.isfinite(res.fun):
                    results.append(res)
            except Exception:
                continue

        if not results:
            raise RuntimeError(
                f"All {n_restarts} restarts failed for kernel='{self.kernel_name}'."
            )

        best = min(results, key=lambda r: r.fun)
        funs = sorted(r.fun for r in results)
        gap_idx = min(2, len(funs) - 1)
        gap = float(funs[gap_idx] - funs[0])

        ll = float(-best.fun)
        n = len(t)
        k = len(self.param_names)
        bic = float(k * np.log(n) - 2 * ll)
        aic = float(2 * k - 2 * ll)

        self.fit_result = GPFit(
            kernel_name=self.kernel_name,
            params=dict(zip(self.param_names, best.x.tolist())),
            log_likelihood=ll,
            aic=aic, bic=bic,
            n_obs=n, n_params=k,
            success=bool(best.success),
            convergence_gap=gap,
            n_restarts=len(results),
            host=host, file=file,
        )
        return self.fit_result

    # ---- sample ----

    def sample(self, t, rng=None, yerr=None, with_jitter=True) -> np.ndarray:
        """Draw a noise realization at times t from the GP prior with
        fitted hyperparameters.

        Parameters
        ----------
        t : array
            Times at which to sample.
        rng : np.random.Generator, optional
            For reproducibility.
        yerr : array or None
            If provided, obs uncertainties at sample times are added in
            quadrature to fitted jitter. Use this when generating
            synthetic data with known obs noise. If None, kernel + jitter only.
        with_jitter : bool
            If False, omit the fitted jitter (pure kernel sample).
        """
        if self.fit_result is None:
            raise RuntimeError("Call fit() before sample().")
        rng = rng or np.random.default_rng()
        t = np.asarray(t, float)
        order = np.argsort(t)
        t_sorted = t[order]

        params = self._params_array()
        kernel_params = params[:-1]
        log_jitter = params[-1]

        if yerr is None:
            eff_yerr = np.full_like(t_sorted, np.exp(log_jitter) if with_jitter else 1e-10)
        else:
            ys = np.asarray(yerr, float)[order]
            if with_jitter:
                eff_yerr = np.sqrt(ys**2 + np.exp(2 * log_jitter))
            else:
                eff_yerr = np.maximum(ys, 1e-10)

        kernel = self.spec.build(kernel_params)
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        gp.compute(t_sorted, yerr=eff_yerr, quiet=True)

        white = rng.standard_normal(len(t_sorted))
        s_sorted = gp.dot_tril(white)

        out = np.empty_like(s_sorted)
        out[order] = s_sorted
        return out

    # ---- goodness of fit (dense Cholesky whitening) ----

    def whiten(self, t, y, sigma, max_n: int = 2500) -> np.ndarray:
        """Cholesky-whitened residuals z = L^-1 y where
        L L^T = K(t,t) + diag(sigma^2 + jitter^2).

        For a correctly-specified GP, z ~ IID N(0,1).
        Uses dense Cholesky; raises if len(t) > max_n.
        """
        if self.fit_result is None:
            raise RuntimeError("Call fit() first.")
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        sigma = np.asarray(sigma, float)
        order = np.argsort(t)
        t_s, y_s, sigma_s = t[order], y[order], sigma[order]
        n = len(t_s)
        if n > max_n:
            raise RuntimeError(f"whiten(): dense Cholesky too costly for N={n} > {max_n}")

        params = self._params_array()
        kernel = self.spec.build(params[:-1])
        log_jitter = params[-1]

        tau = np.abs(t_s[:, None] - t_s[None, :])
        K = kernel.get_value(tau)
        K = K + np.diag(sigma_s**2 + np.exp(2 * log_jitter))

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            jitter_add = 1e-10 * (np.trace(K) / n)
            K = K + np.eye(n) * jitter_add
            L = np.linalg.cholesky(K)

        z_s = la.solve_triangular(L, y_s, lower=True)
        out = np.empty_like(z_s)
        out[order] = z_s
        return out

    def goodness_of_fit(self, t, y, sigma) -> GoodnessOfFit:
        """Whitened-residual normality (KS) and independence (Ljung-Box)."""
        z = self.whiten(t, y, sigma)
        n = len(z)
        ks_stat, ks_p = st.kstest(z, 'norm')

        max_lag = min(20, n // 5)
        if max_lag < 2:
            lb_stat, lb_p, lb_df = float('nan'), float('nan'), 0
        else:
            z_c = z - z.mean()
            denom = float(np.dot(z_c, z_c))
            acf = np.empty(max_lag + 1)
            acf[0] = 1.0
            for k in range(1, max_lag + 1):
                acf[k] = float(np.dot(z_c[:-k], z_c[k:]) / denom)
            lb_stat = float(n * (n + 2) * np.sum(acf[1:]**2 / (n - np.arange(1, max_lag + 1))))
            lb_p = float(1.0 - st.chi2.cdf(lb_stat, df=max_lag))
            lb_df = int(max_lag)

        return GoodnessOfFit(
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            lb_statistic=lb_stat,
            lb_pvalue=lb_p,
            lb_df=lb_df,
            whitened_std=float(np.std(z)),
            whitened_kurt_excess=float(st.kurtosis(z, fisher=True)),
            n=int(n),
        )

    # ---- diagnostic on samples ----

    def diagnose_samples(self, t, n_draws=200, rng=None) -> dict:
        """Sample many noise realizations; report std + kurtosis."""
        rng = rng or np.random.default_rng(0)
        samples = np.array([self.sample(t, rng=rng) for _ in range(n_draws)])
        flat = samples.ravel()
        return dict(
            std=float(np.std(flat)),
            kurt_excess=float(st.kurtosis(flat, fisher=True)),
            min=float(flat.min()),
            max=float(flat.max()),
        )


# =========================================================================== #
# Multi-kernel fitting + selection                                             #
# =========================================================================== #

def fit_all_kernels(t, residuals, sigma, host=None, file=None,
                    kernels: Sequence[str] = ('sho', 'matern32', 'rotation', 'sho+matern32'),
                    n_restarts: int = 5, seed: int = 0,
                    verbose: bool = False) -> Tuple[GPNoiseModel, Dict[str, GPFit]]:
    """Fit each kernel; return (best_model_by_BIC, dict_of_all_fits)."""
    fits = {}
    models = {}
    for kname in kernels:
        m = GPNoiseModel(kernel=kname)
        try:
            fr = m.fit(t, residuals, sigma, host=host, file=file,
                       n_restarts=n_restarts, seed=seed)
            fits[kname] = fr
            models[kname] = m
            if verbose:
                print(f"    {kname:>20}: logL={fr.log_likelihood:8.2f}  "
                      f"BIC={fr.bic:7.2f}  AIC={fr.aic:7.2f}  "
                      f"gap={fr.convergence_gap:.2g}  k={fr.n_params}")
        except Exception as e:
            if verbose:
                print(f"    {kname:>20}: FAILED ({e})")
    if not fits:
        raise RuntimeError("All kernels failed.")
    best_kname = min(fits, key=lambda k: fits[k].bic)
    return models[best_kname], fits


# =========================================================================== #
# Library: drop-in for BootstrapNoiseModel                                     #
# =========================================================================== #

class GPNoiseLibrary:
    """
    Library of fitted GPs. .sample(t, rng) picks a random fit and samples
    noise at the requested times. Drop-in for BootstrapNoiseModel in
    synthetic_rv.py.
    """

    def __init__(self, fits: Sequence[GPFit]):
        if not fits:
            raise ValueError("Library empty.")
        self.fits = list(fits)

    @classmethod
    def from_json(cls, path) -> "GPNoiseLibrary":
        """Read JSON written by gp_corpus_fit.py. Accepts both formats:
        flat list of GPFit dicts, or rich system-level dicts with
        'best_kernel' and 'all_fits'."""
        with open(path) as f:
            records = json.load(f)
        fits = []
        for r in records:
            if 'best_kernel' in r:
                best = r['best_kernel']
                fd = dict(r['all_fits'][best])
                fd['host'] = r.get('host')
                fd['file'] = r.get('file')
                fits.append(GPFit(**fd))
            else:
                fits.append(GPFit(**r))
        return cls(fits)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump([fr.to_dict() for fr in self.fits], f, indent=2)

    def __len__(self):
        return len(self.fits)

    def sample(self, t, rng=None, which=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        idx = which if which is not None else int(rng.integers(len(self.fits)))
        fr = self.fits[idx]
        m = GPNoiseModel(kernel=fr.kernel_name)
        m.fit_result = fr
        return m.sample(t, rng=rng)


# =========================================================================== #
# Time-lag DCF (Edelson & Krolik 1988) for irregular sampling                  #
# =========================================================================== #

def time_lag_dcf(t, y, n_bins: int = 20, max_lag=None, log_spaced: bool = True):
    """
    Discrete Correlation Function (Edelson & Krolik 1988, ApJ 333, 646).

    For an irregularly-sampled time series, the time-lag autocorrelation
    is estimated by binning all pairs of points by their time separation.
    Returns (bin_centers, dcf, dcf_err, counts).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    y_c = y - y.mean()
    var = float(np.var(y))
    if var == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    i_idx, j_idx = np.triu_indices(n, k=1)
    lags = t[j_idx] - t[i_idx]
    prods = y_c[i_idx] * y_c[j_idx] / var

    if max_lag is None:
        max_lag = float(lags.max()) / 2.0
    min_lag = float(lags.min()) if lags.min() > 0 else 1e-3

    if log_spaced and min_lag > 0:
        edges = np.logspace(np.log10(max(min_lag, 1e-3)),
                             np.log10(max(max_lag, min_lag * 10)), n_bins + 1)
    else:
        edges = np.linspace(min_lag, max_lag, n_bins + 1)

    centers, means, errs, counts = [], [], [], []
    for k in range(n_bins):
        mask = (lags >= edges[k]) & (lags < edges[k + 1])
        c = int(mask.sum())
        if c < 2:
            continue
        if log_spaced:
            centers.append(float(np.sqrt(edges[k] * edges[k + 1])))
        else:
            centers.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(float(prods[mask].mean()))
        errs.append(float(prods[mask].std() / np.sqrt(c)))
        counts.append(c)

    return np.array(centers), np.array(means), np.array(errs), np.array(counts)
