"""Fixed-length features for unevenly sampled radial-velocity observations."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft
from scipy.interpolate import UnivariateSpline


def spectral_feature_names(d: int) -> list[str]:
    """Return names for the first ``d`` non-DC Fourier power bins."""
    if d <= 0:
        raise ValueError("d must be positive")
    return [f"spectral_power_{i:03d}" for i in range(1, d + 1)]


def spectral_features(
    t,
    y,
    d: int = 64,
    grid_size: int = 1024,
    smoothing: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode an uneven time series as a fixed-length Fourier power vector.

    The observations are sorted, duplicate timestamps are averaged, and a
    smoothing spline is evaluated on a uniform grid. The DC component is
    removed before the first ``d`` non-zero-frequency power bins are returned.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if smoothing is not None and smoothing < 0:
        raise ValueError("smoothing must be non-negative or None")

    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    if len(t) < 2:
        raise ValueError("at least two observations are required")
    if not (np.isfinite(t).all() and np.isfinite(y).all()):
        raise ValueError("t and y must contain only finite values")

    order = np.argsort(t, kind="stable")
    t = t[order]
    y = y[order]

    # UnivariateSpline requires strictly increasing x values. Multiple RV
    # measurements at the same timestamp are represented by their mean.
    t_unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if len(t_unique) < 2:
        raise ValueError("at least two distinct observation times are required")
    if len(t_unique) != len(t):
        y_sum = np.zeros(len(t_unique), dtype=float)
        np.add.at(y_sum, inverse, y)
        y = y_sum / counts
        t = t_unique

    spline_degree = min(3, len(t) - 1)
    spline = UnivariateSpline(t, y, k=spline_degree, s=smoothing)
    t_uniform = np.linspace(t[0], t[-1], grid_size)
    y_fit = np.asarray(spline(t_uniform), dtype=float)
    if not np.isfinite(y_fit).all():
        raise ValueError("spline interpolation produced non-finite values")

    y_fit -= y_fit.mean()
    if np.allclose(y_fit, 0.0, rtol=1e-10, atol=1e-12):
        return np.zeros(d, dtype=np.float64)

    # Bin zero is the DC component. It should be zero after mean subtraction
    # and is excluded so every returned dimension represents real variation.
    power = np.abs(rfft(y_fit)) ** 2
    power = power[1:]

    if normalize:
        power_sum = float(power.sum())
        if power_sum > 0:
            power = power / power_sum

    features = np.zeros(d, dtype=np.float64)
    n_copy = min(d, len(power))
    features[:n_copy] = power[:n_copy]
    return features


PHASE_SCALAR_NAMES = ["phase_ptp", "phase_skew", "phase_half_diff"]


def phase_fold_bin_names(n_bins: int) -> list[str]:
    """Return names for ``n_bins`` orbital-phase RV mean bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [f"phase_bin_{i + 1:03d}" for i in range(n_bins)]


def phase_fold_feature_names(n_bins: int = 32) -> list[str]:
    """All phase-fold feature names: bins plus shape scalars."""
    return [*phase_fold_bin_names(n_bins), *PHASE_SCALAR_NAMES]


def phase_fold_anchor_from_curve(
    t,
    y,
    P_days: float,
    *,
    n_bins: int = 32,
) -> float:
    """Epoch-free fold anchor: phase origin at the max of a provisional fold.

    Provisional fold uses ``t.min()`` as origin; the phase of the maximum
    bin-mean RV becomes the new zero. Identical convention on synthetic and
    real curves, so omega remains identifiable from waveform asymmetry.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if P_days <= 0 or not np.isfinite(P_days):
        raise ValueError("P_days must be positive and finite")
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(t) != len(y) or len(t) < 2:
        raise ValueError("need at least two (t, y) observations")
    if not (np.isfinite(t).all() and np.isfinite(y).all()):
        raise ValueError("t and y must be finite")

    t0 = float(np.min(t))
    phase = ((t - t0) / float(P_days)) % 1.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means = np.full(n_bins, np.nan, dtype=np.float64)
    for j in range(n_bins):
        if j < n_bins - 1:
            mask = (phase >= edges[j]) & (phase < edges[j + 1])
        else:
            mask = (phase >= edges[j]) & (phase <= edges[j + 1])
        if mask.any():
            bin_means[j] = float(np.mean(y[mask]))
    if not np.isfinite(bin_means).any():
        return t0
    j_max = int(np.nanargmax(bin_means))
    phase_max = 0.5 * (edges[j_max] + edges[j_max + 1])
    return t0 + phase_max * float(P_days)


def phase_fold_features(
    t,
    y,
    P_days: float,
    *,
    n_bins: int = 32,
    t_peri: float | None = None,
    epoch_free: bool = False,
) -> np.ndarray:
    """Encode an RV curve in orbital phase for eccentricity / omega shape analysis.

    Phase is ``((t - t_peri) / P) mod 1``. By default ``t_peri`` is required
    (catalog periapsis). With ``epoch_free=True``, the fold origin is the
    phase of maximum RV on a provisional fold (see
    ``phase_fold_anchor_from_curve``), so real curves without catalog
    ``t_peri`` can be featurized under the same convention as synthetics.

    Returns ``n_bins`` mean-RV bins plus three shape scalars (35 features when
    ``n_bins=32``): peak-to-peak, skewness, and first-half minus second-half mean.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if P_days <= 0 or not np.isfinite(P_days):
        raise ValueError("P_days must be positive and finite")

    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(t) != len(y) or len(t) < 2:
        raise ValueError("need at least two (t, y) observations")
    if not (np.isfinite(t).all() and np.isfinite(y).all()):
        raise ValueError("t and y must be finite")

    if epoch_free:
        t_peri = phase_fold_anchor_from_curve(t, y, P_days, n_bins=n_bins)
    elif t_peri is None or not np.isfinite(t_peri):
        raise ValueError("t_peri is required for phase-fold features (or pass epoch_free=True)")

    phase = ((t - float(t_peri)) / float(P_days)) % 1.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means = np.zeros(n_bins, dtype=np.float64)
    for j in range(n_bins):
        if j < n_bins - 1:
            mask = (phase >= edges[j]) & (phase < edges[j + 1])
        else:
            mask = (phase >= edges[j]) & (phase <= edges[j + 1])
        if mask.any():
            bin_means[j] = float(np.mean(y[mask]))

    ptp = float(bin_means.max() - bin_means.min())
    std = float(np.std(bin_means))
    if std > 1e-12:
        skew = float(np.mean(((bin_means - bin_means.mean()) / std) ** 3))
    else:
        skew = 0.0
    half = n_bins // 2
    half_diff = float(bin_means[:half].mean() - bin_means[half:].mean())

    return np.concatenate([bin_means, np.array([ptp, skew, half_diff], dtype=np.float64)])


def phase_fold_curve(
    t,
    y,
    P_days: float,
    *,
    t_peri: float,
    n_plot: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase_grid, rv_on_grid) for plotting a folded curve."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    phase = ((t - float(t_peri)) / float(P_days)) % 1.0
    order = np.argsort(phase)
    phase_sorted = phase[order]
    y_sorted = y[order]
    grid = np.linspace(0.0, 1.0, n_plot)
    rv_grid = np.interp(grid, phase_sorted, y_sorted, period=1.0)
    return grid, rv_grid
