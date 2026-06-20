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
