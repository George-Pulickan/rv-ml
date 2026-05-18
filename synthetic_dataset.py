"""
synthetic_dataset.py — Synthetic RV generator for encoder pre-training.

Samples (P, K, e, ω) from realistic priors, generates a Keplerian RV curve
via models/kepler_torch.py, injects GP noise from GPNoiseLibrary, and
returns the same (4×256) tensor format as RVDataset in preprocess.py.

The synthetic dataset is unlimited in size and provides clean ground-truth
labels — essential for pre-training the encoder before fine-tuning on the
631 real systems.

Priors (motivated by the corpus statistics in data/splits.csv):
    P      ~ LogUniform(1, 3000)   days
    K      ~ LogUniform(1, 300)    m/s
    e      ~ Beta(2, 5)            low-eccentricity prior; mode at 0.17
    ω      ~ Uniform(0, 2π)
    T_peri ~ Uniform(t_min, t_min + P)   (random phase)
    γ      = 0  (median subtracted in the tensor, so γ is irrelevant)

Time grids are drawn from the empirical distribution of the real corpus:
    baseline ~ LogUniform(100, 4000) d  (trimmed real data range)
    n_obs    ~ DiscreteUniform(15, 200)  (trimmed)
    σ_obs    ~ LogNormal fit to real σ distribution (median≈4.6, p95≈16 m/s)
    cadence  ~ clustered Poisson-gap model seeded from real gaps

GP noise is injected via GPNoiseLibrary when a library JSON is available.
Falls back to white Gaussian noise scaled to σ_obs if the library is absent.

Usage
-----
    from synthetic_dataset import SyntheticRVDataset, make_synthetic_batch

    ds = SyntheticRVDataset(n_samples=50_000, seed=42)
    x, theta, info = ds[0]          # x: (4,256) float32 tensor

    # For quick batching without DataLoader:
    X, Theta = make_synthetic_batch(batch_size=64, rng=np.random.default_rng(0))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import T_MAX, THETA_NAMES

_GP_LIB_PATH  = Path("data/gp_fits.json")
_STATS_PATH   = Path("data/dataset_stats.json")

# Empirical cadence model (from 146 real systems, see synthetic_dataset.py header)
_BASELINE_LOG_MIN  = np.log(100.0)
_BASELINE_LOG_MAX  = np.log(4000.0)
_N_OBS_MIN         = 15
_N_OBS_MAX         = 200
_SIGMA_LOG_MEAN    = np.log(4.62)   # ln(median σ_obs)
_SIGMA_LOG_STD     = 0.75           # calibrated to span p5=0.9 → p95=16 m/s


# ---------------------------------------------------------------------------
# Prior samplers
# ---------------------------------------------------------------------------

def _sample_orbital_params(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Sample n sets of (P, K, e, ω, T_peri_phase) from the prior."""
    P      = np.exp(rng.uniform(np.log(1.0), np.log(3000.0), size=n))
    K      = np.exp(rng.uniform(np.log(1.0), np.log(300.0),  size=n))
    e      = rng.beta(2, 5, size=n).clip(0.0, 0.99)
    omega  = rng.uniform(0.0, 2 * np.pi, size=n)
    phase  = rng.uniform(0.0, 1.0, size=n)   # T_peri = t_min + phase * P
    return {"P": P, "K": K, "e": e, "omega": omega, "phase": phase}


def _sample_time_grid(rng: np.random.Generator) -> np.ndarray:
    """
    Draw a realistic observation time grid for one system.

    Uses a clustered Poisson-gap model: most observations come in runs
    (observing seasons) separated by longer gaps, mimicking real campaigns.
    """
    baseline = float(np.exp(rng.uniform(_BASELINE_LOG_MIN, _BASELINE_LOG_MAX)))
    n_obs    = int(rng.integers(_N_OBS_MIN, _N_OBS_MAX + 1))

    # ~3 observing seasons per year; season length ~100 d
    n_seasons = max(1, round(baseline / 365.25 * 3))
    season_starts = np.sort(rng.uniform(0, baseline - 100.0, size=n_seasons).clip(0, baseline))

    points = []
    for s_start in season_starts:
        season_end = min(s_start + 100.0, baseline)
        k = rng.integers(1, max(2, n_obs // n_seasons + 2))
        pts = rng.uniform(s_start, season_end, size=k)
        points.append(pts)
    t = np.sort(np.concatenate(points))

    # Thin or pad to exactly n_obs
    if len(t) >= n_obs:
        idx = np.sort(rng.choice(len(t), size=n_obs, replace=False))
        t   = t[idx]
    else:
        extra = rng.uniform(0, baseline, size=n_obs - len(t))
        t     = np.sort(np.concatenate([t, extra]))

    return t.astype(np.float64)


def _sample_sigma(rng: np.random.Generator, n_obs: int) -> np.ndarray:
    """Per-observation measurement uncertainty drawn from the empirical distribution."""
    log_sig = rng.normal(_SIGMA_LOG_MEAN, _SIGMA_LOG_STD, size=n_obs)
    return np.exp(log_sig).astype(np.float64)


# ---------------------------------------------------------------------------
# GP noise injection
# ---------------------------------------------------------------------------

def _load_gp_library():
    """Load GPNoiseLibrary lazily; returns None if the JSON is absent."""
    if not _GP_LIB_PATH.exists():
        return None
    try:
        from gp_noise_model import GPNoiseLibrary
        return GPNoiseLibrary.from_json(str(_GP_LIB_PATH))
    except Exception:
        return None


_GP_LIBRARY = None  # module-level cache; populated on first use


def _inject_noise(t: np.ndarray, sigma: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Sample additive noise for a time grid.

    If GPNoiseLibrary is available, draw from a random real system's GP.
    Otherwise fall back to i.i.d. N(0, σ²) white noise.
    """
    global _GP_LIBRARY
    if _GP_LIBRARY is None:
        _GP_LIBRARY = _load_gp_library()

    if _GP_LIBRARY is not None:
        try:
            s = _GP_LIBRARY.sample(t, rng=rng).astype(np.float64)
            if not np.isnan(s).any():
                return s
        except Exception:
            pass  # fall through to white noise
    return rng.normal(0.0, sigma).astype(np.float64)


# ---------------------------------------------------------------------------
# Single-sample generator
# ---------------------------------------------------------------------------

def generate_one(
    params: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate one synthetic RV sample.

    Parameters
    ----------
    params : dict with keys P, K, e, omega, phase
    rng    : numpy Generator

    Returns
    -------
    x     : (4, T_MAX) float32  — input tensor in preprocess.py format
    theta : (5,) float32        — [log10_P, log10_K, e, cos_ω, sin_ω]
    info  : dict                — metadata for debugging
    """
    from kepler_check import rv_keplerian as rv_np   # numpy reference

    P, K, e, omega = params["P"], params["K"], params["e"], params["omega"]

    t       = _sample_time_grid(rng)
    sigma   = _sample_sigma(rng, len(t))
    t_peri  = t.min() + params["phase"] * P

    rv_clean = rv_np(t, P, K, e, omega, t_peri)        # noiseless Kepler
    noise    = _inject_noise(t, sigma, rng)
    rv_obs   = rv_clean + noise

    # --- pack into (4, T_MAX) tensor (same as RVDataset) ---
    n_real  = len(t)
    t_min   = float(t.min())
    t_span  = float(t.max() - t.min()) if len(t) > 1 else 1.0
    rv_med  = float(np.median(rv_obs))
    rv_std  = float(np.std(rv_obs)) or 1.0

    t_norm  = (t - t_min) / t_span
    rv_norm = (rv_obs - rv_med) / rv_std
    sig_norm = sigma / rv_std

    x = np.zeros((4, T_MAX), dtype=np.float32)
    n_pad = min(n_real, T_MAX)
    x[0, :n_pad] = t_norm[:n_pad]
    x[1, :n_pad] = rv_norm[:n_pad]
    x[2, :n_pad] = sig_norm[:n_pad]
    x[3, :n_pad] = 1.0

    theta = np.array([
        np.log10(P),
        np.log10(K),
        e,
        np.cos(omega),
        np.sin(omega),
    ], dtype=np.float32)

    info = {
        "P": P, "K": K, "e": e, "omega_deg": np.degrees(omega),
        "t_peri": t_peri, "n_obs": n_real,
        "baseline_d": t_span, "snr": K / float(np.median(sigma)),
        "t_min": t_min, "rv_std": rv_std,
        # Canonical keys matching RVDataset (used by train.py collate_fn)
        "t_span_days": t_span,
        "t_min_days":  t_min,
        "rv_std_ms":   rv_std,
        "valid":       True,
    }
    return x, theta, info


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticRVDataset(Dataset):
    """
    Torch Dataset of synthetic RV systems with known orbital parameters.

    Parameters
    ----------
    n_samples : total number of samples in the epoch
    seed      : base random seed (each sample uses seed + index)
    stats     : optional normalization dict from data/dataset_stats.json;
                if provided, theta is normalized to zero-mean/unit-std using
                training-split statistics (same as RVDataset)
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
        stats: dict | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.seed      = seed
        self.stats     = stats or _load_stats()

        # Pre-sample all orbital parameters at construction time for
        # reproducibility (time grids are sampled lazily per __getitem__).
        rng = np.random.default_rng(seed)
        self._params = _sample_orbital_params(rng, n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        params = {k: float(v[idx]) for k, v in self._params.items()}
        rng    = np.random.default_rng(self.seed + idx)
        x, theta, info = generate_one(params, rng)

        if self.stats is not None:
            theta = _normalise_theta(theta, self.stats)

        return torch.from_numpy(x), torch.from_numpy(theta), info


def _load_stats() -> dict | None:
    if _STATS_PATH.exists():
        return json.loads(_STATS_PATH.read_text())
    return None


def _normalise_theta(theta: np.ndarray, stats: dict) -> np.ndarray:
    out = theta.copy()
    for i, name in enumerate(THETA_NAMES):
        mu  = stats[name]["mean"]
        std = stats[name]["std"] or 1.0
        out[i] = (theta[i] - mu) / std
    return out


# ---------------------------------------------------------------------------
# Batch helper (numpy, no DataLoader overhead)
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of synthetic samples.

    Returns
    -------
    X     : (B, 4, T_MAX) float32
    Theta : (B, 5) float32
    """
    rng    = rng or np.random.default_rng()
    stats  = stats or _load_stats()
    params = _sample_orbital_params(rng, batch_size)
    X_list, Theta_list = [], []
    for i in range(batch_size):
        p   = {k: float(v[i]) for k, v in params.items()}
        sub = np.random.default_rng(rng.integers(0, 2**31))
        x, theta, _ = generate_one(p, sub)
        if stats is not None:
            theta = _normalise_theta(theta, stats)
        X_list.append(x)
        Theta_list.append(theta)
    return np.stack(X_list), np.stack(Theta_list)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Generating 200 synthetic samples …")
    t0  = time.perf_counter()
    ds  = SyntheticRVDataset(n_samples=200, seed=0)
    dt  = time.perf_counter() - t0

    x0, th0, info0 = ds[0]
    print(f"  Construction time: {dt:.2f} s")
    print(f"  x shape:     {tuple(x0.shape)}  (want (4, {T_MAX}))")
    print(f"  theta shape: {tuple(th0.shape)}  (want (5,))")
    print(f"  theta names: {THETA_NAMES}")
    print(f"  sample[0] info:")
    for k, v in info0.items():
        print(f"    {k:12s} = {v:.4g}" if isinstance(v, float) else f"    {k:12s} = {v}")

    # Distribution sanity checks
    all_theta = np.stack([ds[i][1].numpy() for i in range(200)])
    print(f"\n  Theta stats over 200 samples (normalised):")
    for i, name in enumerate(THETA_NAMES):
        col = all_theta[:, i]
        print(f"    {name:12s}  mean={col.mean():.2f}  std={col.std():.2f}")

    print("\n  Batch helper (B=8) …")
    X, Theta = make_synthetic_batch(batch_size=8)
    print(f"  X shape: {X.shape}  Theta shape: {Theta.shape}")

    print("\nDone.")
