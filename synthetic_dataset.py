"""
synthetic_dataset.py — Synthetic RV generator for encoder pre-training.

Samples (P, K, e, ω) from physically motivated priors, generates a Keplerian
RV curve via models/kepler_torch.py, injects GP noise from GPNoiseLibrary, and
returns the same (4×256, LSP_N) tensor format as RVDataset in preprocess.py.

Priors
------
    P      ~ LogUniform(1, 3000) d
    K      ~ LogUniform(1, 300) m/s
    e      ~ Beta(2, 5)   low-eccentricity prior (Kipping 2013, MNRAS 434, L51)
    ω      ~ Uniform(0, 2π)
    T_peri ~ Uniform(t_min, t_min + P)   (random orbital phase)
    γ = 0  (median-subtracted in normalised tensor)

Time grids
----------
    Bootstrapped from the real training corpus (data/splits.csv + data/rv_raw).
    Using actual cadences is more realistic than any heuristic model and ensures
    the encoder pre-trains on exactly the same observing patterns as fine-tuning.
    Falls back to a seasonal Poisson-gap model if real grids are unavailable.

Noise
-----
    GP noise drawn from GPNoiseLibrary (Matérn-3/2 / SHO fitted to real residuals).
    Falls back to i.i.d. N(0, σ²) white noise if GP library is absent or returns NaN.

Usage
-----
    from synthetic_dataset import SyntheticRVDataset, make_synthetic_batch

    ds = SyntheticRVDataset(n_samples=50_000, seed=42)
    x, lsp, theta, info = ds[0]     # x:(4,256) lsp:(LSP_N,) theta:(5,)

    X, Lsp, Theta = make_synthetic_batch(batch_size=64, rng=np.random.default_rng(0))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import LSP_N, LSP_PERIODS, T_MAX, THETA_NAMES, compute_lsp

_GP_LIB_PATH = Path("data/gp_fits.json")
_STATS_PATH  = Path("data/dataset_stats.json")
_SPLITS_CSV  = Path("data/splits.csv")
_RV_DIR      = Path("data/rv_raw")

# σ distribution fit to 146 real systems: ln(σ) ~ N(μ, σ²)
# Calibrated: median σ ≈ 4.6 m/s, p5 ≈ 0.9 m/s, p95 ≈ 16 m/s
_SIGMA_LOG_MEAN = np.log(4.62)
_SIGMA_LOG_STD  = 0.75


# ---------------------------------------------------------------------------
# Prior samplers
# ---------------------------------------------------------------------------

def _sample_orbital_params(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """
    Sample n sets of (P, K, e, ω, phase) from the prior.

    e ~ Beta(2, 5) following Kipping (2013, MNRAS 434, L51), which provides
    an informative prior from the observed exoplanet eccentricity distribution.
    """
    P     = np.exp(rng.uniform(np.log(1.0),    np.log(3000.0), size=n))
    K     = np.exp(rng.uniform(np.log(1.0),    np.log(300.0),  size=n))
    e     = rng.beta(2, 5, size=n).clip(0.0, 0.99)
    omega = rng.uniform(0.0, 2 * np.pi, size=n)
    phase = rng.uniform(0.0, 1.0, size=n)
    return {"P": P, "K": K, "e": e, "omega": omega, "phase": phase}


# ---------------------------------------------------------------------------
# Real-cadence bootstrap
# ---------------------------------------------------------------------------

_REAL_TIME_GRIDS: list[np.ndarray] | None = None   # module-level cache


def _load_real_time_grids() -> list[np.ndarray]:
    """
    Load sorted observation time arrays from the training split.

    Times are shifted to t_min = 0 for portability.  Cached after first call.
    """
    global _REAL_TIME_GRIDS
    if _REAL_TIME_GRIDS is not None:
        return _REAL_TIME_GRIDS

    if not _SPLITS_CSV.exists() or not _RV_DIR.exists():
        _REAL_TIME_GRIDS = []
        return []

    try:
        import pandas as pd
        from parse_and_label import parse_tbl

        df    = pd.read_csv(_SPLITS_CSV)
        files = df.loc[df["split"] == "train", "file"].tolist()

        grids: list[np.ndarray] = []
        for fname in files:
            path = _RV_DIR / fname
            if not path.exists():
                continue
            try:
                _, t, _, _ = parse_tbl(path)
                t = np.sort(np.asarray(t, dtype=np.float64))
                if len(t) >= 10:
                    grids.append(t - t.min())   # start at 0
            except Exception:
                continue

        _REAL_TIME_GRIDS = grids
        print(f"[synthetic_dataset] loaded {len(grids)} real time grids from training split")
    except Exception as exc:
        print(f"[synthetic_dataset] could not load real grids ({exc}); using heuristic fallback")
        _REAL_TIME_GRIDS = []

    return _REAL_TIME_GRIDS


def _sample_time_grid(rng: np.random.Generator) -> np.ndarray:
    """
    Return a realistic observation time grid.

    Bootstraps directly from the real training corpus.  Falls back to a
    seasonal Poisson-gap model if the corpus is unavailable.
    """
    grids = _load_real_time_grids()
    if grids:
        return grids[int(rng.integers(0, len(grids)))].copy()
    return _sample_time_grid_heuristic(rng)


def _sample_time_grid_heuristic(rng: np.random.Generator) -> np.ndarray:
    """
    Heuristic time grid for testing / offline use (no .tbl files needed).

    Generates a seasonal observing campaign: ~3 seasons/year, each ~90 d long.
    """
    baseline = float(np.exp(rng.uniform(np.log(100.0), np.log(4000.0))))
    n_obs    = int(rng.integers(15, 201))

    season_len = 90.0
    n_seasons  = max(1, round(baseline / 365.25 * 3))
    # Ensure season starts fit within the baseline
    s_max = max(0.0, baseline - season_len)
    season_starts = np.sort(rng.uniform(0.0, s_max + 1e-6, size=n_seasons))

    pts_list: list[np.ndarray] = []
    for s0 in season_starts:
        s1 = min(s0 + season_len, baseline)
        k  = max(1, rng.integers(1, max(2, n_obs // n_seasons + 2)))
        pts_list.append(rng.uniform(s0, s1, size=k))

    t = np.sort(np.concatenate(pts_list)) if pts_list else np.array([0.0])

    if len(t) >= n_obs:
        t = t[np.sort(rng.choice(len(t), size=n_obs, replace=False))]
    else:
        t = np.sort(np.concatenate([t, rng.uniform(0, baseline, size=n_obs - len(t))]))

    return t.astype(np.float64)


def _sample_sigma(rng: np.random.Generator, n_obs: int) -> np.ndarray:
    """Per-observation σ_obs drawn from the empirical log-normal distribution."""
    log_sig = rng.normal(_SIGMA_LOG_MEAN, _SIGMA_LOG_STD, size=n_obs)
    return np.exp(log_sig).astype(np.float64)


# ---------------------------------------------------------------------------
# GP noise injection
# ---------------------------------------------------------------------------

_GP_LIBRARY = None


def _load_gp_library():
    if not _GP_LIB_PATH.exists():
        return None
    try:
        from gp_noise_model import GPNoiseLibrary
        return GPNoiseLibrary.from_json(str(_GP_LIB_PATH))
    except Exception:
        return None


def _inject_noise(t: np.ndarray, sigma: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """
    GP noise from GPNoiseLibrary (Matérn-3/2 or SHO).
    Falls back to white N(0, σ²) if GP sample fails or is NaN.
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
            pass
    return rng.normal(0.0, sigma).astype(np.float64)


# ---------------------------------------------------------------------------
# Single-sample generator
# ---------------------------------------------------------------------------

def generate_one(
    params: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate one synthetic RV sample.

    Returns
    -------
    x     : (4, T_MAX) float32  — [t_norm, rv_norm, sig_norm, mask]
    lsp   : (LSP_N,) float32    — GLS power spectrum (Zechmeister & Kürster 2009)
    theta : (5,) float32        — [log10_P, log10_K, e, cos_ω, sin_ω]
    info  : dict
    """
    from kepler_check import rv_keplerian as rv_np

    P, K, e, omega = params["P"], params["K"], params["e"], params["omega"]

    t       = _sample_time_grid(rng)
    sigma   = _sample_sigma(rng, len(t))
    t_peri  = float(t.min()) + params["phase"] * P

    rv_clean = rv_np(t, P, K, e, omega, t_peri)
    noise    = _inject_noise(t, sigma, rng)
    rv_obs   = rv_clean + noise

    # Compute GLS periodogram on the same fixed grid as RVDataset
    lsp = compute_lsp(t, rv_obs, sigma)   # (LSP_N,)

    # Pack into (4, T_MAX) tensor
    n_real  = len(t)
    t_min   = float(t.min())
    t_span  = float(t.max() - t.min()) if n_real > 1 else 1.0
    rv_med  = float(np.median(rv_obs))
    rv_std  = float(np.std(rv_obs, ddof=1)) if n_real > 1 else 1.0
    rv_std  = max(rv_std, 1e-6)

    t_norm   = (t - t_min) / t_span
    rv_norm  = (rv_obs - rv_med) / rv_std
    sig_norm = sigma / rv_std

    x = np.zeros((4, T_MAX), dtype=np.float32)
    n = min(n_real, T_MAX)
    x[0, :n] = t_norm[:n]
    x[1, :n] = rv_norm[:n]
    x[2, :n] = sig_norm[:n]
    x[3, :n] = 1.0

    theta = np.array([
        np.log10(P),
        np.log10(K),
        e,
        np.cos(omega),
        np.sin(omega),
    ], dtype=np.float32)

    # SNR here is K / σ_GP (measurement noise), NOT K / total noise amplitude.
    snr_meas = K / float(np.median(sigma))
    info = {
        "P": P, "K": K, "e": e, "omega_deg": np.degrees(omega),
        "t_peri": t_peri, "n_obs": n_real,
        "baseline_d": t_span, "snr_meas": snr_meas,
        "t_span_days": t_span, "t_min_days": t_min, "rv_std_ms": rv_std,
        "valid": True,
    }
    return x, lsp, theta, info


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticRVDataset(Dataset):
    """
    Torch Dataset of synthetic RV systems with known orbital parameters.

    Each item is (x, lsp, theta, info), matching the interface of RVDataset.

    Parameters
    ----------
    n_samples : total samples per epoch
    seed      : base RNG seed (item i uses seed + i for reproducibility)
    stats     : normalisation dict from data/dataset_stats.json;
                loaded automatically if not supplied
    """

    def __init__(self, n_samples: int = 100_000, seed: int = 42,
                 stats: dict | None = None) -> None:
        self.n_samples = n_samples
        self.seed      = seed
        self.stats     = stats or _load_stats()

        rng = np.random.default_rng(seed)
        self._params = _sample_orbital_params(rng, n_samples)

        # Eagerly load real time grids so the first __getitem__ doesn't print
        _load_real_time_grids()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        params = {k: float(v[idx]) for k, v in self._params.items()}
        rng    = np.random.default_rng(self.seed + idx)
        x, lsp, theta, info = generate_one(params, rng)

        if self.stats is not None:
            theta = _normalise_theta(theta, self.stats)

        return (torch.from_numpy(x), torch.from_numpy(lsp),
                torch.from_numpy(theta), info)


def _load_stats() -> dict | None:
    if _STATS_PATH.exists():
        return json.loads(_STATS_PATH.read_text())
    return None


def _normalise_theta(theta: np.ndarray, stats: dict) -> np.ndarray:
    out = theta.copy()
    for i, name in enumerate(THETA_NAMES):
        mu  = stats[name]["mean"]
        std = max(stats[name]["std"], 1e-8)
        out[i] = (theta[i] - mu) / std
    return out


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch without a DataLoader.

    Returns
    -------
    X     : (B, 4, T_MAX) float32
    Lsp   : (B, LSP_N)    float32
    Theta : (B, 5) float32
    """
    rng   = rng or np.random.default_rng()
    stats = stats or _load_stats()
    p     = _sample_orbital_params(rng, batch_size)
    X_l, L_l, T_l = [], [], []
    for i in range(batch_size):
        pm = {k: float(v[i]) for k, v in p.items()}
        sub = np.random.default_rng(rng.integers(0, 2**31))
        x, lsp, theta, _ = generate_one(pm, sub)
        if stats is not None:
            theta = _normalise_theta(theta, stats)
        X_l.append(x); L_l.append(lsp); T_l.append(theta)
    return np.stack(X_l), np.stack(L_l), np.stack(T_l)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Generating 200 synthetic samples …")
    t0 = time.perf_counter()
    ds = SyntheticRVDataset(n_samples=200, seed=0)
    dt = time.perf_counter() - t0

    x0, lsp0, th0, info0 = ds[0]
    print(f"  Construction: {dt:.2f} s")
    print(f"  x shape:   {tuple(x0.shape)}")
    print(f"  lsp shape: {tuple(lsp0.shape)}")
    print(f"  theta:     {th0.numpy().round(3)}")
    print(f"  lsp range: [{lsp0.min().item():.3f}, {lsp0.max().item():.3f}]")

    nan_x   = sum(1 for i in range(200) if ds[i][0].isnan().any())
    nan_lsp = sum(1 for i in range(200) if ds[i][1].isnan().any())
    print(f"  NaN x: {nan_x}/200   NaN lsp: {nan_lsp}/200   (want 0/0)")

    X, Lsp, Theta = make_synthetic_batch(batch_size=8)
    print(f"  batch: X {X.shape}  Lsp {Lsp.shape}  Theta {Theta.shape}")
    print("Done.")
