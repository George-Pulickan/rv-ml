"""
synthetic_dataset.py — Synthetic RV generator for encoder pre-training.

Samples (P, K, e, ω) from physically motivated priors, generates a Keplerian
RV curve, injects GP noise, and returns the same (4×256, LSP_N) tensor format
as RVDataset in preprocess.py.

Priors (applied to every planet, primary and companion alike)
-----
    P      ~ LogUniform(1, 3000) d
    K      ~ LogUniform(1, 300) m/s
    e      ~ Beta(0.867, 3.03)   eccentricity prior (Kipping 2013, MNRAS 434, L51)
    ω      ~ Uniform(0, 2π)
    T_peri ~ Uniform(t_min, t_min + P)   (random orbital phase)
    γ = 0  (median-subtracted in normalised tensor)

Multi-planet injection
----------------------
With probability f_multi=0.30, one or two companion planets are drawn from
the same priors and their Keplerian signals are added to the primary signal
before noise injection.  The label always corresponds to the dominant planet
— the one with the highest RV semi-amplitude K — consistent with the
definition in preprocess._usable_systems.  This trains the encoder to be
robust to companion contamination without requiring explicit signal separation.

Multiplicity distribution (following Howard et al. 2010, Science 330, 653,
which found ~30% of planet-hosting stars have multiple detected companions):
    P(0 companions) = 1 - f_multi           ≈ 0.70
    P(1 companion)  = f_multi × 0.75        ≈ 0.225
    P(2 companions) = f_multi × 0.25        ≈ 0.075
→ E[N_planets] ≈ 1.4, consistent with the observed RV multiplicity function.

The single noise realisation (GP or white) is shared across all planets.
Orbital stability (Hill stability) is not enforced; period near-degeneracy
has measure zero under continuous priors and is an acknowledged limitation.

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
    from synthetic_dataset import SyntheticRVDataset, generate_cache

    ds = SyntheticRVDataset(n_samples=50_000, seed=42, f_multi=0.30)
    x, lsp, theta, info = ds[0]     # x:(4,256) lsp:(LSP_N,) theta:(5,)

    # Pre-generate cache for fast training
    generate_cache(500_000, "data/pretrain_cache.pt", seed=42, f_multi=0.30)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import LSP_N, T_MAX, THETA_NAMES, compute_lsp

_GP_LIB_PATH = Path("data/gp_fits.json")
_STATS_PATH  = Path("data/dataset_stats.json")
_SPLITS_CSV  = Path("data/splits.csv")
_RV_DIR      = Path("data/rv_raw")

# σ distribution fit to 146 real systems: ln(σ) ~ N(μ, s)
# Calibrated: median σ ≈ 4.6 m/s, p5 ≈ 0.9 m/s, p95 ≈ 16 m/s
_SIGMA_LOG_MEAN = np.log(4.62)
_SIGMA_LOG_STD  = 0.75


# ---------------------------------------------------------------------------
# Prior sampler (identical for primary and companion planets)
# ---------------------------------------------------------------------------

def _sample_orbital_params(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """
    Sample n independent sets of (P, K, e, ω, phase) from the prior.

    P and K use log-uniform priors, appropriate for parameters whose
    uncertainty spans orders of magnitude (Jeffreys prior).  e follows
    Beta(0.867, 3.03) from Kipping (2013, MNRAS 434, L51) Table 1 (MLE fit to
    the observed RV exoplanet eccentricity distribution). This is J-shaped —
    peaked at e=0 — matching the concentration of near-circular orbits in the
    NASA confirmed-planet catalog. Note: Beta(2, 5) is sometimes used as a
    simpler weakly-informative prior but is peaked at e≈0.2, which does NOT
    match the catalog and inflates the eccentricity mismatch seen in diagnostics.
    ω and phase are uniform — no preferred orientation or epoch.

    The same function is called for both primary and companion planets,
    ensuring all planets are drawn from the same marginal prior.
    """
    P     = np.exp(rng.uniform(np.log(1.0),    np.log(3000.0), size=n))
    K     = np.exp(rng.uniform(np.log(1.0),    np.log(300.0),  size=n))
    e     = rng.beta(0.867, 3.03, size=n).clip(0.0, 0.99)
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
    Only training-split files are used — val/test grids are excluded to
    prevent cadence leakage.
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
                    grids.append(t - t.min())   # shift to t_min = 0
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
    Heuristic time grid: seasonal observing campaign (no real .tbl files needed).

    Generates ~3 observing seasons per year, each ~90 d long, with uniform
    random observation times within each season.
    """
    baseline  = float(np.exp(rng.uniform(np.log(100.0), np.log(4000.0))))
    n_obs     = int(rng.integers(15, 201))
    season_len = 90.0
    n_seasons  = max(1, round(baseline / 365.25 * 3))
    s_max      = max(0.0, baseline - season_len)
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


_SIGMA_OBS_JITTER_LOG_STD = 0.10   # ~10% per-observation variation within a system


def _sample_sigma(rng: np.random.Generator, n_obs: int) -> np.ndarray:
    """Per-observation σ_obs drawn from a two-level hierarchical model.

    Each *system* has an intrinsic precision floor σ_system (set by the
    instrument and host star), drawn once per call from the population
    log-normal LogN(_SIGMA_LOG_MEAN, _SIGMA_LOG_STD). Per-observation σ then
    fluctuates around σ_system with a small log-jitter representing
    night-to-night noise variation.

    This matches the real data: per-system median σ spans a wide range
    (HARPS ≈ 0.5 m/s vs HIRES ≈ 3 m/s vs older surveys ≈ 10 m/s), while
    within a single system the σ values are tightly clustered.

    Drawing σ independently per observation (the previous bug) collapsed
    the per-system median to the population mean, leaving the synthetic
    σ distribution far too narrow.
    """
    log_sys  = rng.normal(_SIGMA_LOG_MEAN, _SIGMA_LOG_STD)
    log_obs  = log_sys + rng.normal(0.0, _SIGMA_OBS_JITTER_LOG_STD, size=n_obs)
    return np.exp(log_obs).astype(np.float64)


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
    GP noise from GPNoiseLibrary (Matérn-3/2 or SHO kernel).
    Falls back to i.i.d. N(0, σ²) white noise if GP sample fails or is NaN.

    A single noise draw is shared across all planets in a multi-planet system
    — the noise process is a property of the instrument and stellar activity,
    independent of the planetary configuration.
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
# Single-sample generator with companion injection
# ---------------------------------------------------------------------------

def generate_one(
    params: dict[str, float],
    rng: np.random.Generator,
    f_multi: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate one synthetic RV sample with optional companion injection.

    With probability f_multi, 1 or 2 companion planets are added to the
    primary Keplerian signal.  All planets are drawn from the same (P, K, e, ω)
    prior via _sample_orbital_params.  The label theta corresponds to the
    dominant planet — the one with the highest K — consistent with the
    definition in preprocess._usable_systems.

    The RV curve is the linear superposition of all Keplerian signals plus
    a single shared noise realisation.  Normalisation (rv_med, rv_std) is
    computed on the combined observed signal so the encoder receives the same
    representation it would see for a real multi-planet system.

    Companion count given has_companions:
        P(1 companion | has_companions) = 3/4   (dominant case in RV surveys)
        P(2 companions | has_companions) = 1/4

    Parameters
    ----------
    params   : primary planet parameters (P, K, e, omega, phase)
    rng      : per-sample RNG (seeded with seed + idx for reproducibility)
    f_multi  : probability of injecting companions (default 0.30)

    Returns
    -------
    x     : (4, T_MAX) float32  — [t_norm, rv_norm, sig_norm, mask]
    lsp   : (LSP_N,) float32    — GLS power spectrum (Zechmeister & Kürster 2009)
    theta : (5,) float32        — dominant planet [log10_P, log10_K, e, cos_ω, sin_ω]
    info  : dict
    """
    from kepler_check import rv_keplerian as rv_np

    # ---- Build planet list: primary + optional companions ----
    all_planets: list[dict[str, float]] = [params]
    if rng.random() < f_multi:
        # 1 companion with prob 3/4, 2 companions with prob 1/4
        # (Mayor et al. 2011 found ~75% of multi-planet RV systems have exactly 2 planets)
        n_comp = 1 if rng.random() < 0.75 else 2
        comp   = _sample_orbital_params(rng, n_comp)
        for i in range(n_comp):
            all_planets.append({k: float(v[i]) for k, v in comp.items()})

    # Dominant planet = highest K (consistent with preprocess._usable_systems)
    dom_idx = int(np.argmax([pl["K"] for pl in all_planets]))
    dom     = all_planets[dom_idx]

    # ---- Shared time grid and per-observation errors ----
    t     = _sample_time_grid(rng)
    sigma = _sample_sigma(rng, len(t))

    # ---- Sum Keplerian signals (linear superposition, test-particle limit) ----
    rv_clean = np.zeros(len(t), dtype=np.float64)
    t_peri_list: list[float] = []
    for pl in all_planets:
        # t_peri is set to within one period of the first observation so that
        # the orbital phase is uniformly distributed at t[0], independent of P.
        tp = float(t.min()) + pl["phase"] * pl["P"]
        t_peri_list.append(tp)
        rv_clean += rv_np(t, pl["P"], pl["K"], pl["e"], pl["omega"], tp)
    t_peri_dom = t_peri_list[dom_idx]

    # ---- Single noise draw, shared across all planet signals ----
    noise  = _inject_noise(t, sigma, rng)
    rv_obs = rv_clean + noise

    # ---- GLS periodogram on the combined multi-planet + noise signal ----
    # The LSP will show peaks from all planets; the encoder must learn to
    # focus on the dominant one during training.
    lsp = compute_lsp(t, rv_obs, sigma)

    # ---- Normalise on the combined signal and pack into (4, T_MAX) tensor ----
    # rv_med and rv_std are computed on the combined signal, matching what the
    # encoder will receive for real multi-planet systems.
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

    # ---- Label: dominant planet's parameters ----
    theta = np.array([
        np.log10(dom["P"]),
        np.log10(dom["K"]),
        dom["e"],
        np.cos(dom["omega"]),
        np.sin(dom["omega"]),
    ], dtype=np.float32)

    n_companions = len(all_planets) - 1
    info = {
        "P": dom["P"], "K": dom["K"], "e": dom["e"],
        "omega_deg":   np.degrees(dom["omega"]),
        "t_peri":      t_peri_dom,
        "rv_med_ms":   rv_med,
        "n_obs":       n_real,
        "baseline_d":  t_span,
        "snr_meas":    dom["K"] / float(np.median(sigma)),
        "t_span_days": t_span,
        "t_min_days":  t_min,
        "rv_std_ms":   rv_std,
        "n_planets":   len(all_planets),
        "n_companions": n_companions,
        "has_ecc":     True,   # synthetic e always sampled from prior, never imputed
        "valid":       True,
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
    stats     : normalisation dict from data/dataset_stats.json
    f_multi   : fraction of samples with companion planets (default 0.30)
    """

    def __init__(self, n_samples: int = 100_000, seed: int = 42,
                 stats: dict | None = None, f_multi: float = 0.30) -> None:
        self.n_samples = n_samples
        self.seed      = seed
        self.f_multi   = f_multi
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
        x, lsp, theta, info = generate_one(params, rng, f_multi=self.f_multi)

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
# Pre-generated cache
# ---------------------------------------------------------------------------

class PregenSyntheticDataset(Dataset):
    """
    Torch Dataset backed by a pre-generated .pt cache file.

    Identical interface to SyntheticRVDataset but reads from disk instead of
    generating on-the-fly.  After initial load (~1 s for 500K samples), each
    __getitem__ is a tensor slice with no CPU overhead during training.

    The cache stores n_companions per sample so the multi-planet mix can be
    verified at load time.  Old caches without n_companions are accepted
    (treated as all-single-planet).
    """

    def __init__(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._x            = data["x"]          # (N, 4, T_MAX)
        self._lsp          = data["lsp"]         # (N, LSP_N)
        self._theta        = data["theta"]       # (N, 5) — already normalised
        self._t_span       = data["t_span"]      # (N,)
        self._t_min        = data["t_min"]       # (N,)
        self._rv_std       = data["rv_std"]      # (N,)
        self._n_companions = data.get("n_companions", None)  # (N,) int8 or None

        n = len(self._x)
        f_multi_stored = float(data.get("f_multi", 0.0))
        if self._n_companions is not None:
            frac = float((self._n_companions > 0).float().mean())
            print(f"[PregenSyntheticDataset] {n:,} samples  "
                  f"multi-planet fraction: {frac:.3f} "
                  f"(cache f_multi={f_multi_stored:.2f})")
        else:
            print(f"[PregenSyntheticDataset] {n:,} samples  "
                  f"(legacy cache — companion metadata absent)")

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        n_comp = (int(self._n_companions[idx])
                  if self._n_companions is not None else 0)
        info = {
            "t_span_days":  float(self._t_span[idx]),
            "t_min_days":   float(self._t_min[idx]),
            "rv_std_ms":    float(self._rv_std[idx]),
            "n_companions": n_comp,
            "n_planets":    1 + n_comp,
            "has_ecc":      True,
            "valid":        True,
        }
        return self._x[idx], self._lsp[idx], self._theta[idx], info


def generate_cache(
    n_samples: int,
    path: str | Path,
    seed: int = 42,
    stats: dict | None = None,
    f_multi: float = 0.30,
) -> None:
    """
    Generate n_samples synthetic RV samples and save to a .pt cache file.

    The cache is deterministic: sample i is generated with
    np.random.default_rng(seed + i), so any subset of samples can be
    regenerated identically.  f_multi and n_companions are stored in the
    cache for verification at load time.

    Parameters
    ----------
    n_samples : number of samples to generate
    path      : output .pt file path
    seed      : master RNG seed
    stats     : normalisation dict; loaded from disk if None
    f_multi   : fraction of samples with companion injection (default 0.30)
    """
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = stats or _load_stats()

    rng    = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n_samples)
    _load_real_time_grids()   # warm module cache before the loop

    X_list, L_list, T_list    = [], [], []
    ts_list, tm_list, rs_list = [], [], []
    nc_list: list[int]        = []

    t0 = time.perf_counter()
    for i in range(n_samples):
        p = {k: float(v[i]) for k, v in params.items()}
        r = np.random.default_rng(seed + i)
        x, lsp, theta, info = generate_one(p, r, f_multi=f_multi)
        if stats is not None:
            theta = _normalise_theta(theta, stats)
        X_list.append(x)
        L_list.append(lsp)
        T_list.append(theta)
        ts_list.append(info["t_span_days"])
        tm_list.append(info["t_min_days"])
        rs_list.append(info["rv_std_ms"])
        nc_list.append(info["n_companions"])
        if (i + 1) % 5_000 == 0 or i == n_samples - 1:
            elapsed  = time.perf_counter() - t0
            rate     = (i + 1) / elapsed
            n_multi  = sum(1 for c in nc_list if c > 0)
            frac     = n_multi / len(nc_list)
            print(f"  {i+1:>7,}/{n_samples:,}  {elapsed:5.0f}s  "
                  f"({rate:.0f} samp/s)  multi-planet: {frac:.2%}")

    data = {
        "x":           torch.from_numpy(np.stack(X_list)),
        "lsp":         torch.from_numpy(np.stack(L_list)),
        "theta":       torch.from_numpy(np.stack(T_list)),
        "t_span":      torch.tensor(ts_list,  dtype=torch.float32),
        "t_min":       torch.tensor(tm_list,  dtype=torch.float32),
        "rv_std":      torch.tensor(rs_list,  dtype=torch.float32),
        "n_companions": torch.tensor(nc_list, dtype=torch.int8),
        "n_samples":   n_samples,
        "seed":        seed,
        "f_multi":     f_multi,
    }
    torch.save(data, path)

    elapsed = time.perf_counter() - t0
    size_mb = path.stat().st_size / 1e6
    n_multi = sum(1 for c in nc_list if c > 0)
    print(f"\nSaved {n_samples:,} samples → {path}  ({size_mb:.0f} MB, {elapsed:.0f}s)")
    print(f"Multi-planet fraction: {n_multi/n_samples:.3f} (target {f_multi:.2f})")


# ---------------------------------------------------------------------------
# Batch helper (for use without a DataLoader)
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    stats: dict | None = None,
    f_multi: float = 0.30,
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
        pm  = {k: float(v[i]) for k, v in p.items()}
        sub = np.random.default_rng(rng.integers(0, 2**31))
        x, lsp, theta, _ = generate_one(pm, sub, f_multi=f_multi)
        if stats is not None:
            theta = _normalise_theta(theta, stats)
        X_l.append(x); L_l.append(lsp); T_l.append(theta)
    return np.stack(X_l), np.stack(L_l), np.stack(T_l)


# ---------------------------------------------------------------------------
# CLI smoke-test / cache generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate-cache", metavar="PATH",
                    help="Generate a .pt cache file at PATH and exit")
    ap.add_argument("--n-samples", type=int, default=500_000,
                    help="Number of samples to generate (default: 500,000)")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--f-multi",   type=float, default=0.30,
                    help="Companion injection probability (default: 0.30)")
    args = ap.parse_args()

    if args.generate_cache:
        print(f"Generating {args.n_samples:,} synthetic samples → {args.generate_cache}")
        print(f"  f_multi = {args.f_multi}  (companion injection probability)")
        print(f"  seed    = {args.seed}")
        generate_cache(args.n_samples, args.generate_cache,
                       seed=args.seed, f_multi=args.f_multi)
    else:
        print("Smoke-testing SyntheticRVDataset (200 samples, f_multi=0.30) …")
        t0 = time.perf_counter()
        ds = SyntheticRVDataset(n_samples=200, seed=0, f_multi=0.30)
        dt = time.perf_counter() - t0

        x0, lsp0, th0, info0 = ds[0]
        print(f"  Construction: {dt:.2f} s")
        print(f"  x shape:       {tuple(x0.shape)}")
        print(f"  lsp shape:     {tuple(lsp0.shape)}")
        print(f"  theta:         {th0.numpy().round(3)}")
        print(f"  n_planets:     {info0['n_planets']}  "
              f"n_companions: {info0['n_companions']}")
        print(f"  lsp range:     [{lsp0.min().item():.3f}, {lsp0.max().item():.3f}]")

        nan_x   = sum(1 for i in range(200) if ds[i][0].isnan().any())
        nan_lsp = sum(1 for i in range(200) if ds[i][1].isnan().any())
        n_multi = sum(1 for i in range(200) if ds[i][3]["n_companions"] > 0)
        print(f"  NaN x: {nan_x}/200   NaN lsp: {nan_lsp}/200   (want 0/0)")
        print(f"  Multi-planet: {n_multi}/200  ({n_multi/2:.1f}%, target ~30%)")

        X, Lsp, Theta = make_synthetic_batch(batch_size=8, f_multi=0.30)
        print(f"  batch: X {X.shape}  Lsp {Lsp.shape}  Theta {Theta.shape}")
        print("Done.")
