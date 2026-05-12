"""
synthetic_rv.py
---------------
Generate synthetic RV training data for pretraining the encoder/decoder model,
following Nicolò's suggestion: simulate noiseless Keplerian trajectories using
the same integrator used for validation, then perturb them with a non-parametric
noise model fit on real RV residuals (rather than naive Gaussian noise, which
would create a distribution shift between pretraining and fine-tuning).

Pipeline
--------
  1. Build the noise pool: run the Kepler validator on every file in
     data/rv_raw, keep residuals (data - model) for systems where the model
     is good (RMS/σ < 3). These residuals represent real RV "noise" — a
     mixture of photon shot noise, stellar jitter, and instrumental
     systematics, with whatever non-Gaussian tails reality has.
  2. Sample synthetic Kepler parameters (P, K, e, ω, t_peri, M_*) from
     the empirical distribution of confirmed planets in the catalog, so
     the synthetic systems are statistically representative of the targets.
  3. Sample observation times from real .tbl files, so the synthetic
     cadences inherit real survey patterns (seasonal gaps, dense follow-up
     windows, baseline lengths) rather than using regular grids.
  4. Compute noiseless Kepler RV signals via the integrator.
  5. Add bootstrap-sampled noise from the residual pool.

Output
------
  data/synthetic/synth_NNNNN.npz          # per-system (time, rv, sigma)
  data/synthetic/manifest.csv             # one row per system with labels
  data/noise_pool.npz                     # cached residual pool

Usage
-----
  python synthetic_rv.py --build-noise               # pool only (~5 s)
  python synthetic_rv.py --n 1000 --out data/synthetic   # full dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kepler_check import rv_keplerian, semi_amplitude, validate_one
from parse_and_label import parse_tbl


# ---------------------------------------------------------------------------
# (1) Noise pool from real residuals of validated systems
# ---------------------------------------------------------------------------
def build_noise_pool(rv_dir: Path, labels_path: Path, simbad_cache_path: Path,
                     out_path: Path, rms_over_sigma_max: float = 3.0,
                     verbose: bool = True) -> dict:
    """Collect residuals from every validated file with RMS/σ < threshold.

    Returns a dict with arrays of:
      - residuals : 1D, all residual values pooled across systems (m/s)
      - sigmas    : matching σ_obs values, in case we want to standardize
      - file_ids  : index linking each residual back to its source file
                    (useful for chunk-bootstrap that preserves correlation)
    """
    labels = pd.read_csv(labels_path)
    simbad_cache: dict[str, list[str]] = {}
    if simbad_cache_path.exists():
        simbad_cache = json.loads(simbad_cache_path.read_text())

    files = sorted(rv_dir.glob("UID_*_RVC_*.tbl"))
    if verbose:
        print(f"[noise] sweeping {len(files)} files for residuals "
              f"with RMS/σ < {rms_over_sigma_max}...")

    res_all, sig_all, fid_all = [], [], []
    n_kept = 0
    for i, f in enumerate(files, 1):
        r = validate_one(f, labels, mode="anchor", plot=False, verbose=False,
                         simbad_cache=simbad_cache, return_residuals=True)
        if r["status"] != "ok":
            continue
        if not (0.1 <= r["median_sigma_ms"] <= 100.0):
            continue
        if r["rms_over_sigma"] >= rms_over_sigma_max:
            continue
        res_all.append(r["residuals"])
        sig_all.append(r["sigmas"])
        fid_all.append(np.full(len(r["residuals"]), n_kept, dtype=np.int32))
        n_kept += 1
        if verbose and i % 200 == 0:
            print(f"  [noise] processed {i}/{len(files)}  (kept {n_kept})")

    pool = {
        "residuals": np.concatenate(res_all),
        "sigmas":    np.concatenate(sig_all),
        "file_ids":  np.concatenate(fid_all),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **pool)
    if verbose:
        print(f"[noise] pool size {len(pool['residuals']):,} samples from "
              f"{n_kept} systems  →  {out_path}")
        print(f"  median |residual| = {np.median(np.abs(pool['residuals'])):.2f} m/s")
        print(f"  median σ_obs      = {np.median(pool['sigmas']):.2f} m/s")
        print(f"  excess kurtosis   = "
              f"{_excess_kurtosis(pool['residuals']):.2f}   (0 = Gaussian)")
    return pool


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x)
    mu = x.mean()
    s2 = x.var()
    return float(((x - mu) ** 4).mean() / (s2 * s2) - 3.0) if s2 > 0 else 0.0


# ---------------------------------------------------------------------------
# (2) Bootstrap noise model
# ---------------------------------------------------------------------------
class BootstrapNoiseModel:
    """Sample noise by bootstrap-resampling real RV residuals.

    Two modes:
      - 'iid'   : independent samples (loses temporal correlation)
      - 'chunk' : contiguous chunks from one file at a time (preserves
                  short-timescale correlation, which matters for stellar
                  activity that lives on ~rotation timescales)
    """

    def __init__(self, pool: dict, mode: str = "chunk", chunk_min: int = 5,
                 chunk_max: int = 30):
        self.residuals = np.asarray(pool["residuals"])
        self.file_ids = np.asarray(pool["file_ids"])
        self.mode = mode
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max
        # Pre-index by file for efficient chunk sampling
        self._by_file = {fid: np.where(self.file_ids == fid)[0]
                          for fid in np.unique(self.file_ids)}
        self._files = list(self._by_file)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.mode == "iid":
            return rng.choice(self.residuals, size=n, replace=True)
        # chunk mode
        out = np.empty(n, dtype=self.residuals.dtype)
        filled = 0
        while filled < n:
            fid = self._files[rng.integers(len(self._files))]
            idx = self._by_file[fid]
            if len(idx) == 0:
                continue
            chunk_len = min(rng.integers(self.chunk_min, self.chunk_max + 1),
                            len(idx), n - filled)
            start = rng.integers(0, len(idx) - chunk_len + 1)
            out[filled: filled + chunk_len] = self.residuals[idx[start: start + chunk_len]]
            filled += chunk_len
        return out


# ---------------------------------------------------------------------------
# (3) Empirical Kepler-parameter sampler
# ---------------------------------------------------------------------------
def sample_params(labels: pd.DataFrame, n: int,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Draw n synthetic (P, K, e, ω, t_peri, M*, M sin i) tuples from the
    empirical joint distribution of confirmed exoplanets in the catalog.

    Sampling from real (P, K, e) tuples preserves correlations the catalog
    encodes (e.g. hot Jupiters are mostly circular, cold giants span more e).
    """
    rng = rng or np.random.default_rng()
    df = labels.dropna(subset=["pl_orbper"]).copy()
    # Fill K when missing using the (P, M sin i, M*) formula
    need_K = df["pl_rvamp"].isna() & df["pl_msinij"].notna() & df["st_mass"].notna()
    df.loc[need_K, "pl_rvamp"] = df.loc[need_K].apply(
        lambda r: semi_amplitude(r["pl_msinij"], r["pl_orbper"],
                                  r["pl_orbeccen"] if pd.notna(r["pl_orbeccen"]) else 0.0,
                                  r["st_mass"]),
        axis=1,
    )
    df = df.dropna(subset=["pl_rvamp"]).reset_index(drop=True)

    picks = df.sample(n=n, replace=True, random_state=rng.integers(2 ** 31)) \
              .reset_index(drop=True)
    out = pd.DataFrame({
        "P":         picks["pl_orbper"].astype(float),
        "K":         picks["pl_rvamp"].astype(float),
        "e":         picks["pl_orbeccen"].fillna(0.0).astype(float),
        "omega_deg": picks["pl_orblper"].fillna(90.0).astype(float),
        "msini":     picks["pl_msinij"].astype(float),
        "mstar":     picks["st_mass"].fillna(1.0).astype(float),
    })
    # Random periastron phase, since the absolute epoch is arbitrary for
    # a synthetic system.
    out["t_peri"] = rng.uniform(0.0, out["P"].values, size=n)
    return out


# ---------------------------------------------------------------------------
# (4) Time-grid sampler — inherit real survey cadences
# ---------------------------------------------------------------------------
def sample_times(rv_dir: Path, rng: np.random.Generator | None = None,
                 min_obs: int = 15, max_obs: int = 200,
                 _cache: dict | None = None) -> np.ndarray:
    """Pick a real .tbl file and return its observation times (rebased to 0).
    Caches parsed files in memory so repeated calls are fast.
    """
    rng = rng or np.random.default_rng()
    if _cache is None:
        _cache = sample_times.__dict__.setdefault("_files", {})
        if "_list" not in sample_times.__dict__:
            sample_times.__dict__["_list"] = sorted(rv_dir.glob("UID_*_RVC_*.tbl"))
    files = sample_times.__dict__["_list"]
    if not files:
        raise RuntimeError(f"No .tbl files in {rv_dir}")

    for _ in range(50):
        f = files[rng.integers(len(files))]
        if f.name not in _cache:
            try:
                _, t, _, _ = parse_tbl(f)
                _cache[f.name] = np.sort(t - t[0]) if len(t) else None
            except Exception:  # noqa: BLE001
                _cache[f.name] = None
        t = _cache[f.name]
        if t is None or len(t) < min_obs:
            continue
        if len(t) > max_obs:
            start = rng.integers(0, len(t) - max_obs + 1)
            t = t[start: start + max_obs]
        return t.copy()
    # Fallback: regular grid (rare)
    return np.linspace(0, 1000, min_obs)


# ---------------------------------------------------------------------------
# (5) End-to-end synthetic dataset
# ---------------------------------------------------------------------------
def make_dataset(n_systems: int, rv_dir: Path, labels_path: Path,
                 noise_pool_path: Path, out_dir: Path,
                 noise_mode: str = "chunk", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    labels = pd.read_csv(labels_path)
    pool = dict(np.load(noise_pool_path))
    noise = BootstrapNoiseModel(pool, mode=noise_mode)

    params = sample_params(labels, n_systems, rng=rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, p in params.iterrows():
        times = sample_times(rv_dir, rng=rng)
        omega = np.radians(p["omega_deg"])
        rv_clean = rv_keplerian(times, p["P"], p["K"], p["e"], omega, p["t_peri"])
        rv = rv_clean + noise.sample(len(times), rng=rng)
        # σ for synthetic: draw from the real σ pool to match the noise scale
        sigma = rng.choice(pool["sigmas"], size=len(times), replace=True)
        np.savez(out_dir / f"synth_{i:05d}.npz",
                 time=times, rv=rv, sigma=sigma,
                 P=p["P"], K=p["K"], e=p["e"], omega_deg=p["omega_deg"],
                 t_peri=p["t_peri"], msini=p["msini"], mstar=p["mstar"])
        records.append({"file": f"synth_{i:05d}.npz",
                        "n_obs": len(times),
                        "t_baseline_days": float(times[-1] - times[0]),
                        **p.to_dict()})
        if (i + 1) % 200 == 0:
            print(f"  [synth] {i + 1}/{n_systems}")
    pd.DataFrame(records).to_csv(out_dir / "manifest.csv", index=False)
    print(f"[synth] wrote {n_systems} synthetic systems to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--simbad-cache", type=Path,
                   default=Path("data/simbad_cache.json"))
    p.add_argument("--noise-pool", type=Path, default=Path("data/noise_pool.npz"))
    p.add_argument("--build-noise", action="store_true",
                   help="(Re)build the noise pool from real residuals")
    p.add_argument("--rms-cutoff", type=float, default=3.0,
                   help="Only include systems with RMS/σ < this in noise pool")
    p.add_argument("--n", type=int, default=0,
                   help="Number of synthetic systems to generate (skip if 0)")
    p.add_argument("--noise-mode", choices=("chunk", "iid"), default="chunk")
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.build_noise or (args.n and not args.noise_pool.exists()):
        build_noise_pool(args.rv_dir, args.labels, args.simbad_cache,
                         args.noise_pool, rms_over_sigma_max=args.rms_cutoff)

    if args.n:
        make_dataset(args.n, args.rv_dir, args.labels, args.noise_pool,
                     args.out, noise_mode=args.noise_mode, seed=args.seed)


if __name__ == "__main__":
    main()
