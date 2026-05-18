"""
preprocess.py — Dataset construction for the RV encoder.

Each sample is a real observed RV time series with known orbital parameters
for the dominant planet (highest K amplitude). Series are zero-padded to
T_MAX=256 with a mask channel. Parameters are normalized to a compact
representation suitable for a neural network output head.

Inputs
------
    data/residuals_index.csv      quality-filtered corpus (795 systems)
    data/labels.csv               NASA ps orbital parameters
    data/rv_raw/*.tbl             raw RV files (loaded directly via parse_tbl)

Outputs
-------
    data/splits.csv               host-grouped 70/15/15 manifest
    data/dataset_stats.json       per-parameter normalization constants (train split)

Design notes
------------
Parameter vector theta (5-dim):

    0  log10(P / 1d)          — log-uniform prior; spans ~0.09 to ~8e6 d
    1  log10(K / 1 m/s)       — log-uniform prior; spans ~1 to ~1800 m/s
    2  e                       — eccentricity, ∈ [0, 1)
    3  cos(ω)                  — argument of periastron, circular encoding
    4  sin(ω)                    avoids 2π discontinuity

T_peri is deliberately excluded from theta. It is an observation-epoch-
dependent phase offset (analogous to systemic velocity γ) that is
analytically refittable given (P, K, e, ω). Including T_peri would
require restricting to the 43% of systems with catalog values, and the
catalog values are in BJD which requires careful modular arithmetic.
T_peri will be handled by a fast 1-D refit in the decoder, following
the same approach as kepler_check.py.

e=0 is assumed for systems with missing eccentricity (circular orbit
prior; standard in RV literature). ω is set to 0 for e=0 systems
since it is degenerate for circular orbits.

Input tensor (4 × T_MAX float32):
    row 0  t_norm   = (t - t_min) / t_span   ∈ [0, 1] within each series
    row 1  rv_norm  = (rv - rv_median) / rv_std   median subtraction removes γ
    row 2  sig_norm = sigma / rv_std
    row 3  mask     = 1.0 for real observations, 0.0 for padding

Raw RV from the .tbl files is used (via parse_tbl), NOT the post-Keplerian
residuals in residuals.npz. Residuals have no orbital parameter signal and
are inappropriate as encoder input.

Usage
-----
    from preprocess import make_splits, RVDataset, load_stats

    make_splits()               # run once; writes splits.csv and dataset_stats.json
    ds = RVDataset(split='train')
    x, theta, info = ds.get_numpy(0)
    # x: float32 (4, 256), theta: float32 (5,), info: dict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESID_CSV  = ROOT / 'data' / 'residuals_index.csv'
LABELS_CSV = ROOT / 'data' / 'labels.csv'
RV_DIR     = ROOT / 'data' / 'rv_raw'
SPLITS_CSV = ROOT / 'data' / 'splits.csv'
STATS_JSON = ROOT / 'data' / 'dataset_stats.json'

T_MAX      = 256      # covers 99.0% of corpus at n_obs ≤ 256
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15 (remainder)

THETA_NAMES = ['log10_P', 'log10_K', 'e', 'cos_omega', 'sin_omega']
THETA_DIM   = len(THETA_NAMES)

# Lomb-Scargle periodogram grid — fixed log-spaced periods shared by all systems
# so the encoder always sees the same "period axis" (Scargle 1982, ApJ 263, 835).
# Range 0.5–5000 d covers our full prior; 512 points gives ~1% frequency resolution.
LSP_N       = 512
LSP_PERIODS = np.geomspace(0.5, 5000.0, LSP_N).astype(np.float64)  # days
LSP_FREQS   = (1.0 / LSP_PERIODS)                                    # cycles/day


def compute_lsp(t: np.ndarray, rv: np.ndarray,
                sigma: np.ndarray) -> np.ndarray:
    """
    Compute the generalised Lomb-Scargle power spectrum at the fixed LSP_PERIODS
    grid.  Uses the floating-mean (GLS) formulation of Zechmeister & Kürster
    (2009, A&A 496, 577) as implemented in astropy, which correctly handles
    heteroscedastic errors and a non-zero mean.

    Power is normalised to [0, 1] where 1 corresponds to a perfect sinusoidal
    fit (standard normalisation, Scargle 1982).

    Returns
    -------
    power : (LSP_N,) float32
    """
    from astropy.timeseries import LombScargle
    ls    = LombScargle(t, rv, sigma, normalization='standard',
                        fit_mean=True, center_data=True)
    power = ls.power(LSP_FREQS, assume_regular_frequency=False)
    return np.clip(power, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Host-grouped split                                                            #
# --------------------------------------------------------------------------- #

def make_splits(out_path: Path = SPLITS_CSV, seed: int = 42) -> pd.DataFrame:
    """
    Assign each (host, file) pair to train / val / test.
    All files for a given host go to the same split (no leakage).
    Writes splits.csv and dataset_stats.json; returns the manifest.
    """
    resid  = pd.read_csv(RESID_CSV)
    labels = pd.read_csv(LABELS_CSV)

    usable = _usable_systems(resid, labels)
    if usable.empty:
        raise RuntimeError("No usable systems — check labels.csv and residuals_index.csv")

    # Shuffle unique hosts with a fixed seed
    hosts = np.array(sorted(usable['host'].unique()), dtype=str)
    rng   = np.random.default_rng(seed)
    rng.shuffle(hosts)

    n       = len(hosts)
    n_train = int(np.floor(TRAIN_FRAC * n))
    n_val   = int(np.floor(VAL_FRAC   * n))

    split_map = (
        {h: 'train' for h in hosts[:n_train]}
        | {h: 'val'   for h in hosts[n_train:n_train + n_val]}
        | {h: 'test'  for h in hosts[n_train + n_val:]}
    )
    usable = usable.copy()
    usable['split'] = usable['host'].map(split_map)
    usable.to_csv(out_path, index=False)

    counts = usable['split'].value_counts()
    n_hosts = {s: int((usable[usable['split']==s]['host'].nunique())) for s in ['train','val','test']}
    print(f"Splits saved → {out_path}")
    print(f"  Unique hosts: {n}  "
          f"(train {n_hosts['train']}, val {n_hosts['val']}, test {n_hosts['test']})")
    print(f"  Files: train {counts.get('train',0)}, "
          f"val {counts.get('val',0)}, test {counts.get('test',0)}")

    # Compute and save normalization stats from training split
    stats = compute_stats(usable)
    print(f"Stats saved → {STATS_JSON}")
    return usable


def _usable_systems(resid: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    For each quality-filtered file, identify the dominant planet (highest |K|)
    and extract its parameters. Drops systems with missing P or K.

    Dominant planet = highest K amplitude. For multi-planet systems this is
    the most RV-detectable planet; its signal dominates the time series and
    is the primary prediction target.

    e defaults to 0 (circular orbit) when missing — standard prior in RV
    literature. ω defaults to 0 for e=0 systems (degenerate for circular orbits).
    """
    rows = []
    for _, r in resid.iterrows():
        host = str(r['host']) if pd.notna(r['host']) else None
        if not host or host == 'nan':
            continue
        pl = labels[labels['hostname'] == host]
        if pl.empty:
            continue
        pl_k = pl[pl['pl_rvamp'].notna() & (pl['pl_rvamp'] > 0)]
        if pl_k.empty:
            continue
        dom = pl_k.loc[pl_k['pl_rvamp'].idxmax()]
        if pd.isna(dom['pl_orbper']) or dom['pl_orbper'] <= 0:
            continue
        e       = float(dom['pl_orbeccen']) if pd.notna(dom['pl_orbeccen']) else 0.0
        omega   = float(dom['pl_orblper'])  if pd.notna(dom['pl_orblper'])  else 0.0
        rows.append({
            'file':          str(r['file']),
            'host':          host,
            'n_obs':         int(r['n_obs']),
            'rms_over_sigma': float(r['rms_over_sigma']) if pd.notna(r.get('rms_over_sigma')) else np.nan,
            'P_d':           float(dom['pl_orbper']),
            'K_ms':          float(dom['pl_rvamp']),
            'e':             float(np.clip(e, 0.0, 0.99)),
            'omega_deg':     omega,
            'n_planets':     int(pl['pl_name'].nunique()),
        })
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Parameter normalization                                                       #
# --------------------------------------------------------------------------- #

def compute_stats(splits_df: pd.DataFrame) -> dict:
    """
    Compute per-parameter mean and std on the training split only.
    Normalizing with training stats prevents data leakage.
    """
    train = splits_df[splits_df['split'] == 'train']
    theta = _rows_to_theta(train)

    stats = {}
    for i, name in enumerate(THETA_NAMES):
        col = theta[:, i]
        col = col[np.isfinite(col)]
        stats[name] = {'mean': float(col.mean()), 'std': float(col.std())}

    STATS_JSON.write_text(json.dumps(stats, indent=2))
    return stats


def load_stats() -> dict:
    if not STATS_JSON.exists():
        raise FileNotFoundError(f"{STATS_JSON} not found — run make_splits() first")
    return json.loads(STATS_JSON.read_text())


def _rows_to_theta(df: pd.DataFrame) -> np.ndarray:
    """Convert manifest rows to raw (un-normalized) theta array (N, 5)."""
    omega_rad = np.deg2rad(df['omega_deg'].to_numpy(float))
    theta = np.column_stack([
        np.log10(df['P_d'].to_numpy(float).clip(min=1e-3)),
        np.log10(df['K_ms'].to_numpy(float).clip(min=1e-3)),
        df['e'].to_numpy(float).clip(0.0, 0.99),
        np.cos(omega_rad),
        np.sin(omega_rad),
    ])
    return theta


def normalize_theta(theta: np.ndarray, stats: dict) -> np.ndarray:
    """Standardize theta using training-split statistics (zero-mean, unit-std)."""
    out = np.asarray(theta, dtype=float).copy()
    for i, name in enumerate(THETA_NAMES):
        s = stats[name]
        if s['std'] > 0:
            out[..., i] = (out[..., i] - s['mean']) / s['std']
    return out


def denormalize_theta(theta_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Invert normalize_theta."""
    out = np.asarray(theta_norm, dtype=float).copy()
    for i, name in enumerate(THETA_NAMES):
        s = stats[name]
        out[..., i] = out[..., i] * s['std'] + s['mean']
    return out


# --------------------------------------------------------------------------- #
# Series loading and padding                                                    #
# --------------------------------------------------------------------------- #

def load_raw_rv(fname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (t, rv, sigma) from a .tbl file. Returns raw observed RV — NOT
    post-Keplerian residuals. The encoder requires raw RV to predict orbital
    parameters; residuals contain no orbital parameter information.
    """
    from parse_and_label import parse_tbl
    _, t, rv, err = parse_tbl(RV_DIR / fname)
    return np.asarray(t, float), np.asarray(rv, float), np.asarray(err, float)


def pad_series(t: np.ndarray, rv: np.ndarray, sigma: np.ndarray,
               T: int = T_MAX) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort, normalize, and zero-pad (t, rv, sigma) to length T.

    Normalization:
        t_norm   = (t - t_min) / t_span   ∈ [0, 1]
        rv_norm  = (rv - median(rv)) / std(rv)
        sig_norm = sigma / std(rv)

    Median subtraction removes the systemic velocity γ (instrument-/epoch-
    dependent offset) without requiring a fit. Dividing by std(rv) makes
    the amplitude scale-invariant; sig_norm then represents relative
    measurement precision.

    Returns
    -------
    t_norm, rv_norm, sig_norm : float32 arrays of length T
    mask                      : float32 array, 1.0=real, 0.0=padding
    """
    order = np.argsort(t)
    t, rv, sigma = t[order], rv[order], sigma[order]
    n = min(len(t), T)
    t, rv, sigma = t[:n], rv[:n], sigma[:n]

    t_span = float(t[-1] - t[0]) if n > 1 else 1.0
    t_norm = (t - t[0]) / max(t_span, 1e-6)

    rv_std = float(np.std(rv, ddof=1)) if n > 1 else 1.0
    rv_std = max(rv_std, 1e-6)
    rv_norm  = (rv - float(np.median(rv))) / rv_std
    sig_norm = sigma / rv_std

    def _pad(a: np.ndarray) -> np.ndarray:
        out = np.zeros(T, dtype=np.float32)
        out[:n] = a.astype(np.float32)
        return out

    mask = np.zeros(T, dtype=np.float32)
    mask[:n] = 1.0
    return _pad(t_norm), _pad(rv_norm), _pad(sig_norm), mask


# --------------------------------------------------------------------------- #
# Dataset                                                                       #
# --------------------------------------------------------------------------- #

class RVDataset:
    """
    Dataset of real RV observations with known dominant-planet parameters.

    Loads raw RV from .tbl files (not post-Keplerian residuals).
    Items are (x, theta, info) where:
        x     : float32 (4, T_MAX) — [t_norm, rv_norm, sig_norm, mask]
        theta : float32 (5,)        — normalized parameter vector
        info  : dict               — metadata

    Parameters
    ----------
    split     : 'train' | 'val' | 'test' | 'all'
    normalize : standardize theta using training-split stats (default True)
    """

    def __init__(self,
                 split: Literal['train', 'val', 'test', 'all'] = 'train',
                 splits_path: Path = SPLITS_CSV,
                 normalize: bool = True):
        if not splits_path.exists():
            raise FileNotFoundError(f"{splits_path} — run make_splits() first")
        df = pd.read_csv(splits_path)
        if split != 'all':
            df = df[df['split'] == split].reset_index(drop=True)
        self.df        = df
        self.normalize = normalize
        self.stats     = load_stats() if normalize else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for __getitem__; use get_numpy()")
        x, lsp, theta, info = self.get_numpy(idx)
        return torch.from_numpy(x), torch.from_numpy(lsp), torch.from_numpy(theta), info

    def get_numpy(self, idx: int):
        """Return (x, lsp, theta, info) as numpy float32 arrays."""
        row = self.df.iloc[idx]
        fname = str(row['file'])

        try:
            t, rv, sigma = load_raw_rv(fname)
        except Exception as e:
            x     = np.zeros((4, T_MAX),  dtype=np.float32)
            lsp   = np.zeros(LSP_N,       dtype=np.float32)
            theta = np.zeros(THETA_DIM,   dtype=np.float32)
            return x, lsp, theta, {'host': row['host'], 'file': fname,
                                   'valid': False, 'error': str(e)}

        lsp = compute_lsp(t, rv, sigma)
        t_norm, rv_norm, sig_norm, mask = pad_series(t, rv, sigma, T_MAX)
        x = np.stack([t_norm, rv_norm, sig_norm, mask])   # (4, T_MAX)

        # Recompute scaling metadata needed for the reconstruction loss.
        t_sorted = np.sort(t)[:T_MAX]
        n_use    = len(t_sorted)
        t_span_days = float(t_sorted[-1] - t_sorted[0]) if n_use > 1 else 1.0
        t_min_days  = float(t_sorted[0])
        rv_std_ms   = float(np.std(rv[:T_MAX], ddof=1)) if n_use > 1 else 1.0
        rv_std_ms   = max(rv_std_ms, 1e-6)

        omega_rad = float(np.deg2rad(float(row['omega_deg'])))
        theta_raw = np.array([
            np.log10(max(float(row['P_d']),  1e-3)),
            np.log10(max(float(row['K_ms']), 1e-3)),
            float(np.clip(row['e'], 0.0, 0.99)),
            np.cos(omega_rad),
            np.sin(omega_rad),
        ], dtype=np.float32)

        if self.normalize and self.stats is not None:
            theta_raw = normalize_theta(theta_raw[None], self.stats)[0].astype(np.float32)

        info = {
            'host':         row['host'],
            'file':         fname,
            'n_obs':        int(row['n_obs']),
            'n_planets':    int(row['n_planets']),
            'valid':        True,
            't_span_days':  t_span_days,
            't_min_days':   t_min_days,
            'rv_std_ms':    rv_std_ms,
        }
        return x, lsp, theta_raw, info


# --------------------------------------------------------------------------- #
# Collate for DataLoader                                                        #
# --------------------------------------------------------------------------- #

def collate_fn(batch):
    """Drop invalid items; stack valid ones into batched tensors."""
    import torch
    xs, thetas, infos = zip(*batch)
    valid = [i for i, info in enumerate(infos) if info.get('valid', True)]
    if not valid:
        return None, None, []
    return (torch.stack([xs[i] for i in valid]),
            torch.stack([thetas[i] for i in valid]),
            [infos[i] for i in valid])


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Building host-grouped splits and normalization stats...")
    splits = make_splits(SPLITS_CSV)

    print("\nParameter stats (training split):")
    stats = load_stats()
    for name, s in stats.items():
        print(f"  {name:15s}  mean={s['mean']:+.3f}  std={s['std']:.3f}")

    print(f"\nSplit summary ({len(splits)} total usable systems):")
    print(splits['split'].value_counts().to_string())

    # Sanity check: load 3 items and verify shapes and RV amplitude
    print("\nSanity check (first 3 items, raw RV):")
    ds = RVDataset(split='train', normalize=False)
    for i in range(min(3, len(ds))):
        x, theta, info = ds.get_numpy(i)
        rv_obs = x[1][x[3] == 1]   # unpadded rv_norm
        print(f"  [{i}] {info['host']:20s}  n={info['n_obs']:3d}  "
              f"rv_norm std={rv_obs.std():.3f}  "
              f"log10P={theta[0]:.2f}  log10K={theta[1]:.2f}  "
              f"e={theta[2]:.2f}  valid={info['valid']}")
