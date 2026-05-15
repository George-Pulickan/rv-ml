"""
preprocess.py — Dataset construction for the RV encoder.

Each sample is a real observed RV time series with known orbital parameters
(dominant planet, highest K). Series are zero-padded to T_MAX=256 with a
mask channel. Parameters are normalized to a compact representation suitable
for a neural network output head.

Inputs
------
    data/residuals_index.csv      quality-filtered corpus (795 systems)
    data/labels.csv               NASA ps orbital parameters
    data/rv_raw/*.tbl             raw RV files
    data/gp_fits.json             GP noise library (for synthetic generation)

Outputs
-------
    data/splits.csv               host-grouped 70/15/15 manifest
    data/dataset_stats.json       per-parameter normalization constants

Parameter vector (6-dim, one dominant planet per system)
---------------------------------------------------------
    0  log10(P / 1d)
    1  log10(K / 1 m/s)
    2  e
    3  cos(ω)
    4  sin(ω)
    5  (T_peri mod P) / P        phase in [0, 1]

Input tensor shape: (4, T_MAX)
    row 0  t_norm   = (t - t_min) / t_span     ∈ [0, 1]
    row 1  rv_norm  = (rv - rv_median) / rv_std
    row 2  sig_norm = sigma / rv_std
    row 3  mask     = 1.0 for real obs, 0.0 for padding

Usage
-----
    from preprocess import make_splits, RVDataset, load_stats

    make_splits('data/splits.csv')          # run once

    ds = RVDataset(split='train')
    x, theta, info = ds[0]
    # x: FloatTensor (4, 256)
    # theta: FloatTensor (6,)
    # info: dict with host, file, n_obs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESID_CSV  = ROOT / 'data' / 'residuals_index.csv'
LABELS_CSV = ROOT / 'data' / 'labels.csv'
RV_DIR     = ROOT / 'data' / 'rv_raw'
SPLITS_CSV = ROOT / 'data' / 'splits.csv'
STATS_JSON = ROOT / 'data' / 'dataset_stats.json'

T_MAX      = 256
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

THETA_NAMES = ['log10_P', 'log10_K', 'e', 'cos_omega', 'sin_omega', 'phase_tperi']


# --------------------------------------------------------------------------- #
# Host-grouped split                                                            #
# --------------------------------------------------------------------------- #

def make_splits(out_path: Path = SPLITS_CSV, seed: int = 42) -> pd.DataFrame:
    """
    Assign each (host, file) pair to train / val / test.
    Grouping is by host so no host appears in two splits.
    Returns and saves the manifest DataFrame.
    """
    resid = pd.read_csv(RESID_CSV)
    labels = pd.read_csv(LABELS_CSV)

    # Keep only systems with a usable dominant planet (K required)
    usable = _usable_systems(resid, labels)
    if usable.empty:
        raise RuntimeError("No usable systems found — check labels.csv and residuals_index.csv")

    hosts = usable['host'].dropna().unique().astype(str)
    rng = np.random.default_rng(seed)
    rng.shuffle(hosts)

    n = len(hosts)
    n_train = int(np.floor(TRAIN_FRAC * n))
    n_val   = int(np.floor(VAL_FRAC * n))

    split_map = {}
    for h in hosts[:n_train]:
        split_map[h] = 'train'
    for h in hosts[n_train:n_train + n_val]:
        split_map[h] = 'val'
    for h in hosts[n_train + n_val:]:
        split_map[h] = 'test'

    usable = usable.copy()
    usable['split'] = usable['host'].map(split_map)
    usable.to_csv(out_path, index=False)

    counts = usable['split'].value_counts()
    print(f"Splits saved → {out_path}")
    print(f"  Unique hosts: {n}  (train {counts.get('train',0)}, "
          f"val {counts.get('val',0)}, test {counts.get('test',0)} files)")
    return usable


def _usable_systems(resid: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Join residuals index to labels, find the dominant planet (highest K)
    per file, drop rows where K is missing.
    """
    rows = []
    for _, r in resid.iterrows():
        host = str(r['host']) if pd.notna(r['host']) else None
        if not host:
            continue
        pl = labels[labels['hostname'] == host].copy()
        if pl.empty:
            continue
        # Dominant planet = highest K; fall back to msini if K missing
        pl_k = pl[pl['pl_rvamp'].notna()]
        if pl_k.empty:
            continue
        dom = pl_k.loc[pl_k['pl_rvamp'].abs().idxmax()]
        rows.append({
            'file':        r['file'],
            'host':        host,
            'n_obs':       int(r['n_obs']),
            'rms_over_sigma': r.get('rms_over_sigma', np.nan),
            'P_d':         float(dom['pl_orbper'])       if pd.notna(dom['pl_orbper'])  else np.nan,
            'K_ms':        float(dom['pl_rvamp'])        if pd.notna(dom['pl_rvamp'])   else np.nan,
            'e':           float(dom['pl_orbeccen'])     if pd.notna(dom['pl_orbeccen'])else 0.0,
            'omega_deg':   float(dom['pl_orblper'])      if pd.notna(dom['pl_orblper']) else 0.0,
            'T_peri_d':    float(dom['pl_orbtper'])      if pd.notna(dom['pl_orbtper']) else np.nan,
        })
    df = pd.DataFrame(rows)
    # Drop rows missing P or K (can't form theta without them)
    df = df[df['P_d'].notna() & df['K_ms'].notna() & (df['K_ms'] > 0)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Parameter normalization                                                       #
# --------------------------------------------------------------------------- #

def compute_stats(splits_df: pd.DataFrame) -> dict:
    """
    Compute per-parameter mean and std on the training split.
    Saved to STATS_JSON for use at inference time.
    """
    train = splits_df[splits_df['split'] == 'train']
    theta = _params_to_theta(train)          # (N, 6)

    stats = {}
    for i, name in enumerate(THETA_NAMES):
        col = theta[:, i]
        col = col[np.isfinite(col)]
        stats[name] = {'mean': float(col.mean()), 'std': float(col.std())}

    with open(STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats() -> dict:
    if not STATS_JSON.exists():
        raise FileNotFoundError(f"{STATS_JSON} not found — run make_splits() first")
    with open(STATS_JSON) as f:
        return json.load(f)


def _params_to_theta(df: pd.DataFrame) -> np.ndarray:
    """Convert raw parameter columns to the 6-dim normalized theta array."""
    n = len(df)
    theta = np.full((n, 6), np.nan)
    theta[:, 0] = np.log10(df['P_d'].clip(lower=1e-3))
    theta[:, 1] = np.log10(df['K_ms'].clip(lower=1e-3))
    theta[:, 2] = df['e'].clip(0.0, 1.0)
    omega_rad   = np.deg2rad(df['omega_deg'].fillna(0.0))
    theta[:, 3] = np.cos(omega_rad)
    theta[:, 4] = np.sin(omega_rad)
    # T_peri phase: (T_peri mod P) / P — set to 0 if T_peri missing
    P = df['P_d'].values
    T = df['T_peri_d'].values
    phase = np.where(np.isfinite(T) & (P > 0), (T % P) / P, 0.0)
    theta[:, 5] = phase
    return theta


def normalize_theta(theta: np.ndarray, stats: dict) -> np.ndarray:
    """Standardize theta using training-split statistics."""
    out = theta.copy().astype(float)
    for i, name in enumerate(THETA_NAMES):
        s = stats[name]
        if s['std'] > 0:
            out[..., i] = (out[..., i] - s['mean']) / s['std']
    return out


def denormalize_theta(theta_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Invert normalize_theta."""
    out = theta_norm.copy().astype(float)
    for i, name in enumerate(THETA_NAMES):
        s = stats[name]
        out[..., i] = out[..., i] * s['std'] + s['mean']
    return out


# --------------------------------------------------------------------------- #
# Series padding                                                                #
# --------------------------------------------------------------------------- #

def pad_series(t: np.ndarray, rv: np.ndarray, sigma: np.ndarray,
               T: int = T_MAX) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize and zero-pad (t, rv, sigma) to length T.

    Returns
    -------
    t_norm, rv_norm, sig_norm : (T,) float32 arrays
    mask                      : (T,) float32, 1.0 = real, 0.0 = padding
    """
    n = min(len(t), T)
    order = np.argsort(t)
    t, rv, sigma = t[order][:n], rv[order][:n], sigma[order][:n]

    # Normalize time to [0, 1]
    t_span = float(t[-1] - t[0]) if n > 1 else 1.0
    t_norm = (t - t[0]) / max(t_span, 1e-6)

    # Normalize RV: subtract median, divide by std
    rv_med = float(np.median(rv))
    rv_std = float(np.std(rv)) if n > 1 else 1.0
    rv_std = max(rv_std, 1e-6)
    rv_norm  = (rv - rv_med) / rv_std
    sig_norm = sigma / rv_std

    # Pad to T
    def _pad(arr):
        out = np.zeros(T, dtype=np.float32)
        out[:n] = arr
        return out

    mask = np.zeros(T, dtype=np.float32)
    mask[:n] = 1.0

    return _pad(t_norm), _pad(rv_norm), _pad(sig_norm), mask


# --------------------------------------------------------------------------- #
# PyTorch Dataset                                                               #
# --------------------------------------------------------------------------- #

class RVDataset:
    """
    PyTorch-compatible dataset over real RV observations.

    Parameters
    ----------
    split : 'train' | 'val' | 'test' | 'all'
    splits_path : path to splits.csv (generated by make_splits)
    normalize : if True, standardize theta with training-split stats

    Item
    ----
    x     : FloatTensor (4, T_MAX)  — [t_norm, rv_norm, sig_norm, mask]
    theta : FloatTensor (6,)         — parameter vector (optionally normalized)
    info  : dict                     — {'host', 'file', 'n_obs'}
    """

    def __init__(self,
                 split: Literal['train', 'val', 'test', 'all'] = 'train',
                 splits_path: Path = SPLITS_CSV,
                 normalize: bool = True):
        if not splits_path.exists():
            raise FileNotFoundError(f"{splits_path} not found — run make_splits() first")
        df = pd.read_csv(splits_path)
        if split != 'all':
            df = df[df['split'] == split].reset_index(drop=True)
        self.df = df
        self.normalize = normalize
        self.stats = load_stats() if normalize else None
        self._labels = pd.read_csv(LABELS_CSV)

        # Pre-load cached residuals if available (fast path)
        resid_npz = ROOT / 'data' / 'residuals.npz'
        resid_csv = ROOT / 'data' / 'residuals_index.csv'
        self._cache = {}
        if resid_npz.exists() and resid_csv.exists():
            npz = np.load(resid_npz, allow_pickle=True)
            ridx = pd.read_csv(resid_csv)
            for i, row in ridx.iterrows():
                self._cache[str(row['file'])] = (
                    npz[f't_{i}'], npz[f'resid_{i}'], npz[f'sigma_{i}']
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for RVDataset.__getitem__; "
                              "use get_numpy(idx) without torch")
        x_np, theta_np, info = self.get_numpy(idx)
        return (torch.from_numpy(x_np),
                torch.from_numpy(theta_np),
                info)

    def get_numpy(self, idx: int):
        """Return (x, theta, info) as numpy arrays — no torch dependency."""
        row = self.df.iloc[idx]
        fname = str(row['file'])

        # Fast path: use cached residuals from residuals.npz
        if fname in self._cache:
            t, rv, sigma = self._cache[fname]
            t, rv, sigma = (np.asarray(a, float) for a in (t, rv, sigma))
        else:
            # Slow path: re-run validate_one
            from kepler_check import validate_one
            path = RV_DIR / fname
            res = validate_one(str(path), self._labels,
                               mode='fit', auto_sign=True, fit_tperi=True,
                               trend_order=2, return_residuals=True,
                               plot=False, verbose=False)
            if not isinstance(res, dict) or res.get('status') != 'ok':
                x = np.zeros((4, T_MAX), dtype=np.float32)
                theta = np.zeros(6, dtype=np.float32)
                return x, theta, {'host': row['host'], 'file': fname, 'valid': False}
            t     = np.asarray(res['times'],     float)
            rv    = np.asarray(res['residuals'], float)
            sigma = np.asarray(res['sigmas'],    float)

        t_norm, rv_norm, sig_norm, mask = pad_series(t, rv, sigma, T_MAX)
        x = np.stack([t_norm, rv_norm, sig_norm, mask], axis=0)  # (4, T_MAX)

        # Build theta from the splits manifest row
        theta_raw = np.array([
            np.log10(max(float(row['P_d']),   1e-3)),
            np.log10(max(float(row['K_ms']),  1e-3)),
            float(row['e']),
            float(np.cos(np.deg2rad(float(row['omega_deg'])))),
            float(np.sin(np.deg2rad(float(row['omega_deg'])))),
            0.0 if not np.isfinite(float(row['T_peri_d']))
                else float(row['T_peri_d']) % float(row['P_d']) / float(row['P_d']),
        ], dtype=np.float32)

        if self.normalize and self.stats:
            theta_raw = normalize_theta(theta_raw[None], self.stats)[0].astype(np.float32)

        info = {
            'host':  row['host'],
            'file':  row['file'],
            'n_obs': int(row['n_obs']),
            'valid': True,
        }
        return x, theta_raw, info


# --------------------------------------------------------------------------- #
# Collate function for DataLoader                                               #
# --------------------------------------------------------------------------- #

def collate_fn(batch):
    """Stack valid items; skip invalid ones. For use with torch DataLoader."""
    try:
        import torch
    except ImportError:
        raise ImportError("torch required for collate_fn")
    xs, thetas, infos = zip(*batch)
    valid = [i for i, info in enumerate(infos) if info.get('valid', True)]
    if not valid:
        return None, None, []
    xs     = torch.stack([xs[i]     for i in valid])
    thetas = torch.stack([thetas[i] for i in valid])
    infos  = [infos[i] for i in valid]
    return xs, thetas, infos


# --------------------------------------------------------------------------- #
# CLI: run to build splits + stats                                              #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import pandas as pd

    print("Building host-grouped splits...")
    splits = make_splits(SPLITS_CSV)

    print("\nComputing normalization statistics on training split...")
    stats = compute_stats(splits)
    print(f"Stats saved → {STATS_JSON}")
    print("\nParameter stats (training split):")
    for name, s in stats.items():
        print(f"  {name:20s}  mean={s['mean']:+.3f}  std={s['std']:.3f}")

    print(f"\nSplit summary:")
    print(splits['split'].value_counts().to_string())
    print(f"\nTotal usable systems: {len(splits)}")
