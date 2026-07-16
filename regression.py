"""MLP: RV features (74 / 35 / 109-D) → Kepler params (theta or h/k)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SYNGEN = ROOT / "synthetic_generation"
if str(SYNGEN) not in sys.path:
    sys.path.insert(0, str(SYNGEN))

from generate_synthetic_regression_csv import (  # noqa: E402
    _masked_observations,
    corpus_orbital_params,
    replay_synthetic_sample,
)
from plot_synthetic_regression_csv import collect_real_summary  # noqa: E402
from preprocess import LSP_PERIODS, THETA_NAMES, compute_lsp
from feature_columns import (  # noqa: E402
    BASE_74_COLUMNS,
    FEATURE_SET_COLUMNS,
    PHASE_FOLD_COLUMNS,
    PHASE_FOLD_N_BINS,
    SPECTRAL_DIM,
    SPECTRAL_GRID_SIZE,
)
from theta_loss import (
    apply_hk_constraints,
    apply_theta_constraints,
    e_balance_weights,
    hk_to_theta,
    regression_hk_loss,
    regression_theta_loss,
    theta_loss_weights_hk_numpy,
    theta_loss_weights_numpy,
    theta_to_hk,
    HK_DIM,
)
from time_series_features import phase_fold_features, spectral_features

DEFAULT_CSV = Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv"
DEFAULT_FEATURE_SET = "74"
DEFAULT_OUT = Path("figures") / "regression_synthetic"
DEFAULT_CHECKPOINT = Path("checkpoints") / "regression_mlp.pt"
CHECKPOINT_74 = Path("checkpoints") / "regression_mlp_74.pt"
CHECKPOINT_109 = Path("checkpoints") / "regression_mlp_109.pt"
PHASEFOLD_CSV = Path("synthetic_generation") / "datasets" / "synthetic_regression_10000_phasefold.csv"
CSV_SEED = 123

FEATURE_SETS = FEATURE_SET_COLUMNS

TARGET_LABELS = {
    "log10_P": r"$\log_{10} P$",
    "log10_K": r"$\log_{10} K$",
    "e": r"$e$",
    "cos_omega": r"$\cos\omega$",
    "sin_omega": r"$\sin\omega$",
}

GATE_A_E_R2 = 0.15
GATE_A_OMEGA_R2 = 0.10

E_BANDS = (
    ("e_0.1_0.3", 0.1, 0.3),
    ("e_0.3_0.5", 0.3, 0.5),
    ("e_gt_0.5", 0.5, 1.01),
)


def _snr_for_rows(bundle: DatasetBundle, local_idx: np.ndarray | None = None) -> np.ndarray | None:
    if "median_sigma_ms" not in bundle.df.columns or bundle.y.shape[1] < 2:
        return None
    rows = bundle.row_idx if local_idx is None else bundle.row_idx[local_idx]
    sigma = bundle.df["median_sigma_ms"].to_numpy(dtype=float)[rows]
    log_k = bundle.y[:, 1] if local_idx is None else bundle.y[local_idx, 1]
    return (10 ** log_k) / np.clip(sigma, 1e-6, None)


def _bundle_with_targets(bundle: DatasetBundle, targets: str) -> DatasetBundle:
    if targets == "theta":
        return bundle
    if targets != "hk":
        raise ValueError(f"unknown targets {targets!r}")
    return DatasetBundle(
        bundle.X,
        theta_to_hk(bundle.y),
        row_idx=bundle.row_idx,
        e=bundle.e,
        has_t_peri=bundle.has_t_peri,
        has_ecc=bundle.has_ecc,
        df=bundle.df,
    )


def _to_theta_space(y: np.ndarray, targets: str) -> np.ndarray:
    if targets == "theta":
        return np.asarray(y, dtype=np.float64)
    return hk_to_theta(apply_hk_constraints(y))


def _remap_loss_weights(loss_weights: np.ndarray | None, n_out: int, *, use_hk: bool) -> np.ndarray:
    """Map CLI 5-D weights onto 4-D h/k when needed."""
    if loss_weights is None:
        return np.ones(n_out, dtype=np.float64)
    w = np.asarray(loss_weights, dtype=np.float64)
    if len(w) == n_out:
        return w
    if use_hk and len(w) == 5:
        om = 0.5 * (w[3] + w[4])
        return np.array([w[0], w[1], om, om], dtype=np.float64)
    raise ValueError(f"loss_weights length {len(w)} != {n_out}")


def stratified_omega_report(y_true: np.ndarray, y_pred: np.ndarray, *, snr: np.ndarray | None = None) -> dict:
    e = np.asarray(y_true[:, 2], dtype=np.float64)
    out: dict = {"by_e_band": {}}
    for name, lo, hi in E_BANDS:
        m = (e >= lo) & (e < hi)
        n = int(m.sum())
        if n < 3:
            out["by_e_band"][name] = {"n": n, "omega_mae_deg": float("nan"), "cos_r2": float("nan"), "sin_r2": float("nan")}
            continue
        out["by_e_band"][name] = {
            "n": n,
            "omega_mae_deg": _omega_mae_deg(y_true[m], y_pred[m]),
            "cos_r2": _r2(y_true[m, 3], y_pred[m, 3]),
            "sin_r2": _r2(y_true[m, 4], y_pred[m, 4]),
        }
    if snr is not None:
        snr = np.asarray(snr, dtype=np.float64)
        valid = np.isfinite(snr) & (e > 0.1)
        out["by_snr_tertile"] = {}
        if int(valid.sum()) >= 9:
            qs = np.quantile(snr[valid], [1 / 3, 2 / 3])
            edges = [-np.inf, float(qs[0]), float(qs[1]), np.inf]
            labels = ("low", "mid", "high")
            for i, lab in enumerate(labels):
                m = valid & (snr >= edges[i]) & (snr < edges[i + 1])
                n = int(m.sum())
                out["by_snr_tertile"][lab] = {
                    "n": n,
                    "omega_mae_deg": _omega_mae_deg(y_true[m], y_pred[m]) if n >= 3 else float("nan"),
                    "snr_lo": edges[i] if np.isfinite(edges[i]) else None,
                    "snr_hi": edges[i + 1] if np.isfinite(edges[i + 1]) else None,
                }
    return out


def _print_stratified_omega(report: dict) -> None:
    bands = report.get("by_e_band") or {}
    if bands:
        print("omega by e-band:")
        for name, row in bands.items():
            print(
                f"  {name:12s}  n={row['n']:4d}  MAE={row['omega_mae_deg']:.1f} deg  "
                f"cos R2={row['cos_r2']:.3f}  sin R2={row['sin_r2']:.3f}"
            )
    snr_rows = report.get("by_snr_tertile") or {}
    if snr_rows:
        print("omega by SNR tertile (e>0.1):")
        for name, row in snr_rows.items():
            print(f"  {name:12s}  n={row['n']:4d}  MAE={row['omega_mae_deg']:.1f} deg")


def _feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature-set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]


def encode_rv(time: np.ndarray, rv: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Encode a raw RV curve into the 74-dim classifier feature vector."""
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(rv, dtype=float).reshape(-1)
    sig = np.asarray(sigma, dtype=float).reshape(-1)

    if len(t) < 2 or not (np.isfinite(t).all() and np.isfinite(y).all() and np.isfinite(sig).all()):
        raise ValueError("need at least two finite observations")

    t_span = float(t.max() - t.min())
    if t_span <= 0:
        raise ValueError("observation baseline must be positive")

    t_norm = (t - t.min()) / t_span
    rv_std = float(np.std(y))
    if rv_std < 1e-8:
        rv_std = 1.0
    rv_norm = (y - np.median(y)) / rv_std

    spectral = spectral_features(
        t_norm,
        rv_norm,
        d=SPECTRAL_DIM,
        grid_size=SPECTRAL_GRID_SIZE,
    )

    rv_ms = y
    gaps = np.diff(np.sort(t))
    lsp = compute_lsp(t, rv_ms, sig)

    summary = np.array(
        [
            len(t),
            t_span,
            rv_std,
            float(np.subtract(*np.percentile(rv_ms, [75, 25]))),
            float(np.median(sig)),
            float(np.subtract(*np.percentile(sig, [75, 25]))),
            float(LSP_PERIODS[int(np.argmax(lsp))]),
            float(np.max(lsp)),
            float(np.median(gaps)) if len(gaps) else np.nan,
            float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
        ],
        dtype=np.float64,
    )

    features = np.concatenate([spectral, summary])
    if not np.isfinite(features).all():
        raise ValueError("non-finite encoded features")
    return features


def _theta_from_manifest_row(row: pd.Series) -> np.ndarray:
    omega_rad = np.radians(float(row["omega_deg"]))
    return np.array(
        [
            np.log10(float(row["P"])),
            np.log10(float(row["K"])),
            float(row["e"]),
            np.cos(omega_rad),
            np.sin(omega_rad),
        ],
        dtype=np.float64,
    )


class DatasetBundle:
    """Features, targets, and metadata for subset evaluation."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        row_idx: np.ndarray,
        e: np.ndarray,
        has_t_peri: np.ndarray,
        has_ecc: np.ndarray,
        df: pd.DataFrame,
    ):
        self.X = X
        self.y = y
        self.row_idx = row_idx
        self.e = e
        self.has_t_peri = has_t_peri
        self.has_ecc = has_ecc
        self.df = df


def load_from_csv(csv_path: Path, feature_set: str, *, max_rows: int | None = None) -> DatasetBundle:
    """Load precomputed features and targets from the synthetic regression CSV."""
    df = pd.read_csv(csv_path)
    feature_cols = _feature_columns(feature_set)
    required = [*THETA_NAMES, *feature_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[THETA_NAMES].to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"[load_csv] dropped {n_drop} rows with non-finite values")

    has_t_peri_col = df["has_t_peri"].to_numpy(dtype=float) if "has_t_peri" in df.columns else np.ones(len(df))
    has_ecc = np.ones(len(df), dtype=bool)
    row_idx = np.arange(len(df), dtype=int)[valid]
    X_v, y_v = X[valid], y[valid]
    e_v = df["e"].to_numpy(dtype=float)[valid]
    tp_v = has_t_peri_col[valid]
    ecc_v = has_ecc[valid]
    if max_rows is not None and max_rows < len(X_v):
        print(f"[load_csv] truncating to first {max_rows} rows")
        X_v, y_v = X_v[:max_rows], y_v[:max_rows]
        row_idx = row_idx[:max_rows]
        e_v, tp_v, ecc_v = e_v[:max_rows], tp_v[:max_rows], ecc_v[:max_rows]

    return DatasetBundle(
        X_v,
        y_v,
        row_idx=row_idx,
        e=e_v,
        has_t_peri=tp_v,
        has_ecc=ecc_v,
        df=df,  # keep full CSV for replay indexing
    )


def load_from_npz(data_dir: Path) -> DatasetBundle:
    """Load NPZ corpus and encode each system with encode_rv (74-D only)."""
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    row_idx: list[int] = []

    for i, row in manifest.iterrows():
        npz_path = data_dir / row["file"]
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        try:
            features = encode_rv(data["time"], data["rv"], data["sigma"])
            theta = _theta_from_manifest_row(row)
        except ValueError:
            continue
        X_list.append(features)
        y_list.append(theta)
        row_idx.append(int(i))

    if not X_list:
        raise RuntimeError(f"no valid samples loaded from {data_dir}")
    X = np.stack(X_list)
    y = np.stack(y_list)
    return DatasetBundle(
        X,
        y,
        row_idx=np.asarray(row_idx, dtype=int),
        e=y[:, 2],
        has_t_peri=np.zeros(len(y)),
        has_ecc=np.ones(len(y), dtype=bool),
        df=pd.DataFrame(),
    )


def recompute_phasefold_block(
    row_indices: np.ndarray,
    log10_P: np.ndarray,
    *,
    seed: int,
    n_samples: int,
    f_multi: float = 0.0,
) -> np.ndarray:
    """Recompute phase-fold features folding at ``10**log10_P`` (Gate C)."""
    params = corpus_orbital_params(seed, n_samples)
    out = np.zeros((len(row_indices), len(PHASE_FOLD_COLUMNS)), dtype=np.float64)
    n = len(row_indices)
    report_every = max(1, n // 10)
    for j, (idx, lp) in enumerate(zip(row_indices, log10_P)):
        x, _, _, info = replay_synthetic_sample(int(idx), seed, n_samples, f_multi=f_multi, params=params)
        xm = _masked_observations(x)
        t_days = xm[0] * float(info["t_span_days"])
        rv_ms = xm[1] * float(info["rv_std_ms"])
        P_days = float(10 ** lp)
        phase = phase_fold_features(
            t_days,
            rv_ms,
            P_days,
            n_bins=PHASE_FOLD_N_BINS,
            t_peri=float(info["t_peri"]),
        )
        out[j] = phase
        if (j + 1) % report_every == 0 or (j + 1) == n:
            print(f"    phase-fold progress {j + 1}/{n}", flush=True)
    return out


def replace_phase_features(X: np.ndarray, feature_set: str, phase_block: np.ndarray) -> np.ndarray:
    """Swap the phase-fold columns in ``X`` (109-D layout)."""
    if feature_set == "35":
        return phase_block.copy()
    if feature_set != "109":
        raise ValueError("predicted-P fold only applies to feature sets 35 or 109")
    cols = _feature_columns(feature_set)
    phase_start = cols.index(PHASE_FOLD_COLUMNS[0])
    X_new = X.copy()
    X_new[:, phase_start : phase_start + len(PHASE_FOLD_COLUMNS)] = phase_block
    return X_new


def lsp_peak_log10_P(bundle: DatasetBundle) -> np.ndarray:
    """log10 of the CSV Lomb-Scargle peak period for each loaded row."""
    if "lsp_peak_period_d" not in bundle.df.columns:
        raise ValueError("CSV missing lsp_peak_period_d (needed for --period-source lsp_peak)")
    p = bundle.df["lsp_peak_period_d"].to_numpy(dtype=float)[bundle.row_idx]
    return np.log10(np.clip(p, 1e-6, None))


def resolve_fold_log10_P(
    source: str,
    *,
    pred_log10_P: np.ndarray | None = None,
    lsp_log10_P: np.ndarray | None = None,
    true_log10_P: np.ndarray | None = None,
) -> np.ndarray:
    """Pick fold period: mlp74, lsp_peak, hybrid, or oracle."""
    if source == "oracle":
        if true_log10_P is None:
            raise ValueError("oracle fold requires true_log10_P")
        return np.asarray(true_log10_P, dtype=np.float64)
    if source == "mlp74":
        if pred_log10_P is None:
            raise ValueError("mlp74 fold requires pred_log10_P")
        return np.asarray(pred_log10_P, dtype=np.float64)
    if source == "lsp_peak":
        if lsp_log10_P is None:
            raise ValueError("lsp_peak fold requires lsp_log10_P")
        return np.asarray(lsp_log10_P, dtype=np.float64)
    if source == "hybrid":
        if pred_log10_P is None or lsp_log10_P is None:
            raise ValueError("hybrid fold requires pred_log10_P and lsp_log10_P")
        pred = np.asarray(pred_log10_P, dtype=np.float64)
        lsp = np.asarray(lsp_log10_P, dtype=np.float64)
        close = np.abs(pred - lsp) <= 0.05
        return np.where(close, lsp, pred)
    raise ValueError(f"unknown period source {source!r}")


def apply_log10_p_jitter(
    log10_P: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add samples from residual_pool to log10_P."""
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return np.asarray(log10_P, dtype=np.float64).copy()
    noise = rng.choice(residuals, size=len(log10_P), replace=True)
    return np.asarray(log10_P, dtype=np.float64) + noise


def predict_log10_P_all(
    model: RegressionHead,
    bundle: DatasetBundle,
    preds: dict,
    norm_stats: dict,
    device: torch.device,
    *,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> np.ndarray:
    """log10_P for all rows from a trained stage-1 model."""
    out = np.empty(len(bundle.X), dtype=np.float64)
    out[preds["val_idx"]] = preds["y_pred"][:, 0]
    train_idx = preds["train_idx"]
    if len(train_idx):
        out[train_idx] = predict(
            model,
            bundle.X[train_idx],
            norm_stats,
            device,
            constrain_e=constrain_e,
            constrain_omega=constrain_omega,
        )[:, 0]
    return out


def rebuild_bundle_phasefold(
    bundle: DatasetBundle,
    feature_set: str,
    fold_log10_P: np.ndarray,
    *,
    seed: int = CSV_SEED,
    f_multi: float = 0.0,
) -> DatasetBundle:
    """Copy bundle with phase-fold features recomputed at fold_log10_P."""
    print(f"  recomputing phase-fold for {len(bundle.row_idx):,} rows ...")
    phase_block = recompute_phasefold_block(
        bundle.row_idx,
        fold_log10_P,
        seed=seed,
        n_samples=len(bundle.df),
        f_multi=f_multi,
    )
    X_new = replace_phase_features(bundle.X, feature_set, phase_block)
    return DatasetBundle(
        X_new,
        bundle.y.copy(),
        row_idx=bundle.row_idx.copy(),
        e=bundle.e.copy(),
        has_t_peri=bundle.has_t_peri.copy(),
        has_ecc=bundle.has_ecc.copy(),
        df=bundle.df,
    )


class RegressionHead(nn.Module):
    """MLP head on encoder / phase-fold features."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64), out_dim: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualEHead(nn.Module):
    """Gate + circular (e=0) and eccentric specialists."""

    def __init__(self, gate: RegressionHead, circ: RegressionHead, ecc: RegressionHead):
        super().__init__()
        self.gate = gate
        self.circ = circ
        self.ecc = ecc


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _omega_mae_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_w = np.degrees(np.arctan2(y_true[:, 4], y_true[:, 3]))
    pred_w = np.degrees(np.arctan2(y_pred[:, 4], y_pred[:, 3]))
    diff = (pred_w - true_w + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(diff)))


def _physical_mae(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> float | None:
    j = THETA_NAMES.index(name)
    if name in ("log10_P", "log10_K"):
        return float(np.mean(np.abs(10 ** y_pred[:, j] - 10 ** y_true[:, j])))
    if name in ("cos_omega", "sin_omega"):
        return None
    return float(np.mean(np.abs(y_pred[:, j] - y_true[:, j])))


def _subset_masks(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(bundle.y), dtype=bool),
        "has_ecc": bundle.has_ecc.astype(bool),
        "e_gt_0": bundle.has_ecc & (bundle.e > 0.0),
        "e_gt_0.1": bundle.has_ecc & (bundle.e > 0.1),
        "has_t_peri": bundle.has_t_peri.astype(bool),
        "e_gt_0.1_has_t_peri": bundle.has_t_peri.astype(bool) & (bundle.e > 0.1),
    }


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(THETA_NAMES):
        entry: dict[str, float] = {
            "mse": float(np.mean((y_pred[:, j] - y_true[:, j]) ** 2)),
            "r2": _r2(y_true[:, j], y_pred[:, j]),
        }
        mae_phys = _physical_mae(y_true, y_pred, name)
        if mae_phys is not None:
            entry["mae_physical"] = mae_phys
        out[name] = entry
    if len(y_true) >= 3:
        out["omega_angular"] = {"mae_deg": _omega_mae_deg(y_true, y_pred)}
    else:
        out["omega_angular"] = {"mae_deg": float("nan")}
    return out


def _subset_metrics(
    bundle: DatasetBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict]:
    masks = _subset_masks(bundle)
    out: dict[str, dict] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        out[name] = {
            "n": n,
            "excluded_n": int((~mask).sum()),
            "per_target": _per_target_metrics(y_true[mask], y_pred[mask]) if n >= 1 else {},
        }
    return out


def _denorm_theta(raw: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return raw * y_std + y_mean


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _gate_balanced_weights(
    is_pos: np.ndarray,
    has_ecc: np.ndarray,
) -> np.ndarray:
    """Inverse-frequency weights so e=0 and e>0 contribute equally (mean 1 on has_ecc)."""
    has_ecc = np.asarray(has_ecc, dtype=bool)
    is_pos = np.asarray(is_pos, dtype=bool)
    w = np.zeros(len(is_pos), dtype=np.float64)
    m = has_ecc
    if not m.any():
        return w
    n_pos = int((is_pos & m).sum())
    n_zero = int((~is_pos & m).sum())
    n = int(m.sum())
    if n_pos == 0 or n_zero == 0:
        w[m] = 1.0
        return w
    w[is_pos & m] = 0.5 * n / n_pos
    w[~is_pos & m] = 0.5 * n / n_zero
    return w


def _zero_class_metrics(
    true_zero: np.ndarray,
    pred_zero: np.ndarray,
    has_ecc: np.ndarray,
) -> dict[str, float | int]:
    """Accuracy / precision / recall / F1 for the e=0 class."""
    m = np.asarray(has_ecc, dtype=bool)
    tz = np.asarray(true_zero, dtype=bool)
    pz = np.asarray(pred_zero, dtype=bool)
    n = int(m.sum())
    n_true_zero = int((tz & m).sum())
    n_pred_zero = int((pz & m).sum())
    n_hit = int((pz & tz & m).sum())
    recall = n_hit / n_true_zero if n_true_zero else float("nan")
    precision = n_hit / n_pred_zero if n_pred_zero else float("nan")
    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "n": n,
        "frac_true_zero": float(tz[m].mean()) if n else float("nan"),
        "acc": float((pz[m] == tz[m]).mean()) if n else float("nan"),
        "recall_zero": float(recall),
        "precision_zero": float(precision),
        "f1_zero": float(f1),
    }


def _e_subset_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    """e R² / MAE on all / e>0 / e>0.1 (honest continuous-e scores)."""
    e_t = np.asarray(y_true[:, 2], dtype=np.float64)
    e_p = np.asarray(y_pred[:, 2], dtype=np.float64)

    def _one(mask: np.ndarray) -> dict[str, float | int]:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "r2": float("nan"), "mae": float("nan")}
        return {
            "n": n,
            "r2": float(_r2(e_t[mask], e_p[mask])),
            "mae": float(np.mean(np.abs(e_p[mask] - e_t[mask]))),
        }

    return {
        "all": _one(np.ones(len(e_t), dtype=bool)),
        "e_gt_0": _one(e_t > 0.0),
        "e_gt_0.1": _one(e_t > OMEGA_EVAL_E_MIN),
    }


def _select_gate_threshold(
    logits: np.ndarray,
    true_zero: np.ndarray,
    has_ecc: np.ndarray,
    *,
    true_e: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    false_circ_penalty: float = 1.5,
) -> tuple[float, dict]:
    """Pick P(e>0) threshold: max F1(zero) − penalty for routing e>0.1 as circular."""
    if candidates is None:
        candidates = np.linspace(0.15, 0.85, 15)
    probs = _sigmoid_np(logits)
    true_e = None if true_e is None else np.asarray(true_e, dtype=np.float64)
    has_ecc = np.asarray(has_ecc, dtype=bool)
    best_t = 0.5
    best_score = -1e9
    best: dict = {"f1_zero": float("nan"), "score": best_score, "false_circ_e_gt_0.1": float("nan")}
    for t in candidates:
        pred_zero = probs < float(t)
        m = _zero_class_metrics(true_zero, pred_zero, has_ecc)
        f1 = m["f1_zero"]
        if not np.isfinite(f1):
            continue
        false_circ = 0.0
        recall_ecc = 1.0
        if true_e is not None:
            ecc_m = has_ecc & (true_e > 0.1)
            if ecc_m.any():
                false_circ = float(pred_zero[ecc_m].mean())
                recall_ecc = float((~pred_zero[ecc_m]).mean())
        score = float(f1) + float(recall_ecc) - false_circ_penalty * float(false_circ)
        if score > best_score:
            best_score = score
            best_t = float(t)
            best = {
                **m,
                "score": float(score),
                "false_circ_e_gt_0.1": float(false_circ),
                "recall_ecc_e_gt_0.1": float(recall_ecc),
            }
    return best_t, best


def _frac_forced_omega0(y_true: np.ndarray, y_pred: np.ndarray, *, e_min: float = 0.1) -> float:
    """Fraction of true e>e_min rows whose pred is exactly the circ convention (cos=1, sin=0)."""
    mask = np.asarray(y_true[:, 2], dtype=np.float64) > e_min
    if not mask.any():
        return float("nan")
    forced = (np.abs(y_pred[mask, 3] - 1.0) < 1e-5) & (np.abs(y_pred[mask, 4]) < 1e-5)
    return float(forced.mean())


def predict_dual(
    model: DualEHead,
    X: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    *,
    denorm_targets: bool = True,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> np.ndarray:
    """Gate-route through circ/ecc specialists; always returns 5-D theta."""
    x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float64)
    x_std = np.asarray(norm_stats["x_std"], dtype=np.float64)
    gate_threshold = float(norm_stats.get("gate_threshold", 0.5))
    targets = norm_stats.get("targets", "theta")
    X_n = (X - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_n).float().to(device)
        logits = model.gate(xt).squeeze(-1).cpu().numpy()
        circ_raw = model.circ(xt).cpu().numpy()
        ecc_raw = model.ecc(xt).cpu().numpy()

    if denorm_targets:
        circ_stats = norm_stats["circ"]
        ecc_stats = norm_stats["ecc"]
        circ = _denorm_theta(
            circ_raw,
            np.asarray(circ_stats["y_mean"], dtype=np.float64),
            np.asarray(circ_stats["y_std"], dtype=np.float64),
        )
        ecc = _denorm_theta(
            ecc_raw,
            np.asarray(ecc_stats["y_mean"], dtype=np.float64),
            np.asarray(ecc_stats["y_std"], dtype=np.float64),
        )
    else:
        circ, ecc = circ_raw.copy(), ecc_raw.copy()

    if targets == "hk":
        circ[:, 2] = 0.0
        circ[:, 3] = 0.0
    else:
        circ[:, 2] = 0.0
        circ[:, 3] = 1.0
        circ[:, 4] = 0.0

    use_ecc = _sigmoid_np(logits) >= gate_threshold
    pred = np.where(use_ecc[:, None], ecc, circ)
    if targets == "hk":
        pred = _to_theta_space(pred, "hk")
    if constrain_e or constrain_omega:
        pred = apply_theta_constraints(pred, constrain_e=constrain_e, constrain_omega=constrain_omega)
    return pred


def predict(
    model: nn.Module,
    X: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    *,
    denorm_targets: bool = True,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> np.ndarray:
    """Predict 5-D theta (h/k decoded when targets=hk)."""
    if isinstance(model, DualEHead) or norm_stats.get("e_head") == "dual":
        return predict_dual(
            model,
            X,
            norm_stats,
            device,
            denorm_targets=denorm_targets,
            constrain_e=constrain_e,
            constrain_omega=constrain_omega,
        )

    targets = norm_stats.get("targets", "theta")
    n_out = HK_DIM if targets == "hk" else len(THETA_NAMES)
    x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float64)
    x_std = np.asarray(norm_stats["x_std"], dtype=np.float64)
    X_n = (X - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(X_n).float().to(device)).cpu().numpy()
    pred = raw[:, :n_out].copy()
    if denorm_targets and norm_stats.get("y_mean") is not None:
        y_mean = np.asarray(norm_stats["y_mean"], dtype=np.float64)
        y_std = np.asarray(norm_stats["y_std"], dtype=np.float64)
        pred = pred * y_std + y_mean
        if raw.shape[1] > n_out:
            p_pos = 1.0 / (1.0 + np.exp(-raw[:, n_out]))
            if targets == "hk":
                pred[:, 2:] = np.where(p_pos[:, None] >= 0.5, pred[:, 2:], 0.0)
            else:
                pred[:, 2] = np.where(p_pos >= 0.5, pred[:, 2], 0.0)
    if targets == "hk":
        pred = _to_theta_space(pred, "hk")
    if constrain_e or constrain_omega:
        pred = apply_theta_constraints(pred, constrain_e=constrain_e, constrain_omega=constrain_omega)
    return pred


def _fit_mlp_loop(
    model: RegressionHead,
    *,
    train_ds: TensorDataset,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    loss_fn,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    val_extra: tuple[torch.Tensor, ...] = (),
    log_prefix: str = "",
) -> RegressionHead:
    """AdamW + early stopping on val loss; mutates and returns ``model``."""
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            optim.zero_grad()
            loss_fn(model(batch[0]), *batch[1:]).backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t, *val_extra).cpu())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  {log_prefix}epoch {epoch:4d}  val_loss={val_loss:.5f}")

        if stale >= patience:
            print(f"  {log_prefix}early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _target_norm_stats(y: np.ndarray, *, target_norm: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_mean, y_std, y_fit) for a specialist subset."""
    if target_norm:
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)
        return y_mean, y_std, (y - y_mean) / y_std
    y_mean = np.zeros(y.shape[1], dtype=np.float64)
    y_std = np.ones(y.shape[1], dtype=np.float64)
    return y_mean, y_std, y


def train_dual_e_models(
    bundle: DatasetBundle,
    *,
    feature_set: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    device: torch.device,
    patience: int = 30,
    target_norm: bool = True,
    loss_weights: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    mask_omega: bool = True,
    hard_omega_mask: bool = True,
    circular_omega: bool = True,
    constrain_e: bool = True,
    constrain_omega: bool = True,
    e_balance: bool = False,
    hurdle_bce_weight: float = 1.0,
    gate_threshold: float | None = None,
    targets: str = "theta",
) -> tuple[DualEHead, dict, dict]:
    if targets not in ("theta", "hk"):
        raise ValueError(f"unknown targets {targets!r}")
    bundle = _bundle_with_targets(bundle, targets)
    use_hk = targets == "hk"
    n_out = HK_DIM if use_hk else len(THETA_NAMES)

    X, y = bundle.X, bundle.y
    train_idx, val_idx = _val_split_indices(len(X), val_frac, seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    e_train = bundle.e[train_idx]
    e_val = bundle.e[val_idx]
    has_ecc_train = bundle.has_ecc[train_idx].astype(bool)
    has_ecc_val = bundle.has_ecc[val_idx].astype(bool)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    loss_weights = _remap_loss_weights(loss_weights, n_out, use_hk=use_hk)
    dim_w = torch.from_numpy(loss_weights.astype(np.float32)).to(device)
    in_dim = X.shape[1]

    train_zero = e_train <= 0.0
    train_pos = e_train > 0.0
    val_zero = e_val <= 0.0
    val_pos = e_val > 0.0
    if not train_zero.any() or not train_pos.any():
        raise ValueError("dual e-head needs both e=0 and e>0 rows in the train split")
    if not val_zero.any() or not val_pos.any():
        raise ValueError("dual e-head needs both e=0 and e>0 rows in the val split")

    print(f"  dual: training e=0 vs e>0 gate (class-balanced, targets={targets})")
    gate = RegressionHead(in_dim=in_dim, out_dim=1).to(device)
    gate_y_train = train_pos.astype(np.float32)
    gate_y_val = val_pos.astype(np.float32)
    gate_w_train = _gate_balanced_weights(train_pos, has_ecc_train).astype(np.float32)
    gate_w_val = _gate_balanced_weights(val_pos, has_ecc_val).astype(np.float32)
    gate_ds = TensorDataset(
        torch.from_numpy(X_train_n).float(),
        torch.from_numpy(gate_y_train),
        torch.from_numpy(gate_w_train),
    )
    X_val_t = torch.from_numpy(X_val_n).float().to(device)
    gate_y_val_t = torch.from_numpy(gate_y_val).to(device)
    gate_w_val_t = torch.from_numpy(gate_w_val).to(device)

    def _gate_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return hurdle_bce_weight * (
            nn.functional.binary_cross_entropy_with_logits(
                pred.squeeze(-1), target, weight=w, reduction="sum"
            )
            / w.sum().clamp(min=1.0)
        )

    _fit_mlp_loop(
        gate,
        train_ds=gate_ds,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        device=device,
        loss_fn=_gate_loss,
        X_val_t=X_val_t,
        y_val_t=gate_y_val_t,
        val_extra=(gate_w_val_t,),
        log_prefix="gate ",
    )

    with torch.no_grad():
        gate_logits_val = gate(X_val_t).squeeze(-1).cpu().numpy()
    if gate_threshold is None:
        chosen_threshold, thr_info = _select_gate_threshold(
            gate_logits_val,
            val_zero,
            has_ecc_val,
            true_e=e_val,
        )
        print(
            f"  dual: auto gate_threshold={chosen_threshold:.3f} "
            f"(score={thr_info.get('score', float('nan')):.3f}, "
            f"f1_zero={thr_info.get('f1_zero', float('nan')):.3f}, "
            f"false_circ_e>0.1={thr_info.get('false_circ_e_gt_0.1', float('nan')):.3f})"
        )
    else:
        chosen_threshold = float(gate_threshold)
        print(f"  dual: using gate_threshold={chosen_threshold:.3f}")

    def _theta_sample_w(y_sub: np.ndarray, has_ecc_sub: np.ndarray, e_sub: np.ndarray) -> np.ndarray:
        if use_hk:
            w = theta_loss_weights_hk_numpy(y_sub, has_ecc=has_ecc_sub)
            if e_balance:
                w[:, 2] *= e_balance_weights(e_train[has_ecc_train], e_sub)
                w[:, 3] *= e_balance_weights(e_train[has_ecc_train], e_sub)
            return w
        w = theta_loss_weights_numpy(
            y_sub,
            has_ecc=has_ecc_sub,
            mask_omega=mask_omega,
            hard_omega_mask=hard_omega_mask,
        )
        if e_balance:
            w[:, 2] *= e_balance_weights(e_train[has_ecc_train], y_sub[:, 2])
        return w

    def _fit_specialist(
        name: str,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
    ) -> tuple[RegressionHead, dict]:
        print(f"  dual: training {name} specialist (n_train={int(train_mask.sum())})")
        y_tr = y_train[train_mask]
        y_va = y_val[val_mask]
        y_mean, y_std, y_tr_fit = _target_norm_stats(y_tr, target_norm=target_norm)
        if target_norm:
            y_va_fit = (y_va - y_mean) / y_std
        else:
            y_va_fit = y_va

        w_tr = _theta_sample_w(y_tr, has_ecc_train[train_mask], e_train[train_mask])
        w_va = _theta_sample_w(y_va, has_ecc_val[val_mask], e_val[val_mask])
        y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
        y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)

        ds = TensorDataset(
            torch.from_numpy(X_train_n[train_mask]).float(),
            torch.from_numpy(y_tr_fit.astype(np.float32)),
            torch.from_numpy(w_tr.astype(np.float32)),
        )
        Xv = torch.from_numpy(X_val_n[val_mask]).float().to(device)
        yv = torch.from_numpy(y_va_fit.astype(np.float32)).to(device)
        wv = torch.from_numpy(w_va.astype(np.float32)).to(device)

        def _reg_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            if use_hk:
                return regression_hk_loss(pred[:, :n_out], target, w, dim_w)
            return regression_theta_loss(
                pred[:, :n_out],
                target,
                w,
                dim_w,
                y_mean=y_mean_t,
                y_std=y_std_t,
                circular_omega=circular_omega,
            )

        head = RegressionHead(in_dim=in_dim, out_dim=n_out).to(device)
        _fit_mlp_loop(
            head,
            train_ds=ds,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
            loss_fn=_reg_loss,
            X_val_t=Xv,
            y_val_t=yv,
            val_extra=(wv,),
            log_prefix=f"{name} ",
        )
        stats = {"y_mean": y_mean.tolist(), "y_std": y_std.tolist()}
        return head, stats

    circ, circ_stats = _fit_specialist("circ", train_zero, val_zero)
    ecc, ecc_stats = _fit_specialist("ecc", train_pos, val_pos)

    model = DualEHead(gate, circ, ecc).to(device)
    norm_stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "target_norm": target_norm,
        "feature_set": feature_set,
        "in_dim": in_dim,
        "out_dim": n_out,
        "e_head": "dual",
        "targets": targets,
        "gate_threshold": float(chosen_threshold),
        "circ": circ_stats,
        "ecc": ecc_stats,
        "y_mean": ecc_stats["y_mean"],
        "y_std": ecc_stats["y_std"],
    }

    val_pred = predict(
        model,
        X_val,
        norm_stats,
        device,
        constrain_e=constrain_e,
        constrain_omega=constrain_omega,
    )
    y_val_theta = hk_to_theta(y_val) if use_hk else y_val
    val_bundle = DatasetBundle(
        X_val,
        y_val_theta,
        row_idx=bundle.row_idx[val_idx],
        e=e_val,
        has_t_peri=bundle.has_t_peri[val_idx],
        has_ecc=has_ecc_val,
        df=bundle.df,
    )

    with torch.no_grad():
        logits = model.gate(X_val_t).squeeze(-1).cpu().numpy()
    true_zero = e_val <= 0.0
    pred_zero = _sigmoid_np(logits) < chosen_threshold
    clf = _zero_class_metrics(true_zero, pred_zero, has_ecc_val)
    e_report = _e_subset_report(y_val_theta, val_pred)
    forced_omega0 = _frac_forced_omega0(y_val_theta, val_pred)
    correct_route = (pred_zero == true_zero) & has_ecc_val
    if correct_route.any():
        e_r2_correct = float(_r2(y_val_theta[correct_route, 2], val_pred[correct_route, 2]))
    else:
        e_r2_correct = float("nan")
    e_report["correct_route"] = {
        "n": int(correct_route.sum()),
        "r2": e_r2_correct,
        "mae": (
            float(np.mean(np.abs(val_pred[correct_route, 2] - y_val_theta[correct_route, 2])))
            if correct_route.any()
            else float("nan")
        ),
    }
    e_report["frac_forced_omega0_e_gt_0.1"] = forced_omega0
    strat = stratified_omega_report(y_val_theta, val_pred, snr=_snr_for_rows(val_bundle))

    print(
        f"  dual gate: acc={clf['acc']:.3f}  recall_zero={clf['recall_zero']:.3f}  "
        f"precision_zero={clf['precision_zero']:.3f}  f1_zero={clf['f1_zero']:.3f}"
    )
    print(
        f"  dual e: R2_all={e_report['all']['r2']:.3f}  "
        f"R2_e>0={e_report['e_gt_0']['r2']:.3f}  "
        f"R2_e>0.1={e_report['e_gt_0.1']['r2']:.3f}  "
        f"R2_correct_route={e_r2_correct:.3f}  "
        f"forced_omega0_on_e>0.1={forced_omega0:.3f}"
    )

    metrics: dict = {
        "feature_set": feature_set,
        "targets": targets,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_train_circ": int(train_zero.sum()),
        "n_train_ecc": int(train_pos.sum()),
        "val_mse": float(np.mean((val_pred - y_val_theta) ** 2)),
        "per_target": _per_target_metrics(y_val_theta, val_pred),
        "subsets": _subset_metrics(val_bundle, y_val_theta, val_pred),
        "e_report": e_report,
        "stratified_omega": strat,
        "norm_stats": norm_stats,
        "loss_weights": loss_weights.tolist(),
        "mask_omega": mask_omega,
        "hard_omega_mask": hard_omega_mask,
        "circular_omega": circular_omega and not use_hk,
        "target_norm": target_norm,
        "e_head": "dual",
        "e_balance": bool(e_balance),
        "hurdle_bce_weight": float(hurdle_bce_weight),
        "gate_threshold": float(chosen_threshold),
        "e_zero_classifier": clf,
    }

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_kind": "dual",
                "model": model.state_dict(),
                "norm_stats": norm_stats,
            },
            checkpoint_path,
        )
        print(f"saved checkpoint -> {checkpoint_path}")

    preds = {
        "y_true": y_val_theta,
        "y_pred": val_pred,
        "val_idx": val_idx,
        "train_idx": train_idx,
        "X_val": X_val,
        "val_row_idx": bundle.row_idx[val_idx],
    }
    return model, preds, metrics



def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild a single or dual e-head model from a checkpoint dict."""
    norm_stats = ckpt["norm_stats"]
    in_dim = int(norm_stats["in_dim"])
    out_dim = int(norm_stats.get("out_dim", len(THETA_NAMES)))
    if ckpt.get("model_kind") == "dual" or norm_stats.get("e_head") == "dual":
        model = DualEHead(
            RegressionHead(in_dim=in_dim, out_dim=1),
            RegressionHead(in_dim=in_dim, out_dim=out_dim),
            RegressionHead(in_dim=in_dim, out_dim=out_dim),
        ).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = RegressionHead(in_dim=in_dim, out_dim=out_dim).to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, norm_stats


def train_model(
    bundle: DatasetBundle,
    *,
    feature_set: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    device: torch.device,
    patience: int = 30,
    target_norm: bool = True,
    loss_weights: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    mask_omega: bool = True,
    hard_omega_mask: bool = True,
    circular_omega: bool = True,
    constrain_e: bool = True,
    constrain_omega: bool = True,
    e_head: str = "direct",
    e_balance: bool = False,
    hurdle_bce_weight: float = 1.0,
    gate_threshold: float | None = None,
    targets: str = "theta",
) -> tuple[nn.Module, dict, dict]:
    """Train the MLP and return model, predictions, and metrics."""
    if targets not in ("theta", "hk"):
        raise ValueError(f"unknown targets {targets!r}")
    if e_head not in ("direct", "hurdle", "dual"):
        raise ValueError(f"unknown e_head {e_head!r}; choose 'direct', 'hurdle', or 'dual'")
    if e_head == "dual":
        return train_dual_e_models(
            bundle,
            feature_set=feature_set,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            val_frac=val_frac,
            seed=seed,
            device=device,
            patience=patience,
            target_norm=target_norm,
            loss_weights=loss_weights,
            checkpoint_path=checkpoint_path,
            mask_omega=mask_omega,
            hard_omega_mask=hard_omega_mask,
            circular_omega=circular_omega,
            constrain_e=constrain_e,
            constrain_omega=constrain_omega,
            e_balance=e_balance,
            hurdle_bce_weight=hurdle_bce_weight,
            gate_threshold=gate_threshold,
            targets=targets,
        )

    use_hk = targets == "hk"
    bundle = _bundle_with_targets(bundle, targets)
    n_out = HK_DIM if use_hk else len(THETA_NAMES)

    X, y = bundle.X, bundle.y
    train_idx, val_idx = _val_split_indices(len(X), val_frac, seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    if target_norm:
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)
        y_train_fit = (y_train - y_mean) / y_std
        y_val_fit = (y_val - y_mean) / y_std
    else:
        y_mean = np.zeros(y.shape[1])
        y_std = np.ones(y.shape[1])
        y_train_fit = y_train
        y_val_fit = y_val

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    loss_weights = _remap_loss_weights(loss_weights, n_out, use_hk=use_hk)
    dim_w = torch.from_numpy(loss_weights.astype(np.float32)).to(device)

    has_ecc_train = bundle.has_ecc[train_idx].astype(bool)
    has_ecc_val = bundle.has_ecc[val_idx].astype(bool)
    e_train = bundle.e[train_idx]
    e_val = bundle.e[val_idx]

    if use_hk:
        train_sample_w = theta_loss_weights_hk_numpy(y_train, has_ecc=has_ecc_train)
        val_sample_w = theta_loss_weights_hk_numpy(y_val, has_ecc=has_ecc_val)
        if e_balance:
            train_sample_w[:, 2] *= e_balance_weights(e_train[has_ecc_train], e_train)
            train_sample_w[:, 3] *= e_balance_weights(e_train[has_ecc_train], e_train)
            val_sample_w[:, 2] *= e_balance_weights(e_train[has_ecc_train], e_val)
            val_sample_w[:, 3] *= e_balance_weights(e_train[has_ecc_train], e_val)
        if e_head == "hurdle":
            pos = (e_train > 0).astype(np.float64)
            train_sample_w[:, 2] *= pos
            train_sample_w[:, 3] *= pos
            pos_v = (e_val > 0).astype(np.float64)
            val_sample_w[:, 2] *= pos_v
            val_sample_w[:, 3] *= pos_v
        train_aux = np.stack([e_train > 0, has_ecc_train], axis=1).astype(np.float32)
        val_aux = np.stack([e_val > 0, has_ecc_val], axis=1).astype(np.float32)
    else:
        train_sample_w = theta_loss_weights_numpy(
            y_train,
            has_ecc=has_ecc_train,
            mask_omega=mask_omega,
            hard_omega_mask=hard_omega_mask,
        )
        val_sample_w = theta_loss_weights_numpy(
            y_val,
            has_ecc=has_ecc_val,
            mask_omega=mask_omega,
            hard_omega_mask=hard_omega_mask,
        )
        if e_balance:
            e_fit = y_train[has_ecc_train, 2]
            train_sample_w[:, 2] *= e_balance_weights(e_fit, y_train[:, 2])
            val_sample_w[:, 2] *= e_balance_weights(e_fit, y_val[:, 2])
        if e_head == "hurdle":
            train_sample_w[:, 2] *= (y_train[:, 2] > 0).astype(np.float64)
            val_sample_w[:, 2] *= (y_val[:, 2] > 0).astype(np.float64)
        train_aux = np.stack([y_train[:, 2] > 0, has_ecc_train], axis=1).astype(np.float32)
        val_aux = np.stack([y_val[:, 2] > 0, has_ecc_val], axis=1).astype(np.float32)

    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n).float(),
        torch.from_numpy(y_train_fit).float(),
        torch.from_numpy(train_sample_w.astype(np.float32)),
        torch.from_numpy(train_aux),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    out_dim = n_out + 1 if e_head == "hurdle" else n_out
    in_dim = X.shape[1]
    model = RegressionHead(in_dim=in_dim, out_dim=out_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    def _loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if use_hk:
            loss = regression_hk_loss(pred[:, :n_out], target, w, dim_w)
        else:
            loss = regression_theta_loss(
                pred[:, :n_out],
                target,
                w,
                dim_w,
                y_mean=y_mean_t,
                y_std=y_std_t,
                circular_omega=circular_omega,
            )
        if e_head == "hurdle":
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred[:, n_out], aux[:, 0], weight=aux[:, 1], reduction="sum"
            ) / aux[:, 1].sum().clamp(min=1.0)
            loss = loss + hurdle_bce_weight * bce
        return loss

    X_val_t = torch.from_numpy(X_val_n).float().to(device)
    y_val_fit_t = torch.from_numpy(y_val_fit).float().to(device)
    val_w_t = torch.from_numpy(val_sample_w.astype(np.float32)).to(device)
    val_aux_t = torch.from_numpy(val_aux).to(device)

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb, ab in loader:
            xb, yb, wb, ab = xb.to(device), yb.to(device), wb.to(device), ab.to(device)
            optim.zero_grad()
            loss = _loss(model(xb), yb, wb, ab)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_loss_t = _loss(model(X_val_t), y_val_fit_t, val_w_t, val_aux_t)
        val_loss = float(val_loss_t.cpu())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  val_mse={val_loss:.5f}")

        if stale >= patience:
            print(f"  early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    norm_stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "target_norm": target_norm,
        "feature_set": feature_set,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "e_head": e_head,
        "targets": targets,
    }

    val_pred = predict(
        model,
        X_val,
        norm_stats,
        device,
        constrain_e=constrain_e,
        constrain_omega=constrain_omega,
    )
    y_val_theta = hk_to_theta(y_val) if use_hk else y_val

    val_bundle = DatasetBundle(
        X_val,
        y_val_theta,
        row_idx=bundle.row_idx[val_idx],
        e=bundle.e[val_idx],
        has_t_peri=bundle.has_t_peri[val_idx],
        has_ecc=bundle.has_ecc[val_idx],
        df=bundle.df,
    )
    strat = stratified_omega_report(y_val_theta, val_pred, snr=_snr_for_rows(val_bundle))

    metrics: dict = {
        "feature_set": feature_set,
        "targets": targets,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_mse": float(np.mean((val_pred - y_val_theta) ** 2)),
        "per_target": _per_target_metrics(y_val_theta, val_pred),
        "subsets": _subset_metrics(val_bundle, y_val_theta, val_pred),
        "e_report": _e_subset_report(y_val_theta, val_pred),
        "stratified_omega": strat,
        "norm_stats": norm_stats,
        "loss_weights": loss_weights.tolist(),
        "mask_omega": mask_omega,
        "hard_omega_mask": hard_omega_mask,
        "circular_omega": circular_omega and not use_hk,
        "target_norm": target_norm,
        "e_head": e_head,
        "e_balance": bool(e_balance),
    }

    true_zero = e_val <= 0.0
    if e_head == "hurdle":
        metrics["hurdle_bce_weight"] = float(hurdle_bce_weight)
        with torch.no_grad():
            logits = model(X_val_t)[:, n_out].cpu().numpy()
        pred_zero = logits < 0.0
        metrics["e_zero_classifier"] = _zero_class_metrics(true_zero, pred_zero, has_ecc_val)
    else:
        pred_zero = val_pred[:, 2] <= 1e-3
        metrics["e_zero_classifier"] = _zero_class_metrics(true_zero, pred_zero, has_ecc_val)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "norm_stats": norm_stats}, checkpoint_path)
        print(f"saved checkpoint -> {checkpoint_path}")

    preds = {
        "y_true": y_val_theta,
        "y_pred": val_pred,
        "val_idx": val_idx,
        "train_idx": train_idx,
        "X_val": X_val,
        "val_row_idx": bundle.row_idx[val_idx],
    }
    return model, preds, metrics


def eval_predicted_p_fold(
    model: RegressionHead,
    bundle: DatasetBundle,
    preds: dict,
    norm_stats: dict,
    *,
    feature_set: str,
    seed: int,
    device: torch.device,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> dict:
    """Gate C: replace oracle phase features with folds at predicted P."""
    val_row_idx = preds["val_row_idx"]
    pred_log10_P = preds["y_pred"][:, 0]
    phase_block = recompute_phasefold_block(
        val_row_idx, pred_log10_P, seed=CSV_SEED, n_samples=len(bundle.df)
    )
    X_val_pred = replace_phase_features(preds["X_val"], feature_set, phase_block)
    y_pred = predict(model, X_val_pred, norm_stats, device, constrain_e=constrain_e, constrain_omega=constrain_omega)
    y_true = preds["y_true"]

    val_bundle = DatasetBundle(
        X_val_pred,
        y_true,
        row_idx=val_row_idx,
        e=bundle.e[preds["val_idx"]],
        has_t_peri=bundle.has_t_peri[preds["val_idx"]],
        has_ecc=bundle.has_ecc[preds["val_idx"]],
        df=bundle.df,
    )
    return {
        "fold_period": "predicted",
        "val_mse": float(np.mean((y_pred - y_true) ** 2)),
        "per_target": _per_target_metrics(y_true, y_pred),
        "subsets": _subset_metrics(val_bundle, y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _scatter_limits(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


OMEGA_EVAL_E_MIN = 0.1  # skip near-circular rows when scoring omega


def _omega_eval_mask(y_true: np.ndarray, *, e_min: float = OMEGA_EVAL_E_MIN) -> np.ndarray:
    """True where e > e_min (omega is well-defined)."""
    return np.asarray(y_true[:, 2], dtype=np.float64) > e_min


def _omega_panel_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Arrays/R² for a scatter panel; omega uses e > 0.1 only."""
    j = THETA_NAMES.index(target)
    if target in ("cos_omega", "sin_omega"):
        mask = _omega_eval_mask(y_true)
        yt, yp = y_true[mask, j], y_pred[mask, j]
        return yt, yp, _r2(yt, yp), int(mask.sum())
    yt, yp = y_true[:, j], y_pred[:, j]
    return yt, yp, _r2(yt, yp), len(yt)


def plot_single_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    out_path: Path,
    *,
    title: str | None = None,
    metrics: dict | None = None,
) -> None:
    """True-vs-pred scatter; omega plots use e > 0.1 only."""
    yt, yp, r2, n = _omega_panel_arrays(y_true, y_pred, target)
    if len(yt) == 0:
        print(f"skip plot {out_path}: no rows for {target}")
        return
    mse = float(np.mean((yp - yt) ** 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=14, alpha=0.5, edgecolors="none")
    lo, hi = _scatter_limits(yt, yp)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.65)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true {TARGET_LABELS[target]}")
    ax.set_ylabel(f"predicted {TARGET_LABELS[target]}")
    if title is None:
        if target in ("cos_omega", "sin_omega"):
            title = (
                f"{TARGET_LABELS[target]}  ($e>{OMEGA_EVAL_E_MIN:g}$, n={n})  "
                f"$R^2$={r2:.3f}  MSE={mse:.4f}"
            )
        elif target == "e" and metrics and "e_report" in metrics:
            er = metrics["e_report"]
            title = (
                f"{TARGET_LABELS[target]}  $R^2$={r2:.3f}  "
                f"(e>0: {er['e_gt_0']['r2']:.3f}, n={er['e_gt_0']['n']})"
            )
        else:
            title = f"{TARGET_LABELS[target]}  $R^2$={r2:.3f}  MSE={mse:.4f}"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, metrics: dict) -> None:
    """5-panel validation scatter; omega panels use e > 0.1."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    omega_sub = metrics.get("subsets", {}).get("e_gt_0.1", {}).get("per_target", {})
    e_report = metrics.get("e_report") or _e_subset_report(y_true, y_pred)

    for j, (ax, name) in enumerate(zip(axes, THETA_NAMES)):
        yt, yp, r2_panel, n_panel = _omega_panel_arrays(y_true, y_pred, name)
        ax.scatter(yt, yp, s=8, alpha=0.45, edgecolors="none")
        lo, hi = _scatter_limits(yt, yp)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if name in ("cos_omega", "sin_omega"):
            r2 = omega_sub.get(name, {}).get("r2", r2_panel)
            ax.set_title(f"{TARGET_LABELS[name]} ($e>{OMEGA_EVAL_E_MIN:g}$)\n$R^2$={r2:.3f}  n={n_panel}")
        elif name == "e":
            r2 = metrics["per_target"][name]["r2"]
            r2_pos = e_report["e_gt_0"]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f} (e>0: {r2_pos:.3f})")
        else:
            r2 = metrics["per_target"][name]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.25)

    fig.suptitle(f"pred vs true (val; omega: e>{OMEGA_EVAL_E_MIN:g})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_all_targets_individual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    *,
    prefix: str = "pred_vs_true",
    metrics: dict | None = None,
) -> None:
    """One scatter per target (omega restricted to e > 0.1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in THETA_NAMES:
        plot_single_target(
            y_true,
            y_pred,
            name,
            out_dir / f"{prefix}_{name}.png",
            metrics=metrics,
        )


def plot_combined_scatter(
    synth_true: np.ndarray,
    synth_pred: np.ndarray,
    real_true: np.ndarray,
    real_pred: np.ndarray,
    target: str,
    out_path: Path,
) -> None:
    if target not in THETA_NAMES:
        raise ValueError(f"unknown target {target!r}; choose from {THETA_NAMES}")

    j = THETA_NAMES.index(target)
    yt_s, yp_s = synth_true[:, j], synth_pred[:, j]
    yt_r, yp_r = real_true[:, j], real_pred[:, j]
    mse_s = float(np.mean((yp_s - yt_s) ** 2))
    mse_r = float(np.mean((yp_r - yt_r) ** 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.scatter(yt_s, yp_s, s=12, alpha=0.45, c="red", edgecolors="none",
               label=f"synthetic (MSE={mse_s:.3f})")
    ax.scatter(yt_r, yp_r, s=24, alpha=0.75, c="blue", edgecolors="none",
               label=f"real (MSE={mse_r:.3f})")

    lo = min(yt_s.min(), yp_s.min(), yt_r.min(), yp_r.min())
    hi = max(yt_s.max(), yp_s.max(), yt_r.max(), yp_r.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true {TARGET_LABELS[target]}")
    ax.set_ylabel(f"predicted {TARGET_LABELS[target]}")
    ax.set_title(f"Synthetic-trained MLP: {TARGET_LABELS[target]}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_e_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true[:, 2], y_pred[:, 2], s=10, alpha=0.5, edgecolors="none")
    lo = min(y_true[:, 2].min(), y_pred[:, 2].min())
    hi = max(y_true[:, 2].max(), y_pred[:, 2].max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
    ax.set_xlabel("true e")
    ax.set_ylabel("pred e")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _omega_angles_deg(y: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(y[:, 4], y[:, 3]))


def plot_omega_unit_circle(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, *, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true[:, 3], y_true[:, 4], s=12, alpha=0.45, label="true", edgecolors="none")
    ax.scatter(y_pred[:, 3], y_pred[:, 4], s=12, alpha=0.45, label="pred", edgecolors="none")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$\cos\omega$")
    ax.set_ylabel(r"$\sin\omega$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_omega_error_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    true_w = _omega_angles_deg(y_true)
    pred_w = _omega_angles_deg(y_pred)
    err = (pred_w - true_w + 180.0) % 360.0 - 180.0
    mae = float(np.mean(np.abs(err)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(err, bins=40, alpha=0.75, edgecolor="none")
    ax.axvline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\omega$ error (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"Angular error (MAE={mae:.1f} deg)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_omega_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, *, subset: str = "e>0.1") -> None:
    mask = y_true[:, 2] > 0.1
    if not mask.any():
        return
    yt, yp = y_true[mask], y_pred[mask]
    plot_omega_unit_circle(yt, yp, out_dir / "pred_vs_true_omega_circle.png", title=f"omega on unit circle ({subset})")
    plot_omega_error_hist(yt, yp, out_dir / "pred_vs_true_omega_deg.png")


def _print_omega_headline(metrics: dict) -> None:
    sub = metrics.get("subsets", {}).get("e_gt_0.1_has_t_peri", metrics.get("subsets", {}).get("e_gt_0.1", {}))
    if not sub:
        return
    pt = sub.get("per_target", {})
    n = sub.get("n", 0)
    cos_r2 = pt.get("cos_omega", {}).get("r2", float("nan"))
    sin_r2 = pt.get("sin_omega", {}).get("r2", float("nan"))
    mae_deg = pt.get("omega_angular", {}).get("mae_deg", float("nan"))
    print(f"omega headline (e>0.1, n={n}): angular MAE={mae_deg:.1f} deg  cos R2={cos_r2:.3f}  sin R2={sin_r2:.3f}")


def _print_e_headline(metrics: dict) -> None:
    er = metrics.get("e_report")
    clf = metrics.get("e_zero_classifier")
    if er:
        print(
            f"e headline: R2_all={er['all']['r2']:.3f}  "
            f"R2_e>0={er['e_gt_0']['r2']:.3f} (n={er['e_gt_0']['n']})  "
            f"R2_e>0.1={er['e_gt_0.1']['r2']:.3f}  "
            f"MAE_e>0={er['e_gt_0']['mae']:.4f}"
        )
        if "correct_route" in er:
            cr = er["correct_route"]
            print(f"  e R2 correct_route={cr['r2']:.3f} (n={cr['n']})")
    if clf:
        print(
            f"e=0 gate: acc={clf.get('acc', float('nan')):.3f}  "
            f"recall={clf.get('recall_zero', float('nan')):.3f}  "
            f"precision={clf.get('precision_zero', float('nan')):.3f}  "
            f"f1={clf.get('f1_zero', float('nan')):.3f}"
        )


def eval_two_step(
    bundle_109: DatasetBundle,
    model_74: RegressionHead,
    norm_74: dict,
    model_109: RegressionHead,
    norm_109: dict,
    preds_109: dict,
    device: torch.device,
    *,
    period_source: str = "mlp74",
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> dict:
    """Fold at period_source period, then run the 109-D shape head."""
    X_val = preds_109["X_val"]
    y_true = preds_109["y_true"]
    val_row_idx = preds_109["val_row_idx"]
    val_idx = preds_109["val_idx"]

    n74 = len(BASE_74_COLUMNS)
    X_74 = X_val[:, :n74]
    pred_74 = predict(
        model_74,
        X_74,
        norm_74,
        device,
        constrain_e=constrain_e,
        constrain_omega=constrain_omega,
    )
    pred_log10_P = pred_74[:, 0]
    lsp_log10_P = lsp_peak_log10_P(bundle_109)[val_idx]
    fold_log10_P = resolve_fold_log10_P(
        period_source,
        pred_log10_P=pred_log10_P,
        lsp_log10_P=lsp_log10_P,
        true_log10_P=y_true[:, 0],
    )

    phase_block = recompute_phasefold_block(
        val_row_idx, fold_log10_P, seed=CSV_SEED, n_samples=len(bundle_109.df)
    )
    X_two_step = replace_phase_features(X_val, "109", phase_block)
    pred_109 = predict(
        model_109,
        X_two_step,
        norm_109,
        device,
        constrain_e=constrain_e,
        constrain_omega=constrain_omega,
    )

    y_pred = pred_109.copy()
    y_pred[:, 0] = pred_log10_P if period_source in ("mlp74", "hybrid") else fold_log10_P

    val_bundle = DatasetBundle(
        X_two_step,
        y_true,
        row_idx=val_row_idx,
        e=bundle_109.e[val_idx],
        has_t_peri=bundle_109.has_t_peri[val_idx],
        has_ecc=bundle_109.has_ecc[val_idx],
        df=bundle_109.df,
    )
    p_r2_mlp = _r2(y_true[:, 0], pred_log10_P)
    p_r2_fold = _r2(y_true[:, 0], fold_log10_P)
    frac_5pct = float(np.mean(np.abs(10 ** fold_log10_P / 10 ** y_true[:, 0] - 1.0) <= 0.05))
    return {
        "mode": "two_step",
        "p_stage": "74",
        "period_source": period_source,
        "p_r2_stage1": float(p_r2_mlp),
        "p_r2_fold": float(p_r2_fold),
        "fold_within_5pct": frac_5pct,
        "val_mse": float(np.mean((y_pred - y_true) ** 2)),
        "per_target": _per_target_metrics(y_true, y_pred),
        "subsets": _subset_metrics(val_bundle, y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _omega_r2_e_gt(metrics: dict) -> float:
    sub = metrics.get("subsets", {}).get("e_gt_0.1_has_t_peri", metrics.get("subsets", {}).get("e_gt_0.1", {}))
    pt = sub.get("per_target", {})
    return float(np.nanmean([
        pt.get("cos_omega", {}).get("r2", float("nan")),
        pt.get("sin_omega", {}).get("r2", float("nan")),
    ]))


def _omega_mae_e_gt(metrics: dict) -> float:
    sub = metrics.get("subsets", {}).get("e_gt_0.1_has_t_peri", metrics.get("subsets", {}).get("e_gt_0.1", {}))
    return float(sub.get("per_target", {}).get("omega_angular", {}).get("mae_deg", float("nan")))


def run_two_step_pipeline(args: argparse.Namespace, device: torch.device) -> dict:
    """Train 74-D then 109-D and evaluate the fold-then-shape path."""
    csv_path = PHASEFOLD_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"two-step requires {csv_path}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    train_kw = _theta_train_kwargs(args)
    pred_kw = _predict_kwargs(args)
    loss_w = _parse_loss_weights(args.loss_weights)
    period_source = args.period_source
    stage2_fold = args.stage2_fold
    use_jitter = bool(args.stage2_p_jitter)

    print("=" * 60)
    print("Two-step Stage 1: train 74-D (P/K baseline features)")
    max_rows = getattr(args, "max_rows", None)
    bundle_74 = load_from_csv(csv_path, "74", max_rows=max_rows)
    model_74, preds_74, metrics_74 = train_model(
        bundle_74,
        feature_set="74",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=loss_w,
        checkpoint_path=CHECKPOINT_74,
        **train_kw,
    )
    print(f"  74-D log10_P R2={metrics_74['per_target']['log10_P']['r2']:.3f}")

    pred_p_all = predict_log10_P_all(
        model_74, bundle_74, preds_74, metrics_74["norm_stats"], device, **pred_kw
    )
    lsp_p_all = lsp_peak_log10_P(bundle_74)
    true_p_all = bundle_74.y[:, 0]
    mlp_residuals = pred_p_all[preds_74["train_idx"]] - true_p_all[preds_74["train_idx"]]
    lsp_residuals = lsp_p_all[preds_74["train_idx"]] - true_p_all[preds_74["train_idx"]]
    residual_pool = mlp_residuals if period_source == "mlp74" else lsp_residuals

    if stage2_fold == "oracle":
        fold_base = true_p_all.copy()
    elif stage2_fold == "predicted":
        fold_base = resolve_fold_log10_P(
            period_source,
            pred_log10_P=pred_p_all,
            lsp_log10_P=lsp_p_all,
            true_log10_P=true_p_all,
        )
    elif stage2_fold == "jitter":
        fold_base = apply_log10_p_jitter(
            true_p_all, residual_pool, np.random.default_rng(args.seed + 17)
        )
    else:
        raise ValueError(f"unknown --stage2-fold {stage2_fold!r}")

    fold_train = fold_base.copy()
    if use_jitter and stage2_fold != "jitter":
        rng_j = np.random.default_rng(args.seed + 19)
        fold_train[preds_74["train_idx"]] = apply_log10_p_jitter(
            fold_base[preds_74["train_idx"]], residual_pool, rng_j
        )

    print("=" * 60)
    print(
        f"Two-step Stage 2: train 109-D "
        f"(stage2_fold={stage2_fold}, period_source={period_source}, jitter={use_jitter})"
    )
    bundle_109_oracle = load_from_csv(csv_path, "109", max_rows=max_rows)
    if stage2_fold == "oracle" and not use_jitter:
        bundle_109_train = bundle_109_oracle
    else:
        bundle_109_train = rebuild_bundle_phasefold(
            bundle_109_oracle, "109", fold_train, seed=CSV_SEED
        )

    ckpt_109 = CHECKPOINT_109
    if stage2_fold != "oracle" or period_source != "mlp74" or use_jitter:
        ckpt_109 = Path("checkpoints") / f"regression_mlp_109_{stage2_fold}_{period_source}.pt"

    model_109, preds_109, metrics_109 = train_model(
        bundle_109_train,
        feature_set="109",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=loss_w,
        checkpoint_path=ckpt_109,
        **train_kw,
    )
    print(f"  109-D stage2-train e R2={metrics_109['per_target']['e']['r2']:.3f}")
    _print_omega_headline(metrics_109)

    print("=" * 60)
    print("Oracle-fold check on stage-2 model")
    X_oracle = replace_phase_features(
        preds_109["X_val"],
        "109",
        recompute_phasefold_block(
            preds_109["val_row_idx"],
            preds_109["y_true"][:, 0],
            seed=CSV_SEED,
            n_samples=len(bundle_109_oracle.df),
        ),
    )
    y_oracle = predict(
        model_109, X_oracle, metrics_109["norm_stats"], device, **pred_kw
    )
    oracle_probe = {
        "per_target": _per_target_metrics(preds_109["y_true"], y_oracle),
        "subsets": _subset_metrics(
            DatasetBundle(
                X_oracle,
                preds_109["y_true"],
                row_idx=preds_109["val_row_idx"],
                e=bundle_109_oracle.e[preds_109["val_idx"]],
                has_t_peri=bundle_109_oracle.has_t_peri[preds_109["val_idx"]],
                has_ecc=bundle_109_oracle.has_ecc[preds_109["val_idx"]],
                df=bundle_109_oracle.df,
            ),
            preds_109["y_true"],
            y_oracle,
        ),
    }
    print(f"  oracle-fold omega R2 (e>0.1)={_omega_r2_e_gt(oracle_probe):.3f}")

    print("=" * 60)
    print("Gate C: fold at 109-D predicted P")
    gate_c = eval_predicted_p_fold(
        model_109,
        bundle_109_train,
        preds_109,
        metrics_109["norm_stats"],
        feature_set="109",
        seed=CSV_SEED,
        device=device,
        **pred_kw,
    )
    om_c_r2 = _omega_r2_e_gt(gate_c)
    print(f"  Gate C omega R2 (e>0.1): {om_c_r2:.3f}")

    print("=" * 60)
    print(f"Two-step eval (period_source={period_source})")
    preds_for_ts = {
        **preds_109,
        "X_val": bundle_109_oracle.X[preds_109["val_idx"]],
        "val_idx": preds_109["val_idx"],
    }
    two_step = eval_two_step(
        bundle_109_oracle,
        model_74,
        metrics_74["norm_stats"],
        model_109,
        metrics_109["norm_stats"],
        preds_for_ts,
        device,
        period_source=period_source,
        **pred_kw,
    )
    om_ts_r2 = _omega_r2_e_gt(two_step)
    om_ts_mae = _omega_mae_e_gt(two_step)
    print(f"  two-step P R2 (stage1 MLP)={two_step['p_r2_stage1']:.3f}")
    print(f"  two-step P R2 (fold source)={two_step['p_r2_fold']:.3f}  within5%={two_step['fold_within_5pct']:.3f}")
    print(f"  two-step e R2={two_step['per_target']['e']['r2']:.3f}")
    print(f"  two-step omega R2 (e>0.1)={om_ts_r2:.3f}  angular MAE={om_ts_mae:.1f} deg")
    _print_omega_headline({"subsets": two_step["subsets"]})

    report = {
        "csv": str(csv_path),
        "loss_weights": loss_w.tolist(),
        "circular_omega": not args.no_circular_omega,
        "hard_omega_mask": not args.soft_omega_mask,
        "period_source": period_source,
        "stage2_fold": stage2_fold,
        "stage2_p_jitter": use_jitter,
        "stage1_74": metrics_74,
        "stage2_109_train": metrics_109,
        "stage2_oracle_fold": oracle_probe,
        "gate_c_109_self_p": {k: v for k, v in gate_c.items() if k not in ("y_true", "y_pred")},
        "two_step": {k: v for k, v in two_step.items() if k not in ("y_true", "y_pred")},
        "omega_r2_e_gt_0.1": {
            "stage2_train_val": _omega_r2_e_gt(metrics_109),
            "oracle_fold": _omega_r2_e_gt(oracle_probe),
            "gate_c_self_p": float(om_c_r2),
            "two_step": float(om_ts_r2),
        },
        "omega_mae_e_gt_0.1_deg": {
            "stage2_train_val": _omega_mae_e_gt(metrics_109),
            "two_step": float(om_ts_mae),
        },
    }
    if om_ts_r2 > 0.05 and om_ts_mae < 40.0:
        report["note"] = f"two-step ok (source={period_source}, fold={stage2_fold}): omega MAE={om_ts_mae:.1f} deg"
    elif om_ts_r2 > om_c_r2:
        report["note"] = "two-step better than Gate C, but omega still weak"
    else:
        report["note"] = "omega still collapses at predicted/LSP period; need tighter P or more fold noise in training"

    plot_pred_vs_true(two_step["y_true"], two_step["y_pred"], out_dir / "pred_vs_true_two_step.png", two_step)
    plot_all_targets_individual(two_step["y_true"], two_step["y_pred"], out_dir, prefix="pred_vs_true_two_step")
    plot_omega_diagnostics(two_step["y_true"], two_step["y_pred"], out_dir)

    out_path = out_dir / "two_step_metrics.json"
    _write_benchmark(report, out_path)
    return report


def _gate_pass_a(metrics: dict) -> bool:
    sub = metrics["subsets"].get("e_gt_0.1_has_t_peri", metrics["subsets"].get("e_gt_0.1", {}))
    pt = sub.get("per_target", {})
    e_r2 = pt.get("e", {}).get("r2", float("nan"))
    cos_r2 = pt.get("cos_omega", {}).get("r2", float("nan"))
    sin_r2 = pt.get("sin_omega", {}).get("r2", float("nan"))
    omega_r2 = np.nanmean([cos_r2, sin_r2])
    return (e_r2 > GATE_A_E_R2) or (omega_r2 > GATE_A_OMEGA_R2)


def run_benchmark_gates(args: argparse.Namespace, device: torch.device) -> dict:
    """Run Gates A/B/C and optional ablations; write benchmark.json."""
    phasefold_csv = Path("synthetic_generation") / "datasets" / "synthetic_regression_10000_phasefold.csv"
    csv_path = args.csv if args.csv != DEFAULT_CSV else phasefold_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"benchmark requires phasefold CSV at {csv_path}")
    out_dir = args.out
    train_kw = _theta_train_kwargs(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gate A: 35-D phase-fold only (oracle P)")
    bundle_35 = load_from_csv(csv_path, "35")
    _, preds_a, metrics_a = train_model(
        bundle_35,
        feature_set="35",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=_parse_loss_weights(args.loss_weights),
        checkpoint_path=None,
        **train_kw,
    )
    pass_a = _gate_pass_a(metrics_a)
    print(f"Gate A pass: {pass_a}  (e R2>{GATE_A_E_R2} or omega R2>{GATE_A_OMEGA_R2} on e>0.1)")
    _print_omega_headline(metrics_a)

    benchmark: dict = {
        "csv": str(csv_path),
        "gate_a_35_oracle": {
            "pass": pass_a,
            "metrics": metrics_a,
        },
    }

    if not pass_a:
        benchmark["stopped_after"] = "gate_a"
        benchmark["story"] = (
            "Phase-fold features alone do not recover e/omega above gate thresholds. "
            "Do not proceed to 109-D or predicted-P tests."
        )
        _write_benchmark(benchmark, out_dir / "benchmark.json")
        return benchmark

    print("=" * 60)
    print("Gate B baseline: 74-D (oracle)")
    bundle_74 = load_from_csv(csv_path, "74")
    _, _, metrics_74 = train_model(
        bundle_74,
        feature_set="74",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=_parse_loss_weights(args.loss_weights),
        checkpoint_path=None,
        **train_kw,
    )

    print("Gate B: 109-D (oracle P)")
    bundle_109 = load_from_csv(csv_path, "109")
    model_109, preds_b, metrics_109 = train_model(
        bundle_109,
        feature_set="109",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=_parse_loss_weights(args.loss_weights),
        checkpoint_path=args.checkpoint,
        **train_kw,
    )

    e_r2_74 = metrics_74["per_target"]["e"]["r2"]
    e_r2_109 = metrics_109["per_target"]["e"]["r2"]
    om_sub = metrics_109["subsets"]["e_gt_0.1_has_t_peri"]
    om_r2 = np.nanmean([
        om_sub["per_target"].get("cos_omega", {}).get("r2", float("nan")),
        om_sub["per_target"].get("sin_omega", {}).get("r2", float("nan")),
    ])
    pass_b = e_r2_109 > e_r2_74
    print(f"Gate B pass (e R2 improves): {pass_b}  74={e_r2_74:.3f} 109={e_r2_109:.3f}")
    print(f"  omega R2 (e>0.1, has_t_peri) ~ {om_r2:.3f}")
    _print_omega_headline(metrics_109)

    benchmark["gate_b_74_oracle_baseline"] = metrics_74
    benchmark["gate_b_109_oracle"] = {"pass": pass_b, "metrics": metrics_109}

    print("=" * 60)
    print("Gate C: 109-D eval with predicted-P phase fold")
    metrics_c = eval_predicted_p_fold(
        model_109,
        bundle_109,
        preds_b,
        metrics_109["norm_stats"],
        feature_set="109",
        seed=CSV_SEED,
        device=device,
        **_predict_kwargs(args),
    )
    om_c = metrics_c["subsets"]["e_gt_0.1_has_t_peri"]["per_target"]
    om_c_r2 = np.nanmean([om_c.get("cos_omega", {}).get("r2", float("nan")), om_c.get("sin_omega", {}).get("r2", float("nan"))])
    collapse = om_c_r2 < 0.05
    metrics_c_out = {k: v for k, v in metrics_c.items() if k not in ("y_true", "y_pred")}
    benchmark["gate_c_109_predicted_p"] = {
        "collapse": collapse,
        "metrics": metrics_c_out,
        "omega_r2_e_gt_0.1": float(om_c_r2),
    }
    print(f"Gate C omega R2 (e>0.1, predicted P): {om_c_r2:.3f}  collapse={collapse}")

    if not collapse and args.run_optional_ablations:
        print("=" * 60)
        print("Optional: loss-weight ablation 1,1,5,5,5 on 109-D")
        _, _, metrics_lw = train_model(
            bundle_109,
            feature_set="109",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            seed=args.seed,
            device=device,
            patience=args.patience,
            target_norm=not args.no_target_norm,
            loss_weights=np.array([1.0, 1.0, 5.0, 5.0, 5.0]),
            checkpoint_path=None,
            **train_kw,
        )
        benchmark["optional_loss_weights_1_1_5_5_5"] = metrics_lw

        if args.generate_50k:
            csv_50k = Path("synthetic_generation") / "datasets" / "synthetic_regression_50000_phasefold.csv"
            if csv_50k.exists():
                print("Optional: 109-D on 50K phasefold CSV")
                bundle_50k = load_from_csv(csv_50k, "109")
                _, _, metrics_50k = train_model(
                    bundle_50k,
                    feature_set="109",
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    val_frac=args.val_frac,
                    seed=args.seed,
                    device=device,
                    patience=args.patience,
                    target_norm=not args.no_target_norm,
                    loss_weights=_parse_loss_weights(args.loss_weights),
                    checkpoint_path=None,
                    **train_kw,
                )
                benchmark["optional_50k_109_oracle"] = metrics_50k

    if collapse:
        benchmark["story"] = (
            "Phase-fold features help with oracle period, but omega signal collapses when folding "
            "at predicted P. Deployment needs a known/constrained period before e/omega refinement."
        )
    else:
        benchmark["story"] = (
            "Phase-fold features improve e/omega with oracle period; some omega signal survives "
            "predicted-P folding — promising for two-step P-then-shape workflows."
        )

    plot_e_scatter(preds_b["y_true"], preds_b["y_pred"], out_dir / "combined_scatter_e.png", "e (109-D oracle)")
    e_mask = preds_b["y_true"][:, 2] > 0.1
    plot_e_scatter(
        preds_b["y_true"][e_mask],
        preds_b["y_pred"][e_mask],
        out_dir / "pred_vs_true_e_gt_0.1.png",
        "e (validation, e>0.1)",
    )
    plot_omega_diagnostics(preds_b["y_true"], preds_b["y_pred"], out_dir)

    _write_benchmark(benchmark, out_dir / "benchmark.json")
    return benchmark


def _val_split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def load_checkpoint_and_predict_val(
    bundle: DatasetBundle,
    checkpoint_path: Path,
    *,
    val_frac: float,
    seed: int,
    device: torch.device,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a saved checkpoint and predict on the validation split."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, norm_stats = build_model_from_checkpoint(ckpt, device)

    train_idx, val_idx = _val_split_indices(len(bundle.X), val_frac, seed)
    X_val, y_val = bundle.X[val_idx], bundle.y[val_idx]
    y_pred = predict(
        model,
        X_val,
        norm_stats,
        device,
        constrain_e=constrain_e,
        constrain_omega=constrain_omega,
    )

    val_bundle = DatasetBundle(
        X_val,
        y_val,
        row_idx=bundle.row_idx[val_idx],
        e=bundle.e[val_idx],
        has_t_peri=bundle.has_t_peri[val_idx],
        has_ecc=bundle.has_ecc[val_idx],
        df=bundle.df,
    )
    metrics = {
        "val_mse": float(np.mean((y_pred - y_val) ** 2)),
        "per_target": _per_target_metrics(y_val, y_pred),
        "subsets": _subset_metrics(val_bundle, y_val, y_pred),
        "e_report": _e_subset_report(y_val, y_pred),
        "stratified_omega": stratified_omega_report(y_val, y_pred, snr=_snr_for_rows(val_bundle)),
        "e_head": norm_stats.get("e_head", "direct"),
        "gate_threshold": norm_stats.get("gate_threshold"),
        "targets": norm_stats.get("targets", "theta"),
    }
    true_zero = y_val[:, 2] <= 0.0
    has_ecc_val = bundle.has_ecc[val_idx].astype(bool)
    if isinstance(model, DualEHead):
        x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float64)
        x_std = np.asarray(norm_stats["x_std"], dtype=np.float64)
        thr = float(norm_stats.get("gate_threshold", 0.5))
        with torch.no_grad():
            xt = torch.from_numpy(((X_val - x_mean) / x_std).astype(np.float32)).to(device)
            logits = model.gate(xt).squeeze(-1).cpu().numpy()
        pred_zero = _sigmoid_np(logits) < thr
        metrics["e_zero_classifier"] = _zero_class_metrics(true_zero, pred_zero, has_ecc_val)
    else:
        pred_zero = y_pred[:, 2] <= 1e-3
        metrics["e_zero_classifier"] = _zero_class_metrics(true_zero, pred_zero, has_ecc_val)
    return y_val, y_pred, metrics


def _write_benchmark(benchmark: dict, path: Path) -> None:
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(benchmark, indent=2, default=_default))
    print(f"saved benchmark -> {path}")


def _parse_loss_weights(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError("--loss-weights must have 4 (hk) or 5 (theta) comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def _theta_train_kwargs(args: argparse.Namespace) -> dict:
    return {
        "mask_omega": not args.no_mask_omega,
        "hard_omega_mask": not args.soft_omega_mask,
        "circular_omega": not args.no_circular_omega,
        "constrain_e": not args.no_constrain_e,
        "constrain_omega": not args.no_constrain_omega,
        "e_head": args.e_head,
        "e_balance": args.e_balance,
        "hurdle_bce_weight": args.hurdle_bce_weight,
        "gate_threshold": args.gate_threshold,
        "targets": args.targets,
    }


def _predict_kwargs(args: argparse.Namespace) -> dict:
    return {
        "constrain_e": not args.no_constrain_e,
        "constrain_omega": not args.no_constrain_omega,
    }


PERIOD_TOLERANCE_FRACS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def run_period_tolerance(
    args: argparse.Namespace,
    device: torch.device,
    *,
    rel_fracs: tuple[float, ...] = PERIOD_TOLERANCE_FRACS,
) -> dict:
    """Omega MAE vs fold-period error on an oracle checkpoint."""
    csv_path = args.csv if args.csv != DEFAULT_CSV else PHASEFOLD_CSV
    if args.feature_set not in ("35", "109"):
        raise ValueError("--period-tolerance requires --feature-set 35 or 109")
    if not csv_path.exists():
        raise FileNotFoundError(f"period-tolerance requires {csv_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    out_dir = args.diagnose_out or (args.out / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_kw = _predict_kwargs(args)

    bundle = load_from_csv(csv_path, args.feature_set, max_rows=getattr(args, "max_rows", None))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, norm_stats = build_model_from_checkpoint(ckpt, device)

    _, val_idx = _val_split_indices(len(bundle.X), args.val_frac, args.seed)
    y_true = bundle.y[val_idx]
    row_idx = bundle.row_idx[val_idx]
    true_log10_P = y_true[:, 0]
    rng = np.random.default_rng(args.seed + 31)
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(val_idx))

    curve: list[dict] = []
    print("=" * 60)
    print("Period tolerance (fold P = true P × (1±eps))")
    for frac in rel_fracs:
        if frac == 0.0:
            fold_log10_P = true_log10_P
        else:
            fold_P = (10 ** true_log10_P) * (1.0 + signs * frac)
            fold_log10_P = np.log10(np.clip(fold_P, 1e-6, None))
        phase_block = recompute_phasefold_block(
            row_idx, fold_log10_P, seed=CSV_SEED, n_samples=len(bundle.df)
        )
        X_val = replace_phase_features(bundle.X[val_idx], args.feature_set, phase_block)
        y_pred = predict(model, X_val, norm_stats, device, **pred_kw)
        mask = y_true[:, 2] > 0.1
        mae = _omega_mae_deg(y_true[mask], y_pred[mask]) if mask.any() else float("nan")
        r2s = [_r2(y_true[mask, j], y_pred[mask, j]) for j in (3, 4)] if mask.any() else []
        entry = {
            "rel_period_error": float(frac),
            "n_e_gt_0.1": int(mask.sum()),
            "omega_mae_deg": float(mae),
            "omega_r2_mean": float(np.nanmean(r2s)) if r2s else float("nan"),
            "e_r2": float(_r2(y_true[:, 2], y_pred[:, 2])),
        }
        curve.append(entry)
        print(
            f"  eps={frac * 100:5.1f}%  omega MAE={mae:6.1f} deg  "
            f"omega R2={entry['omega_r2_mean']:.3f}  e R2={entry['e_r2']:.3f}"
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "csv": str(csv_path),
        "feature_set": args.feature_set,
        "curve": curve,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [100.0 * c["rel_period_error"] for c in curve]
    ys = [c["omega_mae_deg"] for c in curve]
    ax.plot(xs, ys, "o-", lw=1.5)
    ax.set_xlabel("relative period error [%]")
    ax.set_ylabel(r"$\omega$ MAE [deg] ($e>0.1$)")
    ax.set_title("Phase-fold omega vs period error")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "period_tolerance_omega.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {plot_path}")

    out_json = out_dir / "period_tolerance.json"
    _write_benchmark(report, out_json)
    return report


def run_e_head_ablate(args: argparse.Namespace, device: torch.device) -> dict:
    """Train direct / balance / hurdle / dual on 109-D; promote best by e>0 R2 + F1."""
    csv_path = args.csv if args.csv != DEFAULT_CSV else PHASEFOLD_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"e-head ablate requires {csv_path}")

    variants = [
        ("baseline", {"e_head": "direct", "e_balance": False}),
        ("e_balance", {"e_head": "direct", "e_balance": True}),
        ("hurdle", {"e_head": "hurdle", "e_balance": False}),
        ("hurdle_e_balance", {"e_head": "hurdle", "e_balance": True}),
        ("dual", {"e_head": "dual", "e_balance": False}),
        ("dual_e_balance", {"e_head": "dual", "e_balance": True}),
    ]
    base_train = {
        k: v
        for k, v in _theta_train_kwargs(args).items()
        if k not in ("e_head", "e_balance")
    }
    bundle = load_from_csv(csv_path, "109", max_rows=getattr(args, "max_rows", None))
    results: dict[str, dict] = {}
    preds_by_name: dict[str, dict] = {}
    metrics_by_name: dict[str, dict] = {}
    out_root = args.out / "e_head_ablate"
    out_root.mkdir(parents=True, exist_ok=True)

    for name, overrides in variants:
        print("=" * 60)
        print(f"e-head ablation: {name}")
        train_kw = {**base_train, **overrides}
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = Path("checkpoints") / f"regression_mlp_109_{name}.pt"
        _, preds, metrics = train_model(
            bundle,
            feature_set="109",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            seed=args.seed,
            device=device,
            patience=args.patience,
            target_norm=not args.no_target_norm,
            loss_weights=_parse_loss_weights(args.loss_weights),
            checkpoint_path=ckpt,
            **train_kw,
        )
        er = metrics["e_report"]
        clf = metrics.get("e_zero_classifier") or {}
        forced = _frac_forced_omega0(preds["y_true"], preds["y_pred"])
        er["frac_forced_omega0_e_gt_0.1"] = forced
        metrics["frac_forced_omega0_e_gt_0.1"] = forced
        slim = {
            "per_target": metrics["per_target"],
            "subsets": metrics.get("subsets", {}),
            "e_report": er,
            "e_zero_classifier": metrics.get("e_zero_classifier"),
            "gate_threshold": metrics.get("gate_threshold"),
            "val_mse": metrics.get("val_mse"),
            "frac_forced_omega0_e_gt_0.1": forced,
            "stratified_omega": metrics.get("stratified_omega"),
            "targets": metrics.get("targets", getattr(args, "targets", "theta")),
        }
        results[name] = slim
        preds_by_name[name] = preds
        metrics_by_name[name] = metrics
        _write_benchmark(slim, out_dir / "metrics.json")
        plot_pred_vs_true(preds["y_true"], preds["y_pred"], out_dir / "pred_vs_true.png", metrics)
        print(
            f"  e R2={metrics['per_target']['e']['r2']:.3f}  "
            f"e>0 R2={er['e_gt_0']['r2']:.3f}  "
            f"f1_zero={clf.get('f1_zero', float('nan')):.3f}  "
            f"forced_omega0={forced:.3f}  "
            f"P R2={metrics['per_target']['log10_P']['r2']:.3f}  "
            f"omega MAE={_omega_mae_e_gt(metrics):.1f}"
        )

    def _ablate_score(name: str) -> float:
        er = results[name].get("e_report") or {}
        clf = results[name].get("e_zero_classifier") or {}
        e_pos = er.get("e_gt_0", {}).get("r2", float("nan"))
        f1 = clf.get("f1_zero", float("nan"))
        forced = results[name].get("frac_forced_omega0_e_gt_0.1", float("nan"))
        e_pos = 0.0 if not np.isfinite(e_pos) else float(e_pos)
        f1 = 0.0 if not np.isfinite(f1) else float(f1)
        forced = 0.0 if not np.isfinite(forced) else float(forced)
        return e_pos + f1 - 1.5 * forced

    best_name = max(results, key=_ablate_score)
    best_preds = preds_by_name[best_name]
    best_metrics = metrics_by_name[best_name]

    main_out = args.out
    main_out.mkdir(parents=True, exist_ok=True)
    plot_pred_vs_true(
        best_preds["y_true"], best_preds["y_pred"], main_out / "pred_vs_true.png", best_metrics
    )
    plot_all_targets_individual(
        best_preds["y_true"], best_preds["y_pred"], main_out, metrics=best_metrics
    )
    plot_omega_diagnostics(best_preds["y_true"], best_preds["y_pred"], main_out)
    try:
        from regression_diagnostics import plot_omega_vs_e, plot_parameter_pair_grid

        pair_out = main_out / "diagnostics"
        plot_omega_vs_e(best_preds["y_true"], best_preds["y_pred"], pair_out)
        plot_parameter_pair_grid(best_preds["y_true"], best_preds["y_pred"], pair_out)
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: diagnostics plots skipped ({exc})")

    best_ckpt_src = Path("checkpoints") / f"regression_mlp_109_{best_name}.pt"
    best_ckpt_dst = Path("checkpoints") / "regression_mlp_109_best_e.pt"
    if best_ckpt_src.exists():
        import shutil

        shutil.copy2(best_ckpt_src, best_ckpt_dst)

    summary = {
        "csv": str(csv_path),
        "targets": getattr(args, "targets", "theta"),
        "best_variant": best_name,
        "best_score": _ablate_score(best_name),
        "variants": {
            name: {
                "e_r2": results[name]["per_target"]["e"]["r2"],
                "e_gt_0_r2": (results[name].get("e_report") or {}).get("e_gt_0", {}).get("r2"),
                "e_gt_0_mae": (results[name].get("e_report") or {}).get("e_gt_0", {}).get("mae"),
                "f1_zero": (results[name].get("e_zero_classifier") or {}).get("f1_zero"),
                "recall_zero": (results[name].get("e_zero_classifier") or {}).get("recall_zero"),
                "frac_forced_omega0_e_gt_0.1": results[name].get("frac_forced_omega0_e_gt_0.1"),
                "log10_P_r2": results[name]["per_target"]["log10_P"]["r2"],
                "log10_K_r2": results[name]["per_target"]["log10_K"]["r2"],
                "omega_mae_e_gt_0.1": _omega_mae_e_gt(results[name]),
                "score": _ablate_score(name),
                "e_zero_classifier": results[name].get("e_zero_classifier"),
                "stratified_omega": results[name].get("stratified_omega"),
            }
            for name in results
        },
        "best_stratified_omega": results[best_name].get("stratified_omega"),
        "note": "winner = argmax(e>0 R2 + zero F1 - 1.5*forced_omega0); main figures overwritten with winner",
    }
    _write_benchmark(best_metrics, main_out / "metrics.json")
    _write_benchmark(summary, out_root / "comparison.json")
    print("=" * 60)
    print("e-head ablation summary")
    for name, row in summary["variants"].items():
        mark = " <-- best" if name == best_name else ""
        f1v = row["f1_zero"]
        f1s = f"{float(f1v):.3f}" if f1v is not None and np.isfinite(f1v) else "nan"
        epos = row["e_gt_0_r2"]
        eposs = f"{float(epos):.3f}" if epos is not None and np.isfinite(epos) else "nan"
        print(
            f"  {name:20s}  e R2={row['e_r2']:.3f}  e>0 R2={eposs}  "
            f"f1_zero={f1s}  "
            f"P R2={row['log10_P_r2']:.3f}  score={row['score']:.3f}{mark}"
        )
    print(f"promoted {best_name} -> {main_out / 'pred_vs_true.png'} and {best_ckpt_dst}")
    print(f"winner stratified omega ({best_name}):")
    _print_stratified_omega(results[best_name].get("stratified_omega") or {})
    return summary



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    src.add_argument("--data-dir", type=Path, help="NPZ corpus (74-D only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET)
    p.add_argument("--fold-period", choices=("oracle", "predicted"), default="oracle")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--loss-weights", default="1,1,1,1,1")
    p.add_argument(
        "--loss-weights-ecc",
        action="store_true",
        help="use 1,1,5,5,5 loss weights for e and omega",
    )
    p.add_argument("--no-target-norm", action="store_true")
    p.add_argument("--no-mask-omega", action="store_true", help="disable omega masking on low-e rows")
    p.add_argument(
        "--soft-omega-mask",
        action="store_true",
        help="sigmoid down-weight for omega on low-e (default: hard zero below 0.05)",
    )
    p.add_argument(
        "--no-circular-omega",
        action="store_true",
        help="use MSE on cos/sin instead of circular 1-cos(delta omega) loss",
    )
    p.add_argument(
        "--e-head",
        choices=("direct", "hurdle", "dual"),
        default="direct",
        help=(
            "direct: single MLP; "
            "hurdle: shared MLP + e>0 classifier; "
            "dual: separate MLPs for e=0 and e≠0 plus a gate"
        ),
    )
    p.add_argument(
        "--targets",
        choices=("theta", "hk"),
        default="theta",
        help="theta: e,cosω,sinω (default); hk: k=e cosω, h=e sinω then decode",
    )
    p.add_argument(
        "--e-balance",
        action="store_true",
        help="inverse-frequency reweighting of the e loss (counters the zero-inflated e prior)",
    )
    p.add_argument(
        "--hurdle-bce-weight",
        type=float,
        default=1.0,
        help="weight of the e>0 classifier BCE term (hurdle / dual gate)",
    )
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help=(
            "P(e>0) decision threshold for --e-head dual "
            "(default: auto-select on val to max zero-class F1)"
        ),
    )
    p.add_argument("--no-constrain-e", action="store_true", help="do not clip e to [0, 0.99] at predict")
    p.add_argument("--no-constrain-omega", action="store_true", help="do not L2-normalize cos/sin omega at predict")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--combined-target", default="log10_K", choices=THETA_NAMES)
    p.add_argument("--real-split", default="all")
    p.add_argument("--benchmark-gates", action="store_true", help="run Gates A/B/C and write benchmark.json")
    p.add_argument("--run-optional-ablations", action="store_true")
    p.add_argument("--generate-50k", action="store_true", help="use 50K CSV if present for optional ablation")
    p.add_argument(
        "--two-step",
        action="store_true",
        help="train 74-D + 109-D and evaluate two-step P-then-shape pipeline",
    )
    p.add_argument(
        "--stage2-fold",
        choices=("oracle", "predicted", "jitter"),
        default="predicted",
        help="train stage-2 on oracle / predicted / jittered folds",
    )
    p.add_argument(
        "--period-source",
        choices=("mlp74", "lsp_peak", "hybrid"),
        default="lsp_peak",
        help="which period to fold at (default: lsp_peak)",
    )
    p.add_argument(
        "--stage2-p-jitter",
        action="store_true",
        help="add residual noise to stage-2 fold periods on the train set",
    )
    p.add_argument(
        "--period-tolerance",
        action="store_true",
        help="plot omega MAE vs fold-period error",
    )
    p.add_argument(
        "--e-head-ablate",
        action="store_true",
        help="compare e-head variants on 109-D",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap CSV rows (debug)",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="load checkpoint and regenerate pred-vs-true plots (no training)",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="run regression diagnostic harness (SNR, P/baseline, LSP, sanity JSON)",
    )
    p.add_argument(
        "--diagnose-out",
        type=Path,
        default=None,
        help="output dir for --diagnose (default: figures/regression_synthetic/diagnostics)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.loss_weights_ecc:
        args.loss_weights = "1,1,5,5,5"

    if args.period_tolerance:
        if args.checkpoint == DEFAULT_CHECKPOINT:
            args.checkpoint = CHECKPOINT_109
        if args.feature_set == DEFAULT_FEATURE_SET:
            args.feature_set = "109"
        if args.csv == DEFAULT_CSV:
            args.csv = PHASEFOLD_CSV
        run_period_tolerance(args, device)
        return

    if args.e_head_ablate:
        if args.csv == DEFAULT_CSV:
            args.csv = PHASEFOLD_CSV
        run_e_head_ablate(args, device)
        return

    if args.benchmark_gates:
        run_benchmark_gates(args, device)
        return

    if args.two_step:
        run_two_step_pipeline(args, device)
        return

    if args.diagnose:
        if args.data_dir is not None:
            raise ValueError("--diagnose requires --csv, not --data-dir")
        from regression_diagnostics import run_regression_diagnostics

        diag_out = args.diagnose_out or (args.out / "diagnostics")
        run_regression_diagnostics(
            csv_path=args.csv,
            checkpoint_path=args.checkpoint,
            feature_set=args.feature_set,
            out_dir=diag_out,
            val_frac=args.val_frac,
            seed=args.seed,
            device=device,
        )
        return

    if args.plot_only:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        if args.data_dir is not None:
            raise ValueError("--plot-only requires --csv, not --data-dir")
        bundle = load_from_csv(args.csv, args.feature_set)
        pred_kw = _predict_kwargs(args)
        y_true, y_pred, metrics = load_checkpoint_and_predict_val(
            bundle,
            args.checkpoint,
            val_frac=args.val_frac,
            seed=args.seed,
            device=device,
            **pred_kw,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        plot_pred_vs_true(y_true, y_pred, args.out / "pred_vs_true.png", metrics)
        plot_all_targets_individual(y_true, y_pred, args.out, metrics=metrics)
        e_mask = y_true[:, 2] > OMEGA_EVAL_E_MIN
        if e_mask.any():
            plot_e_scatter(
                y_true[e_mask],
                y_pred[e_mask],
                args.out / "pred_vs_true_e_gt_0.1.png",
                f"e (e>{OMEGA_EVAL_E_MIN:g})",
            )
        plot_omega_diagnostics(y_true, y_pred, args.out)
        from regression_diagnostics import plot_omega_vs_e, plot_parameter_pair_grid

        pair_out = args.out / "diagnostics"
        plot_omega_vs_e(y_true, y_pred, pair_out)
        plot_parameter_pair_grid(y_true, y_pred, pair_out)
        print("per-target R2 (omega on e>0.1):")
        for name in THETA_NAMES:
            if name in ("cos_omega", "sin_omega"):
                continue
            print(f"  {name:12s}  R2={metrics['per_target'][name]['r2']:.3f}")
        _print_e_headline(metrics)
        _print_omega_headline(metrics)
        strat = metrics.get("stratified_omega")
        if strat is None:
            strat = stratified_omega_report(y_true, y_pred, snr=_snr_for_rows(bundle))
            metrics["stratified_omega"] = strat
        _print_stratified_omega(strat)
        omega_sub = metrics.get("subsets", {}).get("e_gt_0.1", {}).get("per_target", {})
        baseline = {
            "e_min": OMEGA_EVAL_E_MIN,
            "n_e_gt_0.1": int(e_mask.sum()),
            "n_val": int(len(y_true)),
            "cos_omega_r2": omega_sub.get("cos_omega", {}).get("r2"),
            "sin_omega_r2": omega_sub.get("sin_omega", {}).get("r2"),
            "omega_mae_deg": omega_sub.get("omega_angular", {}).get("mae_deg"),
            "stratified_omega": strat,
            "full_sample": {
                "cos_omega_r2": metrics["per_target"]["cos_omega"]["r2"],
                "sin_omega_r2": metrics["per_target"]["sin_omega"]["r2"],
                "omega_mae_deg": metrics["per_target"].get("omega_angular", {}).get("mae_deg"),
            },
            "checkpoint": str(args.checkpoint),
            "csv": str(args.csv),
        }
        pair_out.mkdir(parents=True, exist_ok=True)
        _write_benchmark(baseline, pair_out / "omega_e_gt_0.1.json")
        return

    if args.data_dir is not None:
        print(f"loading NPZ corpus from {args.data_dir} ...")
        bundle = load_from_npz(args.data_dir)
        if args.feature_set != "74":
            raise ValueError("NPZ mode only supports --feature-set 74")
    else:
        print(f"loading CSV from {args.csv} (feature-set={args.feature_set}) ...")
        bundle = load_from_csv(args.csv, args.feature_set, max_rows=args.max_rows)

    print(f"dataset: {len(bundle.X):,} samples, {bundle.X.shape[1]} features -> {bundle.y.shape[1]} targets")

    ckpt = None if args.no_checkpoint else args.checkpoint
    train_kw = _theta_train_kwargs(args)
    pred_kw = _predict_kwargs(args)
    model, preds, metrics = train_model(
        bundle,
        feature_set=args.feature_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
        target_norm=not args.no_target_norm,
        loss_weights=_parse_loss_weights(args.loss_weights),
        checkpoint_path=ckpt,
        **train_kw,
    )

    if args.fold_period == "predicted" and args.feature_set in ("35", "109"):
        print("evaluating with predicted-P phase fold (Gate C style) ...")
        gate_c = eval_predicted_p_fold(
            model,
            bundle,
            preds,
            metrics["norm_stats"],
            feature_set=args.feature_set,
            seed=CSV_SEED,
            device=device,
            **_predict_kwargs(args),
        )
        metrics["predicted_p_fold_eval"] = {
            k: v for k, v in gate_c.items() if k not in ("y_true", "y_pred")
        }
        preds["y_pred"] = gate_c["y_pred"]

    args.out.mkdir(parents=True, exist_ok=True)
    plot_pred_vs_true(preds["y_true"], preds["y_pred"], args.out / "pred_vs_true.png", metrics)
    plot_all_targets_individual(preds["y_true"], preds["y_pred"], args.out, metrics=metrics)
    plot_e_scatter(preds["y_true"], preds["y_pred"], args.out / "combined_scatter_e.png", f"e ({args.feature_set}-D)")
    e_mask = preds["y_true"][:, 2] > 0.1
    if e_mask.any():
        plot_e_scatter(
            preds["y_true"][e_mask],
            preds["y_pred"][e_mask],
            args.out / "pred_vs_true_e_gt_0.1.png",
            "e (e>0.1)",
        )
    plot_omega_diagnostics(preds["y_true"], preds["y_pred"], args.out)

    with_phasefold = args.feature_set in ("35", "109")
    print(f"collecting real reference systems (split={args.real_split}) ...")
    real_df = collect_real_summary(
        args.real_split,
        sigma_min=0.1,
        sigma_max=100.0,
        with_phasefold=with_phasefold,
    )
    print(f"real reference: {len(real_df)} systems")
    feature_cols = _feature_columns(args.feature_set)
    X_real = real_df[feature_cols].to_numpy(dtype=np.float64)
    y_real = real_df[THETA_NAMES].to_numpy(dtype=np.float64)
    valid_real = np.isfinite(X_real).all(axis=1) & np.isfinite(y_real).all(axis=1)
    X_real, y_real = X_real[valid_real], y_real[valid_real]
    n_excluded = int((~valid_real).sum())
    if n_excluded:
        print(f"real transfer: excluded {n_excluded} rows (missing phase-fold / non-finite)")

    if len(y_real) == 0:
        print("real transfer: no real systems with finite features; skipping transfer eval")
        metrics["real_transfer"] = {
            "n_real": 0,
            "n_excluded": n_excluded,
            "real_split": args.real_split,
            "note": "skipped: no real rows with finite features (phase-fold needs t_peri)",
        }
    else:
        y_pred_real = predict(model, X_real, metrics["norm_stats"], device, **pred_kw)
        real_bundle = DatasetBundle(
            X_real,
            y_real,
            row_idx=np.arange(len(y_real)),
            e=y_real[:, 2],
            has_t_peri=(
                real_df["has_t_peri"].to_numpy(dtype=float)[valid_real]
                if "has_t_peri" in real_df.columns
                else np.zeros(len(y_real))
            ),
            has_ecc=np.ones(len(y_real), dtype=bool),
            df=real_df,
        )
        metrics["real_transfer"] = {
            "n_real": int(len(y_real)),
            "n_excluded": n_excluded,
            "real_split": args.real_split,
            "mse": float(np.mean((y_pred_real - y_real) ** 2)),
            "per_target": _per_target_metrics(y_real, y_pred_real),
            "subsets": _subset_metrics(real_bundle, y_real, y_pred_real),
        }

        plot_combined_scatter(
            preds["y_true"],
            preds["y_pred"],
            y_real,
            y_pred_real,
            args.combined_target,
            args.out / f"combined_scatter_{args.combined_target}.png",
        )

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {metrics_path}")
    print(f"validation MSE: {metrics['val_mse']:.5f}")
    for name in THETA_NAMES:
        t = metrics["per_target"][name]
        print(f"  {name:12s}  R2={t['r2']:.3f}  MSE={t['mse']:.5f}")
    _print_e_headline(metrics)
    _print_omega_headline(metrics)
    if metrics.get("stratified_omega"):
        _print_stratified_omega(metrics["stratified_omega"])
    if "per_target" in metrics["real_transfer"]:
        rt = metrics["real_transfer"]["per_target"][args.combined_target]
        print(f"real transfer ({args.combined_target}): R2={rt['r2']:.3f}  MSE={rt['mse']:.5f}")


if __name__ == "__main__":
    main()
