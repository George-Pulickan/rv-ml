"""
regression.py — MLP regression on RV encoder features (74 / 35 / 109-D).

Architecture
------------
    Feature vector  ->  MLP head  ->  5 Kepler params

Feature sets (--feature-set)
----------------------------
    74   spectral (64) + observation summaries (10)  [default]
    35   phase-fold bins + shape scalars only (Gate A sanity)
    109  74 + 35 (oracle or predicted-P phase fold) — recommended for e / omega

Targets (theta, 5-dim)
----------------------
    log10_P, log10_K, e, cos_omega, sin_omega

Usage
-----
    python regression.py
    python regression.py --feature-set 109 --csv synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv
    python regression.py --two-step --loss-weights-ecc
    python regression.py --benchmark-gates
"""

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
    apply_theta_constraints,
    regression_theta_loss,
    theta_loss_weights_numpy,
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


def load_from_csv(csv_path: Path, feature_set: str) -> DatasetBundle:
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

    return DatasetBundle(
        X[valid],
        y[valid],
        row_idx=np.arange(len(df), dtype=int)[valid],
        e=df["e"].to_numpy(dtype=float)[valid],
        has_t_peri=has_t_peri_col[valid],
        has_ecc=has_ecc[valid],
        df=df,
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
    if name == "log10_P":
        return float(np.mean(np.abs(10 ** y_pred[:, j] - 10 ** y_true[:, j])))
    if name == "log10_K":
        return float(np.mean(np.abs(10 ** y_pred[:, j] - 10 ** y_true[:, j])))
    if name in ("cos_omega", "sin_omega"):
        return None
    return float(np.mean(np.abs(y_pred[:, j] - y_true[:, j])))


def _subset_masks(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(bundle.y), dtype=bool),
        "has_ecc": bundle.has_ecc.astype(bool),
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


def predict(
    model: RegressionHead,
    X: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    *,
    denorm_targets: bool = True,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> np.ndarray:
    """Apply the trained MLP; optionally denormalize and project to physical ranges."""
    x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float64)
    x_std = np.asarray(norm_stats["x_std"], dtype=np.float64)
    X_n = (X - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_n).float().to(device)).cpu().numpy()
    if denorm_targets and norm_stats.get("y_mean") is not None:
        y_mean = np.asarray(norm_stats["y_mean"], dtype=np.float64)
        y_std = np.asarray(norm_stats["y_std"], dtype=np.float64)
        pred = pred * y_std + y_mean
    if constrain_e or constrain_omega:
        pred = apply_theta_constraints(pred, constrain_e=constrain_e, constrain_omega=constrain_omega)
    return pred


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
) -> tuple[RegressionHead, dict, dict]:
    """Train the MLP and return model, predictions, and metrics."""
    X, y = bundle.X, bundle.y
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

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

    if loss_weights is None:
        loss_weights = np.ones(y.shape[1], dtype=np.float64)
    dim_w = torch.from_numpy(loss_weights.astype(np.float32)).to(device)

    train_sample_w = theta_loss_weights_numpy(
        y_train,
        has_ecc=bundle.has_ecc[train_idx],
        mask_omega=mask_omega,
        hard_omega_mask=hard_omega_mask,
    )
    val_sample_w = theta_loss_weights_numpy(
        y_val,
        has_ecc=bundle.has_ecc[val_idx],
        mask_omega=mask_omega,
        hard_omega_mask=hard_omega_mask,
    )
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n).float(),
        torch.from_numpy(y_train_fit).float(),
        torch.from_numpy(train_sample_w.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    in_dim = X.shape[1]
    model = RegressionHead(in_dim=in_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = regression_theta_loss(
                pred,
                yb,
                wb,
                dim_w,
                y_mean=y_mean_t,
                y_std=y_std_t,
                circular_omega=circular_omega,
            )
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_pred_fit = model(torch.from_numpy(X_val_n).float().to(device))
            val_w = torch.from_numpy(val_sample_w.astype(np.float32)).to(device)
            val_loss_t = regression_theta_loss(
                val_pred_fit,
                torch.from_numpy(y_val_fit).float().to(device),
                val_w,
                dim_w,
                y_mean=y_mean_t,
                y_std=y_std_t,
                circular_omega=circular_omega,
            )
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

    model.eval()
    with torch.no_grad():
        val_pred_fit = model(torch.from_numpy(X_val_n).float().to(device)).cpu().numpy()
    val_pred = val_pred_fit * y_std + y_mean
    val_pred = apply_theta_constraints(val_pred, constrain_e=constrain_e, constrain_omega=constrain_omega)

    val_bundle = DatasetBundle(
        X_val,
        y_val,
        row_idx=bundle.row_idx[val_idx],
        e=bundle.e[val_idx],
        has_t_peri=bundle.has_t_peri[val_idx],
        has_ecc=bundle.has_ecc[val_idx],
        df=bundle.df,
    )

    norm_stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "target_norm": target_norm,
        "feature_set": feature_set,
        "in_dim": in_dim,
    }

    metrics: dict = {
        "feature_set": feature_set,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_mse": float(np.mean((val_pred - y_val) ** 2)),
        "per_target": _per_target_metrics(y_val, val_pred),
        "subsets": _subset_metrics(val_bundle, y_val, val_pred),
        "norm_stats": norm_stats,
        "loss_weights": loss_weights.tolist(),
        "mask_omega": mask_omega,
        "hard_omega_mask": hard_omega_mask,
        "circular_omega": circular_omega,
        "target_norm": target_norm,
    }

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "norm_stats": norm_stats}, checkpoint_path)
        print(f"saved checkpoint -> {checkpoint_path}")

    preds = {
        "y_true": y_val,
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


def plot_single_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    out_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Save one large true-vs-predicted scatter for a single target."""
    j = THETA_NAMES.index(target)
    yt, yp = y_true[:, j], y_pred[:, j]
    r2 = _r2(yt, yp)
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
    ax.set_title(title or f"{TARGET_LABELS[target]}  $R^2$={r2:.3f}  MSE={mse:.4f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, metrics: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))

    for j, (ax, name) in enumerate(zip(axes, THETA_NAMES)):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ax.scatter(yt, yp, s=8, alpha=0.45, edgecolors="none")
        lo, hi = _scatter_limits(yt, yp)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        r2 = metrics["per_target"][name]["r2"]
        ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.25)

    fig.suptitle("Synthetic regression: predicted vs true (validation split)")
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
) -> None:
    """Save one true-vs-predicted scatter per target."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in THETA_NAMES:
        plot_single_target(
            y_true,
            y_pred,
            name,
            out_dir / f"{prefix}_{name}.png",
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


def eval_two_step(
    bundle_109: DatasetBundle,
    model_74: RegressionHead,
    norm_74: dict,
    model_109: RegressionHead,
    norm_109: dict,
    preds_109: dict,
    device: torch.device,
    *,
    constrain_e: bool = True,
    constrain_omega: bool = True,
) -> dict:
    """
    Two-step deployment eval: P from 74-D model; re-fold at that P; e/omega from 109-D.

    preds_109 must contain X_val (109-D oracle layout), y_true, val_row_idx, val_idx.
    """
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

    phase_block = recompute_phasefold_block(
        val_row_idx, pred_log10_P, seed=CSV_SEED, n_samples=len(bundle_109.df)
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
    y_pred[:, 0] = pred_log10_P

    val_bundle = DatasetBundle(
        X_two_step,
        y_true,
        row_idx=val_row_idx,
        e=bundle_109.e[val_idx],
        has_t_peri=bundle_109.has_t_peri[val_idx],
        has_ecc=bundle_109.has_ecc[val_idx],
        df=bundle_109.df,
    )
    p_r2 = _r2(y_true[:, 0], pred_log10_P)
    return {
        "mode": "two_step",
        "p_stage": "74",
        "p_r2_stage1": float(p_r2),
        "val_mse": float(np.mean((y_pred - y_true) ** 2)),
        "per_target": _per_target_metrics(y_true, y_pred),
        "subsets": _subset_metrics(val_bundle, y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_two_step_pipeline(args: argparse.Namespace, device: torch.device) -> dict:
    """Train 74-D + 109-D oracle models and evaluate two-step vs single-shot Gate C."""
    csv_path = PHASEFOLD_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"two-step requires {csv_path}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    train_kw = _theta_train_kwargs(args)
    pred_kw = _predict_kwargs(args)
    loss_w = _parse_loss_weights(args.loss_weights)

    print("=" * 60)
    print("Two-step Stage 1: train 74-D (P/K baseline features)")
    bundle_74 = load_from_csv(csv_path, "74")
    model_74, _, metrics_74 = train_model(
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

    print("=" * 60)
    print("Two-step Stage 2: train 109-D (oracle phase-fold)")
    bundle_109 = load_from_csv(csv_path, "109")
    model_109, preds_109, metrics_109 = train_model(
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
        loss_weights=loss_w,
        checkpoint_path=CHECKPOINT_109,
        **train_kw,
    )
    print(f"  109-D oracle e R2={metrics_109['per_target']['e']['r2']:.3f}")
    _print_omega_headline(metrics_109)

    print("=" * 60)
    print("Gate C baseline: 109-D fold at P predicted by same 109-D model")
    gate_c = eval_predicted_p_fold(
        model_109,
        bundle_109,
        preds_109,
        metrics_109["norm_stats"],
        feature_set="109",
        seed=CSV_SEED,
        device=device,
        **pred_kw,
    )
    om_c = gate_c["subsets"]["e_gt_0.1_has_t_peri"]["per_target"]
    om_c_r2 = np.nanmean([om_c.get("cos_omega", {}).get("r2", float("nan")), om_c.get("sin_omega", {}).get("r2", float("nan"))])
    print(f"  Gate C omega R2 (e>0.1): {om_c_r2:.3f}")

    print("=" * 60)
    print("Two-step eval: P from 74-D, re-fold, e/omega from 109-D")
    two_step = eval_two_step(
        bundle_109,
        model_74,
        metrics_74["norm_stats"],
        model_109,
        metrics_109["norm_stats"],
        {**preds_109, "val_idx": preds_109["val_idx"]},
        device,
        **pred_kw,
    )
    om_ts = two_step["subsets"]["e_gt_0.1_has_t_peri"]["per_target"]
    om_ts_r2 = np.nanmean([om_ts.get("cos_omega", {}).get("r2", float("nan")), om_ts.get("sin_omega", {}).get("r2", float("nan"))])
    om_ts_mae = om_ts.get("omega_angular", {}).get("mae_deg", float("nan"))
    print(f"  two-step P R2 (stage1)={two_step['p_r2_stage1']:.3f}")
    print(f"  two-step e R2={two_step['per_target']['e']['r2']:.3f}")
    print(f"  two-step omega R2 (e>0.1)={om_ts_r2:.3f}  angular MAE={om_ts_mae:.1f} deg")
    _print_omega_headline({"subsets": two_step["subsets"]})

    report = {
        "csv": str(csv_path),
        "loss_weights": loss_w.tolist(),
        "circular_omega": not args.no_circular_omega,
        "hard_omega_mask": not args.soft_omega_mask,
        "stage1_74": metrics_74,
        "stage2_109_oracle": metrics_109,
        "gate_c_109_self_p": {k: v for k, v in gate_c.items() if k not in ("y_true", "y_pred")},
        "two_step_74_p_109_shape": {k: v for k, v in two_step.items() if k not in ("y_true", "y_pred")},
        "omega_r2_e_gt_0.1": {
            "oracle_109": float(np.nanmean([
                metrics_109["subsets"]["e_gt_0.1_has_t_peri"]["per_target"].get("cos_omega", {}).get("r2", float("nan")),
                metrics_109["subsets"]["e_gt_0.1_has_t_peri"]["per_target"].get("sin_omega", {}).get("r2", float("nan")),
            ])),
            "gate_c_self_p": float(om_c_r2),
            "two_step": float(om_ts_r2),
        },
    }
    if om_ts_r2 > 0.05 and om_ts_r2 > om_c_r2 + 0.05:
        report["story"] = (
            "Two-step inference (P from 74-D, then re-fold) improves omega vs folding at "
            "109-D self-predicted P. Deploy as stage-1 period + stage-2 shape."
        )
    elif om_ts_r2 > om_c_r2:
        report["story"] = (
            "Two-step slightly beats Gate C on omega R2 but both remain poor. Period error "
            "still scrambles phase-fold features; need better P or train stage-2 on predicted-P folds."
        )
    else:
        report["story"] = (
            "Two-step did not beat Gate C on omega. Period accuracy (R2~0.74) is insufficient "
            "for phase folding, and stage-2 was trained on oracle folds only. Next: 512-bin LSP, "
            "period refinement, or fine-tune 109-D on predicted-P phase features."
        )

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
    norm_stats = ckpt["norm_stats"]
    in_dim = int(norm_stats["in_dim"])
    model = RegressionHead(in_dim=in_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

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

    metrics = {
        "val_mse": float(np.mean((y_pred - y_val) ** 2)),
        "per_target": _per_target_metrics(y_val, y_pred),
        "subsets": _subset_metrics(
            DatasetBundle(
                X_val,
                y_val,
                row_idx=bundle.row_idx[val_idx],
                e=bundle.e[val_idx],
                has_t_peri=bundle.has_t_peri[val_idx],
                has_ecc=bundle.has_ecc[val_idx],
                df=bundle.df,
            ),
            y_val,
            y_pred,
        ),
    }
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
    if len(parts) != 5:
        raise ValueError("--loss-weights must have 5 comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def _theta_train_kwargs(args: argparse.Namespace) -> dict:
    return {
        "mask_omega": not args.no_mask_omega,
        "hard_omega_mask": not args.soft_omega_mask,
        "circular_omega": not args.no_circular_omega,
        "constrain_e": not args.no_constrain_e,
        "constrain_omega": not args.no_constrain_omega,
    }


def _predict_kwargs(args: argparse.Namespace) -> dict:
    return {
        "constrain_e": not args.no_constrain_e,
        "constrain_omega": not args.no_constrain_omega,
    }


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
        plot_all_targets_individual(y_true, y_pred, args.out)
        e_mask = y_true[:, 2] > 0.1
        if e_mask.any():
            plot_e_scatter(
                y_true[e_mask],
                y_pred[e_mask],
                args.out / "pred_vs_true_e_gt_0.1.png",
                "e (e>0.1)",
            )
        plot_omega_diagnostics(y_true, y_pred, args.out)
        print("per-target R2:")
        for name in THETA_NAMES:
            print(f"  {name:12s}  R2={metrics['per_target'][name]['r2']:.3f}")
        _print_omega_headline(metrics)
        return

    if args.data_dir is not None:
        print(f"loading NPZ corpus from {args.data_dir} ...")
        bundle = load_from_npz(args.data_dir)
        if args.feature_set != "74":
            raise ValueError("NPZ mode only supports --feature-set 74")
    else:
        print(f"loading CSV from {args.csv} (feature-set={args.feature_set}) ...")
        bundle = load_from_csv(args.csv, args.feature_set)

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
    plot_all_targets_individual(preds["y_true"], preds["y_pred"], args.out)
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
        # Real systems have no catalog t_peri, so phase-fold features are NaN
        # and every row is dropped for feature sets 35 / 109.
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
            has_t_peri=real_df.get("has_t_peri", pd.Series(0.0)).to_numpy(dtype=float)[valid_real],
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
    _print_omega_headline(metrics)
    if "per_target" in metrics["real_transfer"]:
        rt = metrics["real_transfer"]["per_target"][args.combined_target]
        print(f"real transfer ({args.combined_target}): R2={rt['r2']:.3f}  MSE={rt['mse']:.5f}")


if __name__ == "__main__":
    main()
