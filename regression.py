"""
regression.py — MLP regression on Shuaib's 74-feature RV encoder.

Architecture
------------
    Raw RV time series  →  74-dim encoder (frozen)  →  MLP head  →  5 Kepler params

The encoder is the same feature set used by the real-vs-synthetic classifier:
64 spline/FFT spectral bins (time_series_features.py) plus 10 observation
summaries (validate_synthetic_dataset.py).

Targets (theta, 5-dim)
----------------------
    log10_P, log10_K, e, cos_omega, sin_omega

Data sources
------------
    Default: pre-generated CSV with features + labels
        synthetic_generation/datasets/synthetic_regression_10000.csv

    Optional: NPZ corpus from synthetic_rv.py — features computed on the fly
        data/synthetic/manifest.csv + synth_*.npz

Usage
-----
    python regression.py
    python regression.py --csv synthetic_generation/datasets/synthetic_regression_10000.csv
    python regression.py --data-dir data/synthetic --epochs 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import LSP_PERIODS, THETA_NAMES, compute_lsp
from time_series_features import spectral_feature_names, spectral_features
from validate_synthetic_dataset import OBSERVATION_SUMMARY_FEATURES

SPECTRAL_DIM = 64
SPECTRAL_GRID_SIZE = 1024
DEFAULT_CSV = Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv"
DEFAULT_OUT = Path("figures") / "regression_synthetic"

SPECTRAL_COLUMNS = spectral_feature_names(SPECTRAL_DIM)
FEATURE_COLUMNS = [*SPECTRAL_COLUMNS, *OBSERVATION_SUMMARY_FEATURES]
TARGET_LABELS = {
    "log10_P": r"$\log_{10} P$",
    "log10_K": r"$\log_{10} K$",
    "e": r"$e$",
    "cos_omega": r"$\cos\omega$",
    "sin_omega": r"$\sin\omega$",
}


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


def load_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed features and targets from the synthetic regression CSV."""
    df = pd.read_csv(csv_path)
    missing = [c for c in [*THETA_NAMES, *FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[THETA_NAMES].to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"[load_csv] dropped {n_drop} rows with non-finite values")
    return X[valid], y[valid]


def load_from_npz(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NPZ corpus and encode each system with encode_rv."""
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, row in manifest.iterrows():
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

    if not X_list:
        raise RuntimeError(f"no valid samples loaded from {data_dir}")
    return np.stack(X_list), np.stack(y_list)


class RegressionHead(nn.Module):
    """MLP head on top of the frozen 74-dim encoder features."""

    def __init__(self, in_dim: int = 74, hidden: tuple[int, ...] = (128, 64), out_dim: int = 5):
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


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    device: torch.device,
    patience: int = 30,
) -> tuple[RegressionHead, dict[str, np.ndarray], dict]:
    """Train the MLP and return the model, predictions, and metrics."""
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

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = RegressionHead().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val_n).float().to(device)).cpu().numpy()
        val_loss = float(np.mean((val_pred - y_val) ** 2))

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
        val_pred = model(torch.from_numpy(X_val_n).float().to(device)).cpu().numpy()

    metrics: dict = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_mse": float(np.mean((val_pred - y_val) ** 2)),
        "per_target": {},
    }
    for j, name in enumerate(THETA_NAMES):
        metrics["per_target"][name] = {
            "mse": float(np.mean((val_pred[:, j] - y_val[:, j]) ** 2)),
            "r2": _r2(y_val[:, j], val_pred[:, j]),
        }

    norm_stats = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist()}
    return model, {"y_true": y_val, "y_pred": val_pred}, {**metrics, "norm_stats": norm_stats}


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, metrics: dict) -> None:
    """Save a 1x5 scatter panel of true vs predicted orbital parameters."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))

    for j, (ax, name) in enumerate(zip(axes, THETA_NAMES)):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ax.scatter(yt, yp, s=8, alpha=0.45, edgecolors="none")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        lo -= pad
        hi += pad
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="precomputed features + targets CSV (default: 10K synthetic set)",
    )
    src.add_argument(
        "--data-dir",
        type=Path,
        help="NPZ corpus directory (manifest.csv + synth_*.npz); encodes features on the fly",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.data_dir is not None:
        print(f"loading NPZ corpus from {args.data_dir} …")
        X, y = load_from_npz(args.data_dir)
    else:
        print(f"loading CSV from {args.csv} …")
        X, y = load_from_csv(args.csv)

    print(f"dataset: {len(X):,} samples, {X.shape[1]} features -> {y.shape[1]} targets")

    model, preds, metrics = train_model(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    plot_path = args.out / "pred_vs_true.png"
    plot_pred_vs_true(preds["y_true"], preds["y_pred"], plot_path, metrics)

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {metrics_path}")
    print(f"validation MSE: {metrics['val_mse']:.5f}")
    for name in THETA_NAMES:
        t = metrics["per_target"][name]
        print(f"  {name:12s}  R²={t['r2']:.3f}  MSE={t['mse']:.5f}")


if __name__ == "__main__":
    main()
