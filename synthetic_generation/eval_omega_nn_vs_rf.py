"""
Does the RVEncoder NN recover omega better than the random forest?

Both models are evaluated on the *same* real test systems, in physical
(cos_omega, sin_omega) space (R^2 is affine-invariant, so this matches the
normalized space the encoder trains in). We report three nested subsets:

    all       - every valid single-planet test system
    has_ecc   - systems with a measured catalog eccentricity (omega defined)
    e>0.1     - has_ecc AND e>0.1, where omega is actually identifiable
                (for near-circular orbits omega is degenerate with t_peri, so
                 no model can recover it -- see the omega analysis in handover)

RF: trained on the 10k synthetic CSV (spectral64+summary), predicting all five
targets, then applied to the real test features (identical feature construction
as plot_synthetic_regression_csv.collect_real_summary).

Encoder: a trained RVEncoder checkpoint, run on the normalized (x, lsp) of the
same systems; its normalized output is un-normalized to physical theta.

NOTE: the shipped checkpoints are epoch-1 smoke artifacts (best == last); a
converged encoder is required before this comparison is conclusive.

Usage
-----
    python synthetic_generation/eval_omega_nn_vs_rf.py
    python synthetic_generation/eval_omega_nn_vs_rf.py --checkpoint checkpoints/finetune_best.pt --arch resnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from preprocess import RVDataset, THETA_NAMES
from feature_columns import (
    SPECTRAL_COLUMNS,
    SPECTRAL_DIM,
    SPECTRAL_GRID_SIZE,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
)
from time_series_features import spectral_features
from generate_synthetic_regression_csv import (
    _masked_observations,
)
from train_regression_models import _build

from models.encoder import build_encoder, un_normalise_theta

FEATURES = [*SPECTRAL_COLUMNS, *SUMMARY_COLUMNS]
OMEGA_DIMS = {"cos_omega": 3, "sin_omega": 4}


def _summary_row(xm, info, lsp) -> dict:
    from preprocess import LSP_PERIODS
    from feature_columns import PHASE_FOLD_COLUMNS, PHASE_FOLD_N_BINS
    from time_series_features import phase_fold_features

    rv_std = float(info["rv_std_ms"])
    sigma = xm[2] * rv_std
    rv_ms = xm[1] * rv_std
    t_days = xm[0] * float(info["t_span_days"])
    gaps = np.diff(np.sort(t_days))
    spectral = spectral_features(xm[0], xm[1], d=SPECTRAL_DIM, grid_size=SPECTRAL_GRID_SIZE)
    row = {
        "n_obs": int(info["n_obs"]),
        "baseline_d": float(info["t_span_days"]),
        "rv_std_ms": rv_std,
        "rv_iqr_ms": float(np.subtract(*np.percentile(rv_ms, [75, 25]))),
        "median_sigma_ms": float(np.median(sigma)),
        "sigma_iqr_ms": float(np.subtract(*np.percentile(sigma, [75, 25]))),
        "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
        "lsp_peak_power": float(np.max(lsp)),
        "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
        "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
    }
    row.update({n: float(v) for n, v in zip(SPECTRAL_COLUMNS, spectral)})
    # Epoch-free phase-fold at LSP peak P (matches phasefold_epochfree CSV).
    # Extra keys are ignored by RF FEATURES selection; used by 109-D MLP.
    P_fold = float(row["lsp_peak_period_d"])
    if P_fold > 0 and np.isfinite(P_fold):
        phase = phase_fold_features(
            t_days,
            rv_ms,
            P_fold,
            n_bins=PHASE_FOLD_N_BINS,
            epoch_free=True,
        )
        row.update({n: float(v) for n, v in zip(PHASE_FOLD_COLUMNS, phase)})
        row["has_t_peri"] = 1.0
    return row


def collect_test(real_split: str, sigma_min: float, sigma_max: float):
    """Aligned real systems: RF features, encoder inputs, true theta, e, has_ecc."""
    ds_raw = RVDataset(real_split, normalize=False, single_planet=True)
    ds_norm = RVDataset(real_split, normalize=True, single_planet=True)

    feat_rows, enc_inputs, y_true, e_vals, has_ecc = [], [], [], [], []
    for i in range(len(ds_raw)):
        x, lsp, theta, info = ds_raw.get_numpy(i)
        if not info.get("valid", True):
            continue
        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
        if not (sigma_min <= med_sigma <= sigma_max):
            continue

        feat_rows.append(_summary_row(xm, info, lsp))
        y_true.append([float(theta[k]) for k in range(5)])
        e_vals.append(float(theta[2]))
        has_ecc.append(bool(info.get("has_ecc", False)))

        xn, lspn, _, _ = ds_norm.get_numpy(i)
        enc_inputs.append((xn, lspn))

    X = pd.DataFrame(feat_rows, columns=FEATURES).to_numpy(dtype=float)
    return X, enc_inputs, np.asarray(y_true), np.asarray(e_vals), np.asarray(has_ecc)


def encoder_predict(checkpoint: Path, arch: str, enc_inputs) -> np.ndarray:
    stats = json.loads(Path("data/dataset_stats.json").read_text())
    enc = build_encoder(arch)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    enc.load_state_dict(state)
    enc.eval()
    preds = []
    with torch.no_grad():
        for xn, lspn in enc_inputs:
            xt = torch.from_numpy(xn).unsqueeze(0)
            lt = torch.from_numpy(lspn).unsqueeze(0)
            out = enc(xt, lt)
            phys = un_normalise_theta(out, stats)[0].numpy()
            preds.append(phys)
    return np.asarray(preds)


def _metrics(y_true, y_pred, mask, dims):
    out = {}
    yt, yp = y_true[mask], y_pred[mask]
    for name, j in dims.items():
        if len(yt) >= 3:
            out[name] = {
                "r2": float(r2_score(yt[:, j], yp[:, j])),
                "mae": float(mean_absolute_error(yt[:, j], yp[:, j])),
            }
        else:
            out[name] = {"r2": float("nan"), "mae": float("nan")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetune_best.pt"))
    ap.add_argument("--arch", type=str, default="resnet")
    ap.add_argument("--csv", type=Path,
                    default=Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv")
    ap.add_argument("--real-split", default="test", choices=("all", "train", "val", "test"))
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=100.0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_real, enc_inputs, y_true, e_vals, has_ecc = collect_test(
        args.real_split, args.sigma_min, args.sigma_max
    )
    n = len(y_true)
    print(f"real {args.real_split}: {n} systems, has_ecc={int(has_ecc.sum())}, "
          f"e>0.1={int(((e_vals > 0.1) & has_ecc).sum())}")

    # RF trained on synthetic, applied to real test.
    df = pd.read_csv(args.csv)
    Xs = df[FEATURES].to_numpy(dtype=float)
    ys = df[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    rf = _build("separate", args.n_estimators, args.seed, list(TARGET_COLUMNS))
    rf.fit(Xs, ys)
    y_rf = rf.predict(X_real)

    # Encoder.
    if args.checkpoint.exists():
        y_nn = encoder_predict(args.checkpoint, args.arch, enc_inputs)
    else:
        print(f"[warn] checkpoint {args.checkpoint} not found; skipping encoder")
        y_nn = None

    subsets = {
        "all": np.ones(n, bool),
        "has_ecc": has_ecc,
        "e>0.1": has_ecc & (e_vals > 0.1),
    }

    print("\nomega recovery: RF (synthetic-trained) vs NN encoder")
    print("=" * 62)
    for sub, mask in subsets.items():
        print(f"\n[{sub}]  n={int(mask.sum())}")
        rf_m = _metrics(y_true, y_rf, mask, OMEGA_DIMS)
        line = f"  {'param':<12}{'RF R2':>10}{'RF MAE':>10}"
        if y_nn is not None:
            line += f"{'NN R2':>10}{'NN MAE':>10}"
        print(line)
        for name in OMEGA_DIMS:
            row = f"  {name:<12}{rf_m[name]['r2']:>+10.3f}{rf_m[name]['mae']:>10.3f}"
            if y_nn is not None:
                nn_m = _metrics(y_true, y_nn, mask, OMEGA_DIMS)
                row += f"{nn_m[name]['r2']:>+10.3f}{nn_m[name]['mae']:>10.3f}"
            print(row)

    print(f"\ncheckpoint: {args.checkpoint} (arch={args.arch})")
    if args.checkpoint.exists():
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        print(f"  epoch={ck.get('epoch')}  (epoch 1 = untrained smoke artifact)")


if __name__ == "__main__":
    main()
