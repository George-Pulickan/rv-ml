"""
Spectral-resolution experiment: does the full 512-bin Lomb-Scargle power spectrum
recover orbital parameters better than the coarse 64-bin representation?

The 74-D RF baseline showed the 64-bin spectrum is uninformative (R^2 < 0) while
the 10 summaries carry the signal. This asks whether that is a *resolution*
limitation: the RVEncoder NN consumes the full 512-bin LSP. Here we compare, on
the same synthetic draws, five input feature sets:

    summary            - 10 observation summaries
    spectral64         - 64 coarse sum-normalized power bins
    lsp512             - full 512-bin Lomb-Scargle power spectrum
    spectral64+summary - the current CSV representation (74-D)
    lsp512+summary     - full spectrum + summaries (522-D)

For each we report 5-fold cross-validated per-target R^2/MAE/RMSE (separate RFs),
held-out test metrics, and synthetic-trained -> real transfer. For the best set
(lsp512+summary) we additionally fit the joint multi-output RF and produce
joint-vs-separate true-vs-predicted plots, directly comparable to the RVEncoder
NN's joint-vs-separate diagnostics.

A single RF config (max_features="sqrt") is used across all feature sets so the
64-vs-512 comparison is fair and the 512-D fits stay tractable.

Usage
-----
    python synthetic_generation/generate_lsp_regression_csv.py      # build the CSV first
    python synthetic_generation/lsp_resolution_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from preprocess import LSP_PERIODS, RVDataset
from time_series_features import spectral_features

from generate_synthetic_regression_csv import (
    SPECTRAL_COLUMNS,
    SPECTRAL_DIM,
    SPECTRAL_GRID_SIZE,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
    _masked_observations,
)
from generate_lsp_regression_csv import LSP_COLUMNS
from train_regression_models import (
    TARGET_LABELS,
    _build,
    cross_validate,
    per_target_metrics,
    plot_true_vs_pred,
)

MAX_FEATURES = "sqrt"

FEATURE_SETS: dict[str, list[str]] = {
    "summary": SUMMARY_COLUMNS,
    "spectral64": SPECTRAL_COLUMNS,
    "lsp512": LSP_COLUMNS,
    "spectral64+summary": [*SPECTRAL_COLUMNS, *SUMMARY_COLUMNS],
    "lsp512+summary": [*LSP_COLUMNS, *SUMMARY_COLUMNS],
}
BEST_SET = "lsp512+summary"


def collect_real(real_split: str, sigma_min: float, sigma_max: float) -> pd.DataFrame:
    """Real single-planet rows with targets + LSP512 + spectral64 + summaries."""
    ds = RVDataset(split=real_split, normalize=False, single_planet=True)
    all_cols = [*TARGET_COLUMNS, *LSP_COLUMNS, *SPECTRAL_COLUMNS, *SUMMARY_COLUMNS]
    rows: list[dict[str, float]] = []

    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue
        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        rv_std = float(info["rv_std_ms"])
        sigma = xm[2] * rv_std
        med_sigma = float(np.median(sigma))
        if not (sigma_min <= med_sigma <= sigma_max):
            continue
        rv_ms = xm[1] * rv_std
        t_days = xm[0] * float(info["t_span_days"])
        gaps = np.diff(np.sort(t_days))
        spectral = spectral_features(xm[0], xm[1], d=SPECTRAL_DIM, grid_size=SPECTRAL_GRID_SIZE)

        row = {
            "log10_P": float(theta[0]),
            "log10_K": float(theta[1]),
            "e": float(theta[2]),
            "cos_omega": float(theta[3]),
            "sin_omega": float(theta[4]),
            "n_obs": int(info["n_obs"]),
            "baseline_d": float(info["t_span_days"]),
            "rv_std_ms": rv_std,
            "rv_iqr_ms": float(np.subtract(*np.percentile(rv_ms, [75, 25]))),
            "median_sigma_ms": med_sigma,
            "sigma_iqr_ms": float(np.subtract(*np.percentile(sigma, [75, 25]))),
            "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
            "lsp_peak_power": float(np.max(lsp)),
            "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
            "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
        }
        row.update({n: float(v) for n, v in zip(SPECTRAL_COLUMNS, spectral)})
        row.update({n: float(v) for n, v in zip(LSP_COLUMNS, np.asarray(lsp, dtype=float))})
        rows.append(row)

    return pd.DataFrame(rows, columns=all_cols)


def plot_r2_by_featureset(
    cv: dict[str, dict[str, dict[str, dict[str, float]]]],
    targets: list[str],
    out_path: Path,
) -> None:
    sets = list(FEATURE_SETS)
    x = np.arange(len(targets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#bbbbbb", "#ff7f0e", "#1f77b4", "#d62728", "#2ca02c"]
    for k, s in enumerate(sets):
        vals = [cv[s][t]["r2"]["mean"] for t in targets]
        errs = [cv[s][t]["r2"]["std"] for t in targets]
        ax.bar(x + (k - 2) * width, vals, width, yerr=errs, capsize=2,
               label=s, color=colors[k % len(colors)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("cross-validated $R^2$")
    ax.set_title("Parameter recovery by input feature set (separate RFs, 5-fold CV)")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(
    csv_path: Path,
    out_dir: Path,
    fig_dir: Path,
    real_split: str,
    sigma_min: float,
    sigma_max: float,
    n_estimators: int,
    n_folds: int,
    test_size: float,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    targets = list(TARGET_COLUMNS)
    y = df[targets].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"loaded {n} synthetic rows; {len(train_idx)} train / {len(test_idx)} test")

    real = collect_real(real_split, sigma_min, sigma_max)
    y_real = real[targets].to_numpy(dtype=float)
    print(f"collected {len(real)} real reference systems")

    report: dict[str, object] = {
        "csv_path": str(csv_path),
        "n_synthetic": int(n),
        "n_real": int(len(real)),
        "real_split": real_split,
        "n_estimators": n_estimators,
        "max_features": MAX_FEATURES,
        "cv_folds": n_folds,
        "targets": targets,
        "feature_dims": {k: len(v) for k, v in FEATURE_SETS.items()},
        "cv_separate": {},
        "holdout_separate": {},
        "real_transfer_separate": {},
        "cv_joint": {},
        "holdout_joint": {},
        "real_transfer_joint": {},
    }

    for fs_name, cols in FEATURE_SETS.items():
        X = df[cols].to_numpy(dtype=float)
        X_real = real[cols].to_numpy(dtype=float)
        print(f"[separate] {fs_name} ({len(cols)}-D) ...")

        report["cv_separate"][fs_name] = cross_validate(
            "separate", X, y, targets, n_estimators, n_folds, seed, max_features=MAX_FEATURES
        )
        m = _build("separate", n_estimators, seed, targets, max_features=MAX_FEATURES)
        m.fit(X[train_idx], y[train_idx])
        report["holdout_separate"][fs_name] = per_target_metrics(
            y[test_idx], m.predict(X[test_idx]), targets
        )
        tr = _build("separate", n_estimators, seed, targets, max_features=MAX_FEATURES)
        tr.fit(X, y)
        report["real_transfer_separate"][fs_name] = per_target_metrics(
            y_real, tr.predict(X_real), targets
        )

    # Joint vs separate on the best (full-resolution) feature set.
    cols = FEATURE_SETS[BEST_SET]
    X = df[cols].to_numpy(dtype=float)
    X_real = real[cols].to_numpy(dtype=float)
    print(f"[joint] {BEST_SET} ...")
    report["cv_joint"][BEST_SET] = cross_validate(
        "joint", X, y, targets, n_estimators, n_folds, seed, max_features=MAX_FEATURES
    )
    jm = _build("joint", n_estimators, seed, targets, max_features=MAX_FEATURES)
    jm.fit(X[train_idx], y[train_idx])
    y_pred_joint = jm.predict(X[test_idx])
    report["holdout_joint"][BEST_SET] = per_target_metrics(y[test_idx], y_pred_joint, targets)
    jtr = _build("joint", n_estimators, seed, targets, max_features=MAX_FEATURES)
    jtr.fit(X, y)
    report["real_transfer_joint"][BEST_SET] = per_target_metrics(
        y_real, jtr.predict(X_real), targets
    )

    # Held-out separate predictions on BEST_SET for the paired plot.
    sm = _build("separate", n_estimators, seed, targets, max_features=MAX_FEATURES)
    sm.fit(X[train_idx], y[train_idx])
    y_pred_sep = sm.predict(X[test_idx])

    plot_true_vs_pred(
        y[test_idx], y_pred_joint, targets, report["holdout_joint"][BEST_SET],
        f"RF joint (multi-output) - {BEST_SET} - synthetic held-out test",
        fig_dir / "regression_true_vs_pred_lsp_joint.png",
    )
    plot_true_vs_pred(
        y[test_idx], y_pred_sep, targets, report["holdout_separate"][BEST_SET],
        f"RF separate (per-target) - {BEST_SET} - synthetic held-out test",
        fig_dir / "regression_true_vs_pred_lsp_separate.png",
    )
    plot_r2_by_featureset(report["cv_separate"], targets, fig_dir / "lsp_resolution_r2_by_featureset.png")

    (out_dir / "lsp_resolution_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_report(report, out_dir / "lsp_resolution_report.txt")
    print(f"wrote metrics + report to {out_dir}")
    print(f"wrote figures to {fig_dir}")


def _write_report(report: dict[str, object], path: Path) -> None:
    targets = report["targets"]
    lines = [
        "Spectral-resolution experiment (64-bin vs 512-bin power spectrum)",
        "=" * 64,
        f"synthetic rows : {report['n_synthetic']}",
        f"real reference : {report['n_real']} (split={report['real_split']})",
        f"forest         : {report['n_estimators']} trees, max_features={report['max_features']}, "
        f"{report['cv_folds']}-fold CV",
        "feature dims   : " + ", ".join(f"{k}={v}" for k, v in report["feature_dims"].items()),
        "",
        "Cross-validated R^2 by feature set (separate RFs, mean +/- std over folds)",
        "-" * 74,
        f"{'feature set':<22}" + "".join(f"{t:>12}" for t in targets),
    ]
    for fs in FEATURE_SETS:
        cv = report["cv_separate"][fs]
        row = f"{fs:<22}" + "".join(
            f"{cv[t]['r2']['mean']:>+8.3f}±{cv[t]['r2']['std']:.2f}" for t in targets
        )
        lines.append(row)
    lines.append("")

    lines.append("Synthetic-trained -> real transfer R^2 (separate RFs)")
    lines.append("-" * 74)
    lines.append(f"{'feature set':<22}" + "".join(f"{t:>12}" for t in targets))
    for fs in FEATURE_SETS:
        tr = report["real_transfer_separate"][fs]
        lines.append(f"{fs:<22}" + "".join(f"{tr[t]['r2']:>+12.3f}" for t in targets))
    lines.append("")

    lines.append(f"Joint vs separate on {BEST_SET} (held-out test R^2)")
    lines.append("-" * 50)
    hj, hs = report["holdout_joint"][BEST_SET], report["holdout_separate"][BEST_SET]
    lines.append(f"{'target':<12}{'joint':>10}{'separate':>10}")
    for t in targets:
        lines.append(f"{t:<12}{hj[t]['r2']:>+10.3f}{hs[t]['r2']:>+10.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("synthetic_generation") / "datasets" / "synthetic_lsp_regression_10000.csv",
    )
    p.add_argument("--out-dir", type=Path, default=Path("synthetic_generation") / "regression")
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("synthetic_generation") / "figures" / "synthetic_regression_10000",
    )
    p.add_argument("--real-split", choices=("all", "train", "val", "test"), default="all")
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=100.0)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        csv_path=args.csv,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        real_split=args.real_split,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        n_estimators=args.n_estimators,
        n_folds=args.cv_folds,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
