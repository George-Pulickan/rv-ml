"""
Random-forest regression baselines on the synthetic RV input-output CSV.

This is the Random-Forest counterpart to the RVEncoder neural network, built to
Nicolo's specification:

    input  = i)  the time-series power spectrum (64 spectral-power bins), and
             ii) the observation summary features Shuaib's generator produces.
    output = the true Keplerian parameters used to generate each time series
             (log10_P, log10_K, e, cos_omega, sin_omega).

Two model families are compared, mirroring the joint-vs-separate NN comparison
Jovie reported:

    * "joint"    - a single multi-output RandomForestRegressor predicting all
                   five targets at once (targets standardized so no single
                   target dominates the variance-reduction split criterion).
    * "separate" - one independent single-output RandomForestRegressor per
                   target.

Three input feature sets are ablated so we can answer the question directly:
does the power spectrum add predictive value beyond the cheap summaries?

    * "summary"  - the 10 observation-summary features only.
    * "spectral" - the 64 power-spectrum bins only.
    * "both"     - the full spec (74 features).

For every (model family x feature set) we report K-fold cross-validated
per-target MAE / RMSE / R^2 (mean +/- std). We additionally fit on a held-out
train split to produce true-vs-predicted diagnostic plots, and we test the
synthetic-trained model on the *real* reference systems (same feature space) to
measure how well the synthetic input->output mapping transfers to real data.

The compact 74-D representation used here is deliberately distinct from the
~1500-D representation the RVEncoder consumes (512-bin Lomb-Scargle power
spectrum + a (4, 256) summary tensor); see the report footer.

Usage
-----
    python synthetic_generation/train_regression_models.py
    python synthetic_generation/train_regression_models.py --n-estimators 300 --cv-folds 5
    python synthetic_generation/train_regression_models.py --save-models
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_synthetic_regression_csv import (  # noqa: E402
    SPECTRAL_COLUMNS,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
)
from plot_synthetic_regression_csv import collect_real_summary  # noqa: E402


FEATURE_SETS: dict[str, list[str]] = {
    "summary": SUMMARY_COLUMNS,
    "spectral": SPECTRAL_COLUMNS,
    "both": [*SPECTRAL_COLUMNS, *SUMMARY_COLUMNS],
}

# Human-readable target labels for plots.
TARGET_LABELS = {
    "log10_P": r"$\log_{10} P$",
    "log10_K": r"$\log_{10} K$",
    "e": r"$e$",
    "cos_omega": r"$\cos\omega$",
    "sin_omega": r"$\sin\omega$",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
) -> dict[str, dict[str, float]]:
    """Per-target MAE / RMSE / R^2 for an (n, n_targets) prediction block."""
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        out[name] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(root_mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
            "target_std": float(np.std(yt)),
        }
    return out


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_rf(n_estimators: int, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    )


@dataclass
class JointModel:
    """Single multi-output RF with target standardization."""

    n_estimators: int
    seed: int
    _rf: RandomForestRegressor = field(init=False, repr=False)
    _mu: np.ndarray = field(init=False, repr=False)
    _sd: np.ndarray = field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointModel":
        # Standardize targets so a large-variance target (e.g. log10_P) does not
        # dominate the averaged variance-reduction split criterion and starve a
        # small-variance target (e.g. e).
        self._mu = y.mean(axis=0)
        self._sd = np.where(y.std(axis=0) > 0, y.std(axis=0), 1.0)
        self._rf = _make_rf(self.n_estimators, self.seed)
        self._rf.fit(X, (y - self._mu) / self._sd)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._rf.predict(X) * self._sd + self._mu


@dataclass
class SeparateModel:
    """One independent single-output RF per target."""

    n_estimators: int
    seed: int
    targets: list[str]
    _rfs: list[RandomForestRegressor] = field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeparateModel":
        self._rfs = []
        for j in range(y.shape[1]):
            rf = _make_rf(self.n_estimators, self.seed + j)
            rf.fit(X, y[:, j])
            self._rfs.append(rf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([rf.predict(X) for rf in self._rfs])

    def feature_importances(self) -> np.ndarray:
        """(n_targets, n_features) impurity-based importances."""
        return np.vstack([rf.feature_importances_ for rf in self._rfs])


def _build(family: str, n_estimators: int, seed: int, targets: list[str]):
    if family == "joint":
        return JointModel(n_estimators=n_estimators, seed=seed)
    if family == "separate":
        return SeparateModel(n_estimators=n_estimators, seed=seed, targets=targets)
    raise ValueError(f"unknown model family: {family}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    targets: list[str],
    n_estimators: int,
    n_folds: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """K-fold CV -> per-target {metric: {mean, std}} aggregated over folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, dict[str, float]]] = []

    for tr, te in kf.split(X):
        model = _build(family, n_estimators, seed, targets)
        model.fit(X[tr], y[tr])
        y_pred = model.predict(X[te])
        fold_metrics.append(per_target_metrics(y[te], y_pred, targets))

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for name in targets:
        aggregated[name] = {
            metric: _mean_std([fm[name][metric] for fm in fold_metrics])
            for metric in ("mae", "rmse", "r2")
        }
    return aggregated


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    metrics: dict[str, dict[str, float]],
    title: str,
    out_path: Path,
) -> None:
    n = len(targets)
    fig, axs = plt.subplots(1, n, figsize=(4.0 * n, 4.2))
    axs = np.atleast_1d(axs).ravel()
    for ax, j, name in zip(axs, range(n), targets):
        yt, yp = y_true[:, j], y_pred[:, j]
        ax.scatter(yt, yp, s=8, alpha=0.25, color="#1f77b4", linewidths=0)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlabel(f"true {TARGET_LABELS.get(name, name)}")
        ax.set_ylabel(f"predicted {TARGET_LABELS.get(name, name)}")
        m = metrics[name]
        ax.set_title(f"{name}\n$R^2$={m['r2']:.3f}  MAE={m['mae']:.3f}")
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    targets: list[str],
    out_path: Path,
    top_k: int = 15,
) -> None:
    n = len(targets)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 6.0))
    axs = np.atleast_1d(axs).ravel()
    for ax, j, name in zip(axs, range(n), targets):
        imp = importances[j]
        order = np.argsort(imp)[::-1][:top_k]
        names = [feature_names[k] for k in order][::-1]
        vals = imp[order][::-1]
        ax.barh(range(len(names)), vals, color="#4c72b0")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("impurity importance")
        ax.grid(alpha=0.2, axis="x")
    fig.suptitle(f"Per-target feature importance (top {top_k})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
    save_models: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    targets = list(TARGET_COLUMNS)
    y_all = df[targets].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    print(f"loaded {n} synthetic rows from {csv_path}")
    print(f"held-out split: {len(train_idx)} train / {len(test_idx)} test")

    # Real reference systems in the identical 74-D feature space.
    real = collect_real_summary(real_split, sigma_min=sigma_min, sigma_max=sigma_max)
    print(f"collected {len(real)} real reference systems (split={real_split})")
    y_real = real[targets].to_numpy(dtype=float)

    report: dict[str, object] = {
        "csv_path": str(csv_path),
        "n_synthetic": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_real_reference": int(len(real)),
        "real_split": real_split,
        "n_estimators": n_estimators,
        "cv_folds": n_folds,
        "test_size": test_size,
        "seed": seed,
        "targets": targets,
        "feature_dims": {name: len(cols) for name, cols in FEATURE_SETS.items()},
        "cross_validation": {},
        "holdout": {},
        "real_transfer": {},
        "baseline": {},
    }

    # --- Dummy (mean) baseline for context: MAE/RMSE of predicting the train mean.
    baseline_pred = np.zeros_like(y_all[test_idx])
    for j in range(len(targets)):
        dummy = DummyRegressor(strategy="mean").fit(
            np.zeros((len(train_idx), 1)), y_all[train_idx, j]
        )
        baseline_pred[:, j] = dummy.predict(np.zeros((len(test_idx), 1)))
    report["baseline"] = per_target_metrics(y_all[test_idx], baseline_pred, targets)

    # --- Per feature set x model family.
    importances_for_plot: np.ndarray | None = None
    for fs_name, fs_cols in FEATURE_SETS.items():
        X_all = df[fs_cols].to_numpy(dtype=float)
        X_real = real[fs_cols].to_numpy(dtype=float)

        for family in ("joint", "separate"):
            key = f"{family}|{fs_name}"
            print(f"[cv] {key} ...")
            report["cross_validation"][key] = cross_validate(
                family, X_all, y_all, targets, n_estimators, n_folds, seed
            )

            # Held-out fit for point metrics + plots.
            model = _build(family, n_estimators, seed, targets)
            model.fit(X_all[train_idx], y_all[train_idx])
            y_pred_test = model.predict(X_all[test_idx])
            report["holdout"][key] = per_target_metrics(
                y_all[test_idx], y_pred_test, targets
            )

            # Diagnostic plots only for the full-spec "both" feature set.
            if fs_name == "both":
                plot_true_vs_pred(
                    y_all[test_idx],
                    y_pred_test,
                    targets,
                    report["holdout"][key],
                    f"RF ({family}, spectral+summary) - synthetic held-out test",
                    fig_dir / f"regression_true_vs_pred_{family}.png",
                )

            # Synthetic -> real transfer: refit on ALL synthetic, test on real.
            transfer = _build(family, n_estimators, seed, targets)
            transfer.fit(X_all, y_all)
            if len(real):
                y_pred_real = transfer.predict(X_real)
                report["real_transfer"][key] = per_target_metrics(
                    y_real, y_pred_real, targets
                )
                if fs_name == "both" and family == "separate":
                    plot_true_vs_pred(
                        y_real,
                        y_pred_real,
                        targets,
                        report["real_transfer"][key],
                        "RF (separate, spectral+summary) - synthetic-trained, "
                        f"tested on real ({real_split})",
                        fig_dir / "regression_true_vs_pred_real_transfer.png",
                    )

            # Feature importances from the full-spec separate model.
            if fs_name == "both" and family == "separate":
                imp = transfer.feature_importances()
                importances_for_plot = imp
                imp_df = pd.DataFrame(imp.T, index=fs_cols, columns=targets)
                imp_df.index.name = "feature"
                imp_df.to_csv(out_dir / "regression_feature_importances.csv")
                if save_models:
                    import joblib

                    joblib.dump(transfer, out_dir / "rf_separate_both.joblib")

    if importances_for_plot is not None:
        plot_feature_importance(
            importances_for_plot,
            FEATURE_SETS["both"],
            targets,
            fig_dir / "regression_feature_importance.png",
        )

    return report


def write_report(report: dict[str, object], out_dir: Path) -> None:
    (out_dir / "regression_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    targets = report["targets"]
    lines: list[str] = []
    lines.append("Random-forest regression on the synthetic RV input-output CSV")
    lines.append("=" * 64)
    lines.append(f"synthetic rows : {report['n_synthetic']}")
    lines.append(f"held-out split : {report['n_train']} train / {report['n_test']} test")
    lines.append(f"real reference : {report['n_real_reference']} systems (split={report['real_split']})")
    lines.append(f"forest         : {report['n_estimators']} trees, {report['cv_folds']}-fold CV")
    lines.append(
        "feature dims   : "
        + ", ".join(f"{k}={v}" for k, v in report["feature_dims"].items())
    )
    lines.append("")

    def block(title: str, section: str, use_cv: bool) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        header = f"{'config':<20}{'target':<12}{'R2':>10}{'MAE':>10}{'RMSE':>10}"
        lines.append(header)
        for key in sorted(report[section].keys()):
            for name in targets:
                m = report[section][key][name]
                if use_cv:
                    r2 = f"{m['r2']['mean']:.3f}"
                    mae = f"{m['mae']['mean']:.3f}"
                    rmse = f"{m['rmse']['mean']:.3f}"
                else:
                    r2 = f"{m['r2']:.3f}"
                    mae = f"{m['mae']:.3f}"
                    rmse = f"{m['rmse']:.3f}"
                lines.append(f"{key:<20}{name:<12}{r2:>10}{mae:>10}{rmse:>10}")
            lines.append("")

    block("Cross-validated metrics (mean over folds)", "cross_validation", use_cv=True)
    block("Held-out test metrics", "holdout", use_cv=False)
    block("Synthetic-trained -> real transfer", "real_transfer", use_cv=False)

    lines.append("Baseline (predict train mean) held-out test")
    lines.append("-" * 44)
    for name in targets:
        m = report["baseline"][name]
        lines.append(f"  {name:<12} R2={m['r2']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    lines.append("")

    lines.append("Note on feature dimensionality")
    lines.append("-" * 30)
    lines.append(
        "This RF baseline uses the compact 74-D representation stored in the CSV\n"
        "(64 power-spectrum bins + 10 observation summaries). The RVEncoder NN\n"
        "instead consumes ~1500 dims: a 512-bin Lomb-Scargle power spectrum plus a\n"
        "(4, 256) summary tensor. The two are different encodings of the same RV\n"
        "time series; metrics are not directly comparable across the two spaces."
    )

    (out_dir / "regression_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("synthetic_generation") / "regression",
    )
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
    p.add_argument("--save-models", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
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
        save_models=args.save_models,
    )
    write_report(report, args.out_dir)
    print(f"wrote regression metrics + report to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
