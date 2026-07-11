"""
Create real-vs-synthetic visual checks for a synthetic RV regression CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import LSP_PERIODS, RVDataset
from generate_synthetic_regression_csv import (
    CSV_COLUMNS,
    CSV_COLUMNS_PHASEFOLD,
    PHASE_FOLD_COLUMNS,
    PHASE_FOLD_N_BINS,
    SPECTRAL_COLUMNS,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
)
from time_series_features import spectral_features


KEPLER_COLUMNS = TARGET_COLUMNS
COMPARISON_COLUMNS = [*TARGET_COLUMNS, *SUMMARY_COLUMNS]


def _masked_observations(x: np.ndarray) -> np.ndarray:
    mask = x[3] == 1
    return x[:, mask]


def collect_real_summary(
    real_split: str,
    sigma_min: float,
    sigma_max: float,
    *,
    with_phasefold: bool = False,
) -> pd.DataFrame:
    """Collect real single-planet rows using the same regression columns as the CSV."""
    ds = RVDataset(split=real_split, normalize=False, single_planet=True)
    rows: list[dict[str, float]] = []
    rejected_sigma = 0
    columns = CSV_COLUMNS_PHASEFOLD if with_phasefold else CSV_COLUMNS

    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue

        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue

        rv_std = float(info["rv_std_ms"])
        sigma = xm[2] * rv_std
        rv_ms = xm[1] * rv_std
        med_sigma = float(np.median(sigma))
        if not (sigma_min <= med_sigma <= sigma_max):
            rejected_sigma += 1
            continue
        t_days = xm[0] * float(info["t_span_days"]) + float(info["t_min_days"])
        gaps = np.diff(np.sort(t_days))

        spectral = spectral_features(xm[0], xm[1], d=len(SPECTRAL_COLUMNS), grid_size=1024)

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
        row.update({name: float(value) for name, value in zip(SPECTRAL_COLUMNS, spectral)})
        if with_phasefold:
            # Real RVDataset lacks catalog t_peri; phase-fold features are undefined.
            row["has_t_peri"] = 0.0
            row.update({name: np.nan for name in PHASE_FOLD_COLUMNS})
        rows.append(row)

    if rejected_sigma:
        print(
            f"rejected {rejected_sigma} real systems with median sigma outside "
            f"[{sigma_min}, {sigma_max}] m/s"
        )
    return pd.DataFrame(rows, columns=columns)


def _hist_overlay(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    out_path: Path,
    title: str,
) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4.0 * n_rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, col in zip(axs, columns):
        real_vals = real[col].replace([np.inf, -np.inf], np.nan).dropna()
        synth_vals = synthetic[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(real_vals, bins=35, density=True, alpha=0.62, label="real", color="#1f77b4")
        ax.hist(synth_vals, bins=35, density=True, alpha=0.58, label="synthetic", color="#ff7f0e")
        ax.set_title(col)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    for ax in axs[len(columns):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _relationship_checks(real: pd.DataFrame, synthetic: pd.DataFrame, out_path: Path) -> None:
    pairs = [
        ("log10_P", "lsp_peak_period_d", "True log10 period vs LSP peak period"),
        ("log10_K", "rv_std_ms", "True log10 K vs observed RV scatter"),
        ("median_sigma_ms", "lsp_peak_power", "Measurement uncertainty vs LSP peak power"),
        ("baseline_d", "lsp_peak_power", "Observation baseline vs LSP peak power"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.ravel()
    for ax, (x_col, y_col, title) in zip(axs, pairs):
        ax.scatter(
            synthetic[x_col],
            synthetic[y_col],
            s=7,
            alpha=0.18,
            color="#ff7f0e",
            label="synthetic",
            linewidths=0,
        )
        ax.scatter(
            real[x_col],
            real[y_col],
            s=16,
            alpha=0.72,
            color="#1f77b4",
            label="real",
            linewidths=0,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        if y_col == "lsp_peak_period_d":
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _correlation_heatmaps(real: pd.DataFrame, synthetic: pd.DataFrame, out_path: Path) -> None:
    real_corr = real[COMPARISON_COLUMNS].corr(numeric_only=True)
    synth_corr = synthetic[COMPARISON_COLUMNS].corr(numeric_only=True)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, corr, title in [
        (axs[0], real_corr, "real"),
        (axs[1], synth_corr, "synthetic"),
    ]:
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        ax.set_title(f"{title} correlations")

    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.04, label="Pearson r")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _spectral_comparison(real: pd.DataFrame, synthetic: pd.DataFrame, out_path: Path) -> None:
    real_power = real[SPECTRAL_COLUMNS].to_numpy(dtype=float)
    synth_power = synthetic[SPECTRAL_COLUMNS].to_numpy(dtype=float)
    bins = np.arange(1, len(SPECTRAL_COLUMNS) + 1)

    real_q = np.percentile(real_power, [25, 50, 75], axis=0)
    synth_q = np.percentile(synth_power, [25, 50, 75], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bins, real_q[1], label="real median", color="#1f77b4", lw=2)
    ax.fill_between(bins, real_q[0], real_q[2], color="#1f77b4", alpha=0.2, label="real IQR")
    ax.plot(bins, synth_q[1], label="synthetic median", color="#ff7f0e", lw=2)
    ax.fill_between(bins, synth_q[0], synth_q[2], color="#ff7f0e", alpha=0.2, label="synthetic IQR")
    ax.set_xlabel("spectral power bin")
    ax.set_ylabel("normalized power")
    ax.set_title("Real vs synthetic spectral-power input distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plots(
    csv_path: Path,
    out_dir: Path,
    real_split: str,
    sigma_min: float,
    sigma_max: float,
) -> None:
    synthetic = pd.read_csv(csv_path)
    missing = [col for col in CSV_COLUMNS if col not in synthetic.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    synthetic = synthetic[CSV_COLUMNS]

    real = collect_real_summary(real_split, sigma_min=sigma_min, sigma_max=sigma_max)
    if real.empty:
        raise ValueError("No real rows were collected for comparison")

    out_dir.mkdir(parents=True, exist_ok=True)
    real.to_csv(out_dir / f"real_{real_split}_reference_summary.csv", index=False)

    _hist_overlay(
        real,
        synthetic,
        COMPARISON_COLUMNS,
        out_dir / "real_vs_synthetic_target_summary_histograms.png",
        "Real vs synthetic: targets and summary inputs",
    )
    _hist_overlay(
        real,
        synthetic,
        KEPLER_COLUMNS,
        out_dir / "real_vs_synthetic_keplerian_histograms.png",
        "Real vs synthetic: Keplerian target parameters",
    )
    _hist_overlay(
        real,
        synthetic,
        SUMMARY_COLUMNS,
        out_dir / "real_vs_synthetic_summary_histograms.png",
        "Real vs synthetic: observation summary features",
    )
    _relationship_checks(real, synthetic, out_dir / "real_vs_synthetic_relationship_checks.png")
    _correlation_heatmaps(real, synthetic, out_dir / "real_vs_synthetic_correlation_heatmaps.png")
    _spectral_comparison(real, synthetic, out_dir / "real_vs_synthetic_mean_spectral_power.png")

    print(f"real rows: {len(real)}")
    print(f"synthetic rows: {len(synthetic)}")
    print(f"wrote comparison figures to {out_dir}")


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
        default=Path("synthetic_generation") / "figures" / "synthetic_regression_10000",
    )
    p.add_argument(
        "--real-split",
        choices=("all", "train", "val", "test"),
        default="all",
    )
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=100.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(
        args.csv,
        args.out_dir,
        real_split=args.real_split,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )


if __name__ == "__main__":
    main()
