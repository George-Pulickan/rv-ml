"""
Generate a synthetic RV input-output CSV for regression.

Each row contains the true Keplerian parameters used to generate the time
series, followed by fixed-length input features: spectral power bins and
observation-summary features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import LSP_PERIODS
from synthetic_dataset import _sample_orbital_params, generate_one
from time_series_features import (
    phase_fold_feature_names,
    phase_fold_features,
    spectral_feature_names,
    spectral_features,
)


SPECTRAL_DIM = 64
SPECTRAL_GRID_SIZE = 1024

TARGET_COLUMNS = [
    "log10_P",
    "log10_K",
    "e",
    "cos_omega",
    "sin_omega",
]

SPECTRAL_COLUMNS = spectral_feature_names(SPECTRAL_DIM)
PHASE_FOLD_N_BINS = 32
PHASE_FOLD_COLUMNS = phase_fold_feature_names(PHASE_FOLD_N_BINS)

SUMMARY_COLUMNS = [
    "n_obs",
    "baseline_d",
    "rv_std_ms",
    "rv_iqr_ms",
    "median_sigma_ms",
    "sigma_iqr_ms",
    "lsp_peak_period_d",
    "lsp_peak_power",
    "median_gap_d",
    "p90_gap_d",
]

CSV_COLUMNS = [*TARGET_COLUMNS, *SPECTRAL_COLUMNS, *SUMMARY_COLUMNS]
CSV_COLUMNS_PHASEFOLD = [*CSV_COLUMNS, *PHASE_FOLD_COLUMNS, "has_t_peri"]


def _masked_observations(x: np.ndarray) -> np.ndarray:
    """Return the non-padded columns from the synthetic x tensor."""
    mask = x[3] == 1
    return x[:, mask]


def replay_synthetic_sample(
    i: int,
    seed: int,
    f_multi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Replay synthetic sample ``i`` with the same RNG scheme as ``generate_rows``."""
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, i + 1)
    p = {k: float(v[i]) for k, v in params.items()}
    sample_rng = np.random.default_rng(seed + 10_000 + i)
    return generate_one(p, sample_rng, f_multi=f_multi)


def generate_rows(n_samples: int, seed: int, f_multi: float, *, with_phasefold: bool) -> list[dict[str, float]]:
    """
    Generate regression rows from the synthetic RV generator.

    Target columns come from the dominant-planet theta returned by
    synthetic_dataset.generate_one. Input columns are the spectral encoding and
    summary features derived from the generated observation tensor.
    """
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n_samples)
    rows: list[dict[str, float]] = []

    for i in range(n_samples):
        p = {k: float(v[i]) for k, v in params.items()}
        sample_rng = np.random.default_rng(seed + 10_000 + i)
        x, lsp, theta, info = generate_one(p, sample_rng, f_multi=f_multi)
        xm = _masked_observations(x)
        rv_std = float(info["rv_std_ms"])
        sigma = xm[2] * rv_std
        rv_ms = xm[1] * rv_std
        t_days = xm[0] * float(info["t_span_days"])
        gaps = np.diff(np.sort(t_days))
        spectral = spectral_features(
            xm[0],
            xm[1],
            d=SPECTRAL_DIM,
            grid_size=SPECTRAL_GRID_SIZE,
        )

        row = {
            "log10_P": float(theta[0]),
            "log10_K": float(theta[1]),
            "e": float(theta[2]),
            "cos_omega": float(theta[3]),
            "sin_omega": float(theta[4]),
            "n_obs": int(info["n_obs"]),
            "baseline_d": float(info["baseline_d"]),
            "rv_std_ms": rv_std,
            "rv_iqr_ms": float(np.subtract(*np.percentile(rv_ms, [75, 25]))),
            "median_sigma_ms": float(np.median(sigma)),
            "sigma_iqr_ms": float(np.subtract(*np.percentile(sigma, [75, 25]))),
            "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
            "lsp_peak_power": float(np.max(lsp)),
            "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
            "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
        }
        row.update({name: float(value) for name, value in zip(SPECTRAL_COLUMNS, spectral)})
        if with_phasefold:
            phase = phase_fold_features(
                t_days,
                rv_ms,
                float(info["P"]),
                n_bins=PHASE_FOLD_N_BINS,
                t_peri=float(info["t_peri"]),
            )
            row.update({name: float(value) for name, value in zip(PHASE_FOLD_COLUMNS, phase)})
            row["has_t_peri"] = 1.0
        rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"generated {i + 1:,}/{n_samples:,} rows")

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--f-multi", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv",
    )
    p.add_argument(
        "--with-phasefold",
        action="store_true",
        help="append 35 phase-fold features and has_t_peri column",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.n_samples, args.seed, args.f_multi, with_phasefold=args.with_phasefold)
    columns = CSV_COLUMNS_PHASEFOLD if args.with_phasefold else CSV_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(args.out, index=False)

    print(f"wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
