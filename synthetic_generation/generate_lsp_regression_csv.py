"""
Generate a regression CSV that additionally stores the full 512-bin Lomb-Scargle
power spectrum, for the spectral-resolution experiment.

Motivation
----------
The 74-D regression CSV (`synthetic_regression_10000.csv`) encodes the power
spectrum as only 64 coarse, sum-normalized bins, and the RF baseline showed that
representation carries no recoverable parameter signal (R^2 < 0). The RVEncoder
NN instead consumes the full 512-bin LSP. This script produces a matched dataset
storing, per system:

    targets(5) + lsp_power_001..512 (full LSP) + spectral_power_001..064
    (the coarse bins) + the 10 observation summaries.

The same seeds as `generate_synthetic_regression_csv.py` are used (seed=123,
per-sample seed+10000+i, f_multi default 0.0), so the systems are identical and
the 64-bin vs 512-bin comparison is apples-to-apples on the same draws.

Usage
-----
    python synthetic_generation/generate_lsp_regression_csv.py
    python synthetic_generation/generate_lsp_regression_csv.py --n-samples 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import LSP_N, LSP_PERIODS
from synthetic_dataset import _sample_orbital_params, generate_one
from time_series_features import spectral_features

from generate_synthetic_regression_csv import (
    SPECTRAL_COLUMNS,
    SPECTRAL_DIM,
    SPECTRAL_GRID_SIZE,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
    _masked_observations,
)

LSP_COLUMNS = [f"lsp_power_{i + 1:03d}" for i in range(LSP_N)]
CSV_COLUMNS = [*TARGET_COLUMNS, *LSP_COLUMNS, *SPECTRAL_COLUMNS, *SUMMARY_COLUMNS]


def generate_rows(n_samples: int, seed: int, f_multi: float) -> list[dict[str, float]]:
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
        spectral = spectral_features(xm[0], xm[1], d=SPECTRAL_DIM, grid_size=SPECTRAL_GRID_SIZE)

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
        row.update({name: float(v) for name, v in zip(SPECTRAL_COLUMNS, spectral)})
        row.update({name: float(v) for name, v in zip(LSP_COLUMNS, np.asarray(lsp, dtype=float))})
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
        default=Path("synthetic_generation") / "datasets" / "synthetic_lsp_regression_10000.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.n_samples, args.seed, args.f_multi)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows x {df.shape[1]} cols to {args.out}")


if __name__ == "__main__":
    main()
