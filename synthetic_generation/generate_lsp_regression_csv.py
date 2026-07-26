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

from preprocess import LSP_N
from synthetic_dataset import _sample_orbital_params, generate_one
from feature_columns import (
    SPECTRAL_COLUMNS,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
)

# The targets/summary/coarse-spectral block is defined once, next to the 74-D
# generator, so both datasets are guaranteed to describe the same systems the
# same way — that identity is what makes the 64-bin vs 512-bin comparison valid.
from generate_synthetic_regression_csv import (
    base_summary_row,
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

        row, _, _, _ = base_summary_row(x, lsp, theta, info)
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
