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
from feature_columns import (
    CSV_COLUMNS,
    CSV_COLUMNS_PHASEFOLD,
    PHASE_FOLD_COLUMNS,
    PHASE_FOLD_N_BINS,
    SPECTRAL_COLUMNS,
    SPECTRAL_DIM,
    SPECTRAL_GRID_SIZE,
    SUMMARY_COLUMNS,
    TARGET_COLUMNS,
)
from time_series_features import (
    phase_fold_features,
    spectral_features,
)


def _masked_observations(x: np.ndarray) -> np.ndarray:
    """Return the non-padded columns from the synthetic x tensor."""
    mask = x[3] == 1
    return x[:, mask]


def corpus_orbital_params(seed: int, n_samples: int) -> dict[str, np.ndarray]:
    """Redraw the full parameter corpus exactly as ``generate_rows`` did."""
    rng = np.random.default_rng(seed)
    return _sample_orbital_params(rng, n_samples)


def replay_synthetic_sample(
    i: int,
    seed: int,
    n_samples: int,
    f_multi: float = 0.0,
    *,
    params: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Replay synthetic sample ``i`` with the same RNG scheme as ``generate_rows``.

    ``n_samples`` must be the corpus size used at generation time: the shared
    parameter RNG stream depends on it, so drawing fewer samples yields a
    different system for the same ``i``. Batch callers should pass a
    precomputed ``params`` from ``corpus_orbital_params`` to avoid redrawing
    the corpus per sample.
    """
    if params is None:
        params = corpus_orbital_params(seed, n_samples)
    p = {k: float(v[i]) for k, v in params.items()}
    sample_rng = np.random.default_rng(seed + 10_000 + i)
    return generate_one(p, sample_rng, f_multi=f_multi)


def generate_rows(
    n_samples: int,
    seed: int,
    f_multi: float,
    *,
    with_phasefold: bool,
    epoch_free: bool = False,
    fold_period: str = "oracle",
) -> list[dict[str, float]]:
    """
    Generate regression rows from the synthetic RV generator.

    Target columns come from the dominant-planet theta returned by
    synthetic_dataset.generate_one. Input columns are the spectral encoding and
    summary features derived from the generated observation tensor.

    When ``with_phasefold``:
      - ``fold_period='oracle'`` folds at the true dominant period
      - ``fold_period='lsp'`` folds at the LSP peak period (matches deployment)
      - ``epoch_free=True`` anchors the fold at the phase of max RV (no t_peri)
    """
    if fold_period not in ("oracle", "lsp"):
        raise ValueError("fold_period must be 'oracle' or 'lsp'")
    params = corpus_orbital_params(seed, n_samples)
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
            if fold_period == "lsp":
                P_fold = float(row["lsp_peak_period_d"])
            else:
                P_fold = float(info["P"])
            if epoch_free:
                phase = phase_fold_features(
                    t_days,
                    rv_ms,
                    P_fold,
                    n_bins=PHASE_FOLD_N_BINS,
                    epoch_free=True,
                )
                # Omega remains identifiable from waveform asymmetry under the
                # max-RV anchor; mark has_t_peri=1 so Gate-C / omega metrics run.
                row["has_t_peri"] = 1.0
            else:
                phase = phase_fold_features(
                    t_days,
                    rv_ms,
                    P_fold,
                    n_bins=PHASE_FOLD_N_BINS,
                    t_peri=float(info["t_peri"]),
                )
                row["has_t_peri"] = 1.0
            row.update({name: float(value) for name, value in zip(PHASE_FOLD_COLUMNS, phase)})
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
    p.add_argument(
        "--epoch-free",
        action="store_true",
        help="with --with-phasefold: fold origin = phase of max RV (no catalog t_peri)",
    )
    p.add_argument(
        "--fold-period",
        choices=("oracle", "lsp"),
        default="oracle",
        help="with --with-phasefold: fold at true P or LSP peak P",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.epoch_free and not args.with_phasefold:
        raise ValueError("--epoch-free requires --with-phasefold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(
        args.n_samples,
        args.seed,
        args.f_multi,
        with_phasefold=args.with_phasefold,
        epoch_free=args.epoch_free,
        fold_period=args.fold_period,
    )
    columns = CSV_COLUMNS_PHASEFOLD if args.with_phasefold else CSV_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(args.out, index=False)

    print(f"wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
