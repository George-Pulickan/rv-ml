"""
Phase 0: visual sanity check for oracle phase-folded RV curves.

Samples 5 high-e and 5 low-e synthetic systems, folds at true P and t_peri,
and saves figures/regression_synthetic/phasefold_sanity.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_synthetic_regression_csv import (  # noqa: E402
    _masked_observations,
    replay_synthetic_sample,
)
from time_series_features import phase_fold_curve  # noqa: E402


def _pick_indices(seed: int, n_samples: int, n_scan: int, e_high: float, e_low: float) -> tuple[list[int], list[int]]:
    high: list[int] = []
    low: list[int] = []
    for i in range(n_scan):
        _, _, theta, _ = replay_synthetic_sample(i, seed, n_samples=n_samples)
        e = float(theta[2])
        if e > e_high and len(high) < 5:
            high.append(i)
        if e < e_low and len(low) < 5:
            low.append(i)
        if len(high) >= 5 and len(low) >= 5:
            break
    if len(high) < 5 or len(low) < 5:
        raise RuntimeError(
            f"could not find 5 high-e (>{e_high}) and 5 low-e (<{e_low}) samples in {n_scan} draws"
        )
    return high, low


def _folded_curve(i: int, seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    x, _, theta, info = replay_synthetic_sample(i, seed, n_samples=n_samples)
    xm = _masked_observations(x)
    rv_std = float(info["rv_std_ms"])
    t_days = xm[0] * float(info["t_span_days"])
    rv_ms = xm[1] * rv_std
    phase, rv = phase_fold_curve(
        t_days,
        rv_ms,
        float(info["P"]),
        t_peri=float(info["t_peri"]),
    )
    return phase, rv, float(theta[2])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--n-scan", type=int, default=5000)
    p.add_argument("--e-high", type=float, default=0.3)
    p.add_argument("--e-low", type=float, default=0.05)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures") / "regression_synthetic" / "phasefold_sanity.png",
    )
    args = p.parse_args()

    high_idx, low_idx = _pick_indices(args.seed, args.n_samples, args.n_scan, args.e_high, args.e_low)

    fig, axes = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=False)
    for col, i in enumerate(high_idx):
        phase, rv, e = _folded_curve(i, args.seed, args.n_samples)
        ax = axes[0, col]
        ax.plot(phase, rv, color="#d62728", lw=1.5)
        ax.set_title(f"high e={e:.2f}")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("RV (m/s)")

    for col, i in enumerate(low_idx):
        phase, rv, e = _folded_curve(i, args.seed, args.n_samples)
        ax = axes[1, col]
        ax.plot(phase, rv, color="#1f77b4", lw=1.5)
        ax.set_title(f"low e={e:.2f}")
        ax.set_xlabel("orbital phase")
        ax.grid(alpha=0.25)
    axes[1, 0].set_ylabel("RV (m/s)")

    fig.suptitle("Oracle phase-fold sanity: high-e (asymmetric) vs low-e (circular)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    print(f"high-e indices: {high_idx}")
    print(f"low-e indices: {low_idx}")


if __name__ == "__main__":
    main()
