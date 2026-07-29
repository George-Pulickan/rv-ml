"""Is the 64-bin spline+FFT spectral block the reason psi is weak?

Nicolo 2026-07-28: "We need to fix the encoding step. Especially if we say
something in the paper and do something else in the code."

The paper says the encoder stacks "a 64-bin generalised Lomb-Scargle
periodogram, the floating-mean formulation of Zechmeister (2009)". The code
actually feeds `time_series_features.spectral_features`: a UnivariateSpline
through the irregular samples, resampled onto a uniform 1024-point grid, rfft,
first 64 of 512 bins. Two defects follow -- the spline fabricates signal between
sparse observations, and keeping the first 64 bins band-limits the encoding to
periods above baseline/64 (~21 d at the median baseline), which excludes most of
the corpus.

The real GLS already exists (`preprocess.compute_lsp`, astropy, fit_mean=True,
heteroscedastic, on the FIXED log grid LSP_PERIODS = geomspace(0.5, 5000, 512))
and is already computed for every curve -- `regression.py:212` calls it and then
keeps only two scalars from it (peak period, peak power).

This script compares the candidate encodings head to head on identical curves,
with an identical RF, before any pipeline is rebuilt:

    summary10     the 10 scalars alone (the control)
    spline64      what the code does today
    gls64         the GLS max-pooled to 64 bins   (same width as spline64)
    gls512        the full GLS grid
    gls512+sum    the proposed replacement

Max-pooling, not mean, when reducing 512 -> 64: a periodogram peak is narrow and
averaging washes it out.

Usage
-----
    python scripts/check_gls_encoding.py --n 1500 --out figures/paper/gls_encoding_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_generation"))

from conformal import _masked_observations  # noqa: E402
from feature_columns import SPECTRAL_DIM, SPECTRAL_GRID_SIZE  # noqa: E402
from preprocess import LSP_N, LSP_PERIODS  # noqa: E402
from time_series_features import spectral_features  # noqa: E402
from generate_synthetic_regression_csv import (  # noqa: E402
    corpus_orbital_params,
    replay_synthetic_sample,
)

TARGETS = ["log10_P", "log10_K", "e"]


def gls_pool(lsp: np.ndarray, d: int) -> np.ndarray:
    """Max-pool a 512-point GLS onto d contiguous log-period bins."""
    lsp = np.asarray(lsp, dtype=float).reshape(-1)
    edges = np.linspace(0, len(lsp), d + 1).astype(int)
    return np.array([lsp[edges[i]:edges[i + 1]].max() if edges[i + 1] > edges[i] else 0.0
                     for i in range(d)], dtype=float)


def summary10(xm: np.ndarray, info: dict, lsp: np.ndarray) -> np.ndarray:
    rv_std = float(info["rv_std_ms"])
    sigma = xm[2] * rv_std
    rv_ms = xm[1] * rv_std
    t_days = xm[0] * float(info["t_span_days"])
    gaps = np.diff(np.sort(t_days))
    return np.array([
        float(info["n_obs"]),
        float(info["t_span_days"]),
        rv_std,
        float(np.subtract(*np.percentile(rv_ms, [75, 25]))),
        float(np.median(sigma)),
        float(np.subtract(*np.percentile(sigma, [75, 25]))),
        float(LSP_PERIODS[int(np.argmax(lsp))]),
        float(np.max(lsp)),
        float(np.median(gaps)) if len(gaps) else 0.0,
        float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
    ], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--csv-seed", type=int, default=0)
    ap.add_argument("--csv-rows", type=int, default=10000,
                    help="row count of the CSV whose RNG stream we replay")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures" / "paper" / "gls_encoding_check.json")
    args = ap.parse_args()

    print(f"replaying {args.n} synthetic curves (seed={args.csv_seed}) ...")
    t0 = time.time()
    params = corpus_orbital_params(args.csv_seed, args.csv_rows)
    B = {"summary10": [], "spline64": [], "gls64": [], "gls512": []}
    Y = []
    for i in range(args.n):
        x, lsp, theta, info = replay_synthetic_sample(
            i, args.csv_seed, args.csv_rows, f_multi=0.0, params=params)
        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        lsp = np.asarray(lsp, dtype=float)
        B["summary10"].append(summary10(xm, info, lsp))
        B["spline64"].append(spectral_features(xm[0], xm[1], d=SPECTRAL_DIM,
                                               grid_size=SPECTRAL_GRID_SIZE))
        B["gls64"].append(gls_pool(lsp, 64))
        B["gls512"].append(lsp)
        Y.append([float(theta[0]), float(theta[1]), float(theta[2])])
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.n}  ({time.time() - t0:.0f}s)")

    Y = np.asarray(Y, dtype=float)
    for k in B:
        B[k] = np.asarray(B[k], dtype=float)
    print(f"built {len(Y)} rows in {time.time() - t0:.0f}s; "
          f"GLS grid = {LSP_N} bins over {LSP_PERIODS[0]:.2f}-{LSP_PERIODS[-1]:.0f} d")

    blocks = {
        "summary10": B["summary10"],
        "spline64": B["spline64"],
        "gls64": B["gls64"],
        "gls512": B["gls512"],
        "spline64+summary(=today)": np.hstack([B["spline64"], B["summary10"]]),
        "gls64+summary": np.hstack([B["gls64"], B["summary10"]]),
        "gls512+summary": np.hstack([B["gls512"], B["summary10"]]),
    }

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    report = {"n_rows": int(len(Y)), "n_estimators": args.n_estimators,
              "folds": args.folds, "gls_grid": [float(LSP_PERIODS[0]), float(LSP_PERIODS[-1]), int(LSP_N)],
              "blocks": {}}

    print(f"\n{'block':<26} {'dim':>5} " + " ".join(f"{t:>9}" for t in TARGETS))
    print("-" * 62)
    for name, X in blocks.items():
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        preds = np.zeros_like(Y)
        for tr, te in kf.split(X):
            rf = RandomForestRegressor(n_estimators=args.n_estimators,
                                       random_state=args.seed, n_jobs=-1)
            rf.fit(X[tr], Y[tr])
            preds[te] = rf.predict(X[te])
        r2 = {}
        for j, t in enumerate(TARGETS):
            ss_res = float(((Y[:, j] - preds[:, j]) ** 2).sum())
            ss_tot = float(((Y[:, j] - Y[:, j].mean()) ** 2).sum())
            r2[t] = 1.0 - ss_res / max(ss_tot, 1e-12)
        report["blocks"][name] = {"dim": int(X.shape[1]), "r2": r2}
        print(f"{name:<26} {X.shape[1]:>5} " + " ".join(f"{r2[t]:>9.4f}" for t in TARGETS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
