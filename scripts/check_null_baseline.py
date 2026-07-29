"""Phase A diagnostics: is psi carrying information, and is log10 K miswired?

A2  mse_null : median ||h(theta_prior_median) - y||^2 over curves, against
      mse_psi = median ||h(psi(y)) - y||^2. If mse_psi ~= mse_null then psi is a
      near-mean predictor on the reconstruction scale and every downstream
      number is measuring the prior. (The parameter-space version of this test
      is already answered: R2(log10 P) = 0.70-0.79, not < 0.2.)

A3  K wiring : for a well-sampled curve, K is within tens of percent of the
      robust RV half-range, so log10 K should be nearly linear in log10 sigma_RV.
      Fit log10 K ~ a + b*log10 std(y) on synthetic. If that one-parameter model
      reaches R2 > 0.9 and beats psi's log10 K, the MLP is miswired rather than
      facing a hard problem -- stop and find the bug before optimising anything.

Both run on replayed synthetic curves, so psi and the baselines see identical
data.

Usage
-----
    python scripts/check_null_baseline.py \
        --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
        --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
        --n 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_generation"))

from conformal import _curve_from_x, _masked_observations  # noqa: E402
from conformal_shift import _load_mlp_psi  # noqa: E402
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402
from generate_synthetic_regression_csv import (  # noqa: E402
    corpus_orbital_params,
    replay_synthetic_sample,
)


def curve_mse(decoder, theta5s, curves) -> np.ndarray:
    """Mean squared masked residual in rv_std units, per curve."""
    out = np.zeros(len(curves), dtype=float)
    with torch.no_grad():
        for i, (th, c) in enumerate(zip(theta5s, curves)):
            rv_obs = torch.from_numpy(c["rv_obs"][None, :])
            mask = torch.from_numpy(c["mask"][None, :])
            rv_pred = decoder(
                torch.as_tensor(np.asarray(th)[None, :], dtype=torch.float32),
                torch.from_numpy(c["t_norm"][None, :]),
                torch.tensor([c["t_span"]], dtype=torch.float32),
                torch.tensor([c["t_min"]], dtype=torch.float32),
                rv_obs,
                torch.tensor([c["rv_std"]], dtype=torch.float32),
                mask,
            )
            n = float(mask.sum().clamp(min=1.0))
            out[i] = float((((rv_obs - rv_pred) ** 2) * mask).sum() / n)
    return out


def assert_replay_matches_csv(df, params, seed: int, n_rows: int,
                              probe=(0, 1, 7, 50, 777)) -> None:
    """Fail loudly if replayed row i is not CSV row i.

    Pairing CSV feature rows with replayed curves is only meaningful when the
    replay reproduces the row. The parameter RNG stream is shared across the
    corpus, so the wrong --csv-seed yields a *different system* for the same i
    and every downstream R2 comes out negative with no other symptom. Caught
    exactly that way on 2026-07-28 (seed 0 used against a seed-123 corpus).
    """
    bad = []
    for i in probe:
        if i >= n_rows:
            continue
        _x, _lsp, theta, _info = replay_synthetic_sample(
            i, seed, n_rows, f_multi=0.0, params=params)
        if abs(float(df.iloc[i]["log10_P"]) - float(theta[0])) > 1e-6:
            bad.append(i)
    if bad:
        raise SystemExit(
            f"replay/CSV mismatch at rows {bad} with --csv-seed {seed}: the CSV "
            f"was generated with a different seed (conformal_shift.py uses 123). "
            f"Pairing CSV features with replayed curves would be meaningless.")
    print(f"replay/CSV pairing verified at seed {seed} on rows {probe}")


def r2(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500)
    # MUST match the seed the CSV was generated with (conformal_shift.py uses
    # 123). The parameter RNG stream is shared across the corpus, so a wrong
    # seed silently pairs CSV feature rows with a different system's curve --
    # every R2 then comes out negative for no visible reason. Guarded below.
    ap.add_argument("--csv-seed", type=int, default=123)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures" / "paper" / "null_baseline_check.json")
    args = ap.parse_args()

    decoder = KeplerDecoder().eval()
    psi_predict, norm_stats = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    n_rows = len(df)
    n_use = min(args.n, n_rows)

    # Prior median in the 5-vector. omega has no meaningful median (circular);
    # the decoder refits t_peri and gamma anyway, so 0 rad is the neutral choice.
    med = np.array([
        float(df["log10_P"].median()),
        float(df["log10_K"].median()),
        float(df["e"].median()),
        1.0, 0.0,
    ], dtype=float)
    print(f"prior median theta: log10_P={med[0]:.4f} log10_K={med[1]:.4f} e={med[2]:.4f}")

    params = corpus_orbital_params(args.csv_seed, n_rows)
    assert_replay_matches_csv(df, params, args.csv_seed, n_rows)

    print(f"replaying {n_use} curves ...")
    t0 = time.time()
    curves, rows, y_true, rv_scale = [], [], [], []
    for i in range(n_use):
        x, _lsp, theta, info = replay_synthetic_sample(
            i, args.csv_seed, n_rows, f_multi=0.0, params=params)
        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        curves.append(_curve_from_x(x, info))
        rows.append(i)
        y_true.append([float(theta[0]), float(theta[1]), float(theta[2])])
        rv_scale.append(float(np.std(xm[1] * float(info["rv_std_ms"]))))
    y_true = np.asarray(y_true, float)
    rv_scale = np.asarray(rv_scale, float)
    print(f"  {len(curves)} curves in {time.time() - t0:.0f}s")

    X = df.loc[rows, feature_cols].to_numpy(float)
    psi_y = psi_predict(X)

    # ---- A2 -------------------------------------------------------------
    mse_psi = curve_mse(decoder, psi_y, curves)
    mse_null = curve_mse(decoder, np.repeat(med[None, :], len(curves), axis=0), curves)
    ratio = float(np.median(mse_psi) / max(np.median(mse_null), 1e-12))
    print("\n=== A2  reconstruction: psi vs prior-median null ===")
    print(f"  median mse_psi  = {np.median(mse_psi):.4f}  (rv_std^2)")
    print(f"  median mse_null = {np.median(mse_null):.4f}")
    print(f"  ratio psi/null  = {ratio:.4f}   "
          f"({'NEAR-MEAN -- psi adds nothing' if ratio > 0.9 else 'psi beats the null'})")
    print(f"  psi better than null on {float(np.mean(mse_psi < mse_null)):.1%} of curves")

    # ---- A3 -------------------------------------------------------------
    print("\n=== A3  is log10 K miswired? ===")
    lx = np.log10(np.maximum(rv_scale, 1e-9))
    A = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(A, y_true[:, 1], rcond=None)
    lin_pred = A @ coef
    r2_lin = r2(y_true[:, 1], lin_pred)
    r2_psi_K = r2(y_true[:, 1], psi_y[:, 1])
    r2_psi_P = r2(y_true[:, 0], psi_y[:, 0])
    r2_psi_e = r2(y_true[:, 2], psi_y[:, 2])
    print(f"  log10 K ~ {coef[0]:.4f} + {coef[1]:.4f}*log10 std(y)   R2 = {r2_lin:.4f}")
    print(f"  psi  log10 K                                            R2 = {r2_psi_K:.4f}")
    print(f"  psi  log10 P R2 = {r2_psi_P:.4f}   psi e R2 = {r2_psi_e:.4f}")
    verdict = ("LINEAR MODEL BEATS psi -- suspect a wiring error"
               if (r2_lin > 0.9 and r2_lin > r2_psi_K) else
               "no wiring red flag from this test")
    print(f"  -> {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "n_curves": int(len(curves)),
        "prior_median_theta": med.tolist(),
        "A2": {
            "median_mse_psi": float(np.median(mse_psi)),
            "median_mse_null": float(np.median(mse_null)),
            "ratio_psi_over_null": ratio,
            "frac_psi_better": float(np.mean(mse_psi < mse_null)),
        },
        "A3": {
            "linear_intercept": float(coef[0]), "linear_slope": float(coef[1]),
            "r2_linear_logK": r2_lin, "r2_psi_logK": r2_psi_K,
            "r2_psi_logP": r2_psi_P, "r2_psi_e": r2_psi_e,
            "verdict": verdict,
        },
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
