"""Does GD initialised at psi(y) reach the same optimum as GD initialised at
theta_bar -- the one that *defines* theta*?

This decides whether Nicolo's proposed refined predictor psi'(y) = GD_K(psi(y))
is viable or degenerate.

  - If psi'(y) always lands on theta*, the conformity score s = |psi'(y) - theta*|
    collapses to zero and the prediction region becomes vacuously tight: it
    would be measuring optimiser agreement, not parameter uncertainty.
  - If it lands elsewhere on a material fraction of curves, the score keeps its
    meaning and the CP region becomes, in Nicolo's words, a probabilistic
    guarantee of global optimality.

theta* here is exactly the paper's label: GD from theta_bar on synthetic curves
(from the tabulated values on real ones), same decoder, same objective, same
learning rate.

Reported per GD-step-count K:
  med|psi'-theta*|   the conformity score, per coordinate -> sets interval width
  same-basin frac    fraction agreeing with theta* to a tolerance in log10 P
  loss ratio         l(psi')/l(theta*); <1 means psi' found a BETTER optimum
                     than the label, which would mean theta* is not the argmin
  theta_bar covered  fraction with |psi' - theta_bar| <= |psi' - theta*|, i.e.
                     Nicolo's criterion for cross-validating K

Usage
-----
    python scripts/check_gd_from_psi.py \
        --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
        --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
        --n-syn 200 --out figures/paper/gd_from_psi_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_generation"))

from conformal import _curve_from_x, make_real  # noqa: E402
from conformal_shift import _load_mlp_psi, surrogate_fit_gd  # noqa: E402
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402
from generate_synthetic_regression_csv import (  # noqa: E402
    corpus_orbital_params,
    replay_synthetic_sample,
)

COORD_NAMES = ["log10_P", "log10_K", "e"]


def _loss(decoder, theta5s, systems) -> np.ndarray:
    """Mean squared masked residual in rv_std units, per system."""
    out = np.zeros(len(systems), dtype=float)
    with torch.no_grad():
        for i, (th, s) in enumerate(zip(theta5s, systems)):
            c = s["curve"]
            t_norm = torch.from_numpy(c["t_norm"][None, :])
            rv_obs = torch.from_numpy(c["rv_obs"][None, :])
            mask = torch.from_numpy(c["mask"][None, :])
            rv_pred = decoder(
                torch.as_tensor(np.asarray(th)[None, :], dtype=torch.float32),
                t_norm,
                torch.tensor([c["t_span"]], dtype=torch.float32),
                torch.tensor([c["t_min"]], dtype=torch.float32),
                rv_obs,
                torch.tensor([c["rv_std"]], dtype=torch.float32),
                mask,
            )
            resid = (rv_obs - rv_pred) ** 2
            n = float(mask.sum().clamp(min=1.0))
            out[i] = float((resid * mask).sum() / n)
    return out


def _coords(th: np.ndarray) -> np.ndarray:
    """(log10_P, log10_K, e) from the 5-vector."""
    a = np.asarray(th, dtype=float)
    return a[..., :3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="MLP psi checkpoint (required -- the default in other "
                         "scripts points at a degenerate near-mean predictor)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n-syn", type=int, default=200)
    # MUST match the CSV's generation seed (conformal_shift.py uses 123). A
    # wrong seed pairs CSV feature rows with a different system's curve and is
    # silent apart from implausible R2. Guarded below.
    ap.add_argument("--csv-seed", type=int, default=123)
    ap.add_argument("--gd-steps", type=int, default=200,
                    help="steps used to build theta* (the label)")
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--snapshots", default="0,5,10,20,50,100,200",
                    help="ladder of GD step counts K taken from psi(y)")
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "paper" / "gd_from_psi_check.json")
    args = ap.parse_args()

    ladder = [int(s) for s in args.snapshots.split(",") if s.strip()]
    decoder = KeplerDecoder().eval()
    psi_predict, norm_stats = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    print(f"psi: {args.checkpoint}  (in_dim={norm_stats['in_dim']})")

    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    n_rows = len(df)
    n_use = min(args.n_syn, n_rows)
    print(f"csv rows={n_rows}, using {n_use}")

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "csv": str(args.csv),
        "gd_steps_label": args.gd_steps,
        "gd_lr": args.gd_lr,
        "snapshots": ladder,
        "domains": {},
    }

    # ---------------- synthetic: theta_bar is known -------------------------
    print(f"\n=== synthetic: replaying {n_use} rows ===")
    params = corpus_orbital_params(args.csv_seed, n_rows)
    # Pairing CSV features with replayed curves is only valid if replayed row i
    # IS csv row i. Wrong seed -> different system, silent except for negative
    # R2. Bit me on 2026-07-28.
    bad = [i for i in (0, 1, 7, 50, 777) if i < n_rows and abs(
        float(df.iloc[i]["log10_P"]) - float(replay_synthetic_sample(
            i, args.csv_seed, n_rows, f_multi=0.0, params=params)[2][0])) > 1e-6]
    if bad:
        raise SystemExit(f"replay/CSV mismatch at rows {bad} with --csv-seed "
                         f"{args.csv_seed}; the CSV uses seed 123.")
    print(f"replay/CSV pairing verified at seed {args.csv_seed}")
    systems, theta_bars = [], []
    for i in range(n_use):
        x, _lsp, theta, info = replay_synthetic_sample(
            i, args.csv_seed, n_rows, f_multi=0.0, params=params)
        systems.append({"curve": _curve_from_x(x, info)})
        theta_bars.append(np.asarray(theta, dtype=float))
    theta_bars = np.stack(theta_bars)

    print(f"theta* : GD {args.gd_steps} steps from theta_bar ...")
    theta_star = np.stack(surrogate_fit_gd(decoder, list(theta_bars), systems,
                                           args.gd_steps, args.gd_lr))
    loss_star = _loss(decoder, theta_star, systems)

    X = df.loc[: n_use - 1, feature_cols].to_numpy(dtype=float)
    psi_y = psi_predict(X)
    print(f"psi(y) computed for {len(psi_y)} rows")

    report["domains"]["synthetic"] = _sweep(
        decoder, psi_y, theta_star, theta_bars, loss_star, systems, ladder,
        args.gd_lr, label="synthetic")

    # ---------------- real: theta_bar is the tabulated value ---------------
    print(f"\n=== real ({args.real_split}) ===")
    try:
        real = make_real(args.real_split, 0.1, 100.0)
        r_systems = [{"curve": s["curve"]} for s in real]
        r_tab = np.stack([np.asarray(s["theta5"], dtype=float) for s in real])
        print(f"real systems: {len(r_systems)}")
        r_star = np.stack(surrogate_fit_gd(decoder, list(r_tab), r_systems,
                                           args.gd_steps, args.gd_lr))
        r_loss_star = _loss(decoder, r_star, r_systems)
        # make_real already carries the built feature row and the GLS array;
        # resolve column names exactly as paper_rv_figures._feat_row_for_system.
        def _val(fr: dict, lsp: np.ndarray, c: str) -> float:
            if c in fr:
                return float(fr[c])
            if c.startswith("lsp_"):
                return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
            raise KeyError(c)

        Xr = np.asarray(
            [[_val(s["feat_row"], s["lsp"], c) for c in feature_cols] for s in real],
            dtype=float)
        r_psi = psi_predict(Xr)
        report["domains"]["real"] = _sweep(
            decoder, r_psi, r_star, r_tab, r_loss_star, r_systems, ladder,
            args.gd_lr, label="real")
    except Exception as exc:  # noqa: BLE001
        print(f"  real-domain sweep skipped: {type(exc).__name__}: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


def _sweep(decoder, psi_y, theta_star, theta_ref, loss_star, systems, ladder,
           gd_lr, label: str) -> dict:
    """Run the GD ladder from psi(y) and tabulate agreement with theta*."""
    cs_star = _coords(theta_star)
    cs_ref = _coords(theta_ref)
    d0 = np.abs(_coords(psi_y) - cs_star)
    print(f"\n--- {label}: |psi(y) - theta*| (K=0 baseline) ---")
    print("    median per coord: " + "  ".join(
        f"{c}={np.median(d0[:, k]):.4f}" for k, c in enumerate(COORD_NAMES)))

    out: dict = {"n": int(len(systems)), "by_K": {}}
    for K in ladder:
        if K == 0:
            refined = np.asarray(psi_y, dtype=float)
        else:
            refined = np.stack(surrogate_fit_gd(decoder, list(psi_y), systems,
                                                K, gd_lr))
        cr = _coords(refined)
        d_star = np.abs(cr - cs_star)
        d_ref = np.abs(cr - cs_ref)
        loss_r = _loss(decoder, refined, systems)
        ratio = loss_r / np.maximum(loss_star, 1e-12)

        entry = {
            "median_abs_dev_from_theta_star": {
                c: float(np.median(d_star[:, k])) for k, c in enumerate(COORD_NAMES)},
            "median_abs_dev_from_theta_ref": {
                c: float(np.median(d_ref[:, k])) for k, c in enumerate(COORD_NAMES)},
            "same_basin_frac_log10P": {
                f"tol_{t}": float(np.mean(d_star[:, 0] <= t))
                for t in (0.01, 0.05, 0.10)},
            "same_basin_frac_all3_tol0.05": float(
                np.mean((d_star <= 0.05).all(axis=1))),
            "loss_ratio_median": float(np.median(ratio)),
            "frac_better_than_theta_star": float(np.mean(loss_r < loss_star)),
            "frac_ref_closer_than_star": float(
                np.mean(d_ref[:, 0] <= d_star[:, 0])),
        }
        out["by_K"][str(K)] = entry
        print(f"  K={K:<4} med|d-theta*| P/K/e = "
              f"{entry['median_abs_dev_from_theta_star']['log10_P']:.4f}/"
              f"{entry['median_abs_dev_from_theta_star']['log10_K']:.4f}/"
              f"{entry['median_abs_dev_from_theta_star']['e']:.4f}"
              f"   same-basin(0.05, all 3)={entry['same_basin_frac_all3_tol0.05']:.3f}"
              f"   loss ratio={entry['loss_ratio_median']:.3f}"
              f"   better than theta*={entry['frac_better_than_theta_star']:.3f}")
    return out


if __name__ == "__main__":
    main()
