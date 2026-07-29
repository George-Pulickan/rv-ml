"""Is theta* actually the argmin that eq (2) defines it to be?

eq (2) defines theta* = argmin_theta ||h(theta) - y||^2. In the code it is
computed as 200 Adam steps from theta_bar (from the tabulated values on real
curves). That is a LOCAL descent from one starting point.

The 2026-07-28 multi-start sweep showed the score tail inflating when psi is
refined from the top GLS peaks. There are two possible explanations and they
have opposite consequences:

  (a) multi-start lands on genuinely wrong aliases that happen to fit better
      -> the tail is real ambiguity in the data, CP is right to report it;
  (b) multi-start finds LOWER loss than theta* -> theta* is a local optimum,
      not the argmin, the calibration target is mis-computed, and every
      conformity score in the paper is measured from the wrong point.

This distinguishes them by comparing the two losses directly.

Reports, per domain:
  frac_better        fraction of curves where l(psi') < l(theta*)
  median_ratio       median l(psi')/l(theta*)
  gap_when_better    how much lower, and how far away in log10 P, on the
                     curves where psi' wins -- a big parameter gap with a
                     lower loss is the signature of (b)

Usage
-----
    python scripts/check_theta_star_optimality.py \
        --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
        --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
        --n-syn 300 --refine 50 --multistart 5
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

from conformal import make_real, make_synthetic  # noqa: E402
from conformal_shift import (  # noqa: E402
    _batch_losses,
    _load_mlp_psi,
    multistart_fit_gd,
    surrogate_fit_gd,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402


def feat_matrix(systems: list, feature_cols: list[str]) -> np.ndarray:
    def _val(fr, lsp, c):
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)
    return np.asarray([[_val(s["feat_row"], s["lsp"], c) for c in feature_cols]
                       for s in systems], dtype=float)


def report_domain(name, systems, theta_ref, psi_y, decoder, K, M, gd_steps, gd_lr):
    print(f"\n===== {name}: {len(systems)} systems =====")
    star = np.stack(surrogate_fit_gd(decoder, list(theta_ref), systems, gd_steps, gd_lr))
    refined = np.stack(multistart_fit_gd(decoder, list(psi_y), systems, K, gd_lr, M))
    l_star = _batch_losses(decoder, list(star), systems)
    l_ref = _batch_losses(decoder, list(refined), systems)

    better = l_ref < l_star
    dP = np.abs(refined[:, 0] - star[:, 0])
    out = {
        "n": int(len(systems)),
        "frac_psi_better_than_star": float(np.mean(better)),
        "median_loss_ratio": float(np.median(l_ref / np.maximum(l_star, 1e-12))),
        "median_abs_dlog10P": float(np.median(dP)),
    }
    print(f"  l(psi') < l(theta*) on   : {out['frac_psi_better_than_star']:.1%} of curves")
    print(f"  median l(psi')/l(theta*) : {out['median_loss_ratio']:.3f}")
    if better.any():
        out["when_better"] = {
            "n": int(better.sum()),
            "median_loss_ratio": float(np.median(l_ref[better] / np.maximum(l_star[better], 1e-12))),
            "median_abs_dlog10P": float(np.median(dP[better])),
            "max_abs_dlog10P": float(np.max(dP[better])),
        }
        w = out["when_better"]
        print(f"  where psi' wins (n={w['n']}): loss ratio {w['median_loss_ratio']:.3f}, "
              f"median |dlog10 P| = {w['median_abs_dlog10P']:.4f} dex "
              f"(max {w['max_abs_dlog10P']:.4f})")
        verdict = ("theta* is NOT the argmin on these curves -- the calibration "
                   "target is a local optimum"
                   if w["median_abs_dlog10P"] > 0.01 else
                   "psi' wins by a negligible parameter margin (same optimum)")
    else:
        verdict = "theta* is never beaten -- it is the better optimum everywhere"
    out["verdict"] = verdict
    print(f"  -> {verdict}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n-syn", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--refine", type=int, default=50)
    ap.add_argument("--multistart", type=int, default=5)
    ap.add_argument("--gd-steps", type=int, default=200)
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures" / "paper" / "theta_star_optimality.json")
    args = ap.parse_args()

    decoder = KeplerDecoder().eval()
    psi_predict, _ = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    feature_cols = [c for c in pd.read_csv(args.csv, nrows=1).columns
                    if c not in TARGET_COLUMNS]
    print(f"psi'=GD_{args.refine} from top-{args.multistart} GLS peaks; "
          f"theta*=GD_{args.gd_steps} from reference")

    rep = {"refine": args.refine, "multistart": args.multistart,
           "gd_steps": args.gd_steps, "domains": {}}

    syn = make_synthetic(args.n_syn, args.seed)
    rep["domains"]["synthetic"] = report_domain(
        "synthetic", syn, [s["theta5"] for s in syn],
        psi_predict(feat_matrix(syn, feature_cols)),
        decoder, args.refine, args.multistart, args.gd_steps, args.gd_lr)

    real = make_real(args.real_split, 0.1, 100.0)
    rep["domains"]["real"] = report_domain(
        f"real/{args.real_split}", real, [s["theta5"] for s in real],
        psi_predict(feat_matrix(real, feature_cols)),
        decoder, args.refine, args.multistart, args.gd_steps, args.gd_lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
