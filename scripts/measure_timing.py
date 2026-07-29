"""Measure the inference cost of psi, psi' and theta* on the real test curves.

The paper asserts computational savings but has never measured them; the
figures quoted in earlier drafts (6.4 ms / 4.0 s / ~600x) are recorded in the
handover as unverified, and are wrong now that psi' takes 200 descent steps
rather than a single forward pass.

Reports, per system, on one CPU core:
  psi     featurisation + MLP forward pass          (amortised predictor)
  psi'    GD_K multi-start refinement + K refit     (what the paper reports)
  theta*  GD fit from theta_bar                     (the surrogate label)

theta* is the honest reference for "what a per-target fit costs" in our own
pipeline; a full MCMC posterior is far more expensive again, but we do not
measure one here and do not quote a ratio against it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conformal_shift import (  # noqa: E402
    _load_mlp_psi, make_real, multistart_fit_gd, refit_k_analytic,
    surrogate_fit_gd,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from paper_rv_figures import _feat_row_for_system  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--psi-refine", type=int, default=200)
    ap.add_argument("--psi-multistart", type=int, default=5)
    ap.add_argument("--psi-refit-k", action="store_true")
    ap.add_argument("--gd-steps", type=int, default=200)
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    psi_predict, _ = _load_mlp_psi(args.checkpoint, "cpu")
    systems = make_real(args.real_split, 0.1, 100.0)
    n = len(systems)
    print(f"{n} real systems in split {args.real_split!r}")

    df = pd.read_csv(args.csv, nrows=1)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]

    # --- psi: featurisation + forward pass ---------------------------------
    t_feat, t_fwd = [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        X = np.asarray([_feat_row_for_system(s, feature_cols) for s in systems],
                       dtype=float)
        t1 = time.perf_counter()
        th_psi = psi_predict(X)
        t2 = time.perf_counter()
        t_feat.append(t1 - t0)
        t_fwd.append(t2 - t1)
    feat_s, fwd_s = float(np.median(t_feat)), float(np.median(t_fwd))

    # --- psi': multi-start GD refinement (+ optional K refit) --------------
    dec = KeplerDecoder().eval()
    t_ref = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = multistart_fit_gd(dec, list(th_psi), systems, args.psi_refine,
                                args.gd_lr, args.psi_multistart)
        if args.psi_refit_k:
            out = refit_k_analytic(list(np.asarray(out, dtype=float)), systems)
        t_ref.append(time.perf_counter() - t0)
    ref_s = float(np.median(t_ref))

    # --- theta*: GD fit from theta_bar -------------------------------------
    t_star = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        surrogate_fit_gd(dec, [s["theta5"] for s in systems], systems,
                         args.gd_steps, args.gd_lr)
        t_star.append(time.perf_counter() - t0)
    star_s = float(np.median(t_star))

    psi_total = feat_s + fwd_s
    blob = {
        "n_systems": n,
        "repeats": args.repeats,
        "threads": 1,
        "config": {"psi_refine": args.psi_refine,
                   "psi_multistart": args.psi_multistart,
                   "psi_refit_k": bool(args.psi_refit_k),
                   "gd_steps": args.gd_steps},
        "per_system_ms": {
            "featurisation": 1e3 * feat_s / n,
            "psi_forward": 1e3 * fwd_s / n,
            "psi_total": 1e3 * psi_total / n,
            "psi_prime_refine": 1e3 * ref_s / n,
            "theta_star_gd": 1e3 * star_s / n,
        },
        "ratios": {
            "psi_prime_over_psi": ref_s / psi_total,
            "theta_star_over_psi": star_s / psi_total,
            "theta_star_over_psi_prime": star_s / ref_s,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
