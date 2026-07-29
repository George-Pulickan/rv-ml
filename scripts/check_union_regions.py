"""Would a union of narrow intervals beat one wide box?

The reported log10 P region is a single interval spanning ~1.6 dex. It has to
be that wide because it must reach an alias at 2P or P/2 -- but the decade
between the modes is strongly excluded by the data. Nearly all of that width is
empty space between peaks.

The period posterior for a sparsely sampled RV curve is genuinely multimodal, so
forcing the prediction set to be an interval is a shape mismatch, not a
statistical necessity. Conformal prediction does not require a convex set: the
score can be the distance to the NEAREST of M candidates, and the region is then
the union of M small intervals around them. Coverage is calibrated exactly as
before; only the shape changes.

This measures the pay-off before anything is committed:

  score_box    |psi'(y) - theta*|                   -> one interval of half-width q
  score_union  min_j |cand_j(y) - theta*|           -> M intervals of half-width q_u

and compares TOTAL MEASURE, 2*q against 2*M*q_u, at matched empirical coverage.
If the union does not win on measure it is not worth the complication.

Candidates are the multi-start GD fits from the top-M GLS peaks -- the same
objects --psi-multistart already computes, just kept all instead of argmin.

Usage
-----
    python scripts/check_union_regions.py \
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
    _load_mlp_psi,
    _top_peak_periods,
    surrogate_fit_gd,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402

COORDS3 = ["log10_P", "log10_K", "e"]


def feat_matrix(systems, feature_cols):
    def _val(fr, lsp, c):
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)
    return np.asarray([[_val(s["feat_row"], s["lsp"], c) for c in feature_cols]
                       for s in systems], dtype=float)


def candidate_set(decoder, psi_y, systems, K, lr, M):
    """All M+1 refined candidates per system (not just the lowest-loss one)."""
    cands = [np.stack(surrogate_fit_gd(decoder, list(psi_y), systems, K, lr))]
    peaks = [_top_peak_periods(s.get("lsp", np.array([])), M) for s in systems]
    for j in range(M):
        init = []
        for th, pk in zip(psi_y, peaks):
            t2 = np.asarray(th, dtype=float).copy()
            if j < len(pk) and pk[j] > 0:
                t2[0] = float(np.log10(pk[j]))
            init.append(t2)
        cands.append(np.stack(surrogate_fit_gd(decoder, init, systems, K, lr)))
    return np.stack(cands, axis=1)          # (n_systems, M+1, 5)


def merged_measure(centres: np.ndarray, half: float) -> float:
    """Total length of the union of [c - half, c + half], overlaps merged.

    The first version of this script charged the union C disjoint intervals,
    which is the worst case and far too pessimistic: multi-start candidates
    frequently converge to the same optimum, so the intervals overlap heavily
    and the real measure is much smaller. Merging is the honest comparison.
    """
    iv = sorted((float(c) - half, float(c) + half) for c in centres)
    total, lo, hi = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + (hi - lo)


def analyse(name, cands, theta_star, alpha=0.10):
    n, C, _ = cands.shape
    q_lvl = 1.0 - alpha / 3.0               # Bonferroni divisor d = 3
    print(f"\n===== {name}: n={n}, {C} candidates/system, alpha={alpha} =====")
    out = {"n": int(n), "n_candidates": int(C), "alpha": alpha, "coords": {}}
    for k, c in enumerate(COORDS3):
        star = theta_star[:, k]
        # box: distance from the lowest-loss candidate (index 0 = the psi start,
        # matching --psi-multistart 0)
        d_box = np.abs(cands[:, 0, k] - star)
        q_box = float(np.quantile(d_box, q_lvl))
        # union: distance to the NEAREST candidate
        d_uni = np.min(np.abs(cands[:, :, k] - star[:, None]), axis=1)
        q_uni = float(np.quantile(d_uni, q_lvl))

        meas_box = 2.0 * q_box
        per_system = np.array([merged_measure(cands[i, :, k], q_uni)
                               for i in range(n)], dtype=float)
        meas_uni = float(np.median(per_system))
        # How many intervals the union really costs, after overlaps merge.
        eff_modes = meas_uni / max(2.0 * q_uni, 1e-12)
        ratio = meas_uni / max(meas_box, 1e-12)
        out["coords"][c] = {
            "q_box": q_box, "q_union": q_uni,
            "measure_box": meas_box,
            "measure_union_merged_median": meas_uni,
            "measure_union_worstcase": 2.0 * C * q_uni,
            "effective_modes": eff_modes,
            "ratio_union_over_box": ratio,
            "cover_box": float(np.mean(d_box <= q_box)),
            "cover_union": float(np.mean(d_uni <= q_uni)),
        }
        print(f"  {c:<9} q_box={q_box:7.4f} q_union={q_uni:7.4f}  "
              f"measure {meas_box:7.4f} -> {meas_uni:7.4f} ({ratio:.2f}x)  "
              f"eff_modes={eff_modes:.2f}/{C}  "
              f"{'UNION WINS' if ratio < 1 else 'box wins'}")
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
                    default=ROOT / "figures" / "paper" / "union_regions.json")
    args = ap.parse_args()

    decoder = KeplerDecoder().eval()
    psi_predict, _ = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    fcols = [c for c in pd.read_csv(args.csv, nrows=1).columns if c not in TARGET_COLUMNS]
    rep = {"refine": args.refine, "multistart": args.multistart, "domains": {}}

    for name, systems in (("synthetic", make_synthetic(args.n_syn, args.seed)),
                          (f"real/{args.real_split}",
                           make_real(args.real_split, 0.1, 100.0))):
        psi_y = psi_predict(feat_matrix(systems, fcols))
        star = np.stack(surrogate_fit_gd(decoder, [s["theta5"] for s in systems],
                                         systems, args.gd_steps, args.gd_lr))
        cands = candidate_set(decoder, psi_y, systems, args.refine, args.gd_lr,
                              args.multistart)
        rep["domains"][name] = analyse(name, cands, star)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
