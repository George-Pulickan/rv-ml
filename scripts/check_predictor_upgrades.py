"""Three candidate predictor upgrades, measured against the current best.

Current best (2026-07-28): psi' = GD_50 from the top-5 GLS peaks. It fixes the
period (median error 0.179 -> 0.0025 dex) but leaves the amplitude wrong -- on
51 Peg the refined curve has the right period and a third of the true
amplitude, MSE 479 against 52 for the catalogue.

That is structural, not a tuning failure. models/kepler_torch.py refits T_peri
and gamma analytically each forward pass but takes K straight from theta
(`K = 10.0 ** log10_K`) and leaves it to gradient descent -- even though h is
*linear* in K:

    rv(t) = K * shape(t; P, e, omega, T_peri) + gamma

so the optimal (K, gamma) is a closed-form two-parameter least squares, exactly
the trick already used for gamma alone.

Variants compared, all against the SAME theta* (standard decoder, GD_200 from
the reference) so the scores stay comparable:

    A  K=50,  M=5                      current best
    B  K=50,  M=5  + analytic K        upgrade 1
    C  K=200, M=5                      upgrade 2 (never tested: the ladder went
                                       to 200 single-start, the sweep to M=5 at
                                       K<=50)
    D  K=200, M=5  + analytic K        both

The analytic refit is applied post-hoc here rather than inside the decoder, so
this measures the idea without perturbing the shared model that theta* and every
committed result depend on. If it wins, it gets wired in properly.

Upgrade 3 (predict log10 K - log10 sigma_RV) is deliberately NOT tested here: if
the analytic refit works, K stops being predicted at all and the
reparameterisation is moot. Only worth doing if B and D fail.

Usage
-----
    python scripts/check_predictor_upgrades.py \
        --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
        --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
        --n-syn 150
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

from conformal import make_real, make_synthetic  # noqa: E402
from conformal_shift import (  # noqa: E402
    _batch_losses,
    _load_mlp_psi,
    multistart_fit_gd,
    surrogate_fit_gd,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder, fit_t_peri, rv_keplerian  # noqa: E402

COORDS3 = ["log10_P", "log10_K", "e"]


def feat_matrix(systems, fcols):
    def _val(fr, lsp, c):
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)
    return np.asarray([[_val(s["feat_row"], s["lsp"], c) for c in fcols]
                       for s in systems], dtype=float)


@torch.no_grad()
def refit_k_analytic(theta5s, systems):
    """Replace log10_K by its closed-form least-squares value given P, e, omega.

    h is linear in K, so with T_peri fixed the optimal (K, gamma) solves a
    2-parameter LS. K is clamped non-negative: a negative slope means the shape
    is anti-correlated with the data, and the best fit under K >= 0 is flat.
    """
    out = np.asarray(theta5s, dtype=float).copy()
    for i, (th, s) in enumerate(zip(out, systems)):
        c = s["curve"]
        t_norm = torch.from_numpy(c["t_norm"][None, :])
        mask = torch.from_numpy(c["mask"][None, :])
        rv_ms = torch.from_numpy(c["rv_obs"][None, :]) * float(c["rv_std"])
        t_days = t_norm * float(c["t_span"]) + float(c["t_min"])
        P = torch.tensor([10.0 ** th[0]], dtype=torch.float32)
        K0 = torch.tensor([10.0 ** th[1]], dtype=torch.float32)
        e = torch.tensor([float(np.clip(th[2], 0.0, 0.99))], dtype=torch.float32)
        omega = torch.tensor([float(np.arctan2(th[4], th[3]))], dtype=torch.float32)
        t_peri = fit_t_peri(t_days, rv_ms, mask, P, K0, e, omega)
        shape = rv_keplerian(t_days, P, torch.ones_like(P), e, omega, t_peri)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1)
        sm = (shape * mask).sum(dim=1, keepdim=True) / n
        rm = (rv_ms * mask).sum(dim=1, keepdim=True) / n
        cov = (((shape - sm) * (rv_ms - rm)) * mask).sum(dim=1, keepdim=True) / n
        var = (((shape - sm) ** 2) * mask).sum(dim=1, keepdim=True) / n
        k_hat = float((cov / var.clamp(min=1e-12)).clamp(min=1e-6).item())
        out[i, 1] = float(np.log10(max(k_hat, 1e-6)))
    return out


def summarise(name, th, theta_star, decoder, systems, tab_mse=None):
    d = np.abs(np.asarray(th)[:, :3] - theta_star[:, :3])
    mse = _batch_losses(decoder, list(th), systems)
    row = {
        "median_mse": float(np.median(mse)),
        "med_dev": {c: float(np.median(d[:, k])) for k, c in enumerate(COORDS3)},
        "q90_dev": {c: float(np.quantile(d[:, k], 1 - 0.10 / 3)) for k, c in enumerate(COORDS3)},
    }
    extra = ""
    if tab_mse is not None:
        row["frac_at_or_below_tab"] = float(np.mean(mse <= tab_mse))
        extra = f"  beats_tab={row['frac_at_or_below_tab']:.3f}"
    print(f"  {name:<26} mse={row['median_mse']:7.4f}  "
          f"med dP/dK/de = {row['med_dev']['log10_P']:.4f}/"
          f"{row['med_dev']['log10_K']:.4f}/{row['med_dev']['e']:.4f}  "
          f"q90 dK={row['q90_dev']['log10_K']:.4f}{extra}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n-syn", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--multistart", type=int, default=5)
    ap.add_argument("--gd-steps", type=int, default=200)
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures" / "paper" / "predictor_upgrades.json")
    args = ap.parse_args()

    decoder = KeplerDecoder().eval()
    psi_predict, _ = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    fcols = [c for c in pd.read_csv(args.csv, nrows=1).columns if c not in TARGET_COLUMNS]
    rep = {"multistart": args.multistart, "domains": {}}

    for name, systems in (("synthetic", make_synthetic(args.n_syn, args.seed)),
                          (f"real/{args.real_split}", make_real(args.real_split, 0.1, 100.0))):
        print(f"\n===== {name}: {len(systems)} systems =====")
        psi_y = psi_predict(feat_matrix(systems, fcols))
        t0 = time.time()
        star = np.stack(surrogate_fit_gd(decoder, [s["theta5"] for s in systems],
                                         systems, args.gd_steps, args.gd_lr))
        print(f"  theta* in {time.time() - t0:.0f}s")
        tab_mse = _batch_losses(decoder, [s["theta5"] for s in systems], systems)
        print(f"  reference: median tabulated/theta_bar mse = {np.median(tab_mse):.4f}, "
              f"theta* mse = {np.median(_batch_losses(decoder, list(star), systems)):.4f}")

        dom = {}
        dom["psi_raw"] = summarise("psi (no refine)", psi_y, star, decoder, systems, tab_mse)
        for K in (50, 200):
            t0 = time.time()
            th = np.stack(multistart_fit_gd(decoder, list(psi_y), systems, K,
                                            args.gd_lr, args.multistart))
            print(f"  [K={K} M={args.multistart} in {time.time() - t0:.0f}s]")
            dom[f"K{K}_M{args.multistart}"] = summarise(
                f"K={K} M={args.multistart}", th, star, decoder, systems, tab_mse)
            th_k = refit_k_analytic(th, systems)
            dom[f"K{K}_M{args.multistart}_analyticK"] = summarise(
                f"K={K} M={args.multistart} +analyticK", th_k, star, decoder, systems, tab_mse)
        rep["domains"][name] = dom

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
