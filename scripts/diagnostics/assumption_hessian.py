"""Assumption 3.2 diagnostic: is theta* a strict local minimum of eq (2)?

Draws systems from the prior, fits the surrogate label theta* by gradient
descent (procedure (3)), and forms the Hessian of the summed squared
reconstruction loss

    L(theta) = sum_t ( y_t - h(theta)_t )^2                              (eq 2)

at theta*, in *physical* coordinates.  Reports lambda_min, the fraction of
draws with a positive-definite Hessian, and the condition number.

Two coordinate sets, selected with --coords:

  PKew  (log10 P, log10 K, e, omega)  -- the four CP coordinates
  PKe   (log10 P, log10 K, e)         -- omega marginalised out

The second exists because omega is not identifiable at this SNR (median CP
half-width > pi).  Nicolo, 2026-07-26: "The noise and the non-identifiability
of omega may indeed mess up everything ... What happens if we ignore omega?"

Note the 5-vector (log10 P, log10 K, e, cos w, sin w) is a redundant encoding:
the decoder sees omega only through atan2(sin w, cos w), so the Hessian in the
5-vector has an exact null direction along the (cos w, sin w) radius and is
singular by construction.  An earlier kappa(H) ~ 1e6 figure was that artefact.
Always differentiate in the physical coordinates below.

Usage
-----
    python -m scripts.diagnostics.assumption_hessian --n-draws 25 --steps 200
    python -m scripts.diagnostics.assumption_hessian --coords PKe --steps 200 1000 4000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformal import Scorer, make_synthetic  # noqa: E402
from conformal_shift import surrogate_fit_gd  # noqa: E402

COORD_SETS = {
    "PKew": ["log10_P", "log10_K", "e", "omega"],
    "PKe": ["log10_P", "log10_K", "e"],
}


def _theta5_from_physical(phys: torch.Tensor, omega_fixed: float | None) -> torch.Tensor:
    """(log10 P, log10 K, e[, omega]) -> the decoder's 5-vector, differentiably."""
    if omega_fixed is None:
        log_p, log_k, e, omega = phys[0], phys[1], phys[2], phys[3]
    else:
        log_p, log_k, e = phys[0], phys[1], phys[2]
        omega = torch.as_tensor(omega_fixed, dtype=phys.dtype)
    return torch.stack([log_p, log_k, e, torch.cos(omega), torch.sin(omega)])


def loss_at(decoder, phys: torch.Tensor, curve: dict, omega_fixed: float | None) -> torch.Tensor:
    """eq (2): summed squared residual between the curve and h(theta)."""
    th = _theta5_from_physical(phys, omega_fixed).unsqueeze(0)
    t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0)
    rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0)
    mask = torch.from_numpy(curve["mask"]).unsqueeze(0)
    t_span = torch.tensor([curve["t_span"]], dtype=torch.float32)
    t_min = torch.tensor([curve["t_min"]], dtype=torch.float32)
    rv_std = torch.tensor([curve["rv_std"]], dtype=torch.float32)
    rv_pred = decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)[0]
    m = mask[0] > 0.5
    resid = (rv_obs[0][m] - rv_pred[m])
    return (resid ** 2).sum()


def hessian_stats(decoder, theta5_star: np.ndarray, curve: dict, coords: str) -> dict:
    """lambda_min / kappa of the eq-(2) Hessian at theta*, in physical coords."""
    omega_star = float(np.arctan2(theta5_star[4], theta5_star[3]))
    if coords == "PKew":
        phys0 = np.array([theta5_star[0], theta5_star[1], theta5_star[2], omega_star])
        omega_fixed = None
    else:
        phys0 = np.array([theta5_star[0], theta5_star[1], theta5_star[2]])
        omega_fixed = omega_star

    phys = torch.tensor(phys0, dtype=torch.float32, requires_grad=True)
    H = torch.autograd.functional.hessian(
        lambda p: loss_at(decoder, p, curve, omega_fixed), phys
    )
    H = 0.5 * (H + H.T)  # symmetrise away autograd round-off
    ev = torch.linalg.eigvalsh(H.double()).numpy()
    lam_min, lam_max = float(ev.min()), float(ev.max())
    kappa = float(abs(lam_max) / abs(lam_min)) if lam_min != 0 else float("inf")
    return {"lambda_min": lam_min, "lambda_max": lam_max, "kappa": kappa,
            "pd": bool(lam_min > 0)}


def run(n_draws: int, steps_list: list[int], coords: str, seed: int) -> dict:
    decoder = Scorer().decoder
    systems = make_synthetic(n_draws, seed)
    print(f"drew {len(systems)} synthetic systems (seed {seed}); coords={coords} "
          f"({len(COORD_SETS[coords])}-D)")

    out = {"coords": coords, "n_draws": len(systems), "seed": seed, "by_steps": {}}
    for steps in steps_list:
        stars = surrogate_fit_gd(decoder, [s["theta5"] for s in systems], systems,
                                 steps=steps)
        stats = [hessian_stats(decoder, np.asarray(th, dtype=float), s["curve"], coords)
                 for s, th in zip(systems, stars)]
        lam = np.array([st["lambda_min"] for st in stats])
        pd_frac = float(np.mean([st["pd"] for st in stats]))
        kap = np.array([st["kappa"] for st in stats if np.isfinite(st["kappa"])])
        rec = {
            "pd_fraction": pd_frac,
            "lambda_min_median": float(np.median(lam)),
            "lambda_min_min": float(lam.min()),
            "kappa_median": float(np.median(kap)) if kap.size else None,
        }
        out["by_steps"][str(steps)] = rec
        print(f"  steps={steps:5d}  PD fraction={pd_frac:5.2f}  "
              f"median lambda_min={rec['lambda_min_median']:+.4g}  "
              f"min={rec['lambda_min_min']:+.4g}  "
              f"median kappa={rec['kappa_median']:.4g}" if rec["kappa_median"]
              else f"  steps={steps:5d}  PD fraction={pd_frac:5.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-draws", type=int, default=25)
    ap.add_argument("--steps", type=int, nargs="+", default=[200, 1000, 4000],
                    help="GD step counts to sweep (convergence sweep)")
    ap.add_argument("--coords", choices=sorted(COORD_SETS), default="PKew")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args()

    res = run(args.n_draws, args.steps, args.coords, args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
