"""
conformal.py — Step 6: Unsupervised Conformal Prediction for Kepler parameters.

Implements the unsupervised CP of the project's Overleaf draft (§2.2.1): turn the
point predictions of the Step-5 regressor into prediction *sets* with a
finite-sample coverage guarantee, WITHOUT ever using ground-truth parameters at
calibration time.

Method
------
Point predictor (Step 5):  theta_hat = phi(y) = RF(features(y)).
Conformity score (eq 8, unsupervised — needs no true theta):

    s(theta, y) = || h_kepler(theta) - y ||          (reconstruction residual)

where h_kepler is the fixed Kepler integrator (models/kepler_torch.KeplerDecoder),
which refits the phase (t_peri) and offset (gamma) analytically, so the score
measures parameter mismatch, not alignment. Working in each curve's rv_std units
normalises the per-system scale.

Split-conformal calibration on a set of curves (surrogate label theta_hat = phi(y),
eq 7): q = the Bonferroni (1 - alpha/d) quantile of the calibration scores
{ s(theta_hat_j, y_j) }.  The per-coordinate prediction set (eq 9) fixes the other
coordinates at theta_hat and varies coordinate i:

    Gamma_{alpha,i}(y) = { theta_i : s(theta_hat with coord i -> theta_i, y) <= q }

Guarantee (eq 12): Prob(theta_bar in Gamma_alpha) >= 1 - alpha, jointly over the d
coordinates via the Bonferroni correction.

Everything distributional is taken from the empirical corpus histograms H
(synthetic_dataset), never from ad-hoc assumptions:
  * calibration/test synthetic curves are drawn from H (justifies Assumption 2.2,
    exchangeability), and
  * every parameter search grid spans the empirical support of H (period mixture,
    eccentricity histogram, K prior range); omega is uniform on [0, 2pi) because
    the corpus carries no preferred periastron orientation.

Experiments
-----------
  E1  coverage: empirical coverage vs nominal 1 - alpha, calibrated on synthetic
      and tested on synthetic (in-distribution) and on real systems (covariate
      shift). Reports per-coordinate coverage, joint coverage, and set widths.
  E2  monotonicity (Assumption 2.3): mean score vs signed offset theta_i - theta_bar_i
      per coordinate. Expected monotone ("V") for P/K/e, flat for omega (which our
      recovery experiments show is unidentifiable) -> its CP set is maximally wide
      but still valid.

Usage
-----
    python conformal.py                       # E1 + E2, default sizes
    python conformal.py --n-cal 400 --n-test 400 --grid 41
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from preprocess import RVDataset
from models.kepler_torch import KeplerDecoder
from synthetic_dataset import (
    _K_MAX_MS,
    _K_MIN_MS,
    _sample_eccentricity,
    _sample_orbital_params,
    _sample_period,
    generate_one,
)

ROOT = Path(__file__).resolve().parent
SG = ROOT / "synthetic_generation"
if str(SG) not in sys.path:
    sys.path.insert(0, str(SG))

from generate_synthetic_regression_csv import TARGET_COLUMNS  # noqa: E402
from train_regression_models import _build  # noqa: E402
from eval_omega_nn_vs_rf import FEATURES, _summary_row  # noqa: E402
from generate_synthetic_regression_csv import _masked_observations  # noqa: E402

# CP operates on the four physical coordinates; (cos w, sin w) are a redundant
# encoding of the single angle omega, so we vary omega as one coordinate.
COORDS = ["log10_P", "log10_K", "e", "omega"]
D = len(COORDS)


# ---------------------------------------------------------------------------
# System construction: curve tensors (for the decoder) + features + true theta
# ---------------------------------------------------------------------------


def _theta_to_omega(theta5: np.ndarray) -> float:
    return float(np.arctan2(theta5[4], theta5[3]))


def _curve_from_x(x: np.ndarray, info: dict) -> dict:
    return {
        "t_norm": x[0].astype(np.float32),
        "rv_obs": x[1].astype(np.float32),
        "mask": x[3].astype(np.float32),
        "t_span": float(info["t_span_days"]),
        "t_min": float(info["t_min_days"]),
        "rv_std": float(info["rv_std_ms"]),
    }


def make_synthetic(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n)
    systems = []
    for i in range(n):
        p = {k: float(v[i]) for k, v in params.items()}
        x, lsp, theta, info = generate_one(p, np.random.default_rng(seed + 7_000 + i), f_multi=0.0)
        xm = _masked_observations(x)
        feats = _summary_row(xm, info, lsp)
        systems.append({
            "curve": _curve_from_x(x, info),
            "features": np.array([feats[c] for c in FEATURES], dtype=float),
            "theta5": np.asarray(theta, dtype=float),
        })
    return systems


def make_real(split: str, sigma_min: float, sigma_max: float) -> list[dict]:
    ds = RVDataset(split, normalize=False, single_planet=True)
    systems = []
    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue
        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
        if not (sigma_min <= med_sigma <= sigma_max):
            continue
        feats = _summary_row(xm, info, lsp)
        systems.append({
            "curve": _curve_from_x(x, info),
            "features": np.array([feats[c] for c in FEATURES], dtype=float),
            "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
        })
    return systems


# ---------------------------------------------------------------------------
# Conformity score via the fixed Kepler decoder
# ---------------------------------------------------------------------------


class Scorer:
    def __init__(self):
        self.decoder = KeplerDecoder().eval()

    @torch.no_grad()
    def score(self, theta5: np.ndarray, curve: dict) -> np.ndarray:
        """Reconstruction residual (rv_std units) for a batch of candidate theta.

        theta5 : (G, 5) physical [log10_P, log10_K, e, cos_w, sin_w]
        returns: (G,) RMS masked residual ||rv_obs - h_kepler(theta)||.
        """
        g = theta5.shape[0]
        t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0).expand(g, -1)
        rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0).expand(g, -1)
        mask = torch.from_numpy(curve["mask"]).unsqueeze(0).expand(g, -1)
        t_span = torch.full((g,), curve["t_span"], dtype=torch.float32)
        t_min = torch.full((g,), curve["t_min"], dtype=torch.float32)
        rv_std = torch.full((g,), curve["rv_std"], dtype=torch.float32)
        th = torch.as_tensor(theta5, dtype=torch.float32)
        rv_pred = self.decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)
        diff = (rv_obs - rv_pred) ** 2 * mask
        n = mask.sum(dim=1).clamp(min=1.0)
        return torch.sqrt((diff.sum(dim=1) / n)).cpu().numpy()


def _theta_with_coord(theta_hat5: np.ndarray, coord: str, value: float) -> np.ndarray:
    """Copy theta_hat (5,) and overwrite one CP coordinate; return (1, 5)."""
    out = theta_hat5.copy()
    if coord == "log10_P":
        out[0] = value
    elif coord == "log10_K":
        out[1] = value
    elif coord == "e":
        out[2] = np.clip(value, 0.0, 0.99)
    elif coord == "omega":
        out[3], out[4] = np.cos(value), np.sin(value)
    return out[None, :]


# ---------------------------------------------------------------------------
# Empirical (histogram-derived) search grids  — no ad-hoc ranges
# ---------------------------------------------------------------------------


def histogram_grids(grid: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = _sample_period(rng, 40_000)
    e = _sample_eccentricity(rng, 40_000)
    lo_p, hi_p = np.percentile(np.log10(P), [0.5, 99.5])
    hi_e = float(np.percentile(e, 99.5))
    return {
        "log10_P": np.linspace(lo_p, hi_p, grid),
        "log10_K": np.linspace(math.log10(_K_MIN_MS), math.log10(_K_MAX_MS), grid),
        "e": np.linspace(0.0, hi_e, grid),
        "omega": np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False),
    }


def _true_coord(theta5: np.ndarray, coord: str) -> float:
    return {
        "log10_P": theta5[0], "log10_K": theta5[1], "e": theta5[2],
        "omega": _theta_to_omega(theta5),
    }[coord]


# ---------------------------------------------------------------------------
# E1 — coverage
# ---------------------------------------------------------------------------


def _calib_scores(scorer, calib, theta_hats) -> np.ndarray:
    """Surrogate calibration scores s(theta_hat_j, y_j) (alpha-independent)."""
    return np.array([scorer.score(th[None, :], s["curve"])[0]
                     for s, th in zip(calib, theta_hats)])


def _bonferroni_q(calib_scores: np.ndarray, alpha: float) -> float:
    n = len(calib_scores)
    level = 1.0 - alpha / D
    k = min(int(math.ceil((n + 1) * level)), n)          # rank (1-indexed)
    return float(np.sort(calib_scores)[k - 1])


def _precompute_test(scorer, test, theta_hats, grids) -> list[dict]:
    """Per-system alpha-independent scores: s at the true value, and over the grid."""
    pre = []
    for s, th in zip(test, theta_hats):
        rec = {"s_true": {}, "grid_scores": {}}
        for c in COORDS:
            true_v = _true_coord(s["theta5"], c)
            rec["s_true"][c] = float(scorer.score(_theta_with_coord(th, c, true_v), s["curve"])[0])
            cand = np.vstack([_theta_with_coord(th, c, v)[0] for v in grids[c]])
            rec["grid_scores"][c] = scorer.score(cand, s["curve"])
        pre.append(rec)
    return pre


def _coverage_at(pre: list[dict], grids: dict, q: float) -> dict:
    per_cov = {c: [] for c in COORDS}
    per_w = {c: [] for c in COORDS}
    joint = []
    for rec in pre:
        all_c = True
        for c in COORDS:
            cov = rec["s_true"][c] <= q
            per_cov[c].append(cov)
            all_c = all_c and cov
            acc = grids[c][rec["grid_scores"][c] <= q]
            per_w[c].append(float(acc.max() - acc.min()) if acc.size else 0.0)
        joint.append(all_c)
    return {
        "per_coord_coverage": {c: float(np.mean(per_cov[c])) for c in COORDS},
        "per_coord_median_width": {c: float(np.median(per_w[c])) for c in COORDS},
        "joint_coverage": float(np.mean(joint)),
    }


def run_e1(scorer, rf, calib, test_syn, test_real, grids, alphas, out_dir, fig_dir):
    def hats(systems):
        return list(rf.predict(np.vstack([s["features"] for s in systems])))

    calib_scores = _calib_scores(scorer, calib, hats(calib))
    pre_syn = _precompute_test(scorer, test_syn, hats(test_syn), grids)
    pre_real = _precompute_test(scorer, test_real, hats(test_real), grids)

    report = {"d": D, "coords": COORDS, "n_cal": len(calib),
              "n_test_syn": len(test_syn), "n_test_real": len(test_real),
              "alphas": alphas, "synthetic": {}, "real": {}}
    for a in alphas:
        q = _bonferroni_q(calib_scores, a)
        report["synthetic"][f"{a:.2f}"] = {"q": q, **_coverage_at(pre_syn, grids, q)}
        report["real"][f"{a:.2f}"] = {"q": q, **_coverage_at(pre_real, grids, q)}
        print(f"[E1] alpha={a:.2f} target>={1-a:.2f}  "
              f"syn joint={report['synthetic'][f'{a:.2f}']['joint_coverage']:.3f}  "
              f"real joint={report['real'][f'{a:.2f}']['joint_coverage']:.3f}")

    # coverage-vs-nominal figure (joint + per-coord, synthetic vs real)
    nominal = [1 - a for a in alphas]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, dom in zip(axs, ["synthetic", "real"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="nominal")
        ax.plot(nominal, [report[dom][f"{a:.2f}"]["joint_coverage"] for a in alphas],
                "o-", lw=2, label="joint (all 4)")
        for c in COORDS:
            ax.plot(nominal, [report[dom][f"{a:.2f}"]["per_coord_coverage"][c] for a in alphas],
                    ".-", alpha=0.6, label=c)
        ax.set_xlabel("nominal coverage 1 - alpha")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"{dom} test")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("E1 — unsupervised CP coverage vs nominal", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "conformal_e1_coverage.png", dpi=180)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# E2 — monotonicity of the score (Assumption 2.3)
# ---------------------------------------------------------------------------


def run_e2(scorer, systems, out_dir, fig_dir, n_offsets=25, n_sys=250):
    systems = systems[:n_sys]
    offsets = {
        "log10_P": np.linspace(-1.0, 1.0, n_offsets),
        "log10_K": np.linspace(-1.0, 1.0, n_offsets),
        "e": np.linspace(-0.4, 0.4, n_offsets),
        "omega": np.linspace(-np.pi, np.pi, n_offsets),
    }
    curves = {c: np.zeros((len(systems), n_offsets)) for c in COORDS}
    for si, s in enumerate(systems):
        th_true = s["theta5"]
        for c in COORDS:
            base = _true_coord(th_true, c)
            cand = np.vstack([_theta_with_coord(th_true, c, base + d)[0] for d in offsets[c]])
            curves[c][si] = scorer.score(cand, s["curve"])

    fig, axs = plt.subplots(1, D, figsize=(4.2 * D, 4.2))
    mono = {}
    for ax, c in zip(axs, COORDS):
        mean = curves[c].mean(axis=0)
        med = np.median(curves[c], axis=0)
        ax.plot(offsets[c], mean, "o-", label="mean")
        ax.plot(offsets[c], med, ".--", alpha=0.6, label="median")
        ax.axvline(0, color="r", lw=1)
        ax.set_title(c)
        ax.set_xlabel(r"offset $\theta_i - \bar\theta_i$")
        ax.set_ylabel("recon. residual (rv_std)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        # monotonicity score: correlation of |offset| with score (should be ~1)
        half = n_offsets // 2
        left = np.corrcoef(-offsets[c][:half], mean[:half])[0, 1]
        right = np.corrcoef(offsets[c][half + 1:], mean[half + 1:])[0, 1]
        mono[c] = {"rise_left": float(left), "rise_right": float(right),
                   "min_at_offset": float(offsets[c][int(np.argmin(mean))])}
    fig.suptitle("E2 — score vs offset per coordinate (Assumption 2.3 monotonicity)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "conformal_e2_monotonicity.png", dpi=180)
    plt.close(fig)
    return mono


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=SG / "datasets" / "synthetic_regression_10000.csv")
    ap.add_argument("--out-dir", type=Path, default=SG / "regression")
    ap.add_argument("--fig-dir", type=Path, default=SG / "figures" / "synthetic_regression_10000")
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--real-split", default="test", choices=("all", "train", "val", "test"))
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=100.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # Step-5 regressor phi = RF(features -> theta) on the synthetic CSV.
    df = pd.read_csv(args.csv)
    rf = _build("separate", args.n_estimators, args.seed, list(TARGET_COLUMNS))
    rf.fit(df[FEATURES].to_numpy(float), df[list(TARGET_COLUMNS)].to_numpy(float))
    print(f"trained RF phi on {len(df)} synthetic rows")

    scorer = Scorer()
    grids = histogram_grids(args.grid, args.seed)

    print("building calibration / test systems ...")
    calib = make_synthetic(args.n_cal, args.seed + 1)
    test_syn = make_synthetic(args.n_test, args.seed + 2)
    test_real = make_real(args.real_split, args.sigma_min, args.sigma_max)
    print(f"n_cal={len(calib)} n_test_syn={len(test_syn)} n_test_real={len(test_real)}")

    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    e1 = run_e1(scorer, rf, calib, test_syn, test_real, grids, alphas, args.out_dir, args.fig_dir)
    e2 = run_e2(scorer, test_syn, args.out_dir, args.fig_dir)

    report = {"E1_coverage": e1, "E2_monotonicity": e2}
    (args.out_dir / "conformal_metrics.json").write_text(json.dumps(report, indent=2))
    _write_report(report, args.out_dir / "conformal_report.txt")
    print(f"wrote conformal metrics + report to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")


def _write_report(report: dict, path: Path) -> None:
    e1, e2 = report["E1_coverage"], report["E2_monotonicity"]
    lines = ["Unsupervised Conformal Prediction — E1 (coverage) + E2 (monotonicity)",
             "=" * 70,
             f"coordinates (d={e1['d']}): {', '.join(e1['coords'])}  (Bonferroni)",
             f"n_cal={e1['n_cal']}  n_test_syn={e1['n_test_syn']}  n_test_real={e1['n_test_real']}",
             "",
             "E1 — empirical coverage (should be >= nominal 1 - alpha)",
             "-" * 60]
    for dom in ["synthetic", "real"]:
        lines.append(f"[{dom} test]")
        lines.append(f"  {'1-alpha':>8}{'joint':>9}" + "".join(f"{c:>11}" for c in e1["coords"]))
        for a in e1["alphas"]:
            r = e1[dom][f"{a:.2f}"]
            row = f"  {1-a:>8.2f}{r['joint_coverage']:>9.3f}"
            row += "".join(f"{r['per_coord_coverage'][c]:>11.3f}" for c in e1["coords"])
            lines.append(row)
        lines.append("  median set width @ alpha=0.10:")
        w = e1[dom][f"{e1['alphas'][0]:.2f}"]["per_coord_median_width"]
        lines.append("    " + "  ".join(f"{c}={w[c]:.3g}" for c in e1["coords"]))
        lines.append("")
    lines.append("E2 — monotonicity of the reconstruction score (Assumption 2.3)")
    lines.append("-" * 60)
    lines.append("  (rise_left/right ~ +1 => score increases away from truth; ~0 => flat/unidentifiable)")
    for c, m in e2.items():
        lines.append(f"  {c:<10} rise_left={m['rise_left']:+.2f}  rise_right={m['rise_right']:+.2f}  "
                     f"min@offset={m['min_at_offset']:+.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
