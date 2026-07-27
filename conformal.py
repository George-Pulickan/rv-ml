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

from feature_columns import TARGET_COLUMNS  # noqa: E402
from train_regression_models import _build  # noqa: E402
from eval_omega_nn_vs_rf import FEATURES, _summary_row  # noqa: E402
from generate_synthetic_regression_csv import _masked_observations  # noqa: E402

# CP operates on the four physical coordinates; (cos w, sin w) are a redundant
# encoding of the single angle omega, so we vary omega as one coordinate.
COORDS = ["log10_P", "log10_K", "e", "omega"]
D = len(COORDS)

# Named coordinate sets.  omega is not identifiable at this SNR — its median CP
# half-width exceeds pi, i.e. the interval wraps the full circle — so the paper
# reports P, K, e and states omega's non-identifiability rather than tabulating
# a number that reads like a measurement (Nicolò, 2026-07-26: "Let's skip
# showing and using omega as a target").  omega is *not* removed from theta5:
# the decoder still needs it, GD still optimises it, and it still enters the
# reconstruction through h/k.  Only the reported CP coordinates change, which
# also drops the Bonferroni divisor from 4 to 3.
COORD_SETS = {
    "PKew": ["log10_P", "log10_K", "e", "omega"],
    "PKe": ["log10_P", "log10_K", "e"],
}


def set_coords(names: list[str]) -> None:
    """Rebind the reported CP coordinates in place.

    Mutates the COORDS list rather than reassigning it, so modules that did
    `from conformal import COORDS` see the change.  Use len(COORDS) rather than
    the D constant anywhere the Bonferroni divisor matters.
    """
    unknown = [n for n in names if n not in COORD_SETS["PKew"]]
    if unknown:
        raise ValueError(f"unknown coordinates: {unknown}")
    COORDS[:] = list(names)

# Minimum distinct observing nights for a real RV series to be usable.  Series
# below this are single-night transit/RM sequences that cannot constrain an
# orbital period; see make_real.  Label-free by construction.
MIN_NIGHTS = 5


# ---------------------------------------------------------------------------
# System construction: curve tensors (for the decoder) + features + true theta
# ---------------------------------------------------------------------------


def _theta_to_omega(theta5: np.ndarray) -> float:
    return float(np.arctan2(theta5[4], theta5[3]))


def curve_times_days(curve: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """Absolute observation times in days, reconstructed in float64.

    t_norm is stored float32 (fine: it lives in [0, 1]), but t_min is a JD near
    2.45e6 where float32 spacing is 0.25 d.  Doing t_norm*t_span + t_min in
    float32 therefore snaps distinct epochs onto a coarse grid: 20 of the 57
    real test series lost distinct epochs that way (51 Peg 256 -> 110), and one
    collapsed far enough to yield an all-NaN periodogram, which silently
    poisoned delta_c.  Always reconstruct times through this helper.
    """
    m = (curve["mask"] > 0.5) if mask is None else mask
    t_norm = np.asarray(curve["t_norm"], dtype=np.float64)[m]
    return t_norm * float(curve["t_span"]) + float(curve["t_min"])


def _curve_from_x(x: np.ndarray, info: dict) -> dict:
    return {
        "t_norm": x[0].astype(np.float32),
        "rv_obs": x[1].astype(np.float32),
        "sig": x[2].astype(np.float32),   # per-obs measurement sigma, in rv_std units
        "mask": x[3].astype(np.float32),
        "t_span": float(info["t_span_days"]),
        "t_min": float(info["t_min_days"]),
        "rv_std": float(info["rv_std_ms"]),
    }


def _inject_correlated_noise(x: np.ndarray, lsp: np.ndarray, info: dict,
                             frac: float, tau_days: float,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Add a correlated (squared-exponential) component to a synthetic curve.

    This is *noise-model* misspecification, which is what the paper actually
    claims robustness to: the generative assumption y ~ h(theta) + noise(theta)
    still holds exactly, h(theta_bar) remains the correct mean structure, and
    theta_bar remains the right target — only the noise *process* differs from
    the one the calibration set was drawn with.  Physically this stands for
    stellar activity (spots, rotation, magnetic cycles), which is the dominant
    unmodelled systematic in real radial-velocity data.

    Contrast with companion injection (``f_multi``), which corrupts the mean
    structure instead: there theta* = argmin ||h(theta) - y||^2 drifts toward a
    blend of the planets while coverage is still scored against the dominant
    one, so the surrogate strategy is penalised by construction.

    ``frac`` is the correlated amplitude as a fraction of the curve's RV
    standard deviation; ``tau_days`` the correlation timescale.  The curve is
    re-normalised and its periodogram recomputed afterwards, exactly as the
    pipeline would do for a freshly observed curve.
    """
    from preprocess import compute_lsp

    m = x[3] == 1
    if frac <= 0.0 or m.sum() < 2:
        return x, lsp, info

    rv_std0 = float(info["rv_std_ms"])
    t_days = x[0][m].astype(np.float64) * float(info["t_span_days"]) + float(info["t_min_days"])
    rv_ms = x[1][m].astype(np.float64) * rv_std0
    sig_ms = x[2][m].astype(np.float64) * rv_std0

    dt = t_days[:, None] - t_days[None, :]
    K = (frac * rv_std0) ** 2 * np.exp(-0.5 * (dt / max(tau_days, 1e-6)) ** 2)
    K[np.diag_indices_from(K)] += 1e-8 * max((frac * rv_std0) ** 2, 1e-12)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:                     # numerically singular grid
        w, V = np.linalg.eigh(K)
        L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    rv_ms = rv_ms + L @ rng.standard_normal(len(t_days))

    med = float(np.median(rv_ms))
    std = max(float(np.std(rv_ms)), 1e-6)
    x = x.copy()
    x[1][m] = ((rv_ms - med) / std).astype(x.dtype)
    x[2][m] = (sig_ms / std).astype(x.dtype)
    info = dict(info)
    info["rv_std_ms"] = std
    info["rv_med_ms"] = med
    info["noise_mode"] = f"{info.get('noise_mode', '?')}+correlated{frac:g}"
    return x, np.asarray(compute_lsp(t_days, rv_ms, sig_ms), dtype=float), info


def make_synthetic(n: int, seed: int, f_multi: float = 0.0,
                   noise_frac: float = 0.0, noise_tau_days: float = 30.0) -> list[dict]:
    """Draw ``n`` synthetic systems from the empirical priors.

    ``f_multi`` > 0 injects companion planets: with probability f_multi the
    curve contains one or two additional planets, while ``theta5`` still labels
    the dominant one (highest K).  A single-Keplerian decoder h cannot represent
    such a curve, so this is a controlled knob for *mean-structure*
    misspecification — note this breaks the generative assumption itself, so
    theta_bar stops being the parameter that best describes the curve.

    ``noise_frac`` > 0 instead adds a correlated component at that fraction of
    the curve's RV standard deviation (see ``_inject_correlated_noise``): the
    mean structure stays exactly h(theta_bar) and only the *noise process* is
    misspecified, which is the condition the paper's robustness claim is
    actually stated for.
    """
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n)
    systems = []
    for i in range(n):
        p = {k: float(v[i]) for k, v in params.items()}
        x, lsp, theta, info = generate_one(p, np.random.default_rng(seed + 7_000 + i),
                                           f_multi=f_multi)
        if noise_frac > 0.0:
            x, lsp, info = _inject_correlated_noise(
                x, lsp, info, noise_frac, noise_tau_days,
                np.random.default_rng(seed + 900_000 + i))
        xm = _masked_observations(x)
        feats = _summary_row(xm, info, lsp)
        systems.append({
            "curve": _curve_from_x(x, info),
            "features": np.array([feats[c] for c in FEATURES], dtype=float),
            "feat_row": feats,
            "lsp": np.asarray(lsp, dtype=float),
            "theta5": np.asarray(theta, dtype=float),
        })
    return systems


def _distinct_nights(curve: dict) -> int:
    """Number of distinct calendar nights spanned by a curve's observations."""
    m = np.asarray(curve["mask"], dtype=bool)
    t = np.asarray(curve["t_norm"], dtype=float)[m] * float(curve["t_span"]) + float(curve["t_min"])
    if t.size == 0:
        return 0
    return int(np.unique(np.floor(t)).size)


def make_real(split: str, sigma_min: float, sigma_max: float,
              min_nights: int = MIN_NIGHTS, one_per_host: bool = False) -> list[dict]:
    """Real single-planet systems for one split.

    ``min_nights`` drops series whose observations fall on fewer than that many
    distinct nights.  These are Rossiter-McLaughlin / transit-night sequences:
    high cadence within one night, so they carry no information about an orbital
    period of days or longer (e.g. HD 17156 with 13 points in 0.102 d against a
    21.2 d period).  The cut is deliberately label-free — it uses only the
    observing pattern, never the tabulated P — so it selects the test set
    without consulting the values we are later scored against.

    ``one_per_host`` keeps only the best-sampled series per star (most nights,
    then longest baseline).  The corpus stores one row per RV file, so a single
    star can contribute several correlated series that share the same true
    theta; enable this to make the test points independent across stars.
    """
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
        curve = _curve_from_x(x, info)
        n_nights = _distinct_nights(curve)
        if n_nights < min_nights:
            continue
        feats = _summary_row(xm, info, lsp)
        systems.append({
            "curve": curve,
            "features": np.array([feats[c] for c in FEATURES], dtype=float),
            "feat_row": feats,
            "lsp": np.asarray(lsp, dtype=float),
            "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
            # Carried so downstream figures can label systems by star.
            "host": str(info.get("host", "")),
            "n_nights": n_nights,
        })
    if one_per_host:
        best: dict[str, dict] = {}
        for s in systems:
            key = s["host"]
            cur = best.get(key)
            if cur is None or (s["n_nights"], s["curve"]["t_span"]) > (cur["n_nights"], cur["curve"]["t_span"]):
                best[key] = s
        systems = [best[h] for h in sorted(best)]
    return systems


# ---------------------------------------------------------------------------
# Conformity score via the fixed Kepler decoder
# ---------------------------------------------------------------------------


class Scorer:
    """Reconstruction-residual conformity score.

    mode="rv_std" : RMS residual in each curve's rv_std units (raw).
    mode="chi2"   : residual weighted by the per-obs measurement sigma, i.e. the
                    reduced-chi RMS sqrt(mean[ ((rv_obs - rv_pred)/sigma)^2 ]).
                    This removes the per-system noise-amplitude scale so the
                    calibration quantile is set by fit quality, not raw jitter,
                    and should tighten the prediction sets.
    """

    def __init__(self, mode: str = "rv_std"):
        self.decoder = KeplerDecoder().eval()
        self.mode = mode

    @torch.no_grad()
    def score(self, theta5: np.ndarray, curve: dict) -> np.ndarray:
        g = theta5.shape[0]
        t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0).expand(g, -1)
        rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0).expand(g, -1)
        mask = torch.from_numpy(curve["mask"]).unsqueeze(0).expand(g, -1)
        t_span = torch.full((g,), curve["t_span"], dtype=torch.float32)
        t_min = torch.full((g,), curve["t_min"], dtype=torch.float32)
        rv_std = torch.full((g,), curve["rv_std"], dtype=torch.float32)
        th = torch.as_tensor(theta5, dtype=torch.float32)
        rv_pred = self.decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)
        resid = rv_obs - rv_pred
        if self.mode == "chi2":
            sig = torch.from_numpy(curve["sig"]).unsqueeze(0).expand(g, -1)
            resid = resid / sig.clamp(min=1e-3)
        diff = resid ** 2 * mask
        n = mask.sum(dim=1).clamp(min=1.0)
        return torch.sqrt((diff.sum(dim=1) / n)).cpu().numpy()


def _set_coord(theta5: np.ndarray, coord: str, value: float) -> np.ndarray:
    """Return a copy of theta5 (5,) with one CP coordinate overwritten."""
    out = theta5.copy()
    if coord == "log10_P":
        out[0] = value
    elif coord == "log10_K":
        out[1] = value
    elif coord == "e":
        out[2] = np.clip(value, 0.0, 0.99)
    elif coord == "omega":
        out[3], out[4] = np.cos(value), np.sin(value)
    return out


def _theta_with_coord(theta_hat5: np.ndarray, coord: str, value: float) -> np.ndarray:
    """Copy theta_hat (5,) and overwrite one CP coordinate; return (1, 5)."""
    return _set_coord(theta_hat5, coord, value)[None, :]


def _set_coord_grid(base_M5: np.ndarray, coord: str, grid: np.ndarray) -> np.ndarray:
    """Broadcast a (M, 5) base and a (G,) grid for `coord` into (M, G, 5)."""
    M = base_M5.shape[0]
    G = grid.shape[0]
    big = np.repeat(base_M5[:, None, :], G, axis=1)          # (M, G, 5)
    if coord == "log10_P":
        big[:, :, 0] = grid[None, :]
    elif coord == "log10_K":
        big[:, :, 1] = grid[None, :]
    elif coord == "e":
        big[:, :, 2] = np.clip(grid, 0.0, 0.99)[None, :]
    elif coord == "omega":
        big[:, :, 3] = np.cos(grid)[None, :]
        big[:, :, 4] = np.sin(grid)[None, :]
    return big


# ---------------------------------------------------------------------------
# Profiled conformity score
# ---------------------------------------------------------------------------
#
# The univariate CP set for coordinate c (eq 9) pins the other coordinates at
# the point estimate theta_hat.  When theta_hat is weak on those nuisance
# coordinates the reconstruction stays a poor fit for every swept value of c, so
# the score is flat and the set fills the whole histogram support (valid but
# uninformative).  The *profiled* score instead minimises the reconstruction
# residual over a chosen nuisance set N at each swept value:
#
#     s_prof(c=v, y) = min_{theta_N}  s( theta_hat with c=v and N refit, y )
#
# implemented as batched coordinate descent over N on the empirical grids.  The
# tested coordinate c is never profiled (it is pinned to v).  With N = {} this
# reduces exactly to the pinned score, so the baseline is recovered bit-for-bit.
# Calibration scores are profiled by the identical procedure (pinning c at its
# own predicted value theta_hat_c), which keeps the split-conformal exchange-
# ability argument intact — hence the quantile q becomes per-coordinate.


def profiled_min(
    scorer: "Scorer",
    base5: np.ndarray,
    coord_c: str,
    values: np.ndarray,
    profile_coords: tuple[str, ...],
    pgrids: dict[str, np.ndarray],
    curve: dict,
    sweeps: int = 2,
) -> np.ndarray:
    """Profiled score for coordinate `coord_c` over a set of pinned `values`.

    Returns (len(values),): for each pinned value v of coord_c, the minimum
    reconstruction score over the nuisance coordinates (profile_coords minus
    coord_c), found by `sweeps` passes of 1-D grid coordinate descent warm-
    started from base5.  profile_coords=() gives the plain pinned score.
    """
    values = np.asarray(values, dtype=float)
    M = values.shape[0]
    th = np.repeat(base5[None, :], M, axis=0)                # (M, 5)
    for k in range(M):
        th[k] = _set_coord(th[k], coord_c, values[k])

    nuisance = [c for c in profile_coords if c != coord_c]
    if not nuisance:
        return scorer.score(th, curve)                       # (M,)

    for _ in range(sweeps):
        for p in nuisance:
            g = pgrids[p]                                    # (G,)
            big = _set_coord_grid(th, p, g)                  # (M, G, 5)
            # Keep the incumbent as a candidate so the descent can never make the
            # score worse than the pinned baseline (base value of p may be off-grid).
            big = np.concatenate([big, th[:, None, :]], axis=1)  # (M, G+1, 5)
            Gp = big.shape[1]
            sc = scorer.score(big.reshape(M * Gp, 5), curve).reshape(M, Gp)
            j = sc.argmin(axis=1)                            # (M,)
            th = big[np.arange(M), j]                        # (M, 5) — best per row
    return scorer.score(th, curve)                           # (M,) final min


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


def _calib_scores(scorer, calib, theta_hats, profile_coords, pgrids) -> dict:
    """Per-coordinate surrogate calibration scores (alpha-independent).

    For coordinate c and calibration curve j: profile the nuisance while pinning
    c at its own predicted value theta_hat_c.  With profile_coords=() every
    coordinate yields the same s(theta_hat_j, y_j) as the pinned baseline.
    """
    out = {c: np.empty(len(calib)) for c in COORDS}
    for i, (s, th) in enumerate(zip(calib, theta_hats)):
        for c in COORDS:
            v = _true_coord(th, c)   # pin c at its predicted value
            out[c][i] = profiled_min(scorer, th, c, np.array([v]),
                                     profile_coords, pgrids, s["curve"])[0]
    return out


def _bonferroni_q(calib_scores_c: np.ndarray, alpha: float) -> float:
    n = len(calib_scores_c)
    level = 1.0 - alpha / len(COORDS)
    k = min(int(math.ceil((n + 1) * level)), n)          # rank (1-indexed)
    return float(np.sort(calib_scores_c)[k - 1])


def _precompute_test(scorer, test, theta_hats, grids, profile_coords, pgrids) -> list[dict]:
    """Per-system alpha-independent profiled scores: at the true value and over the grid."""
    pre = []
    for s, th in zip(test, theta_hats):
        rec = {"s_true": {}, "grid_scores": {}}
        for c in COORDS:
            true_v = _true_coord(s["theta5"], c)
            rec["s_true"][c] = float(profiled_min(scorer, th, c, np.array([true_v]),
                                                  profile_coords, pgrids, s["curve"])[0])
            rec["grid_scores"][c] = profiled_min(scorer, th, c, grids[c],
                                                 profile_coords, pgrids, s["curve"])
        pre.append(rec)
    return pre


def _coverage_at(pre: list[dict], grids: dict, q: dict) -> dict:
    per_cov = {c: [] for c in COORDS}
    per_w = {c: [] for c in COORDS}
    joint = []
    for rec in pre:
        all_c = True
        for c in COORDS:
            cov = rec["s_true"][c] <= q[c]
            per_cov[c].append(cov)
            all_c = all_c and cov
            acc = grids[c][rec["grid_scores"][c] <= q[c]]
            per_w[c].append(float(acc.max() - acc.min()) if acc.size else 0.0)
        joint.append(all_c)
    return {
        "per_coord_coverage": {c: float(np.mean(per_cov[c])) for c in COORDS},
        "per_coord_median_width": {c: float(np.median(per_w[c])) for c in COORDS},
        "joint_coverage": float(np.mean(joint)),
    }


def run_e1(scorer, rf, calib, test_syn, test_real, grids, alphas, out_dir, fig_dir,
           suffix="", profile_coords=(), pgrids=None):
    def hats(systems):
        return list(rf.predict(np.vstack([s["features"] for s in systems])))

    pgrids = pgrids if pgrids is not None else grids
    calib_scores = _calib_scores(scorer, calib, hats(calib), profile_coords, pgrids)
    pre_syn = _precompute_test(scorer, test_syn, hats(test_syn), grids, profile_coords, pgrids)
    pre_real = _precompute_test(scorer, test_real, hats(test_real), grids, profile_coords, pgrids)

    report = {"d": len(COORDS), "coords": list(COORDS), "n_cal": len(calib),
              "n_test_syn": len(test_syn), "n_test_real": len(test_real),
              "profile_coords": list(profile_coords),
              "alphas": alphas, "synthetic": {}, "real": {}}
    for a in alphas:
        q = {c: _bonferroni_q(calib_scores[c], a) for c in COORDS}
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
    fig.suptitle(f"E1 — unsupervised CP coverage vs nominal (score={suffix.strip('_') or 'rv_std'})",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / f"conformal_e1_coverage{suffix}.png", dpi=180)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# E2 — monotonicity of the score (Assumption 2.3)
# ---------------------------------------------------------------------------


def run_e2(scorer, systems, out_dir, fig_dir, n_offsets=25, n_sys=250, suffix="",
           profile_coords=(), pgrids=None):
    systems = systems[:n_sys]
    offsets = {
        "log10_P": np.linspace(-1.0, 1.0, n_offsets),
        "log10_K": np.linspace(-1.0, 1.0, n_offsets),
        "e": np.linspace(-0.4, 0.4, n_offsets),
        "omega": np.linspace(-np.pi, np.pi, n_offsets),
    }
    pgrids = pgrids if pgrids is not None else {}
    curves = {c: np.zeros((len(systems), n_offsets)) for c in COORDS}
    for si, s in enumerate(systems):
        th_true = s["theta5"]
        for c in COORDS:
            base = _true_coord(th_true, c)
            vals = base + offsets[c]
            curves[c][si] = profiled_min(scorer, th_true, c, vals,
                                         profile_coords, pgrids, s["curve"])

    n_c = len(COORDS)
    fig, axs = plt.subplots(1, n_c, figsize=(4.2 * n_c, 4.2))
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
    fig.suptitle(f"E2 — score vs offset per coordinate (score={suffix.strip('_') or 'rv_std'})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / f"conformal_e2_monotonicity{suffix}.png", dpi=180)
    plt.close(fig)
    return mono


def plot_width_comparison(report_by_mode: dict, alpha: float, fig_dir: Path) -> None:
    """Median set width per coordinate, rv_std vs chi2 score, on synthetic + real."""
    modes = list(report_by_mode)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dom in zip(axs, ["synthetic", "real"]):
        x = np.arange(len(COORDS))
        w = 0.8 / len(modes)
        for k, m in enumerate(modes):
            widths = report_by_mode[m][dom][f"{alpha:.2f}"]["per_coord_median_width"]
            ax.bar(x + (k - 0.5) * w, [widths[c] for c in COORDS], w, label=f"score={m}")
        ax.set_xticks(x)
        ax.set_xticklabels(COORDS)
        ax.set_ylabel(f"median set width @ 1-alpha={1-alpha:.2f}")
        ax.set_title(f"{dom} test")
        ax.grid(alpha=0.2, axis="y")
        ax.legend()
    fig.suptitle("Prediction-set width by score config (pinned vs profiled)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "conformal_width_comparison.png", dpi=180)
    plt.close(fig)


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
    ap.add_argument("--profile", default="K", choices=("none", "K", "Keomega"),
                    help="nuisance set profiled by the conformity score: "
                         "none (pinned baseline only), K (log10_K, default), "
                         "Keomega (log10_K + e + omega)")
    ap.add_argument("--profile-grid", type=int, default=33,
                    help="grid resolution for the profiled nuisance minimisation")
    ap.add_argument("--sweeps", type=int, default=2,
                    help="coordinate-descent passes for multi-coord profiling")
    ap.add_argument("--chi2", action="store_true",
                    help="also run the sigma-normalized (chi2) score variant")
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

    grids = histogram_grids(args.grid, args.seed)
    pgrids = histogram_grids(args.profile_grid, args.seed)

    print("building calibration / test systems ...")
    calib = make_synthetic(args.n_cal, args.seed + 1)
    test_syn = make_synthetic(args.n_test, args.seed + 2)
    test_real = make_real(args.real_split, args.sigma_min, args.sigma_max)
    print(f"n_cal={len(calib)} n_test_syn={len(test_syn)} n_test_real={len(test_real)}")

    profile_map = {"none": (), "K": ("log10_K",),
                   "Keomega": ("log10_K", "e", "omega")}

    # Each config = (label, score-mode, profiled nuisance coords).  The pinned
    # baseline (no profiling) is always run so the width comparison shows the
    # profiling effect; --profile adds the profiled config, --chi2 the variant.
    configs: list[tuple[str, str, tuple[str, ...]]] = [("rv_std", "rv_std", ())]
    if args.profile != "none":
        configs.append((f"rv_std+prof_{args.profile}", "rv_std",
                        profile_map[args.profile]))
    if args.chi2:
        configs.append(("chi2", "chi2", ()))
        if args.profile != "none":
            configs.append((f"chi2+prof_{args.profile}", "chi2",
                            profile_map[args.profile]))

    import time
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    modes = [label for label, _, _ in configs]
    e1_by_mode, e2_by_mode = {}, {}
    for label, mode, prof in configs:
        print(f"\n===== config: {label}  (score={mode}, profile={list(prof) or 'none'}) =====")
        t0 = time.perf_counter()
        scorer = Scorer(mode=mode)
        e1_by_mode[label] = run_e1(scorer, rf, calib, test_syn, test_real, grids,
                                   alphas, args.out_dir, args.fig_dir, suffix=f"_{label}",
                                   profile_coords=prof, pgrids=pgrids)
        e2_by_mode[label] = run_e2(scorer, test_syn, args.out_dir, args.fig_dir,
                                   suffix=f"_{label}", profile_coords=prof, pgrids=pgrids)
        print(f"  [{label}] done in {time.perf_counter()-t0:.0f}s")

    plot_width_comparison(e1_by_mode, 0.1, args.fig_dir)

    report = {"modes": modes, "E1_coverage": e1_by_mode, "E2_monotonicity": e2_by_mode}
    (args.out_dir / "conformal_metrics.json").write_text(json.dumps(report, indent=2))
    _write_report_multi(report, args.out_dir / "conformal_report.txt")
    print(f"\nwrote conformal metrics + report to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")


def _write_report_multi(report: dict, path: Path) -> None:
    lines = ["Unsupervised Conformal Prediction — score-config comparison",
             "=" * 72, ""]
    for mode in report["modes"]:
        lines.append(f"##### CONFIG: {mode} #####")
        lines.append("")
        lines.extend(_report_lines({"E1_coverage": report["E1_coverage"][mode],
                                    "E2_monotonicity": report["E2_monotonicity"][mode]}))
        lines.append("")
    # width comparison at alpha=0.10
    lines.append("Median set width @ 1-alpha=0.90 — by score config")
    lines.append("-" * 52)
    for dom in ["synthetic", "real"]:
        lines.append(f"[{dom}]")
        for c in COORDS:
            ws = {m: report["E1_coverage"][m][dom]["0.10"]["per_coord_median_width"][c]
                  for m in report["modes"]}
            lines.append("  " + f"{c:<10}" + "  ".join(f"{m}={ws[m]:.3g}" for m in report["modes"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_lines(report: dict) -> list[str]:
    e1, e2 = report["E1_coverage"], report["E2_monotonicity"]
    lines = [f"coordinates (d={e1['d']}): {', '.join(e1['coords'])}  (Bonferroni)",
             f"n_cal={e1['n_cal']}  n_test_syn={e1['n_test_syn']}  n_test_real={e1['n_test_real']}",
             "", "E1 — empirical coverage (should be >= nominal 1 - alpha)", "-" * 60]
    for dom in ["synthetic", "real"]:
        lines.append(f"[{dom} test]")
        lines.append(f"  {'1-alpha':>8}{'joint':>9}" + "".join(f"{c:>11}" for c in e1["coords"]))
        for a in e1["alphas"]:
            r = e1[dom][f"{a:.2f}"]
            row = f"  {1-a:>8.2f}{r['joint_coverage']:>9.3f}"
            row += "".join(f"{r['per_coord_coverage'][c]:>11.3f}" for c in e1["coords"])
            lines.append(row)
        w = e1[dom][f"{e1['alphas'][0]:.2f}"]["per_coord_median_width"]
        lines.append("  median set width @ alpha=0.10: "
                     + "  ".join(f"{c}={w[c]:.3g}" for c in e1["coords"]))
    lines.append("")
    lines.append("E2 — monotonicity (Assumption 2.3; rise~+1 => identifiable, ~0 => flat)")
    for c, m in e2.items():
        lines.append(f"  {c:<10} rise_left={m['rise_left']:+.2f}  rise_right={m['rise_right']:+.2f}  "
                     f"min@offset={m['min_at_offset']:+.3f}")
    return lines


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
