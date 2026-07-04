"""
conformal_shift.py — Step 6 v2: split-CP calibrated on fake data, tested on real,
per Nicolò's 2026-07 spec.

Compares two conformity-score strategies, both built around the Step-5 point
predictor psi(y) = RF(features(y)); psi's feature columns are taken from the
--csv dataset (default: the 512-bin raw-LSP dataset, per Nicolò's OK on more
Fourier bins — the 64-bin sum-normalized spectrum loses the period peak):

  (i)  naive      s_c = | psi(y)_c - theta_bar_c |   with theta_bar the ground-
       truth data-generating parameter (known for every synthetic curve);
  (ii) surrogate  s_c = | psi(y)_c - theta*_c |      with theta* the numerical
       solution of argmin_t || y - kepler(t) ||  (batched coordinate descent on
       the empirical grids, warm-started at psi(y)) — computable on real curves
       too, hence usable under distribution shift.

Calibration uses ONLY synthetic (fake) curves drawn from the empirical priors H
(which are themselves fit on the real TRAIN split only); real val/test systems
are reserved for testing the intervals.  Scores are per-coordinate over the four
physical coordinates (log10_P, log10_K, e, omega — circular distance for omega)
with a Bonferroni 1 - alpha/d quantile; the interval for coordinate c is
psi(y)_c ± q_c, intersected with the empirical support of H.

Covariate-shift reweighting (Tibshirani, Barber, Candès, Ramdas 2019): the
calibration scores are reweighted by the likelihood ratio w(x) = p_real(x) /
p_fake(x), estimated by a logistic real-vs-synthetic discriminator on the
summary features via the odds p/(1-p) (fit on a *separate* synthetic sample vs
the real TRAIN split — never on the real test systems).  The weighted quantile
puts mass w(x_test) at +infinity, so sets can be infinite when the test point
looks very real relative to the calibration cloud; weights are clipped
(--clip-weights) and the effective sample size is reported.

Noise-model normalization (Nicolò's s' — NOTE: he wrote s' = s/(gamma + s),
which is a strictly monotone transform of s and therefore changes no split-CP
set; we read it as the standard locally-normalized score using the uncertainty
proxy v he defines):

    s'_c = s_c / (gamma_reg + v),   v = RMS predictive std of the trained SVGP
    residual noise model evaluated on (kepler(psi(y)), psi(y), t), in units of
    the curve's rv_std (falls back to the median measurement sigma when the
    checkpoint is unavailable).  gamma_reg > 0 is tuned on a held-out synthetic
    tuning set to minimize the mean support-normalized median interval width.

Usage
-----
    python conformal_shift.py                          # full run, n=400
    python conformal_shift.py --n-cal 60 --n-test 60 --n-tune 30   # quick
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from conformal import (
    COORDS,
    D,
    SG,
    Scorer,
    _set_coord_grid,
    _theta_to_omega,
    _true_coord,
    histogram_grids,
    make_real,
    make_synthetic,
)
from generate_synthetic_regression_csv import TARGET_COLUMNS  # noqa: E402
from train_regression_models import _build  # noqa: E402

ROOT = Path(__file__).resolve().parent

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
STRATEGIES = ["naive", "surrogate"]


# ---------------------------------------------------------------------------
# Per-coordinate score: absolute error in theta space (circular for omega)
# ---------------------------------------------------------------------------


def _coord_abs_err(theta_a5: np.ndarray, theta_b5: np.ndarray, coord: str) -> float:
    a, b = _true_coord(theta_a5, coord), _true_coord(theta_b5, coord)
    if coord == "omega":
        return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))
    return float(abs(a - b))


# ---------------------------------------------------------------------------
# Surrogate label theta* = argmin_theta || y - kepler(theta) ||
# ---------------------------------------------------------------------------


def surrogate_fit(scorer: Scorer, base5: np.ndarray, pgrids: dict, curve: dict,
                  sweeps: int = 2) -> np.ndarray:
    """Coordinate-descent minimisation of the reconstruction score over all four
    coordinates on the empirical grids, warm-started at base5 (= psi(y)).  The
    incumbent is always kept as a candidate, so the fit can only improve."""
    th = base5.copy()[None, :]                                   # (1, 5)
    for _ in range(sweeps):
        for c in COORDS:
            big = _set_coord_grid(th, c, pgrids[c])              # (1, G, 5)
            big = np.concatenate([big, th[:, None, :]], axis=1)  # (1, G+1, 5)
            sc = scorer.score(big[0], curve)
            th = big[:, int(sc.argmin())]
    return th[0]


# ---------------------------------------------------------------------------
# Noise-model uncertainty proxy v(kepler(theta_hat), theta_hat, t)
# ---------------------------------------------------------------------------


class NoiseProxy:
    """v = RMS predictive std of the trained SVGP residual model on the
    predicted curve's feature rows, divided by the curve's rv_std (dimensionless
    noise fraction).  Falls back to the median per-obs measurement sigma (also
    in rv_std units) when the checkpoint can't be loaded."""

    def __init__(self):
        from synthetic_dataset import _load_gp_residual_sampler

        self.decoder = Scorer().decoder
        self.sampler = _load_gp_residual_sampler()
        self.source = "gp_residual_svgp" if self.sampler is not None else "median_sigma"

    @torch.no_grad()
    def _pred_curve_ms(self, theta5: np.ndarray, curve: dict) -> tuple[np.ndarray, np.ndarray]:
        """Decoder reconstruction at theta5 in m/s, plus the day-valued times,
        masked to the observed entries."""
        t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0)
        rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0)
        mask = torch.from_numpy(curve["mask"]).unsqueeze(0)
        t_span = torch.tensor([curve["t_span"]], dtype=torch.float32)
        t_min = torch.tensor([curve["t_min"]], dtype=torch.float32)
        rv_std = torch.tensor([curve["rv_std"]], dtype=torch.float32)
        th = torch.as_tensor(theta5[None, :], dtype=torch.float32)
        rv_pred = self.decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)[0].numpy()
        m = curve["mask"] > 0.5
        t_days = curve["t_norm"][m] * curve["t_span"] + curve["t_min"]
        return rv_pred[m] * curve["rv_std"], t_days

    @torch.no_grad()
    def value(self, theta5: np.ndarray, curve: dict) -> float:
        m = curve["mask"] > 0.5
        fallback = float(np.median(curve["sig"][m]))
        if self.sampler is None:
            return max(fallback, 1e-6)
        try:
            from synthetic_dataset import _gp_residual_features

            rv_ms, t_days = self._pred_curve_ms(theta5, curve)
            params = {"P": 10.0 ** theta5[0], "K": 10.0 ** theta5[1],
                      "e": theta5[2], "omega": _theta_to_omega(theta5)}
            X = _gp_residual_features(t_days, rv_ms, params)
            X = (X - self.sampler["mean"]) / np.maximum(self.sampler["std"], 1e-8)
            latent = self.sampler["model"](torch.as_tensor(X, dtype=torch.float32))
            var = latent.variance
            lik = self.sampler["likelihood"]
            try:
                df = float(lik.deg_free)
                noise = float(lik.noise)
                var = var + noise * (df / (df - 2.0) if df > 2.0 else 1.0)
            except Exception:
                pass
            v_ms = float(torch.sqrt(var.mean()))
            return max(v_ms / curve["rv_std"], 1e-6)
        except Exception:
            return max(fallback, 1e-6)


# ---------------------------------------------------------------------------
# Likelihood-ratio weights via a real-vs-fake discriminator
# ---------------------------------------------------------------------------


def fit_weight_model(synth_feats: np.ndarray, real_feats: np.ndarray, seed: int):
    """Logistic real(1)-vs-fake(0) discriminator on standardized log-ish summary
    features; w(x) = odds * class-balance correction = p_real/p_fake ratio."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([synth_feats, real_feats])
    y = np.r_[np.zeros(len(synth_feats)), np.ones(len(real_feats))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    clf.fit(X, y)
    n0, n1 = float(len(synth_feats)), float(len(real_feats))

    def w(feats: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(feats)[:, 1].clip(1e-6, 1 - 1e-6)
        return (p / (1.0 - p)) * (n0 / n1)

    return w, clf


def weighted_quantile(scores: np.ndarray, w_cal: np.ndarray, w_test: float,
                      level: float) -> float:
    """Tibshirani et al. 2019 weighted conformal quantile: normalized weights
    p_i = w_i / (sum_j w_j + w_test) on the calibration scores plus mass
    w_test / (...) at +infinity.  All-ones weights reduce to the standard
    ceil((n+1)*level) rank quantile."""
    order = np.argsort(scores)
    s, ww = scores[order], w_cal[order]
    total = float(ww.sum() + w_test)
    k = int(np.searchsorted(np.cumsum(ww), level * total, side="left"))
    if k >= len(s):
        return math.inf
    return float(s[k])


# ---------------------------------------------------------------------------
# Interval construction + evaluation
# ---------------------------------------------------------------------------


def _support(grids: dict) -> dict:
    sup = {c: (float(grids[c].min()), float(grids[c].max())) for c in COORDS}
    sup["omega"] = (0.0, 2.0 * np.pi)
    return sup


def _interval_width(coord: str, center: float, half: float, sup: dict) -> float:
    lo, hi = sup[coord]
    if coord == "omega":
        return float(min(2.0 * half, hi - lo))
    return float(max(0.0, min(center + half, hi) - max(center - half, lo)))


def evaluate(cal_scores: dict, v_cal: np.ndarray, systems: list, theta_hats: list,
             v_sys: np.ndarray, sup: dict, gamma: float | None,
             w_cal: np.ndarray | None = None, w_test: np.ndarray | None = None) -> dict:
    """Coverage/width of the per-coordinate intervals psi(y)_c ± q_c at each alpha.

    gamma=None  -> raw score s;  gamma=float -> normalized s' = s/(gamma+v).
    w_cal/w_test=None -> unweighted split-CP.
    """
    n_test = len(systems)
    ones = np.ones(len(v_cal))
    w_cal = ones if w_cal is None else w_cal
    w_test = np.ones(n_test) if w_test is None else w_test

    out = {}
    for a in ALPHAS:
        level = 1.0 - a / D
        per_cov = {c: [] for c in COORDS}
        per_w = {c: [] for c in COORDS}
        n_inf = 0
        joint = []
        for i, (s, th) in enumerate(zip(systems, theta_hats)):
            all_c = True
            any_inf = False
            for c in COORDS:
                sc = cal_scores[c] if gamma is None else cal_scores[c] / (gamma + v_cal)
                q = weighted_quantile(sc, w_cal, float(w_test[i]), level)
                half = q if gamma is None else q * (gamma + v_sys[i])
                if not math.isfinite(half):
                    any_inf = True
                    half = sup[c][1] - sup[c][0]     # cap at full support
                err = _coord_abs_err(s["theta5"], th, c)
                cov = err <= half
                per_cov[c].append(bool(cov))
                per_w[c].append(_interval_width(c, _true_coord(th, c), half, sup))
                all_c = all_c and cov
            n_inf += int(any_inf)
            joint.append(all_c)
        out[f"{a:.2f}"] = {
            "per_coord_coverage": {c: float(np.mean(per_cov[c])) for c in COORDS},
            "per_coord_median_width": {c: float(np.median(per_w[c])) for c in COORDS},
            "joint_coverage": float(np.mean(joint)),
            "frac_infinite": n_inf / max(n_test, 1),
        }
    return out


def tune_gamma(cal_scores: dict, v_cal: np.ndarray, tune_sys: list, tune_hats: list,
               v_tune: np.ndarray, sup: dict, alpha: float = 0.10) -> float:
    """Pick gamma_reg minimizing the mean (over coords) support-normalized median
    width on the synthetic tuning set, at the reference alpha."""
    level = 1.0 - alpha / D
    grid = np.median(v_cal) * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    best_g, best_obj = float(grid[0]), math.inf
    for g in grid:
        obj = 0.0
        for c in COORDS:
            q = weighted_quantile(cal_scores[c] / (g + v_cal),
                                  np.ones(len(v_cal)), 1.0, level)
            widths = [_interval_width(c, _true_coord(th, c), q * (g + v_tune[i]), sup)
                      for i, th in enumerate(tune_hats)]
            obj += float(np.median(widths)) / (sup[c][1] - sup[c][0])
        if obj < best_obj:
            best_obj, best_g = obj, float(g)
    return best_g


# ---------------------------------------------------------------------------
# Figures + report
# ---------------------------------------------------------------------------


def plot_coverage(results: dict, fig_dir: Path) -> None:
    doms = ["synthetic_unweighted", "real_unweighted", "real_weighted"]
    fig, axs = plt.subplots(len(STRATEGIES), len(doms),
                            figsize=(4.6 * len(doms), 4.2 * len(STRATEGIES)))
    nominal = [1 - a for a in ALPHAS]
    for r, strat in enumerate(STRATEGIES):
        for cidx, dom in enumerate(doms):
            ax = axs[r][cidx]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            for norm in ["raw", "vnorm"]:
                res = results[strat][norm].get(dom)
                if res is None:
                    continue
                ax.plot(nominal, [res[f"{a:.2f}"]["joint_coverage"] for a in ALPHAS],
                        "o-", label=f"{norm} joint")
            ax.set_title(f"{strat} — {dom}", fontsize=10)
            ax.set_xlabel("nominal 1-alpha")
            ax.set_ylabel("empirical coverage")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Weighted split-CP (calibrate on fake, test on real) — joint coverage",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "conformal_shift_coverage.png", dpi=180)
    plt.close(fig)


def plot_widths(results: dict, fig_dir: Path, alpha: float = 0.10) -> None:
    doms = ["synthetic_unweighted", "real_unweighted", "real_weighted"]
    labels, series = [], []
    for strat in STRATEGIES:
        for norm in ["raw", "vnorm"]:
            labels.append(f"{strat}/{norm}")
            series.append(results[strat][norm])
    fig, axs = plt.subplots(1, len(doms), figsize=(5.2 * len(doms), 4.6))
    x = np.arange(D)
    w = 0.8 / len(labels)
    for ax, dom in zip(axs, doms):
        for k, (lab, res) in enumerate(zip(labels, series)):
            if res.get(dom) is None:
                continue
            ws = res[dom][f"{alpha:.2f}"]["per_coord_median_width"]
            ax.bar(x + (k - len(labels) / 2 + 0.5) * w, [ws[c] for c in COORDS],
                   w, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(COORDS)
        ax.set_title(dom)
        ax.set_ylabel(f"median width @ 1-alpha={1-alpha:.2f}")
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("Interval widths by strategy / score normalization", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "conformal_shift_widths.png", dpi=180)
    plt.close(fig)


def write_report(report: dict, path: Path) -> None:
    lines = ["Split-CP calibrated on fake, tested on real — naive vs surrogate",
             "=" * 72,
             f"n_cal={report['n_cal']}  n_tune={report['n_tune']}  "
             f"n_test_syn={report['n_test_syn']}  n_test_real={report['n_test_real']}",
             f"noise proxy source: {report['proxy_source']}",
             f"weights: ESS={report['weights']['ess']:.1f} of n_cal={report['n_cal']}, "
             f"clipped to [{report['weights']['clip_lo']:g}, {report['weights']['clip_hi']:g}], "
             f"{report['weights']['frac_clipped']:.1%} clipped",
             ""]
    for strat in STRATEGIES:
        lines.append(f"##### STRATEGY: {strat} "
                     f"(gamma_reg={report['gamma_reg'][strat]:.4g}) #####")
        med = report["cal_score_median"][strat]
        lines.append("calibration score median per coord: "
                     + "  ".join(f"{c}={med[c]:.3g}" for c in COORDS))
        for norm in ["raw", "vnorm"]:
            for dom in ["synthetic_unweighted", "real_unweighted", "real_weighted"]:
                res = report["results"][strat][norm].get(dom)
                if res is None:
                    continue
                lines.append(f"[{norm} | {dom}]")
                lines.append(f"  {'1-alpha':>8}{'joint':>9}"
                             + "".join(f"{c:>11}" for c in COORDS) + f"{'%inf':>7}")
                for a in ALPHAS:
                    r = res[f"{a:.2f}"]
                    row = f"  {1-a:>8.2f}{r['joint_coverage']:>9.3f}"
                    row += "".join(f"{r['per_coord_coverage'][c]:>11.3f}" for c in COORDS)
                    row += f"{r['frac_infinite']:>7.2f}"
                    lines.append(row)
                wdt = res["0.10"]["per_coord_median_width"]
                lines.append("  median width @ 1-alpha=0.90: "
                             + "  ".join(f"{c}={wdt[c]:.3g}" for c in COORDS))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=SG / "datasets" / "synthetic_lsp_regression_10000.csv",
                    help="psi training CSV; feature columns are taken from it "
                         "(default: 512-bin LSP dataset, per Nicolò's OK on more "
                         "Fourier bins; pass synthetic_regression_10000.csv for "
                         "the 64-bin variant)")
    ap.add_argument("--out-dir", type=Path, default=SG / "regression")
    ap.add_argument("--fig-dir", type=Path,
                    default=SG / "figures" / "synthetic_regression_10000")
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-tune", type=int, default=100)
    ap.add_argument("--n-weight-synth", type=int, default=400,
                    help="fresh synthetic sample size for the weight discriminator")
    ap.add_argument("--grid", type=int, default=33,
                    help="grid resolution for the surrogate coordinate descent")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--clip-weights", type=float, default=20.0,
                    help="clip likelihood-ratio weights to [1/x, x]")
    ap.add_argument("--real-split", default="test", choices=("all", "train", "val", "test"))
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=100.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # Step-5 point predictor psi. Feature columns come from the CSV itself, so
    # the 512-bin LSP dataset and the 64-bin one both work; per-system vectors
    # are assembled from the stored summary row + raw LSP to match.
    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]

    def feat_matrix(systems: list) -> np.ndarray:
        rows = []
        for s in systems:
            fr, lsp = s["feat_row"], s["lsp"]
            rows.append([fr[c] if c in fr else lsp[int(c.rsplit("_", 1)[1]) - 1]
                         for c in feature_cols])
        return np.asarray(rows, dtype=float)

    rf = _build("separate", args.n_estimators, args.seed, list(TARGET_COLUMNS))
    rf.fit(df[feature_cols].to_numpy(float), df[list(TARGET_COLUMNS)].to_numpy(float))
    print(f"trained RF psi on {len(df)} synthetic rows, {len(feature_cols)} features")

    grids = histogram_grids(args.grid, args.seed)
    sup = _support(grids)

    print("building systems ...")
    calib = make_synthetic(args.n_cal, args.seed + 1)
    tune = make_synthetic(args.n_tune, args.seed + 11)
    test_syn = make_synthetic(args.n_test, args.seed + 2)
    wsynth = make_synthetic(args.n_weight_synth, args.seed + 21)
    test_real = make_real(args.real_split, args.sigma_min, args.sigma_max)
    real_train = make_real("train", args.sigma_min, args.sigma_max)
    print(f"n_cal={len(calib)} n_tune={len(tune)} n_test_syn={len(test_syn)} "
          f"n_test_real={len(test_real)} (weight fit: {len(wsynth)} synth vs "
          f"{len(real_train)} real-train)")

    def hats(systems):
        return list(rf.predict(feat_matrix(systems)))

    hat = {k: hats(v) for k, v in
           [("cal", calib), ("tune", tune), ("syn", test_syn), ("real", test_real)]}

    # Uncertainty proxy v per system (from the trained noise model).
    proxy = NoiseProxy()
    print(f"noise proxy source: {proxy.source}")
    v = {k: np.array([proxy.value(th, s["curve"]) for s, th in zip(sys_, hat[k])])
         for k, sys_ in [("cal", calib), ("tune", tune), ("syn", test_syn),
                         ("real", test_real)]}

    # Likelihood-ratio weights (fit: fresh synth vs real TRAIN; applied to cal +
    # real test). Deliberately kept on the 74-dim summary FEATURES (s["features"])
    # rather than psi's possibly 586-dim set: a 586-dim discriminator on ~700
    # points separates the classes too well and degenerates the weights.
    w_fn, _ = fit_weight_model(np.vstack([s["features"] for s in wsynth]),
                               np.vstack([s["features"] for s in real_train]),
                               args.seed)
    clip = args.clip_weights
    w_cal_raw = w_fn(np.vstack([s["features"] for s in calib]))
    w_real_raw = w_fn(np.vstack([s["features"] for s in test_real]))
    w_cal = np.clip(w_cal_raw, 1.0 / clip, clip)
    w_real = np.clip(w_real_raw, 1.0 / clip, clip)
    ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum())
    frac_clipped = float(np.mean((w_cal_raw < 1 / clip) | (w_cal_raw > clip)))
    print(f"weights: ESS={ess:.1f}/{len(w_cal)}  clipped={frac_clipped:.1%}")

    # Surrogate labels on the calibration set (the only place they are needed).
    scorer = Scorer()
    print("fitting surrogate labels on calibration curves ...")
    t0 = time.perf_counter()
    theta_star = [surrogate_fit(scorer, th, grids, s["curve"], args.sweeps)
                  for s, th in zip(calib, hat["cal"])]
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    # Calibration scores per strategy and coordinate.
    cal_scores = {
        "naive": {c: np.array([_coord_abs_err(hat["cal"][j], calib[j]["theta5"], c)
                               for j in range(len(calib))]) for c in COORDS},
        "surrogate": {c: np.array([_coord_abs_err(hat["cal"][j], theta_star[j], c)
                                   for j in range(len(calib))]) for c in COORDS},
    }

    results, gamma_reg = {}, {}
    for strat in STRATEGIES:
        cs = cal_scores[strat]
        g = tune_gamma(cs, v["cal"], tune, hat["tune"], v["tune"], sup)
        gamma_reg[strat] = g
        print(f"[{strat}] gamma_reg={g:.4g}")
        results[strat] = {}
        for norm, gval in [("raw", None), ("vnorm", g)]:
            results[strat][norm] = {
                "synthetic_unweighted": evaluate(cs, v["cal"], test_syn, hat["syn"],
                                                 v["syn"], sup, gval),
                "real_unweighted": evaluate(cs, v["cal"], test_real, hat["real"],
                                            v["real"], sup, gval),
                "real_weighted": evaluate(cs, v["cal"], test_real, hat["real"],
                                          v["real"], sup, gval,
                                          w_cal=w_cal, w_test=w_real),
            }
            for dom in ["synthetic_unweighted", "real_unweighted", "real_weighted"]:
                r = results[strat][norm][dom]["0.10"]
                print(f"  [{strat}/{norm}/{dom}] joint@0.90={r['joint_coverage']:.3f} "
                      f"inf={r['frac_infinite']:.2f}")

    report = {
        "n_cal": len(calib), "n_tune": len(tune),
        "n_test_syn": len(test_syn), "n_test_real": len(test_real),
        "alphas": ALPHAS, "coords": COORDS,
        "proxy_source": proxy.source,
        "gamma_reg": gamma_reg,
        "weights": {"ess": ess, "frac_clipped": frac_clipped,
                    "clip_lo": 1.0 / clip, "clip_hi": clip},
        "cal_score_median": {s: {c: float(np.median(cal_scores[s][c])) for c in COORDS}
                             for s in STRATEGIES},
        "results": results,
    }
    (args.out_dir / "conformal_shift_metrics.json").write_text(
        json.dumps(report, indent=2))
    write_report(report, args.out_dir / "conformal_shift_report.txt")
    plot_coverage(results, args.fig_dir)
    plot_widths(results, args.fig_dir)
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s — wrote metrics/report to "
          f"{args.out_dir}, figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
