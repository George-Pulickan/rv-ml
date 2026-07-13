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
       solution of argmin_t mean_i | y_i - kepler(t)_i |  (batched Adam gradient
       descent through the differentiable KeplerDecoder, initialized at the
       data-generating / tabulated values — Nicolò's 2026-07 spec) — computable
       on real curves too (tabulated init), hence usable under shift.

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

Noise-model normalization (Nicolò confirmed 2026-07 that s' = s/(gamma + v)
is what he meant), two variants:

    vnorm   s'_c = s_c / (gamma + v_y)
    v2norm  s'_c = s_c / (gamma + v_y + v_c)     (his two-factor version)

with v_y = RMS predictive std of the trained SVGP residual noise model
evaluated on (kepler(psi(y)), psi(y), t), in units of the curve's rv_std
(falls back to the median measurement sigma when the checkpoint is
unavailable), and v_c = a per-coordinate model of the surrogate-label error
E|theta_bar_c - theta*_c| (an RF on the summary features, fit on the synthetic
tuning set where theta_bar is known).  gamma > 0 is tuned per variant on the
same tuning set to minimize the mean support-normalized median interval width.

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
    _theta_to_omega,
    _true_coord,
    histogram_grids,
    make_real,
    make_synthetic,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from train_regression_models import _build  # noqa: E402

ROOT = Path(__file__).resolve().parent

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
STRATEGIES = ["naive", "surrogate"]
NORMS = ["raw", "vnorm", "v2norm"]


# ---------------------------------------------------------------------------
# Per-coordinate score: absolute error in theta space (circular for omega)
# ---------------------------------------------------------------------------


def _coord_abs_err(theta_a5: np.ndarray, theta_b5: np.ndarray, coord: str) -> float:
    a, b = _true_coord(theta_a5, coord), _true_coord(theta_b5, coord)
    if coord == "omega":
        return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))
    return float(abs(a - b))


# ---------------------------------------------------------------------------
# Surrogate label theta* = argmin_theta mean_t | y_t - kepler(theta)_t |
# ---------------------------------------------------------------------------


def _gd_batch(decoder, init5s: np.ndarray, curves: list, steps: int,
              lr: float) -> np.ndarray:
    """Batched Adam minimisation of the masked mean-absolute reconstruction
    error (Nicolò's E_t |y_t - kepler(theta', t)|, in rv_std units) over a set
    of curves sharing one padded length.  t_peri / gamma are refit analytically
    inside the decoder every step (their refit is detached — envelope-style;
    gradients flow through the RV evaluation).  The best iterate per curve is
    kept, so the fit can only improve on the initialization."""
    th = torch.as_tensor(np.asarray(init5s), dtype=torch.float32).clone()  # (B,5)
    t_norm = torch.from_numpy(np.stack([c["t_norm"] for c in curves]))
    rv_obs = torch.from_numpy(np.stack([c["rv_obs"] for c in curves]))
    mask = torch.from_numpy(np.stack([c["mask"] for c in curves]))
    t_span = torch.tensor([c["t_span"] for c in curves], dtype=torch.float32)
    t_min = torch.tensor([c["t_min"] for c in curves], dtype=torch.float32)
    rv_std = torch.tensor([c["rv_std"] for c in curves], dtype=torch.float32)
    n = mask.sum(dim=1).clamp(min=1.0)

    def losses(theta: torch.Tensor) -> torch.Tensor:
        rv_pred = decoder(theta, t_norm, t_span, t_min, rv_obs, rv_std, mask)
        return ((rv_obs - rv_pred).abs() * mask).sum(dim=1) / n            # (B,)

    th.requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    best_loss = torch.full((th.shape[0],), np.inf)
    best_th = th.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        loss = losses(th)
        with torch.no_grad():
            better = loss < best_loss
            best_loss[better] = loss.detach()[better]
            best_th[better] = th.detach()[better]
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            th[:, 2].clamp_(0.0, 0.99)
    with torch.no_grad():
        loss = losses(th)
        better = loss < best_loss
        best_th[better] = th.detach()[better]
    return best_th.numpy().astype(float)


def surrogate_fit_gd(decoder, init5s: list, systems: list, steps: int = 200,
                     lr: float = 0.02) -> list:
    """Surrogate labels for a list of systems, batched by padded curve length,
    initialized at init5s (= theta_bar on synthetic curves, tabulated values on
    real ones, per Nicolò 2026-07)."""
    out: list = [None] * len(systems)
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(systems):
        by_len.setdefault(len(s["curve"]["t_norm"]), []).append(i)
    for idx in by_len.values():
        fitted = _gd_batch(decoder, np.asarray([init5s[i] for i in idx]),
                           [systems[i]["curve"] for i in idx], steps, lr)
        for k, i in enumerate(idx):
            out[i] = fitted[k]
    return out


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
        self.wants_sigma = (self.sampler is not None
                            and "log10_sigma" in self.sampler["feature_names"])

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
            sig_ms = curve["sig"][m] * curve["rv_std"] if self.wants_sigma else None
            X = _gp_residual_features(t_days, rv_ms, params, sigma=sig_ms)
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
# Surrogate-label error model v_c(y) ~ E | theta_bar_c - theta*_c |
# ---------------------------------------------------------------------------


def fit_vk_models(feats: np.ndarray, theta_bars: list, theta_stars: list,
                  seed: int, n_estimators: int = 200):
    """Per-coordinate model of the surrogate-label error (the second reweighting
    factor in Nicolò's s'_c = s_c / (gamma + v_y + v_c), 2026-07): an RF
    regressor of |theta_bar_c - theta*_c| on the summary features, fit on the
    synthetic tuning set where theta_bar is known.  Returns vk(feats) ->
    {coord: (n,) nonnegative array}."""
    from sklearn.ensemble import RandomForestRegressor

    models = {}
    for c in COORDS:
        errs = np.array([_coord_abs_err(tb, ts, c)
                         for tb, ts in zip(theta_bars, theta_stars)])
        m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=-1)
        m.fit(feats, errs)
        models[c] = m

    def vk(f: np.ndarray) -> dict:
        return {c: np.maximum(models[c].predict(f), 0.0) for c in COORDS}

    return vk


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


def evaluate(cal_scores: dict, systems: list, theta_hats: list, sup: dict,
             den_cal: dict | None = None, den_sys: dict | None = None,
             w_cal: np.ndarray | None = None, w_test: np.ndarray | None = None) -> dict:
    """Coverage/width of the per-coordinate intervals psi(y)_c ± q_c at each alpha.

    den_cal/den_sys=None -> raw score s;  otherwise per-coordinate denominator
    arrays (gamma + v_y [+ v_c]) and the normalized score s' = s/den is used
    (interval half-width scales back by den at the test point).
    w_cal/w_test=None -> unweighted split-CP.
    """
    n_test = len(systems)
    n_cal = len(next(iter(cal_scores.values())))
    w_cal = np.ones(n_cal) if w_cal is None else w_cal
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
                sc = cal_scores[c] if den_cal is None else cal_scores[c] / den_cal[c]
                q = weighted_quantile(sc, w_cal, float(w_test[i]), level)
                half = q if den_sys is None else q * float(den_sys[c][i])
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


def tune_gamma(cal_scores: dict, base_cal: dict, tune_hats: list,
               base_tune: dict, sup: dict, alpha: float = 0.10) -> float:
    """Pick gamma minimizing the mean (over coords) support-normalized median
    width on the synthetic tuning set, at the reference alpha.  base_cal /
    base_tune are per-coordinate denominator bases (v_y or v_y + v_c); the
    tuned denominator is gamma + base."""
    level = 1.0 - alpha / D
    med = np.median(np.concatenate([base_cal[c] for c in COORDS]))
    grid = med * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    best_g, best_obj = float(grid[0]), math.inf
    for g in grid:
        obj = 0.0
        for c in COORDS:
            q = weighted_quantile(cal_scores[c] / (g + base_cal[c]),
                                  np.ones(len(base_cal[c])), 1.0, level)
            widths = [_interval_width(c, _true_coord(th, c),
                                      q * (g + float(base_tune[c][i])), sup)
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
            for norm in NORMS:
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
        for norm in NORMS:
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
             "v_c median (cal): " + "  ".join(
                 f"{c}={report['vk_median']['cal'][c]:.3g}" for c in COORDS),
             ""]
    for strat in STRATEGIES:
        gr = report["gamma_reg"][strat]
        lines.append(f"##### STRATEGY: {strat} "
                     f"(gamma_vnorm={gr['vnorm']:.4g}, "
                     f"gamma_v2norm={gr['v2norm']:.4g}) #####")
        med = report["cal_score_median"][strat]
        lines.append("calibration score median per coord: "
                     + "  ".join(f"{c}={med[c]:.3g}" for c in COORDS))
        for norm in NORMS:
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
                    help="resolution of the empirical histogram grids (support)")
    ap.add_argument("--gd-steps", type=int, default=200,
                    help="Adam steps for the surrogate-label gradient descent")
    ap.add_argument("--gd-lr", type=float, default=0.02)
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

    # Surrogate labels theta* by gradient descent (L1 objective, initialized at
    # the data-generating theta_bar — Nicolò 2026-07): on the calibration set
    # (for the surrogate scores) and on the tuning set (to fit the v_c model).
    print("fitting surrogate labels by gradient descent ...")
    t0 = time.perf_counter()
    theta_star = surrogate_fit_gd(proxy.decoder, [s["theta5"] for s in calib],
                                  calib, args.gd_steps, args.gd_lr)
    theta_star_tune = surrogate_fit_gd(proxy.decoder, [s["theta5"] for s in tune],
                                       tune, args.gd_steps, args.gd_lr)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    # Surrogate-label error model v_c (second factor of the v2norm denominator),
    # fit on the tuning set, evaluated everywhere on the summary features.
    vk_fn = fit_vk_models(np.vstack([s["features"] for s in tune]),
                          [s["theta5"] for s in tune], theta_star_tune,
                          args.seed, args.n_estimators)
    vk = {k: vk_fn(np.vstack([s["features"] for s in sys_]))
          for k, sys_ in [("cal", calib), ("tune", tune), ("syn", test_syn),
                          ("real", test_real)]}
    print("v_c median (cal): " + "  ".join(
        f"{c}={float(np.median(vk['cal'][c])):.3g}" for c in COORDS))

    # Calibration scores per strategy and coordinate.
    cal_scores = {
        "naive": {c: np.array([_coord_abs_err(hat["cal"][j], calib[j]["theta5"], c)
                               for j in range(len(calib))]) for c in COORDS},
        "surrogate": {c: np.array([_coord_abs_err(hat["cal"][j], theta_star[j], c)
                                   for j in range(len(calib))]) for c in COORDS},
    }

    # Denominator bases per norm variant: vnorm = v_y, v2norm = v_y + v_c.
    def base(kind: str, key: str) -> dict:
        if kind == "vnorm":
            return {c: v[key] for c in COORDS}
        return {c: v[key] + vk[key][c] for c in COORDS}

    results, gamma_reg = {}, {}
    for strat in STRATEGIES:
        cs = cal_scores[strat]
        gamma_reg[strat] = {
            kind: tune_gamma(cs, base(kind, "cal"), hat["tune"], base(kind, "tune"), sup)
            for kind in ["vnorm", "v2norm"]}
        print(f"[{strat}] gamma_vnorm={gamma_reg[strat]['vnorm']:.4g} "
              f"gamma_v2norm={gamma_reg[strat]['v2norm']:.4g}")
        results[strat] = {}
        for norm in NORMS:
            if norm == "raw":
                dens = {"cal": None, "syn": None, "real": None}
            else:
                g = gamma_reg[strat][norm]
                dens = {key: {c: g + base(norm, key)[c] for c in COORDS}
                        for key in ["cal", "syn", "real"]}
            results[strat][norm] = {
                "synthetic_unweighted": evaluate(cs, test_syn, hat["syn"], sup,
                                                 dens["cal"], dens["syn"]),
                "real_unweighted": evaluate(cs, test_real, hat["real"], sup,
                                            dens["cal"], dens["real"]),
                "real_weighted": evaluate(cs, test_real, hat["real"], sup,
                                          dens["cal"], dens["real"],
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
        "vk_median": {k: {c: float(np.median(vk[k][c])) for c in COORDS}
                      for k in ["cal", "real"]},
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
