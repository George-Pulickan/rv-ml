"""
Paper figures + Earth-like table for the AAAI exoplanet experiment.

Uses the MLP as psi and conformal quantiles from conformal_shift.py
(--psi mlp). Produces:

  figures/paper/rv_heldout_phasefold.png   (Figure 1)
  figures/paper/rv_pred_vs_true.png        (Figure 2)
  figures/paper/earthlike_top10.csv
  figures/paper/earthlike_top10.tex

Usage
-----
    python scripts/paper_rv_figures.py
    python scripts/paper_rv_figures.py --host "HD 2952"
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformal import (  # noqa: E402
    COORDS,
    Scorer,
    _theta_to_omega,
    _true_coord,
    curve_times_days,
    make_real,
)
from conformal_shift import _load_mlp_psi  # noqa: E402
from feature_columns import TARGET_COLUMNS  # noqa: E402
from kepler_check import rv_keplerian  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402
from preprocess import RVDataset  # noqa: E402
from regression import (  # noqa: E402
    load_from_csv,
    plot_pred_vs_true,
    predict,
    _per_target_metrics,
    _subset_metrics,
    _e_subset_report,
)
from synthetic_dataset import _inject_noise  # noqa: E402

DEFAULT_CKPT = ROOT / "checkpoints" / "regression_mlp_74.pt"
DEFAULT_CSV = ROOT / "synthetic_generation" / "datasets" / "synthetic_regression_10000.csv"
DEFAULT_Q = ROOT / "figures" / "paper" / "mlp_cp_quantiles.json"
DEFAULT_METRICS = ROOT / "synthetic_generation" / "regression" / "mlp_psi" / "conformal_shift_metrics.json"
OUT_DIR = ROOT / "figures" / "paper"


def _theta5_to_params(th: np.ndarray) -> dict[str, float]:
    omega = _theta_to_omega(th)
    return {
        "P": float(10.0 ** th[0]),
        "K": float(10.0 ** th[1]),
        "e": float(np.clip(th[2], 0.0, 0.99)),
        "omega": float(omega),
        "phase": 0.0,
    }


def _set_omega(th: np.ndarray, omega: float) -> np.ndarray:
    out = np.asarray(th, dtype=float).copy()
    out[3] = math.cos(omega)
    out[4] = math.sin(omega)
    return out


def _match_system_widths(
    theta_hat: np.ndarray,
    widths_blob: dict | None,
    fallback: dict | None = None,
    alpha: str = "0.40",
) -> dict[str, float] | None:
    """Papernorm half-widths for the system whose psi(y) is closest to theta_hat.

    Non-finite per-system entries (re-encode failures) fall back per-coordinate
    to ``fallback`` (global raw quantiles) when provided.
    """
    if not widths_blob or not widths_blob.get("systems"):
        return None
    th = np.asarray(theta_hat, dtype=float)
    best, best_d = None, np.inf
    for row in widths_blob["systems"]:
        d = float(np.linalg.norm(np.asarray(row["theta5"], dtype=float) - th))
        if d < best_d:
            best_d, best = d, row
    if best is None or best_d > 1.0:  # loose match guard
        return None
    out = {}
    for c in ("log10_P", "log10_K", "e", "omega"):
        v = float(best["halfwidths"][alpha][c])
        if not math.isfinite(v) and fallback is not None:
            v = float(fallback[c])
        out[c] = v
    return out


def _sample_region(center5: np.ndarray, q: dict[str, float], n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Uniform samples in the Bonferroni box Γ_α around psi(y)."""
    samples = []
    for _ in range(n):
        th = np.asarray(center5, dtype=float).copy()
        for c in COORDS:
            half = float(q[c])
            ctr = _true_coord(th, c)
            if c == "omega":
                th = _set_omega(th, (ctr + rng.uniform(-half, half)) % (2.0 * math.pi))
            elif c == "e":
                th[2] = float(np.clip(ctr + rng.uniform(-half, half), 0.0, 0.99))
            elif c == "log10_P":
                th[0] = ctr + rng.uniform(-half, half)
            elif c == "log10_K":
                th[1] = ctr + rng.uniform(-half, half)
        samples.append(th)
    return samples


def _obs_ms(curve: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = curve["mask"] > 0.5
    t = curve_times_days(curve, m)
    rv = curve["rv_obs"][m] * curve["rv_std"]
    sig = curve["sig"][m] * curve["rv_std"]
    return t.astype(float), rv.astype(float), sig.astype(float)


def _kepler_on_grid(th: np.ndarray, t: np.ndarray, t_peri: float) -> np.ndarray:
    p = _theta5_to_params(th)
    return rv_keplerian(t, p["P"], p["K"], p["e"], p["omega"], t_peri)


def _anchor_t_peri(th: np.ndarray, t: np.ndarray, rv: np.ndarray) -> float:
    """Pick t_peri so the Kepler model matches the data median offset (γ free)."""
    p = _theta5_to_params(th)
    # Try a fine phase grid; choose the one minimizing MAD after γ-centering.
    best_tp, best_mad = float(t.min()), np.inf
    for phase in np.linspace(0.0, 1.0, 64, endpoint=False):
        tp = float(t.min()) + phase * p["P"]
        model = rv_keplerian(t, p["P"], p["K"], p["e"], p["omega"], tp)
        resid = rv - model
        resid = resid - np.median(resid)
        mad = float(np.median(np.abs(resid)))
        if mad < best_mad:
            best_mad, best_tp = mad, tp
    return best_tp


def _phase(t: np.ndarray, P: float, t_peri: float) -> np.ndarray:
    return ((t - t_peri) / P) % 1.0


def _feat_row_for_system(system: dict, feature_cols: list[str]) -> np.ndarray:
    fr, lsp = system["feat_row"], system["lsp"]

    def _val(c: str) -> float:
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)

    return np.asarray([_val(c) for c in feature_cols], dtype=float)


def load_quantiles(path: Path, metrics_path: Path) -> dict:
    if path.exists():
        blob = json.loads(path.read_text())
        return blob
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        q = m["quantiles_unweighted"]["surrogate"]
        blob = {
            "psi": m.get("psi"),
            "checkpoint": m.get("checkpoint"),
            "n_cal": m["n_cal"],
            "strategy": "surrogate",
            "norm": "raw",
            "quantiles": q,
            "source": str(metrics_path),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2))
        return blob
    raise FileNotFoundError(f"missing quantiles at {path} and metrics at {metrics_path}")


def pick_system(host: str | None) -> tuple[dict, dict]:
    """Return (conformal-style system dict, RVDataset info) for a test host."""
    ds = RVDataset("test", normalize=False, single_planet=True)
    systems = make_real("test", 0.1, 100.0)
    # Align by iterating dataset in the same filter order as make_real.
    aligned = []
    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue
        from conformal import _masked_observations, _curve_from_x
        from eval_omega_nn_vs_rf import _summary_row

        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
        if not (0.1 <= med_sigma <= 100.0):
            continue
        feats = _summary_row(xm, info, lsp)
        sys_ = {
            "curve": _curve_from_x(x, info),
            "feat_row": feats,
            "lsp": np.asarray(lsp, dtype=float),
            "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
            "info": info,
        }
        aligned.append(sys_)

    if host:
        for s in aligned:
            if s["info"]["host"].lower() == host.lower():
                return s, s["info"]
        raise ValueError(f"host {host!r} not found in filtered real test set")

    # Prefer moderate e, enough points, and a host where psi gets P roughly right.
    scored = []
    for s in aligned:
        e = float(s["theta5"][2])
        n = int(s["curve"]["mask"].sum())
        if n < 30 or not (0.08 <= e <= 0.45):
            continue
        scored.append((n, e, s))
    if not scored:
        return aligned[0], aligned[0]["info"]
    # Prefer HD 139357 when present (clean held-out demo); else most points near e~0.25.
    for _, _, s in scored:
        if s["info"]["host"] == "HD 139357":
            return s, s["info"]
    scored.sort(key=lambda z: (-z[0], abs(z[1] - 0.25)))
    s = scored[0][2]
    return s, s["info"]


def figure1(
    system: dict,
    info: dict,
    psi_predict,
    feature_cols: list[str],
    q04: dict[str, float],
    out_path: Path,
    *,
    n_region: int = 20,
    n_noisy: int = 12,
    seed: int = 0,
    widths_blob: dict | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    X = _feat_row_for_system(system, feature_cols)[None, :]
    th_tab = system["theta5"]
    th_psi = psi_predict(X)[0]

    # Prefer per-system papernorm widths when available (tighter for well-measured hosts).
    q_region = _match_system_widths(th_psi, widths_blob, fallback=q04, alpha="0.40") or q04
    # Cap omega half-width at π so Γ traces stay visually interpretable.
    q_region = dict(q_region)
    q_region["omega"] = float(min(q_region["omega"], math.pi))

    t, rv, sig = _obs_ms(system["curve"])
    P_fold = float(10.0 ** th_tab[0])
    t_peri = _anchor_t_peri(th_tab, t, rv)
    phase_obs = _phase(t, P_fold, t_peri)

    # Dense phase grid for smooth Kepler overlays (evaluate in time via t_peri).
    phase_grid = np.linspace(0.0, 1.0, 400)
    t_grid = t_peri + phase_grid * P_fold

    def folded_model(th: np.ndarray) -> np.ndarray:
        # Physical model at its own P/K/e/ω, plotted vs tabulated-P phase.
        return _kepler_on_grid(th, t_grid, t_peri)

    def model_on_obs(th: np.ndarray) -> np.ndarray:
        return _kepler_on_grid(th, t, t_peri)

    region = _sample_region(th_psi, q_region, n_region, rng)
    # Phase-fold display: freeze P at the fold period so Γ traces stay
    # single-valued in phase (varying P on a fixed phase grid creates
    # multi-cycle nonsense). Widths on K/e/ω still show conformal uncertainty.
    P_log_fold = float(np.log10(P_fold))
    region = [np.array([P_log_fold, th[1], th[2], th[3], th[4]], dtype=float) for th in region]

    # Noisy simulator draws at psi(y): Kepler + residual noise on the real cadence.
    p_psi = _theta5_to_params(th_psi)
    p_psi["phase"] = ((t_peri - float(t.min())) / p_psi["P"]) % 1.0
    noisy_folds = []
    for i in range(n_noisy):
        clean = rv_keplerian(t, p_psi["P"], p_psi["K"], p_psi["e"], p_psi["omega"], t_peri)
        noise, _ = _inject_noise(t, sig, np.random.default_rng(seed + 100 + i),
                                 dominant_params=p_psi, rv_clean_dominant=clean)
        y = clean + noise
        # Align γ like observations (median residual vs data).
        y = y - np.median(y - rv)
        noisy_folds.append((phase_obs, y))

    # Global γ so tabulated model matches observation median residual.
    gamma_tab = float(np.median(rv - model_on_obs(th_tab)))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # (iv) light region traces
    for th in region:
        y = folded_model(th) + gamma_tab
        ax.plot(phase_grid, y, color="0.75", lw=0.7, alpha=0.55, zorder=1)
    # (v) noisy draws
    for ph, y in noisy_folds:
        ax.scatter(ph, y, s=8, c="tab:orange", alpha=0.35, edgecolors="none", zorder=2)
    # (i) observations
    ax.errorbar(phase_obs, rv, yerr=sig, fmt="o", ms=3.5, color="k",
                ecolor="0.55", elinewidth=0.6, capsize=0, label="observations", zorder=5)
    # (ii) tabulated
    ax.plot(phase_grid, folded_model(th_tab) + gamma_tab, color="tab:blue", lw=2.0,
            label=r"$h(\theta_{\mathrm{tab}})$", zorder=4)
    # (iii) predicted
    gamma_psi = float(np.median(rv - model_on_obs(th_psi)))
    ax.plot(phase_grid, folded_model(th_psi) + gamma_psi, color="tab:red", lw=2.0,
            label=r"$h(\psi(y))$", zorder=4)

    ax.set_xlabel("orbital phase (folded at tabulated $P$)")
    ax.set_ylabel("RV (m/s)")
    ax.set_title(
        f"Held-out real: {info['host']}  "
        f"($P_{{\\mathrm{{tab}}}}$={P_fold:.1f} d, $e_{{\\mathrm{{tab}}}}$={th_tab[2]:.2f}; "
        r"light traces $\sim\mathrm{Unif}(\Gamma_{0.4})$)"
    )
    # Legend: add proxies for region / noisy
    ax.plot([], [], color="0.75", lw=1.2, label=r"$\theta\sim\mathrm{Unif}(\Gamma_{0.4})$")
    ax.scatter([], [], s=20, c="tab:orange", alpha=0.7, label="noisy sim. at $\\psi(y)$")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 -> {out_path}")


def figure2(checkpoint: Path, csv_path: Path, out_path: Path, device: torch.device) -> None:
    """Refresh MLP pred-vs-true: P/K/e panels only (74-D has no periapsis epoch)."""
    from regression import (
        TARGET_LABELS,
        DatasetBundle,
        build_model_from_checkpoint,
        _scatter_limits,
    )

    paper_targets = ("log10_P", "log10_K", "e")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model, norm_stats = build_model_from_checkpoint(ckpt, device)
    feature_set = str(norm_stats.get("feature_set", "74"))
    bundle = load_from_csv(csv_path, feature_set)
    rng = np.random.default_rng(42)
    n = len(bundle.X)
    idx = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = np.sort(idx[:n_val])
    X_val = bundle.X[val_idx]
    y_true = bundle.y[val_idx]
    y_pred = predict(model, X_val, norm_stats, device)
    val_bundle = DatasetBundle(
        X_val,
        y_true,
        row_idx=np.asarray(bundle.row_idx)[val_idx],
        e=np.asarray(bundle.e)[val_idx],
        has_t_peri=np.asarray(bundle.has_t_peri)[val_idx],
        has_ecc=np.asarray(bundle.has_ecc)[val_idx],
        df=bundle.df.iloc[val_idx].reset_index(drop=True),
    )
    metrics = {
        "per_target": _per_target_metrics(y_true, y_pred),
        "subsets": _subset_metrics(val_bundle, y_true, y_pred),
        "e_report": _e_subset_report(y_true, y_pred),
    }
    e_report = metrics["e_report"]

    # Option A: omit cos/sin ω panels — 74-D features lack periapsis-epoch
    # information, so ω is nearly unidentifiable (negative R²).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    name_to_j = {"log10_P": 0, "log10_K": 1, "e": 2}
    for ax, name in zip(axes, paper_targets):
        j = name_to_j[name]
        yt, yp = y_true[:, j], y_pred[:, j]
        ax.scatter(yt, yp, s=8, alpha=0.45, edgecolors="none")
        lo, hi = _scatter_limits(yt, yp)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if name == "e":
            r2 = metrics["per_target"][name]["r2"]
            r2_pos = e_report["e_gt_0"]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f} (e>0: {r2_pos:.3f})")
        else:
            r2 = metrics["per_target"][name]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.25)
    fig.suptitle(
        r"pred vs true (74-D MLP val; $\omega$ omitted — needs periapsis epoch)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 (P/K/e) -> {out_path}")
    # Keep a full 5-panel diagnostic copy for internal use.
    plot_pred_vs_true(y_true, y_pred, out_path.with_name("rv_pred_vs_true_full5.png"), metrics)


def _recon_mse(theta5: np.ndarray, curve: dict) -> tuple[float, float]:
    """Reconstruction error of a parameter set against the observed curve.

    Both t_peri and gamma are refit analytically inside KeplerDecoder, so a
    parameter set is never penalised for phase or systemic-velocity convention.
    That is what makes this a fair label-free comparison between the tabulated
    parameters and psi(y): neither side is scored against the other.

    Returns (mse_rv_std, chi2_red): mean squared residual in units of the
    curve's own RV scatter, and the sigma-weighted reduced chi-square.
    """
    decoder = KeplerDecoder().eval()
    t_norm = torch.from_numpy(curve["t_norm"]).unsqueeze(0)
    rv_obs = torch.from_numpy(curve["rv_obs"]).unsqueeze(0)
    mask = torch.from_numpy(curve["mask"]).unsqueeze(0)
    t_span = torch.tensor([curve["t_span"]], dtype=torch.float32)
    t_min = torch.tensor([curve["t_min"]], dtype=torch.float32)
    rv_std = torch.tensor([curve["rv_std"]], dtype=torch.float32)
    th = torch.as_tensor(np.asarray(theta5, dtype=float)[None, :], dtype=torch.float32)
    with torch.no_grad():
        rv_pred = decoder(th, t_norm, t_span, t_min, rv_obs, rv_std, mask)[0].numpy()
    m = curve["mask"] > 0.5
    resid = curve["rv_obs"][m] - rv_pred[m]          # rv_std units
    sig = np.maximum(curve["sig"][m], 1e-6)          # rv_std units
    return float(np.mean(resid ** 2)), float(np.mean((resid / sig) ** 2))


def figure_mse_scatter(
    psi_predict,
    feature_cols: list[str],
    out_path: Path,
    *,
    real_split: str = "test",
    sigma_min: float = 0.1,
    sigma_max: float = 100.0,
    surrogate: bool = True,
    gd_steps: int = 200,
    gd_lr: float = 0.02,
) -> dict:
    """Per-planet reconstruction MSE: tabulated (x) vs our prediction (y).

    Nicolo's 2026-07-25 point: scoring both our prediction and the tabulated
    parameters against GD-inferred labels that were themselves initialised at
    the tabulated values is circular. This figure removes labels from the
    comparison entirely — every parameter set is judged only by how well it
    reconstructs the observed RV curve. Points below the diagonal are planets
    where we explain the data better than the catalog does.

    The GD surrogate theta* is overlaid as the achievable floor: it is the
    argmin of the same objective, so no parameter set can sit below it.
    """
    systems = make_real(real_split, sigma_min, sigma_max)
    if not systems:
        raise RuntimeError(f"no real systems in split {real_split!r}")

    X = np.asarray([_feat_row_for_system(s, feature_cols) for s in systems], dtype=float)
    th_psi = psi_predict(X)

    rows = []
    for i, s in enumerate(systems):
        mse_tab, chi_tab = _recon_mse(s["theta5"], s["curve"])
        mse_psi, chi_psi = _recon_mse(th_psi[i], s["curve"])
        rows.append({
            "host": s.get("host", ""),
            "mse_tab": mse_tab, "mse_psi": mse_psi,
            "chi2_tab": chi_tab, "chi2_psi": chi_psi,
        })

    if surrogate:
        from conformal_shift import surrogate_fit_gd

        decoder = KeplerDecoder().eval()
        stars = surrogate_fit_gd(decoder, [s["theta5"] for s in systems], systems,
                                 gd_steps, gd_lr)
        for i, s in enumerate(systems):
            rows[i]["mse_star"], rows[i]["chi2_star"] = _recon_mse(stars[i], s["curve"])

    df = pd.DataFrame(rows)
    x = df["mse_tab"].to_numpy()
    y = df["mse_psi"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    frac_better = float(np.mean(y[ok] <= x[ok])) if ok.any() else float("nan")

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lo = float(np.nanmin(np.concatenate([x[ok], y[ok]]))) * 0.6
    hi = float(np.nanmax(np.concatenate([x[ok], y[ok]]))) * 1.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1, label="equal fit")
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color="tab:green", alpha=0.06, zorder=0)
    if surrogate and "mse_star" in df:
        ax.scatter(x[ok], df["mse_star"].to_numpy()[ok], s=26, marker="^",
                   color="0.55", alpha=0.8, edgecolors="none", zorder=2,
                   label=r"$\theta^{*}$ (GD floor)")
    ax.scatter(x[ok], y[ok], s=44, color="tab:blue", edgecolor="k", linewidth=0.4,
               zorder=3, label=r"$\psi(y)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"reconstruction MSE, tabulated $\theta_{\mathrm{tab}}$  [$\mathrm{rv\_std}^2$]")
    ax.set_ylabel(r"reconstruction MSE, predicted $\psi(y)$  [$\mathrm{rv\_std}^2$]")
    ax.set_title(f"Held-out {real_split}: who explains the data better?\n"
                 f"{frac_better:.0%} of planets on or below the diagonal "
                 f"(n={int(ok.sum())})")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(out_path.with_suffix(".csv"), index=False)
    summary = {
        "n": int(ok.sum()),
        "frac_psi_at_or_below_tab": frac_better,
        "median_mse_tab": float(np.median(x[ok])),
        "median_mse_psi": float(np.median(y[ok])),
        "median_chi2_tab": float(np.median(df["chi2_tab"])),
        "median_chi2_psi": float(np.median(df["chi2_psi"])),
    }
    if surrogate and "mse_star" in df:
        summary["median_mse_star"] = float(np.median(df["mse_star"]))
    print(f"MSE scatter -> {out_path}  ({summary})")
    return summary


# Parameter pairs for the 2-D prediction-region figure (Nicolo's ask: K/T, T/e, omega/K).
BOX_PAIRS = [("P", "K"), ("P", "e"), ("K", "omega")]

_BOX_AXES = {
    # key: (csv predicted col, csv tabulated col, label, log-scale?)
    "P": ("P_pred_d", "P_tab_d", r"$P$ [d]", True),
    "K": ("K_pred_ms", "K_tab_ms", r"$K$ [m/s]", True),
    "e": ("e_pred", "e_tab", r"$e$", False),
    "omega": ("omega_pred_rad", "omega_tab_rad", r"$\omega$ [rad]", False),
}


def _cp_bounds(row: pd.Series, key: str) -> tuple[float, float]:
    """Physical CP interval for one coordinate, from the alpha=0.1 half-widths.

    P and K are symmetric in log10, so the physical region is asymmetric; the
    CSV carries the precomputed bounds. e and omega are linear.
    """
    if key == "P":
        return float(row["P_cp_lo_d"]), float(row["P_cp_hi_d"])
    if key == "K":
        return float(row["K_cp_lo_ms"]), float(row["K_cp_hi_ms"])
    if key == "e":
        hw = float(row["halfwidth_e_a01"])
        return float(row["e_pred"]) - hw, float(row["e_pred"]) + hw
    hw = float(row["halfwidth_omega_a01"])
    return float(row["omega_pred_rad"]) - hw, float(row["omega_pred_rad"]) + hw


def _bayes_bounds(row: pd.Series, key: str, sigma_scale: float) -> tuple[float, float]:
    """Catalog interval = tabulated value +/- sigma_scale * published 1-sigma."""
    col = {"P": "P_tab_err_d", "K": "K_tab_err_ms", "e": "e_tab_err"}.get(key)
    centre = float(row[_BOX_AXES[key][1]])
    if col is None or col not in row or not np.isfinite(row.get(col, np.nan)):
        return float("nan"), float("nan")
    half = sigma_scale * float(row[col])
    return centre - half, centre + half


def _to_relative(key: str, value: float, centre: float) -> float:
    """Express a coordinate relative to that planet's tabulated value.

    P and K become log10(x / x_tab) (dex), e and omega become x - x_tab. This
    puts every planet in a common frame centred on the catalog value, so the
    two region sizes can be compared on one axis despite spanning orders of
    magnitude in absolute terms.
    """
    if key in ("P", "K"):
        if value <= 0 or centre <= 0:
            return float("nan")
        return math.log10(value / centre)
    if key == "omega":
        return float((value - centre + math.pi) % (2 * math.pi) - math.pi)
    return float(value - centre)


def figure_region_boxes(
    cp_csv: Path,
    out_path: Path,
    *,
    sigma_scale: float = 1.6449,
    pairs: list[tuple[str, str]] | None = None,
    space: str = "relative",
) -> None:
    """2-D CP boxes vs catalog (Bayesian) boxes for the Earth-like sample.

    One panel per parameter pair. For each planet the conformal region is the
    rectangle spanned by its two alpha=0.1 half-widths; the catalog region is
    the rectangle spanned by the published 1-sigma uncertainties scaled to the
    same nominal level. Planets are colour-coded and drawn in physical units,
    with log axes for P and K because the two region sizes differ by orders of
    magnitude — which is itself the result.
    """
    pairs = pairs or BOX_PAIRS
    if space not in ("relative", "physical"):
        raise ValueError(f"unknown space {space!r}")
    df = pd.read_csv(cp_csv, float_precision="round_trip")
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(df))]
    rel = space == "relative"

    def _axis_label(key: str) -> str:
        if not rel:
            return _BOX_AXES[key][2]
        if key in ("P", "K"):
            return rf"$\log_{{10}}({key}/{key}_{{\mathrm{{tab}}}})$ [dex]"
        sym = r"\omega" if key == "omega" else key
        unit = " [rad]" if key == "omega" else ""
        return rf"${sym} - {sym}_{{\mathrm{{tab}}}}${unit}"

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (kx, ky) in zip(axes, pairs):
        for i, (_, row) in enumerate(df.iterrows()):
            c = colors[i]
            cx = float(row[_BOX_AXES[kx][1]])
            cy = float(row[_BOX_AXES[ky][1]])

            def conv(key, val, centre):
                return _to_relative(key, val, centre) if rel else val

            x_lo, x_hi = (conv(kx, v, cx) for v in _cp_bounds(row, kx))
            y_lo, y_hi = (conv(ky, v, cy) for v in _cp_bounds(row, ky))
            # Unfilled in relative space: ten overlapping conformal regions of
            # similar size become unreadable if filled.
            ax.add_patch(plt.Rectangle(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                facecolor="none" if rel else c, alpha=1.0 if rel else 0.10,
                edgecolor=c, lw=1.3, zorder=2))

            bx = _bayes_bounds(row, kx, sigma_scale)
            by = _bayes_bounds(row, ky, sigma_scale)
            if np.isfinite(bx[0]) and np.isfinite(by[0]):
                bx_lo, bx_hi = (conv(kx, v, cx) for v in bx)
                by_lo, by_hi = (conv(ky, v, cy) for v in by)
                ax.add_patch(plt.Rectangle(
                    (bx_lo, by_lo), max(bx_hi - bx_lo, 1e-12), max(by_hi - by_lo, 1e-12),
                    facecolor=c, alpha=0.9, edgecolor="k", lw=0.6, zorder=4))

            ax.scatter([conv(kx, cx, cx)], [conv(ky, cy, cy)],
                       marker="*", s=70, color=c, edgecolor="k", linewidth=0.4, zorder=5)
            ax.scatter([conv(kx, float(row[_BOX_AXES[kx][0]]), cx)],
                       [conv(ky, float(row[_BOX_AXES[ky][0]]), cy)],
                       marker="x", s=34, color=c, linewidth=1.4, zorder=5)

        if rel:
            ax.axhline(0.0, color="0.6", lw=0.7, ls=":", zorder=1)
            ax.axvline(0.0, color="0.6", lw=0.7, ls=":", zorder=1)
        else:
            if _BOX_AXES[kx][3]:
                ax.set_xscale("log")
            if _BOX_AXES[ky][3]:
                ax.set_yscale("log")
        ax.set_xlabel(_axis_label(kx))
        ax.set_ylabel(_axis_label(ky))
        ax.set_title(f"{_BOX_AXES[kx][2]} vs {_BOX_AXES[ky][2]}")
        ax.grid(alpha=0.22, which="both")
        ax.autoscale_view()

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=11, mfc="0.7", mec="0.3", alpha=0.35,
                   label=r"conformal region ($\alpha=0.1$)"),
        plt.Line2D([], [], marker="s", ls="", ms=7, mfc="0.35", mec="k",
                   label=f"catalog {sigma_scale:g}$\\sigma$ region"),
        plt.Line2D([], [], marker="*", ls="", ms=11, mfc="0.5", mec="k", label="tabulated"),
        plt.Line2D([], [], marker="x", ls="", ms=8, color="0.35", label=r"$\psi(y)$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    space_note = ("each planet centred on its own tabulated value"
                  if space == "relative" else "physical units, log axes")
    fig.suptitle("Conformal vs catalog prediction regions, Earth-like held-out "
                 f"planets\n({space_note})", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Region-box figure -> {out_path}")


def figure_trajectories(
    psi_predict,
    feature_cols: list[str],
    out_path: Path,
    *,
    real_split: str = "test",
    n_systems: int = 6,
    n_cycles: float = 10.0,
    sigma_min: float = 0.1,
    sigma_max: float = 100.0,
    q_box: dict | None = None,
    n_box_samples: int = 12,
) -> None:
    """RV trajectories in time — no projection onto a single period.

    Nicolo's 2026-07-25 preference: show the observations as they were taken,
    with the tabulated and predicted Keplerian curves overlaid, rather than
    phase-folding. Systems with the most observations are chosen so the
    sampling pattern is visible.
    """
    systems = make_real(real_split, sigma_min, sigma_max)
    # One panel per star: several .tbl files can share a host, and duplicate
    # panels of the same system waste the figure.
    by_host: dict[str, dict] = {}
    for s in systems:
        h = s.get("host") or ""
        if h not in by_host or int(s["curve"]["mask"].sum()) > int(by_host[h]["curve"]["mask"].sum()):
            by_host[h] = s

    def _best_window(s: dict) -> tuple[float, int, float]:
        """(window start, points inside, observed span) for the densest window."""
        t, _, _ = _obs_ms(s["curve"])
        P = float(10.0 ** s["theta5"][0])
        win = min(n_cycles * P, float(t.max() - t.min()))
        if win <= 0:
            return float(t.min()), len(t), float(t.max() - t.min())
        starts = np.linspace(t.min(), max(t.max() - win, t.min()), 200)
        counts = [int(((t >= a) & (t <= a + win)).sum()) for a in starts]
        t0 = float(starts[int(np.argmax(counts))])
        sel = (t >= t0) & (t <= t0 + win)
        return t0, int(sel.sum()), float(t[sel].max() - t[sel].min()) if sel.any() else 0.0

    # A panel is only informative if the window holds enough points AND they
    # actually spread across it — several hosts have all their observations
    # clumped into two nights, which yields a degenerate near-zero-span window.
    scored = []
    for s in by_host.values():
        P = float(10.0 ** s["theta5"][0])
        t, _, _ = _obs_ms(s["curve"])
        win = min(n_cycles * P, float(t.max() - t.min()))
        baseline = float(t.max() - t.min())
        _, n_in, span_in = _best_window(s)
        # >= 2 full orbits of baseline: transit-survey targets often have their
        # whole RV series inside a single night, which shows no orbital motion.
        if n_in >= 20 and win > 0 and span_in >= 0.3 * win and baseline >= 2.0 * P:
            scored.append((n_in, s))
    if not scored:  # fall back to raw point count rather than failing outright
        scored = [(int(s["curve"]["mask"].sum()), s) for s in by_host.values()]
    scored.sort(key=lambda z: -z[0])
    systems = [s for _, s in scored[:n_systems]]
    if not systems:
        raise RuntimeError(f"no real systems in split {real_split!r}")

    X = np.asarray([_feat_row_for_system(s, feature_cols) for s in systems], dtype=float)
    th_psi = psi_predict(X)

    ncol = 2
    nrow = int(np.ceil(len(systems) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 2.9 * nrow), squeeze=False)
    for k, s in enumerate(systems):
        ax = axes[k // ncol][k % ncol]
        t, rv, sig = _obs_ms(s["curve"])

        # Zoom to the densest window spanning ~n_cycles orbits. Plotting the full
        # baseline is unreadable when it holds hundreds of cycles (a 4 d planet
        # over 3000 d is a solid band) — this keeps the trajectory legible
        # without projecting onto a single period.
        P_tab = float(10.0 ** s["theta5"][0])
        win = min(n_cycles * P_tab, float(t.max() - t.min()))
        if win > 0 and win < (t.max() - t.min()):
            starts = np.linspace(t.min(), t.max() - win, 200)
            counts = [int(((t >= a) & (t <= a + win)).sum()) for a in starts]
            t0 = float(starts[int(np.argmax(counts))])
            sel = (t >= t0) & (t <= t0 + win)
            t, rv, sig = t[sel], rv[sel], sig[sel]

        span = max(t.max() - t.min(), 1e-6)
        t_grid = np.linspace(t.min() - 0.02 * span, t.max() + 0.02 * span, 2000)

        th_tab = s["theta5"]
        tp_tab = _anchor_t_peri(th_tab, t, rv)
        tp_psi = _anchor_t_peri(th_psi[k], t, rv)
        y_tab = _kepler_on_grid(th_tab, t_grid, tp_tab)
        y_psi = _kepler_on_grid(th_psi[k], t_grid, tp_psi)
        y_tab += float(np.median(rv - _kepler_on_grid(th_tab, t, tp_tab)))
        y_psi += float(np.median(rv - _kepler_on_grid(th_psi[k], t, tp_psi)))

        # Nicolo 2026-07-26: overlay a random sample of parameter vectors drawn
        # from the CP box, so the figure shows the *region* rather than a point.
        # Drawn first and faint so the two headline curves stay legible.
        if q_box:
            rng_box = np.random.default_rng(1234 + k)
            for _ in range(n_box_samples):
                th_s = np.array(th_psi[k], dtype=float).copy()
                for ci, cname in enumerate(("log10_P", "log10_K", "e")):
                    hw = float(q_box.get(cname, 0.0))
                    if hw > 0:
                        th_s[ci] += rng_box.uniform(-hw, hw)
                th_s[2] = float(np.clip(th_s[2], 0.0, 0.99))
                tp_s = _anchor_t_peri(th_s, t, rv)
                y_s = _kepler_on_grid(th_s, t_grid, tp_s)
                y_s += float(np.median(rv - _kepler_on_grid(th_s, t, tp_s)))
                ax.plot(t_grid, y_s, color="tab:red", lw=0.6, alpha=0.16, zorder=1)

        # MSE against the observations, so the legend carries the comparison
        # Nicolo asked for rather than requiring a cross-reference to the scatter.
        mse_tab = float(np.mean((rv - _kepler_on_grid(th_tab, t, tp_tab)
                                 - float(np.median(rv - _kepler_on_grid(th_tab, t, tp_tab)))) ** 2))
        mse_psi = float(np.mean((rv - _kepler_on_grid(th_psi[k], t, tp_psi)
                                 - float(np.median(rv - _kepler_on_grid(th_psi[k], t, tp_psi)))) ** 2))
        ax.plot(t_grid, y_tab, color="tab:blue", lw=1.2, alpha=0.9,
                label=rf"$h(\theta_{{\mathrm{{tab}}}})$  MSE={mse_tab:.3g}")
        ax.plot(t_grid, y_psi, color="tab:red", lw=1.2, alpha=0.9,
                label=rf"$h(\psi(y))$  MSE={mse_psi:.3g}")
        ax.errorbar(t, rv, yerr=sig, fmt="o", ms=3.0, color="k", ecolor="0.6",
                    elinewidth=0.6, capsize=0, zorder=5)
        host = s.get("host") or f"system {k}"
        ax.set_title(f"{host}  ($P_{{\\mathrm{{tab}}}}$={10 ** th_tab[0]:.1f} d, "
                     f"{len(t)} obs in a {span:.0f} d window)", fontsize=9)
        ax.set_xlabel("BJD [d]", fontsize=8)
        ax.set_ylabel("RV [m/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22)
        if k == 0:
            ax.legend(loc="best", fontsize=7, frameon=False)

    for k in range(len(systems), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Held-out {real_split} RV trajectories (unfolded)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory figure -> {out_path}")


def earth_likeness(row: pd.Series) -> float:
    """Lower is more Earth-like (P~365 d, low e, mass~1 Mearth when known)."""
    P = float(row["pl_orbper"]) if pd.notna(row.get("pl_orbper")) else np.nan
    e = float(row["pl_orbeccen"]) if pd.notna(row.get("pl_orbeccen")) else 0.0
    mass_j = row.get("pl_bmassj")
    if pd.isna(mass_j):
        mass_j = row.get("pl_msinij")
    mearth = float(mass_j) * 317.8 if pd.notna(mass_j) else np.nan
    score = 0.0
    if np.isfinite(P) and P > 0:
        score += abs(math.log10(P / 365.25))
    else:
        score += 5.0
    score += abs(e)
    if np.isfinite(mearth) and mearth > 0:
        score += abs(math.log10(mearth / 1.0))
    else:
        score += 2.0
    return float(score)


def _mean_abs_err(err1, err2) -> float | None:
    """Mean of |err1|, |err2| from NASA archive asymmetric uncertainties."""
    vals = []
    for v in (err1, err2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            vals.append(abs(fv))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _log10_hw_to_physical(center: float, hw_log10: float) -> tuple[float, float]:
    """Physical bounds of a log10-symmetric half-width around ``center``.

    The conformal region is symmetric in log10, so in physical units it is the
    asymmetric interval [center*10^-hw, center*10^+hw] — it has no single
    half-width, and the lower bound stays positive by construction.
    """
    factor = 10.0 ** float(hw_log10)
    return float(center / factor), float(center * factor)


def earthlike_table(
    psi_predict,
    feature_cols: list[str],
    q01: dict[str, float],
    out_csv: Path,
    out_tex: Path,
    top_k: int = 10,
    widths_blob: dict | None = None,
) -> None:
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    # Restrict to single-planet hosts in our RV corpus with usable Kepler params.
    sp = splits
    if "n_planets" in sp.columns:
        sp = sp[sp["n_planets"] == 1]
    hosts = set(sp["host"].astype(str))
    need = ["pl_orbper", "pl_rvamp", "pl_orbeccen", "pl_orblper"]
    lab = labels[labels["hostname"].astype(str).isin(hosts)].copy()
    for c in need:
        lab = lab[lab[c].notna()]
    lab = lab[lab["pl_rvamp"] > 0]
    lab["earth_score"] = lab.apply(earth_likeness, axis=1)
    lab = lab.sort_values("earth_score")

    systems_by_host: dict[str, dict] = {}
    # Restrict to held-out hosts (val/test) so the table is not train-set leakage.
    for split in ("val", "test"):
        ds = RVDataset(split, normalize=False, single_planet=True)
        for i in range(len(ds)):
            x, lsp, theta, info = ds.get_numpy(i)
            if not info.get("valid", True):
                continue
            from conformal import _masked_observations, _curve_from_x
            from eval_omega_nn_vs_rf import _summary_row

            xm = _masked_observations(x)
            if xm.shape[1] < 10:
                continue
            med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
            if not (0.1 <= med_sigma <= 100.0):
                continue
            feats = _summary_row(xm, info, lsp)
            systems_by_host[info["host"]] = {
                "curve": _curve_from_x(x, info),
                "feat_row": feats,
                "lsp": np.asarray(lsp, dtype=float),
                "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
                "info": info,
                "split": split,
            }
    rows = []
    for _, lab_row in lab.iterrows():
        host = str(lab_row["hostname"])
        if host not in systems_by_host:
            continue
        s = systems_by_host[host]
        X = _feat_row_for_system(s, feature_cols)[None, :]
        pred = psi_predict(X)[0]
        tab = s["theta5"]
        hw = _match_system_widths(pred, widths_blob, fallback=q01, alpha="0.10")
        source = "papernorm_per_system" if hw is not None else "global_raw"
        if hw is None:
            hw = q01

        P_tab = float(10 ** tab[0])
        K_tab = float(10 ** tab[1])
        e_tab = float(tab[2])
        P_pred = float(10 ** pred[0])
        K_pred = float(10 ** pred[1])
        e_pred = float(pred[2])

        P_tab_err = _mean_abs_err(lab_row.get("pl_orbpererr1"), lab_row.get("pl_orbpererr2"))
        K_tab_err = _mean_abs_err(lab_row.get("pl_rvamperr1"), lab_row.get("pl_rvamperr2"))
        e_tab_err = _mean_abs_err(lab_row.get("pl_orbeccenerr1"), lab_row.get("pl_orbeccenerr2"))
        # Asymmetric NASA errs (err2 typically negative lower, err1 positive upper).
        P_tab_err_lo = lab_row.get("pl_orbpererr2")
        P_tab_err_hi = lab_row.get("pl_orbpererr1")
        K_tab_err_lo = lab_row.get("pl_rvamperr2")
        K_tab_err_hi = lab_row.get("pl_rvamperr1")
        e_tab_err_lo = lab_row.get("pl_orbeccenerr2")
        e_tab_err_hi = lab_row.get("pl_orbeccenerr1")

        P_cp_lo, P_cp_hi = _log10_hw_to_physical(P_pred, hw["log10_P"])
        K_cp_lo, K_cp_hi = _log10_hw_to_physical(K_pred, hw["log10_K"])

        rows.append({
            "host": host,
            "pl_name": lab_row.get("pl_name", ""),
            "split": s["split"],
            "earth_score": float(lab_row["earth_score"]),
            "P_tab_d": P_tab,
            "P_tab_err_d": P_tab_err,
            "P_tab_err_lo_d": float(P_tab_err_lo) if pd.notna(P_tab_err_lo) else None,
            "P_tab_err_hi_d": float(P_tab_err_hi) if pd.notna(P_tab_err_hi) else None,
            "K_tab_ms": K_tab,
            "K_tab_err_ms": K_tab_err,
            "K_tab_err_lo_ms": float(K_tab_err_lo) if pd.notna(K_tab_err_lo) else None,
            "K_tab_err_hi_ms": float(K_tab_err_hi) if pd.notna(K_tab_err_hi) else None,
            "e_tab": e_tab,
            "e_tab_err": e_tab_err,
            "e_tab_err_lo": float(e_tab_err_lo) if pd.notna(e_tab_err_lo) else None,
            "e_tab_err_hi": float(e_tab_err_hi) if pd.notna(e_tab_err_hi) else None,
            "omega_tab_rad": float(_theta_to_omega(tab)),
            "P_pred_d": P_pred,
            "K_pred_ms": K_pred,
            "e_pred": e_pred,
            "omega_pred_rad": float(_theta_to_omega(pred)),
            "halfwidth_log10_P_a01": float(hw["log10_P"]),
            "halfwidth_log10_K_a01": float(hw["log10_K"]),
            "halfwidth_e_a01": float(hw["e"]),
            "halfwidth_omega_a01": float(hw["omega"]),
            "P_cp_lo_d": P_cp_lo,
            "P_cp_hi_d": P_cp_hi,
            "K_cp_lo_ms": K_cp_lo,
            "K_cp_hi_ms": K_cp_hi,
            "widths_source": source,
        })
        if len(rows) >= top_k:
            break

    if not rows:
        raise RuntimeError("no Earth-like systems matched the RV corpus (val/test)")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_tex.write_text(render_earthlike_tex(rows))
    print(f"Earth-like table -> {out_csv} , {out_tex}")


def _fmt_num(x: float, digits: int = 2) -> str:
    """Plain fixed-point; everything emitted here must be text-mode safe."""
    return f"{x:.{digits}f}"


def _fmt_pm(val: float, err: float | None, kind: str = "P") -> str:
    """Catalog value as ``val ± err``, matching the err's precision."""
    if err is None or not math.isfinite(err):
        return _fmt_num(val)
    if kind == "P" and err < 0.01:
        return f"{val:.4f} $\\pm$ {err:.4f}"
    if kind == "e" and err < 0.005:
        return f"{val:.3f} $\\pm$ {err:.3f}"
    return f"{_fmt_num(val)} $\\pm$ {_fmt_num(err)}"


def _fmt_dex(val: float, hw_log10: float) -> str:
    """Prediction with its log10-symmetric CP half-width, stated in dex."""
    return f"{_fmt_num(val)} $\\pm$ {hw_log10:.2f} dex"


def render_earthlike_tex(rows: list[dict]) -> str:
    """LaTeX for the Earth-like table: tab +- NASA catalog err, pred +- CP half-width.

    Pure function of the rows written to ``earthlike_top10.csv``, so the table can
    be re-rendered from that CSV without reloading the (gitignored) MLP checkpoint.
    """
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Host & split & $P_{\mathrm{tab}}$ & $P_{\mathrm{pred}}$ "
        r"& $K_{\mathrm{tab}}$ & $K_{\mathrm{pred}}$ "
        r"& $e_{\mathrm{tab}}$ & $e_{\mathrm{pred}}$ \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['host']} & {r['split']} "
            f"& {_fmt_pm(r['P_tab_d'], r['P_tab_err_d'], 'P')} "
            f"& {_fmt_dex(r['P_pred_d'], r['halfwidth_log10_P_a01'])} "
            f"& {_fmt_pm(r['K_tab_ms'], r['K_tab_err_ms'], 'K')} "
            f"& {_fmt_dex(r['K_pred_ms'], r['halfwidth_log10_K_a01'])} "
            f"& {_fmt_pm(r['e_tab'], r['e_tab_err'], 'e')} "
            f"& {_fmt_pm(r['e_pred'], r['halfwidth_e_a01'], 'e')} \\\\"
        )
    omega_hw = float(np.median([r["halfwidth_omega_a01"] for r in rows]))
    lines += [
        r"\hline",
        r"\multicolumn{8}{l}{\footnotesize Tabulated $\pm$: NASA archive published "
        r"uncertainties ($\sim$1$\sigma$). Predicted $\pm$: per-system papernorm "
        r"half-widths of the $\alpha{=}0.1$ conformal region; for $P$ and $K$ these are "
        r"symmetric in $\log_{10}$ (dex), i.e. the physical region is the asymmetric "
        r"interval $[\hat{x}10^{-\mathrm{hw}},\,\hat{x}10^{+\mathrm{hw}}]$ "
        r"(tabulated in the accompanying CSV). "
        f"$\\omega$ CP half-width $\\approx{omega_hw:.2f}$~rad (near-vacuous, omitted). "
        r"Held-out (val/test) hosts only.} \\",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--quantiles", type=Path, default=DEFAULT_Q)
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument(
        "--widths",
        type=Path,
        default=ROOT / "synthetic_generation" / "regression" / "mlp_psi" / "per_system_widths_papernorm.json",
        help="optional per-system papernorm widths JSON from conformal_shift",
    )
    ap.add_argument("--host", default=None, help="held-out host for Figure 1 (default: auto)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-split", default="test", choices=("all", "train", "val", "test"),
                   help="held-out split for the MSE-scatter and trajectory figures")
    ap.add_argument("--sigma-scale", type=float, default=1.6449,
                   help="catalog 1-sigma -> interval scale in the region-box figure "
                        "(default 1.6449 = two-sided 90%%, matching alpha=0.1)")
    ap.add_argument("--n-trajectories", type=int, default=6)
    ap.add_argument("--traj-cycles", type=float, default=10.0,
                   help="orbits shown per trajectory panel; the densest window of "
                        "this length is chosen (full baseline if shorter)")
    ap.add_argument("--n-box-samples", type=int, default=12,
                   help="parameter vectors drawn from the CP box and overlaid on "
                        "each trajectory panel (Nicolo 2026-07-26)")
    ap.add_argument("--no-box-samples", action="store_true",
                   help="draw only h(theta_tab) and h(psi(y)), no box samples")
    ap.add_argument("--no-surrogate-floor", action="store_true",
                   help="skip the GD theta* floor in the MSE scatter (much faster)")
    ap.add_argument(
        "--only",
        choices=("all", "phasefold", "predtrue", "table", "mse", "boxes", "trajectories"),
        default="all",
        help="regenerate a single figure (boxes needs no checkpoint)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The region-box figure is a pure function of committed artifacts (the CP csv
    # + catalog sigmas), so it runs without the psi checkpoint — which matters
    # because that checkpoint is not in the repo (see README, issue #10).
    if args.only == "boxes":
        for sp in ("relative", "physical"):
            suffix = "" if sp == "relative" else "_physical"
            figure_region_boxes(OUT_DIR / "earthlike_top10.csv",
                                OUT_DIR / f"rv_region_boxes{suffix}.png",
                                sigma_scale=args.sigma_scale, space=sp)
        return

    q_blob = load_quantiles(args.quantiles, args.metrics)
    q01 = q_blob["quantiles"]["0.10"]
    q04 = q_blob["quantiles"]["0.40"]
    widths_blob = None
    if args.widths.exists():
        widths_blob = json.loads(args.widths.read_text())
        print(f"loaded per-system widths ({widths_blob.get('n_systems')} systems) from {args.widths}")

    psi_predict, norm_stats = _load_mlp_psi(args.checkpoint, device)
    df = pd.read_csv(args.csv, nrows=1)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    in_dim = int(norm_stats["in_dim"])
    if len(feature_cols) != in_dim:
        raise ValueError(f"csv features {len(feature_cols)} != MLP in_dim {in_dim}")

    want = args.only

    if want in ("all", "phasefold"):
        system, info = pick_system(args.host)
        figure1(system, info, psi_predict, feature_cols, q04,
                OUT_DIR / "rv_heldout_phasefold.png",
                seed=args.seed, widths_blob=widths_blob)
    if want in ("all", "predtrue"):
        figure2(args.checkpoint, args.csv, OUT_DIR / "rv_pred_vs_true.png", device)
    if want in ("all", "mse"):
        figure_mse_scatter(psi_predict, feature_cols, OUT_DIR / "rv_mse_scatter.png",
                           real_split=args.real_split,
                           surrogate=not args.no_surrogate_floor)
    if want in ("all", "trajectories"):
        figure_trajectories(psi_predict, feature_cols, OUT_DIR / "rv_trajectories.png",
                            real_split=args.real_split, n_systems=args.n_trajectories,
                            n_cycles=args.traj_cycles,
                            q_box=None if args.no_box_samples else q01,
                            n_box_samples=args.n_box_samples)
    if want in ("all", "table"):
        earthlike_table(psi_predict, feature_cols, q01,
                        OUT_DIR / "earthlike_top10.csv",
                        OUT_DIR / "earthlike_top10.tex",
                        widths_blob=widths_blob)
    if want == "all":
        for sp in ("relative", "physical"):
            suffix = "" if sp == "relative" else "_physical"
            figure_region_boxes(OUT_DIR / "earthlike_top10.csv",
                                OUT_DIR / f"rv_region_boxes{suffix}.png",
                                sigma_scale=args.sigma_scale, space=sp)


if __name__ == "__main__":
    main()
