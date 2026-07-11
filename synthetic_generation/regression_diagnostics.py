"""
regression_diagnostics.py — automated diagnostics for the synthetic regression MLP.

Quantifies physics-limited vs fixable errors (SNR, P/baseline, LSP baseline,
e prior banding, raw output saturation, sanity checks) before any h/k refactor.

Usage
-----
    python synthetic_generation/regression_diagnostics.py \\
        --csv synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv \\
        --checkpoint checkpoints/regression_mlp_109.pt \\
        --feature-set 109

Or via regression.py:  python regression.py --diagnose --feature-set 109 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import THETA_NAMES  # noqa: E402
from regression import (  # noqa: E402
    DatasetBundle,
    RegressionHead,
    TARGET_LABELS,
    _omega_mae_deg,
    _per_target_metrics,
    _r2,
    _subset_metrics,
    _val_split_indices,
    load_from_csv,
    predict,
)
from synthetic_dataset import _load_eccentricity_prior  # noqa: E402
from theta_loss import apply_theta_constraints  # noqa: E402


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    raise TypeError(type(o))


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"saved -> {path}")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[RegressionHead, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_stats = ckpt["norm_stats"]
    in_dim = int(norm_stats["in_dim"])
    model = RegressionHead(in_dim=in_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, norm_stats


def _metadata_for_indices(bundle: DatasetBundle, indices: np.ndarray) -> pd.DataFrame:
    rows = bundle.row_idx[indices]
    return bundle.df.iloc[rows].reset_index(drop=True)


def _compute_snr(meta: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    k_ms = 10.0 ** y[:, 1]
    sigma = meta["median_sigma_ms"].to_numpy(dtype=float)
    sigma = np.where(sigma > 0, sigma, np.nan)
    return k_ms / sigma


def _omega_mae_deg_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return _omega_mae_deg(y_true[mask], y_pred[mask])


def plot_e_prior_train_hist(train_e: np.ndarray, out_dir: Path) -> dict:
    """Training-set e histogram with empirical prior bin edges overlaid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = _load_eccentricity_prior()

    fig, ax = plt.subplots(figsize=(7, 4))
    counts, edges, _ = ax.hist(train_e, bins=50, range=(0, 1), alpha=0.7, color="C0", label="train split")

    prior_info: dict[str, Any] = {
        "n_train": int(len(train_e)),
        "frac_e_zero": float(np.mean(train_e == 0)),
        "frac_e_lt_0.05": float(np.mean(train_e < 0.05)),
        "frac_e_gt_0.1": float(np.mean(train_e > 0.1)),
    }

    if prior is not None:
        p_zero = float(prior.get("p_zero", 0))
        prior_info["prior_p_zero"] = p_zero
        prior_info["prior_n_bins"] = int(len(prior.get("probs", [])))
        ax.axvline(0, color="C3", ls="--", lw=1.2, alpha=0.8, label=f"prior mass at e=0 ({p_zero:.2f})")
        left = np.asarray(prior.get("left_edges", []), dtype=float)
        right = np.asarray(prior.get("right_edges", []), dtype=float)
        for lo, hi in zip(left, right):
            ax.axvspan(lo, hi, color="C3", alpha=0.06)
        if len(left):
            prior_info["prior_bin_edges"] = np.concatenate([[0.0], left, [right[-1]]]).tolist()

    ax.set_xlabel("eccentricity e")
    ax.set_ylabel("count")
    ax.set_title("Training-set e distribution vs empirical prior bins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "e_prior_train_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")
    return prior_info


def _scatter_colored(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    color: np.ndarray,
    name: str,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    j = THETA_NAMES.index(name)
    yt, yp = y_true[:, j], y_pred[:, j]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(yt, yp, c=color, s=10, alpha=0.55, cmap="viridis", edgecolors="none")
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel(f"true {TARGET_LABELS.get(name, name)}")
    ax.set_ylabel(f"pred {TARGET_LABELS.get(name, name)}")
    ax.set_title(f"{TARGET_LABELS.get(name, name)}  R²={_r2(yt, yp):.3f}{title_suffix}")
    fig.colorbar(sc, ax=ax, label="SNR (K/σ)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_tertiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    name: str,
    out_path: Path,
) -> None:
    j = THETA_NAMES.index(name)
    valid = np.isfinite(snr)
    qs = np.nanquantile(snr[valid], [1 / 3, 2 / 3])
    bands = [
        ("low SNR", snr <= qs[0]),
        ("mid SNR", (snr > qs[0]) & (snr <= qs[1])),
        ("high SNR", snr > qs[1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, (label, mask) in zip(axes, bands):
        m = mask & valid
        if not m.any():
            ax.set_title(f"{label}\n(n=0)")
            continue
        yt, yp = y_true[m, j], y_pred[m, j]
        ax.scatter(yt, yp, s=8, alpha=0.5, edgecolors="none")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.6)
        ax.set_title(f"{label}\nR²={_r2(yt, yp):.3f}  n={m.sum()}")
        ax.set_xlabel(f"true {TARGET_LABELS.get(name, name)}")
    axes[0].set_ylabel(f"pred {TARGET_LABELS.get(name, name)}")
    fig.suptitle(f"{TARGET_LABELS.get(name, name)} by SNR tertile", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_by_snr_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    *,
    n_bins: int = 5,
    e_mask: np.ndarray | None = None,
) -> dict:
    valid = np.isfinite(snr)
    if e_mask is not None:
        valid &= e_mask
    snr_v = snr[valid]
    if len(snr_v) < n_bins:
        return {"bins": [], "note": "insufficient SNR samples"}

    edges = np.quantile(snr_v, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    bins_out: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = valid & (snr >= lo) & (snr <= hi if i == n_bins - 1 else snr < hi)
        n = int(mask.sum())
        entry: dict[str, Any] = {
            "bin": i,
            "snr_lo": float(lo),
            "snr_hi": float(hi),
            "n": n,
        }
        if n >= 2:
            entry["per_target"] = _per_target_metrics(y_true[mask], y_pred[mask])
        bins_out.append(entry)
    return {"bins": bins_out, "n_bins": n_bins}


def plot_snr_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"global": _per_target_metrics(y_true, y_pred)}

    for name in THETA_NAMES:
        _scatter_colored(
            y_true, y_pred, snr, name,
            out_dir / f"pred_vs_true_{name}_by_snr.png",
        )
        _scatter_tertiles(
            y_true, y_pred, snr, name,
            out_dir / f"pred_vs_true_{name}_by_snr_tertiles.png",
        )

    quintiles = metrics_by_snr_bins(y_true, y_pred, snr, n_bins=5)
    report["by_snr_quintile"] = quintiles

    e_mask = y_true[:, 2] > 0.1
    report["omega_mae_e_gt_0.1_global"] = _omega_mae_deg_masked(y_true, y_pred, e_mask)
    report["by_snr_quintile_e_gt_0.1"] = metrics_by_snr_bins(
        y_true, y_pred, snr, n_bins=5, e_mask=e_mask,
    )

    # R² vs SNR bin bar chart for log10_P
    fig, ax = plt.subplots(figsize=(6, 3.5))
    r2s = []
    labels = []
    for b in quintiles.get("bins", []):
        if b.get("n", 0) >= 2 and "per_target" in b:
            r2s.append(b["per_target"]["log10_P"]["r2"])
            labels.append(f"{b['snr_lo']:.1f}–{b['snr_hi']:.1f}")
    if r2s:
        ax.bar(range(len(r2s)), r2s, color="C0", alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(r"$R^2$")
        ax.set_xlabel("SNR (K/σ) quintile")
        ax.set_title(r"$\log_{10} P$ $R^2$ by SNR quintile")
        ax.axhline(_r2(y_true[:, 0], y_pred[:, 0]), color="k", ls="--", lw=0.8, label="global")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "r2_log10_P_by_snr_quintile.png", dpi=150)
    plt.close(fig)

    _write_json(out_dir / "metrics_by_snr.json", report)
    return report


def plot_p_baseline_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_true = 10.0 ** y_true[:, 0]
    p_pred = 10.0 ** y_pred[:, 0]
    baseline = meta["baseline_d"].to_numpy(dtype=float)
    median_gap = meta["median_gap_d"].to_numpy(dtype=float)
    p_over_baseline = p_true / np.maximum(baseline, 1e-6)
    resid = y_pred[:, 0] - y_true[:, 0]

    report: dict[str, Any] = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m1 = np.isfinite(p_over_baseline) & np.isfinite(resid)
    axes[0].scatter(p_over_baseline[m1], resid[m1], s=8, alpha=0.45, edgecolors="none")
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$P$ / baseline")
    axes[0].set_ylabel(r"$\log_{10} P$ residual (pred − true)")
    axes[0].set_title("Period error vs P/baseline")

    m2 = np.isfinite(median_gap) & np.isfinite(resid) & (median_gap > 0)
    axes[1].scatter(median_gap[m2], resid[m2], s=8, alpha=0.45, edgecolors="none")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("median gap (days)")
    axes[1].set_ylabel(r"$\log_{10} P$ residual")
    axes[1].set_title("Period error vs cadence gap")
    fig.tight_layout()
    fig.savefig(out_dir / "log10_P_residual_vs_identifiability.png", dpi=150)
    plt.close(fig)

    masks = {
        "all": np.ones(len(y_true), dtype=bool),
        "p_over_baseline_le_1": p_over_baseline <= 1.0,
        "p_over_baseline_gt_1": p_over_baseline > 1.0,
        "p_over_baseline_gt_2": p_over_baseline > 2.0,
        "p_lt_2x_median_gap": p_true < 2.0 * median_gap,
        "identifiable": (p_over_baseline <= 1.0) & (p_true >= 2.0 * median_gap),
    }
    for name, mask in masks.items():
        n = int(mask.sum())
        entry = {"n": n, "excluded_n": int((~mask).sum())}
        if n >= 2:
            entry["log10_P"] = {
                "r2": _r2(y_true[mask, 0], y_pred[mask, 0]),
                "mae_log10": float(np.mean(np.abs(resid[mask]))),
            }
        report[name] = entry

    # Spearman: |residual| vs P/baseline
    if m1.sum() >= 5:
        rho, pval = stats.spearmanr(p_over_baseline[m1], np.abs(resid[m1]))
        report["spearman_abs_residual_vs_p_over_baseline"] = {"rho": float(rho), "p": float(pval)}

    _write_json(out_dir / "p_baseline_metrics.json", report)
    return report


def period_recovery_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_true = 10.0 ** y_true[:, 0]
    p_mlp = 10.0 ** y_pred[:, 0]
    p_lsp = meta["lsp_peak_period_d"].to_numpy(dtype=float)

    valid = np.isfinite(p_lsp) & (p_lsp > 0) & np.isfinite(p_true) & (p_true > 0)
    dlog_mlp = np.log10(p_mlp / p_true)
    dlog_lsp = np.log10(p_lsp / p_true)
    dlog_mlp_v = dlog_mlp[valid]
    dlog_lsp_v = dlog_lsp[valid]

    def _frac_within(dlog: np.ndarray, pct: float) -> float:
        return float(np.mean(np.abs(dlog) <= np.log10(1 + pct / 100)))

    report = {
        "n_valid": int(valid.sum()),
        "mlp": {
            "mae_dlog10_P": float(np.mean(np.abs(dlog_mlp_v))) if len(dlog_mlp_v) else float("nan"),
            "r2_log10_P": _r2(y_true[valid, 0], y_pred[valid, 0]) if valid.sum() >= 2 else float("nan"),
            "frac_within_5pct": _frac_within(dlog_mlp_v, 5),
            "frac_within_10pct": _frac_within(dlog_mlp_v, 10),
        },
        "lsp_peak": {
            "mae_dlog10_P": float(np.mean(np.abs(dlog_lsp_v))) if len(dlog_lsp_v) else float("nan"),
            "frac_within_5pct": _frac_within(dlog_lsp_v, 5),
            "frac_within_10pct": _frac_within(dlog_lsp_v, 10),
        },
        "lsp_wins_count": int(np.sum(np.abs(dlog_lsp_v) < np.abs(dlog_mlp_v))),
        "mlp_wins_count": int(np.sum(np.abs(dlog_mlp_v) < np.abs(dlog_lsp_v))),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(np.log10(p_true[valid]), np.log10(p_mlp[valid]), s=8, alpha=0.45, label="MLP", edgecolors="none")
    axes[0].scatter(np.log10(p_true[valid]), np.log10(p_lsp[valid]), s=8, alpha=0.35, label="LSP peak", edgecolors="none")
    lo = np.log10(p_true[valid].min())
    hi = np.log10(p_true[valid].max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].set_xlabel(r"$\log_{10} P_{\rm true}$")
    axes[0].set_ylabel(r"$\log_{10} P_{\rm pred}$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Period recovery: MLP vs LSP argmax")

    axes[1].scatter(dlog_lsp_v, dlog_mlp_v, s=8, alpha=0.45, edgecolors="none")
    lim = max(np.abs(dlog_lsp_v).max(), np.abs(dlog_mlp_v).max(), 0.1)
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].set_xlabel(r"$\log_{10}(P_{\rm LSP}/P_{\rm true})$")
    axes[1].set_ylabel(r"$\log_{10}(P_{\rm MLP}/P_{\rm true})$")
    axes[1].set_title("|LSP error| vs |MLP error|")
    fig.tight_layout()
    fig.savefig(out_dir / "period_recovery_lsp_vs_mlp.png", dpi=150)
    plt.close(fig)

    _write_json(out_dir / "period_recovery.json", report)
    return report


def plot_raw_output_histograms(
    y_raw: np.ndarray,
    y_constrained: np.ndarray,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["e", "cos_omega", "sin_omega"]
    idx = [2, 3, 4]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    stats_out: dict[str, Any] = {}

    for col, (ax_pre, ax_post), j, name in zip(range(3), zip(axes[0], axes[1]), idx, names):
        pre, post = y_raw[:, j], y_constrained[:, j]
        ax_pre.hist(pre, bins=60, alpha=0.75, color="C0")
        ax_pre.set_title(f"raw {name}")
        ax_post.hist(post, bins=60, alpha=0.75, color="C1")
        ax_post.set_title(f"constrained {name}")

    # Omega vector norm
    pre_norm = np.sqrt(y_raw[:, 3] ** 2 + y_raw[:, 4] ** 2)
    post_norm = np.sqrt(y_constrained[:, 3] ** 2 + y_constrained[:, 4] ** 2)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if np.ptp(pre_norm) > 1e-12:
        ax2.hist(pre_norm, bins=60, alpha=0.6, label="pre-constraint", color="C0")
    else:
        ax2.axvline(float(pre_norm[0]), color="C0", lw=2, label="pre-constraint (constant)")
    if np.ptp(post_norm) > 1e-12:
        ax2.hist(post_norm, bins=60, alpha=0.6, label="post-constraint", color="C1")
    else:
        ax2.axvline(float(post_norm[0]), color="C1", lw=2, label="post-constraint (constant=1)")
    ax2.axvline(1.0, color="k", ls="--", lw=0.8)
    ax2.set_xlabel(r"$\|(\cos\omega, \sin\omega)\|$")
    ax2.set_ylabel("count")
    ax2.set_title("Omega vector norm (pre vs post L2 normalize)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "omega_vector_norm_hist.png", dpi=150)
    plt.close(fig2)

    stats_out["omega_norm_pre"] = {
        "mean": float(np.mean(pre_norm)),
        "std": float(np.std(pre_norm)),
        "frac_near_1": float(np.mean(np.abs(pre_norm - 1) < 0.05)),
        "frac_lt_0.5": float(np.mean(pre_norm < 0.5)),
    }
    stats_out["omega_norm_post"] = {
        "mean": float(np.mean(post_norm)),
        "frac_near_1": float(np.mean(np.abs(post_norm - 1) < 1e-6)),
    }
    stats_out["e_raw"] = {
        "frac_negative": float(np.mean(y_raw[:, 2] < 0)),
        "frac_gt_0.99": float(np.mean(y_raw[:, 2] > 0.99)),
    }

    for ax in axes.ravel():
        ax.set_ylabel("count")
    fig.suptitle("Pre- vs post-constraint output distributions (validation)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_output_hist.png", dpi=150)
    plt.close(fig)

    _write_json(out_dir / "raw_output_stats.json", stats_out)
    return stats_out


def _spearman_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    covariates: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t_idx, t_name in enumerate(THETA_NAMES):
        resid = np.abs(y_pred[:, t_idx] - y_true[:, t_idx])
        out[t_name] = {}
        for c_name, c_vals in covariates.items():
            m = np.isfinite(c_vals) & np.isfinite(resid)
            if m.sum() < 5:
                out[t_name][c_name] = {"rho": float("nan"), "p": float("nan"), "n": int(m.sum())}
                continue
            rho, pval = stats.spearmanr(c_vals[m], resid[m])
            out[t_name][c_name] = {"rho": float(rho), "p": float(pval), "n": int(m.sum())}
    return out


def omega_mae_vs_e_bins(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    e = y_true[:, 2]
    mask_ecc = e > 0.05
    e_bins = np.linspace(0.05, 0.99, 10)
    bin_report: list[dict] = []
    maes: list[float] = []
    centers: list[float] = []

    for i in range(len(e_bins) - 1):
        lo, hi = e_bins[i], e_bins[i + 1]
        m = mask_ecc & (e >= lo) & (e < hi if i < len(e_bins) - 2 else e <= hi)
        n = int(m.sum())
        entry = {"e_lo": float(lo), "e_hi": float(hi), "n": n}
        if n >= 3:
            mae = _omega_mae_deg(y_true[m], y_pred[m])
            entry["omega_mae_deg"] = mae
            maes.append(mae)
            centers.append(0.5 * (lo + hi))
        bin_report.append(entry)

    fig, ax = plt.subplots(figsize=(6, 4))
    if maes:
        ax.plot(centers, maes, "o-", color="C0")
    ax.set_xlabel("true eccentricity e")
    ax.set_ylabel(r"$\omega$ MAE (deg)")
    ax.set_title(r"Angular error vs $e$ (validation, $e > 0.05$)")
    fig.tight_layout()
    fig.savefig(out_dir / "omega_mae_vs_e.png", dpi=150)
    plt.close(fig)

    return {"bins": bin_report}


def build_sanity_report(
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    meta_val: pd.DataFrame,
    snr_val: np.ndarray,
    y_raw_val: np.ndarray,
    log_path: Path | None,
) -> dict:
    train_metrics = _per_target_metrics(y_train, y_pred_train)
    val_metrics = _per_target_metrics(y_val, y_pred_val)

    gap: dict[str, float] = {}
    for name in THETA_NAMES:
        gap[name] = train_metrics[name]["r2"] - val_metrics[name]["r2"]

    covariates = {
        "n_obs": meta_val["n_obs"].to_numpy(dtype=float),
        "median_sigma_ms": meta_val["median_sigma_ms"].to_numpy(dtype=float),
        "snr_K_over_sigma": snr_val,
        "e": y_val[:, 2],
        "rv_std_ms": meta_val["rv_std_ms"].to_numpy(dtype=float),
    }
    spearman = _spearman_residuals(y_val, y_pred_val, covariates)

    k_true = 10.0 ** y_val[:, 1]
    rv_std = meta_val["rv_std_ms"].to_numpy(dtype=float)
    k_resid = np.abs(10.0 ** y_pred_val[:, 1] - k_true)
    m_k = np.isfinite(rv_std) & np.isfinite(k_true)
    m_kr = m_k & np.isfinite(k_resid)

    pre_norm = np.sqrt(y_raw_val[:, 3] ** 2 + y_raw_val[:, 4] ** 2)

    report: dict[str, Any] = {
        "train_val_r2_gap": gap,
        "train_per_target": train_metrics,
        "val_per_target": val_metrics,
        "residual_spearman": spearman,
        "k_rv_std_correlation": {
            "corr_rv_std_true_K": float(np.corrcoef(rv_std[m_k], k_true[m_k])[0, 1]) if m_k.sum() >= 3 else float("nan"),
            "corr_rv_std_K_residual": float(np.corrcoef(rv_std[m_kr], k_resid[m_kr])[0, 1]) if m_kr.sum() >= 3 else float("nan"),
        },
        "omega_vector_norm_pre": {
            "mean": float(np.mean(pre_norm)),
            "median": float(np.median(pre_norm)),
        },
        "leakage_notes": {
            "synthetic_split": "i.i.d. row permutation (no host grouping)",
            "eccentricity_prior": "fit on real train split only (synthetic_dataset._load_eccentricity_prior)",
            "cadence_bootstrap": "train-only .tbl profiles in synthetic_dataset",
            "real_transfer": "host-grouped splits in preprocess.py (separate from synthetic CSV)",
        },
        "learning_curves": {"available": False, "log_path": str(log_path) if log_path else None},
    }
    return report


def run_regression_diagnostics(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    feature_set: str,
    out_dir: Path,
    val_frac: float = 0.2,
    seed: int = 42,
    device: torch.device | None = None,
    log_path: Path | None = None,
) -> dict:
    """Run full diagnostic suite; write plots and JSON to out_dir."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_from_csv(csv_path, feature_set)
    model, norm_stats = _load_model(checkpoint_path, device)

    train_idx, val_idx = _val_split_indices(len(bundle.X), val_frac, seed)
    X_train, y_train = bundle.X[train_idx], bundle.y[train_idx]
    X_val, y_val = bundle.X[val_idx], bundle.y[val_idx]

    y_pred_train = predict(model, X_train, norm_stats, device)
    y_pred_val = predict(model, X_val, norm_stats, device)
    y_raw_val = predict(
        model, X_val, norm_stats, device,
        constrain_e=False, constrain_omega=False,
    )
    y_constrained_val = apply_theta_constraints(y_raw_val.copy())

    meta_val = _metadata_for_indices(bundle, val_idx)
    snr_val = _compute_snr(meta_val, y_val)

    summary: dict[str, Any] = {
        "csv": str(csv_path),
        "checkpoint": str(checkpoint_path),
        "feature_set": feature_set,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_frac": val_frac,
        "seed": seed,
    }

    print("=== e prior (train split) ===")
    summary["e_prior"] = plot_e_prior_train_hist(y_train[:, 2], out_dir)

    print("=== SNR-sliced errors ===")
    summary["snr"] = plot_snr_diagnostics(y_val, y_pred_val, snr_val, out_dir)

    print("=== P / baseline identifiability ===")
    summary["p_baseline"] = plot_p_baseline_diagnostics(y_val, y_pred_val, meta_val, out_dir)

    print("=== LSP vs MLP period recovery ===")
    summary["period_recovery"] = period_recovery_diagnostics(y_val, y_pred_val, meta_val, out_dir)

    print("=== Raw output histograms ===")
    summary["raw_outputs"] = plot_raw_output_histograms(y_raw_val, y_constrained_val, out_dir)

    print("=== Sanity report ===")
    val_bundle = DatasetBundle(
        X_val, y_val,
        row_idx=bundle.row_idx[val_idx],
        e=bundle.e[val_idx],
        has_t_peri=bundle.has_t_peri[val_idx],
        has_ecc=bundle.has_ecc[val_idx],
        df=bundle.df,
    )
    omega_bins = omega_mae_vs_e_bins(y_val, y_pred_val, out_dir)
    sanity = build_sanity_report(
        bundle, train_idx, val_idx,
        y_train, y_pred_train, y_val, y_pred_val,
        meta_val, snr_val, y_raw_val, log_path,
    )
    sanity["omega_mae_vs_e"] = omega_bins
    sanity["val_subsets"] = _subset_metrics(val_bundle, y_val, y_pred_val)
    _write_json(out_dir / "sanity_report.json", sanity)
    summary["sanity"] = {
        "train_val_r2_gap": sanity["train_val_r2_gap"],
        "k_rv_std_correlation": sanity["k_rv_std_correlation"],
    }

    _write_json(out_dir / "diagnostics_summary.json", summary)
    print(f"\nDiagnostics complete -> {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=ROOT / "synthetic_generation" / "datasets" / "synthetic_regression_10000_phasefold.csv")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "regression_mlp_109.pt")
    p.add_argument("--feature-set", choices=["74", "35", "109"], default="109")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "regression_synthetic" / "diagnostics")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-path", type=Path, default=None, help="optional training log for learning curves")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_regression_diagnostics(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        feature_set=args.feature_set,
        out_dir=args.out,
        val_frac=args.val_frac,
        seed=args.seed,
        device=torch.device(args.device),
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
