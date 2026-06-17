"""
validate_synthetic_dataset.py
-----------------------------
Smoke-test and compare synthetic RV samples against the real preprocessed
corpus before using synthetic data for training.

This script intentionally validates the simplest regime first:

    f_multi = 0.0

That means every synthetic sample is a single-planet Keplerian signal plus
noise. Companion injection and encoder training are later steps.

Outputs are written to a split-named directory under
figures/synthetic_validation/ by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from kepler_check import rv_keplerian as _rv_keplerian
from preprocess import LSP_PERIODS, RVDataset
from synthetic_dataset import (
    _GP_LIB_PATH,
    _load_gp_library,
    _load_real_time_grids,
    _sample_orbital_params,
    generate_one,
)
from time_series_features import spectral_feature_names, spectral_features


VALIDATION_OUT = Path("figures") / "synthetic_validation"
DEFAULT_OUT = VALIDATION_OUT / "real_all"
SPECTRAL_DIM = 64
SPECTRAL_GRID_SIZE = 1024
OBSERVATION_SUMMARY_FEATURES = [
    "n_obs",
    "baseline_d",
    "rv_std_ms",
    "rv_iqr_ms",
    "median_sigma_ms",
    "sigma_iqr_ms",
    "lsp_peak_period_d",
    "lsp_peak_power",
    "median_gap_d",
    "p90_gap_d",
]


def _masked(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = x[3] == 1
    return x[:, mask], mask


def collect_real(
    real_split: str = "all",
    sigma_min: float = 0.1,
    sigma_max: float = 100.0,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    np.ndarray,
]:
    """Collect real single-planet RVDataset samples into comparable metrics.

    Files with median σ outside [sigma_min, sigma_max] m/s are rejected — the
    same physically-motivated filter that synthetic_rv.build_noise_pool applies
    to keep instrument-precision junk (σ=0.01 placeholders, absolute-RV files
    with σ≈30 km/s) out of the comparison set.
    """
    ds = RVDataset(split=real_split, normalize=False, single_planet=True)
    rows = []
    examples = []
    encoded_series = []
    n_rejected_sigma = 0

    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue

        xm, _ = _masked(x)
        if xm.shape[1] < 10:
            continue

        rv_std = float(info["rv_std_ms"])
        sigma = xm[2] * rv_std
        med_sigma = float(np.median(sigma))
        if not (sigma_min <= med_sigma <= sigma_max):
            n_rejected_sigma += 1
            continue
        K = float(10 ** theta[1])
        t_days = xm[0] * float(info["t_span_days"])
        gaps = np.diff(np.sort(t_days))
        encoded_series.append(
            spectral_features(
                xm[0],
                xm[1],
                d=SPECTRAL_DIM,
                grid_size=SPECTRAL_GRID_SIZE,
            )
        )

        rows.append(
            {
                "kind": "real",
                "real_split": real_split,
                "idx": i,
                "host": info.get("host", ""),
                "file": info.get("file", ""),
                "has_ecc": bool(info.get("has_ecc", True)),
                "log10_P": float(theta[0]),
                "log10_K": float(theta[1]),
                "e": float(theta[2]),
                "P_d": float(10 ** theta[0]),
                "K_ms": K,
                "n_obs": int(info["n_obs"]),
                "baseline_d": float(info["t_span_days"]),
                "rv_std_ms": rv_std,
                "rv_iqr_ms": float(np.subtract(*np.percentile(xm[1] * rv_std, [75, 25]))),
                "median_sigma_ms": med_sigma,
                "sigma_iqr_ms": float(np.subtract(*np.percentile(sigma, [75, 25]))),
                "snr_K_over_sigma": K / med_sigma if med_sigma > 0 else np.nan,
                "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
                "lsp_peak_power": float(np.max(lsp)),
                "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
                "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
            }
        )

        if len(examples) < 6:
            examples.append((x, lsp, theta, info))

    if n_rejected_sigma:
        print(f"[collect_real:{real_split}] rejected {n_rejected_sigma} systems with "
              f"median σ outside [{sigma_min}, {sigma_max}] m/s "
              f"(placeholders / absolute-RV files)")
    return pd.DataFrame(rows), examples, np.asarray(encoded_series, dtype=np.float64)


def collect_synthetic(
    n_samples: int,
    seed: int,
    f_multi: float,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    np.ndarray,
]:
    """Generate synthetic samples and collect the same metrics as real data."""
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n_samples)
    rows = []
    examples = []
    encoded_series = []

    for i in range(n_samples):
        p = {k: float(v[i]) for k, v in params.items()}
        r = np.random.default_rng(seed + 10_000 + i)
        x, lsp, theta, info = generate_one(p, r, f_multi=f_multi)

        xm, _ = _masked(x)
        rv_std = float(info["rv_std_ms"])
        sigma = xm[2] * rv_std
        med_sigma = float(np.median(sigma))
        K = float(10 ** theta[1])
        t_days = xm[0] * float(info["t_span_days"])
        gaps = np.diff(np.sort(t_days))
        encoded_series.append(
            spectral_features(
                xm[0],
                xm[1],
                d=SPECTRAL_DIM,
                grid_size=SPECTRAL_GRID_SIZE,
            )
        )

        rows.append(
            {
                "kind": "synthetic",
                "idx": i,
                "host": "synthetic",
                "file": "",
                "has_ecc": True,
                "log10_P": float(theta[0]),
                "log10_K": float(theta[1]),
                "e": float(theta[2]),
                "P_d": float(10 ** theta[0]),
                "K_ms": K,
                "n_obs": int(info["n_obs"]),
                "baseline_d": float(info["baseline_d"]),
                "rv_std_ms": rv_std,
                "rv_iqr_ms": float(np.subtract(*np.percentile(xm[1] * rv_std, [75, 25]))),
                "median_sigma_ms": med_sigma,
                "sigma_iqr_ms": float(np.subtract(*np.percentile(sigma, [75, 25]))),
                "snr_K_over_sigma": float(info["snr_meas"]),
                "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
                "lsp_peak_power": float(np.max(lsp)),
                "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
                "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
            }
        )

        if len(examples) < 12:
            examples.append((x, lsp, theta, info))

    return pd.DataFrame(rows), examples, np.asarray(encoded_series, dtype=np.float64)


def hist_overlay(
    ax,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    col: str,
    title: str,
    bins=35,
    logx: bool = False,
) -> None:
    """Plot real and synthetic density histograms for one metric."""
    r = real[col].replace([np.inf, -np.inf], np.nan).dropna()
    s = synth[col].replace([np.inf, -np.inf], np.nan).dropna()

    if logx:
        r = r[r > 0]
        s = s[s > 0]
        if len(r) and len(s):
            lo = min(r.min(), s.min())
            hi = max(r.max(), s.max())
            bins = np.geomspace(lo, hi, bins)
        ax.set_xscale("log")

    ax.hist(r, bins=bins, alpha=0.55, density=True, label="real")
    ax.hist(s, bins=bins, alpha=0.55, density=True, label="synthetic")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def make_distribution_plots(real: pd.DataFrame, synth: pd.DataFrame, out: Path) -> None:
    """Save parameter and signal-scale comparison plots."""
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    real_known_e = real[real["has_ecc"]].copy() if "has_ecc" in real.columns else real
    hist_overlay(axs[0, 0], real, synth, "log10_P", "log10 period")
    hist_overlay(axs[0, 1], real, synth, "log10_K", "log10 K")
    hist_overlay(axs[0, 2], real_known_e, synth, "e", "eccentricity (real known-e only)")
    hist_overlay(axs[1, 0], real, synth, "snr_K_over_sigma", "K / median sigma", logx=True)
    hist_overlay(axs[1, 1], real, synth, "rv_std_ms", "RV std [m/s]", logx=True)
    hist_overlay(axs[1, 2], real, synth, "lsp_peak_period_d", "LSP peak period [d]", logx=True)
    axs[0, 0].legend()
    fig.suptitle("Real vs synthetic: parameter and signal-scale checks")
    fig.tight_layout()
    fig.savefig(out / "real_vs_synthetic_parameters.png", dpi=180)
    plt.close(fig)


def make_cadence_plots(real: pd.DataFrame, synth: pd.DataFrame, out: Path) -> None:
    """Save cadence and window-function proxy comparison plots."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    hist_overlay(axs[0, 0], real, synth, "n_obs", "number of observations", bins=35)
    hist_overlay(axs[0, 1], real, synth, "baseline_d", "baseline [days]", bins=35, logx=True)
    hist_overlay(axs[1, 0], real, synth, "median_gap_d", "median gap [days]", bins=35, logx=True)

    axs[1, 1].scatter(real["baseline_d"], real["n_obs"], s=12, alpha=0.45, label="real")
    axs[1, 1].scatter(synth["baseline_d"], synth["n_obs"], s=12, alpha=0.45, label="synthetic")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlabel("baseline [days]")
    axs[1, 1].set_ylabel("n_obs")
    axs[1, 1].grid(alpha=0.25)
    axs[1, 1].legend()

    fig.suptitle("Real vs synthetic: cadence checks")
    fig.tight_layout()
    fig.savefig(out / "real_vs_synthetic_cadence.png", dpi=180)
    plt.close(fig)


def make_noise_plots(real: pd.DataFrame, synth: pd.DataFrame, out: Path) -> None:
    """Save noise-scale comparison plots."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    hist_overlay(axs[0], real, synth, "median_sigma_ms", "median sigma [m/s]", bins=35, logx=True)
    hist_overlay(axs[1], real, synth, "rv_std_ms", "RV std [m/s]", bins=35, logx=True)

    ratio_real = real["rv_std_ms"] / real["median_sigma_ms"]
    ratio_syn = synth["rv_std_ms"] / synth["median_sigma_ms"]
    axs[2].hist(ratio_real.replace([np.inf, -np.inf], np.nan).dropna(), bins=35, alpha=0.55, density=True, label="real")
    axs[2].hist(ratio_syn.replace([np.inf, -np.inf], np.nan).dropna(), bins=35, alpha=0.55, density=True, label="synthetic")
    axs[2].set_xscale("log")
    axs[2].set_title("RV std / median sigma")
    axs[2].grid(alpha=0.25)
    axs[0].legend()

    fig.suptitle("Real vs synthetic: noise-scale checks")
    fig.tight_layout()
    fig.savefig(out / "real_vs_synthetic_noise.png", dpi=180)
    plt.close(fig)


def _overlay_exact_curve(ax, theta: np.ndarray, info: dict) -> None:
    """Overlay the noiseless Keplerian curve if t_peri is available in info."""
    t_peri = info.get("t_peri")
    if t_peri is None:
        return
    P       = 10 ** float(theta[0])
    K       = 10 ** float(theta[1])
    e       = float(theta[2])
    omega   = float(np.arctan2(theta[4], theta[3]))
    rv_med  = float(info.get("rv_med_ms", 0.0))
    rv_std  = float(info["rv_std_ms"])
    t_min   = float(info.get("t_min_days", 0.0))
    baseline = float(info["baseline_d"])

    t_dense_rel = np.linspace(0.0, baseline, 500)
    rv_exact = _rv_keplerian(t_dense_rel + t_min, P, K, e, omega, t_peri)
    ax.plot(t_dense_rel, (rv_exact - rv_med) / rv_std,
            color="crimson", lw=1.5, alpha=0.85, zorder=5, label="exact")
    ax.legend(loc="best", fontsize=7)


def make_examples_pdf(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    out: Path,
) -> None:
    """Save per-sample RV and periodogram examples."""
    with PdfPages(out / "examples_single_planet.pdf") as pdf:
        for start in range(0, len(examples), 4):
            chunk = examples[start:start + 4]
            fig, axs = plt.subplots(len(chunk), 2, figsize=(12, 3.0 * len(chunk)))
            if len(chunk) == 1:
                axs = np.array([axs])

            for row, (x, lsp, theta, info) in enumerate(chunk):
                xm, _ = _masked(x)
                t = xm[0] * float(info["baseline_d"])
                rv = xm[1]
                sig = xm[2]

                ax = axs[row, 0]
                ax.errorbar(t, rv, yerr=sig, fmt=".", ms=4, alpha=0.75, label="obs")
                _overlay_exact_curve(ax, theta, info)
                ax.set_xlabel("days since first observation")
                ax.set_ylabel("normalized RV")
                ax.grid(alpha=0.25)
                ax.set_title(
                    f"P={10**theta[0]:.2f} d, K={10**theta[1]:.2f} m/s, "
                    f"e={theta[2]:.2f}, n={info['n_obs']}"
                )

                ax = axs[row, 1]
                ax.plot(LSP_PERIODS, lsp, lw=1.2)
                ax.axvline(10 ** theta[0], color="crimson", ls="--", lw=1, label="true P")
                ax.set_xscale("log")
                ax.set_xlabel("period [days]")
                ax.set_ylabel("LSP power")
                ax.grid(alpha=0.25)
                ax.legend(loc="best", fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_lsp_examples(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    out: Path,
) -> None:
    """Save a compact grid of synthetic Lomb-Scargle examples."""
    fig, axs = plt.subplots(3, 4, figsize=(15, 8))
    for ax, (_, lsp, theta, info) in zip(axs.ravel(), examples[:12]):
        ax.plot(LSP_PERIODS, lsp, lw=1.1)
        ax.axvline(10 ** theta[0], color="crimson", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"P={10**theta[0]:.1f}d SNR={info['snr_meas']:.1f}", fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("Synthetic Lomb-Scargle examples; red dashed line is true period")
    fig.tight_layout()
    fig.savefig(out / "lsp_examples.png", dpi=180)
    plt.close(fig)


def make_classifier_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    real_spectral: np.ndarray,
    synth_spectral: np.ndarray,
    out: Path,
) -> dict:
    """Distinguish real from synthetic data using observations only.

    Each unevenly sampled RV series contributes a fixed-length spline/FFT
    power vector. This is combined with summaries computed directly from the
    observations and their uncertainties. Catalogued or injected Kepler
    parameters are deliberately excluded from the classifier inputs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    if len(real_spectral) != len(real) or len(synth_spectral) != len(synth):
        raise ValueError("spectral feature rows must align with classifier samples")
    if real_spectral.ndim != 2 or synth_spectral.ndim != 2:
        raise ValueError("spectral features must be two-dimensional")
    if real_spectral.shape[1] != synth_spectral.shape[1]:
        raise ValueError("real and synthetic spectral dimensions must match")

    spectral_dim = int(real_spectral.shape[1])
    spectral_names = spectral_feature_names(spectral_dim)
    features = [*spectral_names, *OBSERVATION_SUMMARY_FEATURES]

    summary = pd.concat(
        [
            real[OBSERVATION_SUMMARY_FEATURES],
            synth[OBSERVATION_SUMMARY_FEATURES],
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan)
    spectral = np.vstack([real_spectral, synth_spectral])
    valid = summary.notna().all(axis=1).to_numpy() & np.isfinite(spectral).all(axis=1)

    X = np.column_stack([spectral[valid], summary.to_numpy(dtype=float)[valid]])
    y_all = np.concatenate(
        [
            np.ones(len(real), dtype=int),
            np.zeros(len(synth), dtype=int),
        ]
    )
    y = y_all[valid]

    def _new_classifier(seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=seed,
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    group_columns = {
        "spectral_power": np.arange(spectral_dim),
        **{
            name: np.array([spectral_dim + i])
            for i, name in enumerate(OBSERVATION_SUMMARY_FEATURES)
        },
    }
    group_drops = {name: [] for name in group_columns}
    scores = []
    p_real = np.full(len(y), np.nan, dtype=float)

    # Permute each complete feature group on held-out folds. This makes the
    # 64-bin spectral block comparable with each one-column summary feature.
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_clf = _new_classifier(42 + fold)
        fold_clf.fit(X[train_idx], y[train_idx])
        baseline = balanced_accuracy_score(y[test_idx], fold_clf.predict(X[test_idx]))
        scores.append(baseline)
        real_class_idx = int(np.where(fold_clf.classes_ == 1)[0][0])
        p_real[test_idx] = fold_clf.predict_proba(X[test_idx])[:, real_class_idx]

        rng = np.random.default_rng(42 + fold)
        for name, columns in group_columns.items():
            repeat_scores = []
            for _ in range(5):
                X_permuted = X[test_idx].copy()
                source = X_permuted[:, columns].copy()
                X_permuted[:, columns] = source[rng.permutation(len(test_idx))]
                repeat_scores.append(
                    balanced_accuracy_score(y[test_idx], fold_clf.predict(X_permuted))
                )
            group_drops[name].append(baseline - float(np.mean(repeat_scores)))

    scores = np.asarray(scores, dtype=float)
    clf = _new_classifier(42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_n = min(20, len(features))
    top_idx = idx[:top_n]

    grouped_importances = {
        name: float(np.mean(drops))
        for name, drops in group_drops.items()
    }
    grouped_importance_std = {
        name: float(np.std(drops))
        for name, drops in group_drops.items()
    }
    grouped_sorted = sorted(
        grouped_importances.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    df_for_plots = pd.concat([real, synth], ignore_index=True).iloc[valid].copy()
    df_for_plots["p_real_oof"] = p_real

    fig, (ax_top, ax_group) = plt.subplots(2, 1, figsize=(12, 9))
    ax_top.bar(range(top_n), importances[top_idx])
    ax_top.set_xticks(range(top_n))
    ax_top.set_xticklabels(
        [features[i] for i in top_idx],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax_top.set_ylabel("importance")
    ax_top.set_title("Top individual observation-based classifier features")
    ax_top.grid(alpha=0.25)

    group_names = [name for name, _ in grouped_sorted]
    group_values = [value for _, value in grouped_sorted]
    group_errors = [grouped_importance_std[name] for name in group_names]
    ax_group.bar(range(len(group_names)), group_values, yerr=group_errors, capsize=3)
    ax_group.axhline(0.0, color="black", linewidth=0.8)
    ax_group.set_xticks(range(len(group_names)))
    ax_group.set_xticklabels(group_names, rotation=35, ha="right", fontsize=9)
    ax_group.set_ylabel("balanced-accuracy drop")
    ax_group.set_title(
        f"Cross-validated grouped permutation importance: "
        f"balanced-acc = {scores.mean():.3f} +/- {scores.std():.3f} "
        f"(0.50 = indistinguishable)"
    )
    ax_group.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "classifier_feature_importance.png", dpi=180)
    plt.close(fig)

    # Probability calibration/diagnostic views requested by the supervisor.
    # These use out-of-fold probabilities, so each sample is scored by a model
    # that did not train on that sample.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 31)
    real_probs = df_for_plots.loc[df_for_plots["kind"] == "real", "p_real_oof"]
    synth_probs = df_for_plots.loc[df_for_plots["kind"] == "synthetic", "p_real_oof"]
    ax.hist(synth_probs, bins=bins, alpha=0.55, density=True, label="synthetic")
    ax.hist(real_probs, bins=bins, alpha=0.55, density=True, label="real")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("out-of-fold predicted probability of being real")
    ax.set_ylabel("density")
    ax.set_title("Real-vs-synthetic classifier probability distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "classifier_probability_histogram.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    param_specs = [
        ("log10_P", "log10 period [d]"),
        ("log10_K", "log10 K [m/s]"),
        ("e", "eccentricity"),
    ]
    for ax, (col, xlabel) in zip(axs, param_specs):
        for kind, face, edge, label in [
            ("synthetic", "white", "black", "synthetic"),
            ("real", "black", "black", "real"),
        ]:
            part = df_for_plots[df_for_plots["kind"] == kind]
            ax.scatter(
                part[col],
                part["p_real_oof"],
                s=18,
                facecolors=face,
                edgecolors=edge,
                linewidths=0.6,
                alpha=0.55,
                label=label,
            )
        ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P(real)") if ax is axs[0] else ax.set_ylabel("")
        ax.grid(alpha=0.25)
    axs[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Classifier probability versus Kepler diagnostic parameters")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "classifier_probability_vs_kepler.png", dpi=180)
    plt.close(fig)

    print(f"[classifier] balanced accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"[classifier] top individual feature: {features[idx[0]]}")
    print(f"[classifier] top feature group: {grouped_sorted[0][0]}")
    return {
        "purpose": "real_vs_synthetic_observation_discriminator",
        "class_labels": {"0": "synthetic", "1": "real"},
        "input_type": "spectral_plus_observation_summaries",
        "spectral_method": "smoothing_spline_uniform_grid_rfft_power",
        "spectral_dimension": spectral_dim,
        "spectral_grid_size": SPECTRAL_GRID_SIZE,
        "spectral_normalized": True,
        "summary_features": OBSERVATION_SUMMARY_FEATURES,
        "excluded_from_classifier": [
            "log10_P",
            "log10_K",
            "e",
            "P_d",
            "K_ms",
            "snr_K_over_sigma",
        ],
        "balanced_accuracy_mean": float(scores.mean()),
        "balanced_accuracy_std": float(scores.std()),
        "probability_diagnostics": [
            "classifier_probability_histogram.png",
            "classifier_probability_vs_kepler.png",
        ],
        "top_feature": features[int(idx[0])],
        "top_feature_group": grouped_sorted[0][0],
        "grouped_importance_method": "cross_validated_permutation_accuracy_drop",
        "grouped_feature_importances": dict(grouped_sorted),
        "grouped_feature_importance_std": grouped_importance_std,
        "feature_importances": {
            features[int(i)]: float(importances[int(i)])
            for i in idx
        },
        "n_real": int((y == 1).sum()),
        "n_synthetic": int((y == 0).sum()),
    }


def summarize(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    out: Path,
    args: argparse.Namespace,
    gp_exists: bool,
    gp_loaded: bool,
    n_grids: int,
    classifier_report: dict,
) -> None:
    """Write CSV summaries and a machine-readable generation-mode note."""
    combined = pd.concat([real, synth], ignore_index=True)
    combined.to_csv(out / "summary_real_vs_synthetic_samples.csv", index=False)

    rows = []
    for kind, df in [("real", real), ("synthetic", synth)]:
        for col in [
            "log10_P",
            "log10_K",
            "e",
            "n_obs",
            "baseline_d",
            "median_sigma_ms",
            "sigma_iqr_ms",
            "rv_std_ms",
            "rv_iqr_ms",
            "snr_K_over_sigma",
            "lsp_peak_period_d",
            "lsp_peak_power",
            "median_gap_d",
        ]:
            vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "kind": kind,
                    "metric": col,
                    "n": len(vals),
                    "median": vals.median(),
                    "p05": vals.quantile(0.05),
                    "p95": vals.quantile(0.95),
                    "mean": vals.mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out / "summary_metric_quantiles.csv", index=False)

    notes = {
        "n_synthetic": args.n_samples,
        "f_multi": args.f_multi,
        "seed": args.seed,
        "real_split": args.real_split,
        "n_real_single_planet_valid": int(len(real)),
        "real_time_grids_loaded": int(n_grids),
        "gp_fits_path": str(_GP_LIB_PATH),
        "gp_fits_exists": bool(gp_exists),
        "gp_library_loaded": bool(gp_loaded),
        "noise_mode": "GPNoiseLibrary" if gp_loaded else "white_gaussian_fallback",
        "classifier": classifier_report,
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "generation_mode_summary.json").write_text(json.dumps(notes, indent=2))

    with open(out / "README_synthetic_validation.txt", "w", encoding="utf-8") as f:
        f.write("RV-ML synthetic validation smoke run\n")
        f.write("===================================\n\n")
        f.write(f"Scope: synthetic generation with f_multi={args.f_multi}.\n")
        f.write(f"Real comparison split: {args.real_split}\n")
        f.write(f"Synthetic samples: {args.n_samples}\n")
        f.write(f"Valid real single-planet comparison samples: {len(real)}\n")
        f.write(f"Real time grids loaded for synthetic cadence bootstrap: {n_grids}\n")
        f.write(f"GP fits exists: {gp_exists}\n")
        f.write(f"GP library loaded: {gp_loaded}\n")
        f.write(f"Noise mode used by generator: {notes['noise_mode']}\n\n")
        f.write("Observation-based classifier diagnostic:\n")
        f.write(
            f"- Inputs: {classifier_report['spectral_dimension']} normalized spectral "
            "power bins plus observation-derived summaries.\n"
        )
        f.write(
            "- Kepler parameters and K/measurement-uncertainty are excluded "
            "from classifier inputs.\n"
        )
        f.write(
            f"- Balanced accuracy: {classifier_report['balanced_accuracy_mean']:.3f} "
            f"+/- {classifier_report['balanced_accuracy_std']:.3f}\n"
        )
        f.write(f"- Top individual discriminator: {classifier_report['top_feature']}\n")
        f.write(f"- Top feature group: {classifier_report['top_feature_group']}\n\n")
        f.write("Additional classifier diagnostics:\n")
        f.write("- classifier_probability_histogram.png shows out-of-fold P(real) by class.\n")
        f.write(
            "- classifier_probability_vs_kepler.png shows out-of-fold P(real) "
            "against Kepler diagnostic parameters, which are not classifier inputs.\n\n"
        )
        f.write("Important interpretation:\n")
        f.write("- This is a smoke/diagnostic validation run, not a training cache.\n")
        if gp_loaded:
            f.write("- GP noise path loaded successfully.\n")
        else:
            f.write("- Because gp_fits.json is absent or unloadable, noise is white Gaussian fallback.\n")
        f.write("- Next scientific step is to inspect plots and decide whether priors, cadence, or noise need adjustment.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--f-multi", type=float, default=0.0)
    p.add_argument(
        "--real-split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Real RVDataset split to compare against.",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = (
            DEFAULT_OUT
            if args.real_split == "all"
            else VALIDATION_OUT / f"real_{args.real_split}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    gp_exists = _GP_LIB_PATH.exists()
    gp_lib = _load_gp_library()
    gp_loaded = gp_lib is not None
    grids = _load_real_time_grids()

    real, _, real_spectral = collect_real(args.real_split)
    synth, synth_examples, synth_spectral = collect_synthetic(
        args.n_samples,
        args.seed,
        args.f_multi,
    )

    make_distribution_plots(real, synth, args.out)
    make_cadence_plots(real, synth, args.out)
    make_noise_plots(real, synth, args.out)
    make_examples_pdf(synth_examples, args.out)
    make_lsp_examples(synth_examples, args.out)
    classifier_report = make_classifier_report(
        real,
        synth,
        real_spectral,
        synth_spectral,
        args.out,
    )
    summarize(real, synth, args.out, args, gp_exists, gp_loaded, len(grids), classifier_report)

    print(f"Wrote synthetic validation outputs to {args.out}")
    print(f"Real comparison samples: {len(real)}")
    print(f"Real comparison split: {args.real_split}")
    print(f"Synthetic samples: {len(synth)}")
    print(f"Real time grids loaded: {len(grids)}")
    print(f"Noise mode: {'GPNoiseLibrary' if gp_loaded else 'white_gaussian_fallback'}")


if __name__ == "__main__":
    main()
