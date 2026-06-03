"""
validate_synthetic_dataset.py
-----------------------------
Smoke-test and compare synthetic RV samples against the real preprocessed
corpus before using synthetic data for training.

This script intentionally validates the simplest regime first:

    f_multi = 0.0

That means every synthetic sample is a single-planet Keplerian signal plus
noise. Companion injection and encoder training are later steps.

Outputs are written to data/synthetic_validation/ by default.
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

from preprocess import LSP_PERIODS, RVDataset
from synthetic_dataset import (
    _GP_LIB_PATH,
    _load_gp_library,
    _load_real_time_grids,
    _sample_orbital_params,
    generate_one,
)


DEFAULT_OUT = Path("data") / "synthetic_validation"


def _masked(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = x[3] == 1
    return x[:, mask], mask


def collect_real() -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]]:
    """Collect real single-planet RVDataset samples into comparable metrics."""
    ds = RVDataset(split="all", normalize=False, single_planet=True)
    rows = []
    examples = []

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
        K = float(10 ** theta[1])
        t_days = xm[0] * float(info["t_span_days"])
        gaps = np.diff(np.sort(t_days))

        rows.append(
            {
                "kind": "real",
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
                "median_sigma_ms": med_sigma,
                "snr_K_over_sigma": K / med_sigma if med_sigma > 0 else np.nan,
                "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
                "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
                "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
            }
        )

        if len(examples) < 6:
            examples.append((x, lsp, theta, info))

    return pd.DataFrame(rows), examples


def collect_synthetic(
    n_samples: int,
    seed: int,
    f_multi: float,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]]:
    """Generate synthetic samples and collect the same metrics as real data."""
    rng = np.random.default_rng(seed)
    params = _sample_orbital_params(rng, n_samples)
    rows = []
    examples = []

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
                "median_sigma_ms": med_sigma,
                "snr_K_over_sigma": float(info["snr_meas"]),
                "lsp_peak_period_d": float(LSP_PERIODS[int(np.argmax(lsp))]),
                "median_gap_d": float(np.median(gaps)) if len(gaps) else np.nan,
                "p90_gap_d": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
            }
        )

        if len(examples) < 12:
            examples.append((x, lsp, theta, info))

    return pd.DataFrame(rows), examples


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
                ax.errorbar(t, rv, yerr=sig, fmt=".", ms=4, alpha=0.75)
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


def summarize(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    out: Path,
    args: argparse.Namespace,
    gp_exists: bool,
    gp_loaded: bool,
    n_grids: int,
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
            "rv_std_ms",
            "snr_K_over_sigma",
            "lsp_peak_period_d",
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
        "n_real_single_planet_valid": int(len(real)),
        "real_time_grids_loaded": int(n_grids),
        "gp_fits_path": str(_GP_LIB_PATH),
        "gp_fits_exists": bool(gp_exists),
        "gp_library_loaded": bool(gp_loaded),
        "noise_mode": "GPNoiseLibrary" if gp_loaded else "white_gaussian_fallback",
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "generation_mode_summary.json").write_text(json.dumps(notes, indent=2))

    with open(out / "README_synthetic_validation.txt", "w", encoding="utf-8") as f:
        f.write("RV-ML synthetic validation smoke run\n")
        f.write("===================================\n\n")
        f.write(f"Scope: synthetic generation with f_multi={args.f_multi}.\n")
        f.write(f"Synthetic samples: {args.n_samples}\n")
        f.write(f"Valid real single-planet comparison samples: {len(real)}\n")
        f.write(f"Real time grids loaded for synthetic cadence bootstrap: {n_grids}\n")
        f.write(f"GP fits exists: {gp_exists}\n")
        f.write(f"GP library loaded: {gp_loaded}\n")
        f.write(f"Noise mode used by generator: {notes['noise_mode']}\n\n")
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
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gp_exists = _GP_LIB_PATH.exists()
    gp_lib = _load_gp_library()
    gp_loaded = gp_lib is not None
    grids = _load_real_time_grids()

    real, _ = collect_real()
    synth, synth_examples = collect_synthetic(args.n_samples, args.seed, args.f_multi)

    make_distribution_plots(real, synth, args.out)
    make_cadence_plots(real, synth, args.out)
    make_noise_plots(real, synth, args.out)
    make_examples_pdf(synth_examples, args.out)
    make_lsp_examples(synth_examples, args.out)
    summarize(real, synth, args.out, args, gp_exists, gp_loaded, len(grids))

    print(f"Wrote synthetic validation outputs to {args.out}")
    print(f"Real comparison samples: {len(real)}")
    print(f"Synthetic samples: {len(synth)}")
    print(f"Real time grids loaded: {len(grids)}")
    print(f"Noise mode: {'GPNoiseLibrary' if gp_loaded else 'white_gaussian_fallback'}")


if __name__ == "__main__":
    main()
