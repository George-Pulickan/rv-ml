"""
2D PCA visualization of real vs synthetic RV systems (Nicolo's suggestion).

Projects real and synthetic systems onto their first two principal components
and plots PC1 vs PC2, with white dots for the *true* (real) data and black dots
for the *fake* (synthetic) data. This is an unsupervised companion to the
real-vs-synthetic classifier: if the two clouds overlap in PC space the
synthetic data occupies the same manifold as the real data.

The feature space is the same compact 74-D representation used by the RF
regression baseline: 64 power-spectrum bins + 10 observation summaries. Features
are z-scored (per feature, using pooled real+synthetic statistics) before PCA so
that large-magnitude summaries such as baseline_d and rv_std_ms do not dominate
the projection. PCA is fit on the pooled standardized data, so both clouds are
placed on common axes.

A three-panel feature-block ablation (summaries only / spectrum only / both) is
also produced, to show which block of features drives any real-vs-synthetic
separation.

Usage
-----
    python synthetic_generation/pca_real_vs_synthetic.py
    python synthetic_generation/pca_real_vs_synthetic.py --feature-set spectral
    python synthetic_generation/pca_real_vs_synthetic.py --real-split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_synthetic_regression_csv import (  # noqa: E402
    SPECTRAL_COLUMNS,
    SUMMARY_COLUMNS,
)
from plot_synthetic_regression_csv import collect_real_summary  # noqa: E402


FEATURE_SETS: dict[str, list[str]] = {
    "summary": SUMMARY_COLUMNS,
    "spectral": SPECTRAL_COLUMNS,
    "both": [*SPECTRAL_COLUMNS, *SUMMARY_COLUMNS],
}


def _clean(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return finite feature rows; drop any row with a non-finite value."""
    arr = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def fit_pca(
    real: np.ndarray,
    synth: np.ndarray,
) -> tuple[PCA, StandardScaler, np.ndarray, np.ndarray]:
    """Standardize on pooled data, then fit a 2-component PCA on the pool."""
    pooled = np.vstack([real, synth])
    scaler = StandardScaler().fit(pooled)
    pooled_z = scaler.transform(pooled)
    pca = PCA(n_components=2, random_state=0).fit(pooled_z)
    real_pc = pca.transform(scaler.transform(real))
    synth_pc = pca.transform(scaler.transform(synth))
    return pca, scaler, real_pc, synth_pc


def _set_robust_limits(ax, coords: np.ndarray, lo: float, hi: float) -> int:
    """Zoom axes to the [lo, hi] percentile bulk; return count clipped from view.

    The PCA fit always uses every point; this only affects what is displayed, so
    a handful of extreme synthetic outliers (near-delta spectra) do not compress
    the dense cloud into an unreadable strip.
    """
    x0, x1 = np.percentile(coords[:, 0], [lo, hi])
    y0, y1 = np.percentile(coords[:, 1], [lo, hi])
    mx = 0.05 * (x1 - x0 + 1e-9)
    my = 0.05 * (y1 - y0 + 1e-9)
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    outside = ((coords[:, 0] < x0) | (coords[:, 0] > x1)
               | (coords[:, 1] < y0) | (coords[:, 1] > y1))
    return int(outside.sum())


def _scatter(
    ax,
    real_pc: np.ndarray,
    synth_pc: np.ndarray,
    pca: PCA,
    max_synthetic: int,
    rng: np.random.Generator,
    clip_pct: tuple[float, float] | None = None,
) -> None:
    # Subsample synthetic for legibility only (fit already used all points).
    if len(synth_pc) > max_synthetic:
        sel = rng.choice(len(synth_pc), size=max_synthetic, replace=False)
        synth_plot = synth_pc[sel]
    else:
        synth_plot = synth_pc

    ax.scatter(
        synth_plot[:, 0], synth_plot[:, 1],
        s=6, c="black", alpha=0.30, linewidths=0,
        label=f"synthetic (fake), n={len(synth_pc)}", zorder=1,
    )
    ax.scatter(
        real_pc[:, 0], real_pc[:, 1],
        s=34, facecolors="white", edgecolors="black", linewidths=0.8,
        label=f"real (true), n={len(real_pc)}", zorder=3,
    )
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)")
    ax.grid(alpha=0.2)

    if clip_pct is not None:
        n_clipped = _set_robust_limits(
            ax, np.vstack([real_pc, synth_pc]), clip_pct[0], clip_pct[1]
        )
        if n_clipped:
            ax.text(
                0.99, 0.01, f"{n_clipped} pts off-view",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray",
            )


def top_loadings(pca: PCA, feature_names: list[str], k: int = 10) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for i in range(2):
        comp = pca.components_[i]
        order = np.argsort(np.abs(comp))[::-1][:k]
        out[f"PC{i + 1}"] = [
            {"feature": feature_names[j], "loading": float(comp[j])} for j in order
        ]
    return out


def make_main_figure(
    real: np.ndarray,
    synth: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    real_split: str,
    out_path: Path,
    max_synthetic: int,
    seed: int,
) -> dict[str, object]:
    pca, _, real_pc, synth_pc = fit_pca(real, synth)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    _scatter(ax, real_pc, synth_pc, pca, max_synthetic, rng, clip_pct=(0.2, 99.8))
    ax.set_title(
        f"PCA of real vs synthetic RV systems\n"
        f"feature set: {feature_set} ({len(feature_names)}-D), real split: {real_split}"
    )
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    ev = pca.explained_variance_ratio_
    return {
        "feature_set": feature_set,
        "n_features": len(feature_names),
        "n_real": int(len(real)),
        "n_synthetic": int(len(synth)),
        "explained_variance_ratio": [float(ev[0]), float(ev[1])],
        "cumulative_explained_variance": float(ev[0] + ev[1]),
        "top_loadings": top_loadings(pca, feature_names),
        "real_pc_mean": [float(real_pc[:, 0].mean()), float(real_pc[:, 1].mean())],
        "synthetic_pc_mean": [float(synth_pc[:, 0].mean()), float(synth_pc[:, 1].mean())],
    }, real_pc, synth_pc


def make_ablation_figure(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    real_split: str,
    out_path: Path,
    max_synthetic: int,
    seed: int,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    rng = np.random.default_rng(seed)
    for ax, (name, cols) in zip(axs, FEATURE_SETS.items()):
        real = _clean(real_df, cols)
        synth = _clean(synth_df, cols)
        pca, _, real_pc, synth_pc = fit_pca(real, synth)
        _scatter(ax, real_pc, synth_pc, pca, max_synthetic, rng, clip_pct=(1.0, 99.0))
        ax.set_title(f"{name} ({len(cols)}-D)")
    axs[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Real (white) vs synthetic (black) in PCA space by feature block "
        f"- real split: {real_split}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(
    csv_path: Path,
    fig_dir: Path,
    out_dir: Path,
    feature_set: str,
    real_split: str,
    sigma_min: float,
    sigma_max: float,
    max_synthetic: int,
    seed: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_df = pd.read_csv(csv_path)
    real_df = collect_real_summary(real_split, sigma_min=sigma_min, sigma_max=sigma_max)
    if real_df.empty:
        raise ValueError("No real rows were collected for comparison")

    print(f"synthetic rows: {len(synth_df)}  real rows: {len(real_df)}")

    cols = FEATURE_SETS[feature_set]
    real = _clean(real_df, cols)
    synth = _clean(synth_df, cols)

    summary, real_pc, synth_pc = make_main_figure(
        real, synth, cols, feature_set, real_split,
        fig_dir / f"pca_real_vs_synthetic_{feature_set}.png",
        max_synthetic, seed,
    )

    make_ablation_figure(
        real_df, synth_df, real_split,
        fig_dir / "pca_real_vs_synthetic_feature_blocks.png",
        max_synthetic, seed,
    )

    # Persist transformed coordinates and the numerical summary.
    coords = pd.DataFrame(
        {
            "pc1": np.concatenate([real_pc[:, 0], synth_pc[:, 0]]),
            "pc2": np.concatenate([real_pc[:, 1], synth_pc[:, 1]]),
            "label": (["real"] * len(real_pc)) + (["synthetic"] * len(synth_pc)),
        }
    )
    coords.to_csv(out_dir / f"pca_coords_{feature_set}.csv", index=False)
    (out_dir / f"pca_summary_{feature_set}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    ev = summary["explained_variance_ratio"]
    print(f"explained variance: PC1={ev[0] * 100:.1f}%  PC2={ev[1] * 100:.1f}%  "
          f"(cumulative {summary['cumulative_explained_variance'] * 100:.1f}%)")
    print(f"wrote figures to {fig_dir}")
    print(f"wrote coords + summary to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("synthetic_generation") / "figures" / "synthetic_regression_10000",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("synthetic_generation") / "regression",
    )
    p.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="both")
    p.add_argument("--real-split", choices=("all", "train", "val", "test"), default="all")
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=100.0)
    p.add_argument("--max-synthetic", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        csv_path=args.csv,
        fig_dir=args.fig_dir,
        out_dir=args.out_dir,
        feature_set=args.feature_set,
        real_split=args.real_split,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        max_synthetic=args.max_synthetic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
