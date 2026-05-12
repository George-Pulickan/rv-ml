"""
diagnostics.py
--------------
Corpus-level diagnostic plots for the RV pipeline:

  python diagnostics.py --gallery 12 --out-dir figures/gallery
  python diagnostics.py --scatter         --out figures/rms_vs_params.png

Gallery mode picks representative systems (best, typical, worst) from
`data/validation_summary.csv` and saves per-system validation plots, so
we have visual evidence beyond the canonical 51 Peg test.

Scatter mode plots RMS/σ against orbital period, planet M sin i, and RV
semi-amplitude K, revealing whether the pipeline systematically struggles
in any parameter regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kepler_check import validate_one


# ---------------------------------------------------------------------------
# Gallery: pick representative systems and save per-system plots
# ---------------------------------------------------------------------------
def make_gallery(rv_dir: Path, labels_path: Path, summary_path: Path,
                 simbad_cache_path: Path, out_dir: Path, n: int = 12) -> None:
    labels = pd.read_csv(labels_path)
    summary = pd.read_csv(summary_path)
    simbad_cache: dict[str, list[str]] = {}
    if simbad_cache_path.exists():
        simbad_cache = json.loads(simbad_cache_path.read_text())

    # Quality filter (same as the --all summary)
    ok = summary[summary["status"] == "ok"].copy()
    clean = ok[(ok["n_obs"] >= 10)
               & (ok["median_sigma_ms"].between(0.1, 100.0))]
    clean = clean.sort_values("rms_over_sigma").reset_index(drop=True)
    if len(clean) == 0:
        print("[gallery] no quality-filtered files; run kepler_check.py --all first")
        return

    # Pick three groups: best, near-median, "worst that's still physical"
    k = max(1, n // 3)
    best = clean.head(k)
    mid_start = max(0, (len(clean) - k) // 2)
    typical = clean.iloc[mid_start: mid_start + k]
    # "Worst that's still physical" = trim the very top extremes (those are
    # stellar binaries etc., not informative for validation quality)
    upper_cut = int(len(clean) * 0.97)
    worst = clean.iloc[max(0, upper_cut - k): upper_cut]

    out_dir.mkdir(parents=True, exist_ok=True)
    picks = pd.concat([best.assign(_band="best"),
                        typical.assign(_band="typical"),
                        worst.assign(_band="worst")], ignore_index=True)
    for _, row in picks.iterrows():
        tbl = rv_dir / row["file"]
        save_path = out_dir / f"{row['_band']}_{Path(row['file']).stem}.png"
        validate_one(tbl, labels, mode="anchor", plot=True, save=save_path,
                     verbose=False, simbad_cache=simbad_cache)
    print(f"[gallery] wrote {len(picks)} plots to {out_dir}/")


# ---------------------------------------------------------------------------
# Scatter: RMS/σ versus orbital period / M sin i / K
# ---------------------------------------------------------------------------
def make_scatter(summary_path: Path, index_path: Path, out_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    index = pd.read_csv(index_path)

    # For each file, pick the dominant planet (largest K, falling back to msini)
    index = index.copy()
    index["_sort_key"] = index["pl_rvamp"].fillna(index["pl_msinij"].fillna(0) * 100)
    dominant = (index.sort_values("_sort_key", ascending=False)
                     .drop_duplicates(subset="file", keep="first"))
    cols = ["file", "pl_orbper", "pl_msinij", "pl_rvamp", "hostname"]
    df = summary.merge(dominant[cols], on="file", how="left", suffixes=("", "_y"))
    df = df[(df["status"] == "ok")
            & (df["n_obs"] >= 10)
            & (df["median_sigma_ms"].between(0.1, 100.0))
            & df["pl_orbper"].notna()].copy()
    if df.empty:
        print("[scatter] no usable rows; run parse_and_label and kepler_check --all first")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    plot_specs = [
        ("pl_orbper", "Orbital period [days]", "log"),
        ("pl_msinij", "Planet $M\\sin i$ [$M_\\mathrm{Jup}$]", "log"),
        ("pl_rvamp",  "RV semi-amplitude $K$ [m/s]", "log"),
    ]
    for ax, (col, xlabel, xscale) in zip(axes, plot_specs):
        m = df[col].notna() & (df[col] > 0)
        x = df.loc[m, col]
        y = df.loc[m, "rms_over_sigma"]
        sc = ax.scatter(x, y, s=18, alpha=0.55, edgecolors="none")
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        ax.axhline(1, color="C2", lw=1, ls="--", alpha=0.8,
                   label="RMS = $\\sigma_\\mathrm{obs}$ (photon-noise floor)")
        ax.axhline(3, color="C1", lw=1, ls="--", alpha=0.8,
                   label="RMS = $3\\sigma_\\mathrm{obs}$ (activity floor)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3, which="both")
        # Median trend in log-bins
        try:
            bins = np.geomspace(x.min(), x.max(), 8)
            xm, ym = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (x >= lo) & (x < hi)
                if sel.sum() >= 3:
                    xm.append(np.sqrt(lo * hi))
                    ym.append(y[sel].median())
            ax.plot(xm, ym, "ko-", ms=6, lw=1.5, alpha=0.9, label="median per bin")
        except Exception:  # noqa: BLE001
            pass
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("RMS$_\\mathrm{resid}$ / $\\sigma_\\mathrm{obs}$")
    fig.suptitle(f"Pipeline validation residuals vs dominant-planet parameters "
                 f"  (N = {len(df)} systems)",
                 y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[scatter] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--index", type=Path, default=Path("data/rv_index.csv"))
    p.add_argument("--summary", type=Path,
                   default=Path("data/validation_summary.csv"))
    p.add_argument("--simbad-cache", type=Path,
                   default=Path("data/simbad_cache.json"))
    p.add_argument("--gallery", type=int, default=0,
                   help="Save N validation plots (best/typical/worst mix)")
    p.add_argument("--out-dir", type=Path, default=Path("figures/gallery"))
    p.add_argument("--scatter", action="store_true",
                   help="Plot RMS/σ versus orbital params")
    p.add_argument("--out", type=Path, default=Path("figures/rms_vs_params.png"))
    args = p.parse_args()

    if args.gallery:
        make_gallery(args.rv_dir, args.labels, args.summary,
                     args.simbad_cache, args.out_dir, n=args.gallery)
    if args.scatter:
        make_scatter(args.summary, args.index, args.out)
    if not args.gallery and not args.scatter:
        p.error("Specify --gallery N or --scatter (or both)")


if __name__ == "__main__":
    main()
