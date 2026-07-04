"""
init_experiment.py
------------------
Per Nicolò's request: quantify (a) how much the LS optimizer changes
catalog parameters from their initial values, and (b) what happens when
we initialize T_peri randomly instead of from the catalog.

Runs every file in data/rv_raw twice through validate_one:
  1. With catalog T_peri initialization (current default), recording final
     T_peri values and the RMS that scipy.optimize converges to.
  2. With T_peri replaced by uniform random samples over [0, P] before LM,
     across N seeds, recording the best RMS over those seeds.

Writes data/init_comparison.csv with one row per file and produces a
figure showing the RMS distributions and the T_peri-shift histogram.

Usage
-----
  python init_experiment.py --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kepler_check import validate_one


def run_one(tbl: Path, labels: pd.DataFrame, simbad_cache: dict,
            random_init: bool, seed: int = 0) -> dict | None:
    """One configuration of validate_one with quadratic trend + auto_sign + fit_tperi."""
    r = validate_one(tbl, labels, mode="fit", plot=False, verbose=False,
                     simbad_cache=simbad_cache,
                     auto_sign=True, fit_tperi=True, trend_order=2,
                     random_init=random_init, random_seed=seed)
    return r if r.get("status") == "ok" else None


def compare(rv_dir: Path, labels_path: Path, simbad_cache_path: Path,
            n_seeds: int = 5, out_csv: Path = Path("data/init_comparison.csv"),
            out_fig: Path = Path("figures/init_comparison.png")) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    simbad_cache = (json.loads(simbad_cache_path.read_text())
                     if simbad_cache_path.exists() else {})
    files = sorted(rv_dir.glob("UID_*_RVC_*.tbl"))

    rows = []
    for i, f in enumerate(files, 1):
        # Catalog init
        r_cat = run_one(f, labels, simbad_cache, random_init=False)
        if r_cat is None:
            continue
        # Random init across n_seeds; record best (LM landing in any basin)
        seed_results = []
        for s in range(n_seeds):
            r_rnd = run_one(f, labels, simbad_cache, random_init=True, seed=s)
            if r_rnd is not None:
                seed_results.append(r_rnd["rms_residual_ms"])
        if not seed_results:
            continue
        rows.append({
            "file": f.name,
            "host": r_cat["host"],
            "n_obs": r_cat["n_obs"],
            "n_planets": r_cat["n_planets"],
            "median_sigma_ms": r_cat["median_sigma_ms"],
            "rms_catalog_init": r_cat["rms_residual_ms"],
            "rms_random_best": float(min(seed_results)),
            "rms_random_median": float(np.median(seed_results)),
            "rms_random_worst": float(max(seed_results)),
            "tperi_shifts_d": r_cat.get("tperi_shifts_d", ""),
            "trend_coefs": r_cat.get("trend_coefs", ""),
        })
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}] processed; kept {len(rows)} 'ok' files")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Quality-filter the same way as the corpus summary
    clean = df[(df.n_obs >= 10) & df.median_sigma_ms.between(0.1, 100.0)].copy()
    if len(clean) == 0:
        print("[warn] no quality-filtered rows; skipping figure")
        return df

    same = (np.isclose(clean.rms_catalog_init, clean.rms_random_best, rtol=0.01)
            | (np.abs(clean.rms_catalog_init - clean.rms_random_best) < 0.1))
    cat_wins = (clean.rms_catalog_init < clean.rms_random_best - 0.1)
    rnd_wins = (clean.rms_random_best < clean.rms_catalog_init - 0.1)

    print(f"\nQuality-filtered N = {len(clean)}")
    print(f"  Same RMS  (random found same basin): {same.sum():>5d}  ({same.mean():.1%})")
    print(f"  Catalog better (random missed basin): {cat_wins.sum():>5d}  ({cat_wins.mean():.1%})")
    print(f"  Random better (catalog was wrong):    {rnd_wins.sum():>5d}  ({rnd_wins.mean():.1%})")
    print(f"\nCatalog-init   RMS: median = {clean.rms_catalog_init.median():.2f} m/s")
    print(f"Random-best    RMS: median = {clean.rms_random_best.median():.2f} m/s")
    print(f"Random-median  RMS: median = {clean.rms_random_median.median():.2f} m/s   "
          f"(quantifies basin-hopping risk per seed)")

    # T_peri shift magnitudes (in fraction of period, harder to compute without P;
    # for now report absolute days; per-planet first-shift only)
    first_shifts = []
    for s in clean["tperi_shifts_d"].dropna():
        try:
            first_shifts.append(abs(float(str(s).split(",")[0])))
        except (ValueError, IndexError):
            continue
    first_shifts = np.array(first_shifts)
    print(f"\n|ΔT_peri| from catalog (first planet per file):")
    print(f"  median = {np.median(first_shifts):.3f} d")
    print(f"  90th %ile = {np.percentile(first_shifts, 90):.3f} d")

    # Figure: 2 panels — (a) RMS catalog vs random, (b) shift histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    lim_lo = max(0.1, min(clean.rms_catalog_init.min(), clean.rms_random_best.min()))
    lim_hi = min(1e4, max(clean.rms_catalog_init.max(), clean.rms_random_best.max()))
    ax.scatter(clean.rms_catalog_init, clean.rms_random_best, s=14, alpha=0.5, edgecolors="none")
    line = np.geomspace(lim_lo, lim_hi, 50)
    ax.plot(line, line, "k--", lw=1, alpha=0.5, label="y = x (identical basins)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RMS with catalog T_peri init [m/s]")
    ax.set_ylabel("RMS with random T_peri init (best of N) [m/s]")
    ax.set_title(f"Catalog-init vs random-init  (N = {len(clean)} systems)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    if len(first_shifts):
        bins = np.geomspace(max(first_shifts.min(), 1e-4),
                             first_shifts.max(), 40)
        ax.hist(first_shifts + 1e-6, bins=bins, alpha=0.7, edgecolor="black")
        ax.set_xscale("log")
    ax.set_xlabel("|ΔT_peri| from catalog init  [days]")
    ax.set_ylabel("Count")
    ax.set_title("How far LM moves T_peri from the catalog value")
    ax.grid(alpha=0.3)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_csv} and {out_fig}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--simbad-cache", type=Path,
                   default=Path("data/simbad_cache.json"))
    p.add_argument("--n-seeds", type=int, default=5,
                   help="number of random T_peri seeds per file")
    p.add_argument("--out-csv", type=Path,
                   default=Path("data/init_comparison.csv"))
    p.add_argument("--out-fig", type=Path,
                   default=Path("figures/init_comparison.png"))
    args = p.parse_args()
    compare(args.rv_dir, args.labels, args.simbad_cache,
            n_seeds=args.n_seeds, out_csv=args.out_csv, out_fig=args.out_fig)


if __name__ == "__main__":
    main()
