"""
bayesian_interval_comparison.py
-------------------------------
Compare our conformal-prediction (CP) interval half-widths against the
catalog "Bayesian" intervals for held-out RV hosts, i.e. the published
1-sigma uncertainties (`*err1`/`*err2`) from the NASA Exoplanet Archive.
This is Nicolo's Task 4 / the "comparison with the Bayesian intervals
(let's use the tabulated uncertainty for that)" workstream.

Inputs
------
  --cp-csv    figures/paper/earthlike_top10.csv   (produced by paper_rv_figures.py)
                per held-out system: P/K/e/omega predictions + CP alpha=0.1 half-widths.
  --labels    data/labels.csv                     (produced by parse_and_label.py)
                catalog point values + published err1/err2 for each planet.

Outputs (written to --out-dir, default figures/paper/)
------------------------------------------------------
  bayesian_interval_comparison.csv   tidy per (system, parameter) comparison
  bayesian_interval_comparison.tex   per-parameter summary table for Overleaf
  bayesian_interval_comparison.png   CP vs Bayesian half-width scatter (per parameter)

What "comparable" means here (open conventions — confirm with Nicolo)
---------------------------------------------------------------------
The CP region is reported in the model's target space: log10 dex for P and K,
linear for e, radians for omega. The catalog gives a physical 1-sigma. To put
them on one axis we:
  * convert the catalog sigma into the same space (dex for P/K via
    sigma_dex = sigma_phys / (x * ln 10); rad for omega);
  * scale the catalog 1-sigma to the SAME nominal level as the CP region with
    `--sigma-scale` (default 1.6449 = Gaussian two-sided 90%, matching alpha=0.1).
    Pass `--sigma-scale 1.0` to compare against the raw tabulated 1-sigma.
These three choices (dex vs physical for P/K, the sigma->interval scale, and how
to treat the near-vacuous omega interval) are the things Nicolo may want to pin
down; they are isolated in PARAM_SPECS and `--sigma-scale` so they are easy to change.

Usage
-----
    python scripts/bayesian_interval_comparison.py
    python scripts/bayesian_interval_comparison.py --sigma-scale 1.0 --out-dir /tmp/bayes
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CP_CSV = ROOT / "figures" / "paper" / "earthlike_top10.csv"
DEFAULT_LABELS = ROOT / "data" / "labels.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "paper"

LN10 = math.log(10.0)


@dataclass(frozen=True)
class ParamSpec:
    """How one orbital parameter is compared across the two interval sources.

    space:
      "log10" — compare in log10 dex (P, K). CP half-width is already dex;
                the catalog sigma is converted via sigma/(x*ln10).
      "linear" — compare in physical units (e).
      "angle" — compare in radians with circular wrapping (omega).
    """

    name: str
    pred_col: str          # column in the CP csv (physical units)
    tab_col: str           # column in the CP csv (catalog physical value)
    cp_hw_col: str         # CP alpha=0.1 half-width column
    cat_val_col: str       # catalog value column in labels.csv
    cat_err1_col: str      # catalog +err column
    cat_err2_col: str      # catalog -err column (usually negative)
    space: str
    unit: str              # label for the comparison space (e.g. "dex", "rad")


# Column wiring between earthlike_top10.csv and labels.csv. Catalog omega is
# `pl_orblper` (argument of periastron, DEGREES); everything else is in the
# same physical unit the CP csv reports.
PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("P", "P_pred_d", "P_tab_d", "halfwidth_log10_P_a01",
              "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "log10", "dex"),
    ParamSpec("K", "K_pred_ms", "K_tab_ms", "halfwidth_log10_K_a01",
              "pl_rvamp", "pl_rvamperr1", "pl_rvamperr2", "log10", "dex"),
    ParamSpec("e", "e_pred", "e_tab", "halfwidth_e_a01",
              "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "linear", "-"),
    ParamSpec("omega", "omega_pred_rad", "omega_tab_rad", "halfwidth_omega_a01",
              "pl_orblper", "pl_orblpererr1", "pl_orblpererr2", "angle", "rad"),
]


def _symmetric_sigma(err1: float, err2: float) -> float:
    """Mean of the (usually asymmetric) published +/- uncertainties.

    Returns NaN if neither side is a finite number; uses whichever side is
    present if only one is.
    """
    vals = [abs(e) for e in (err1, err2) if e is not None and np.isfinite(e)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _wrap_to_pi(angle: float) -> float:
    """Wrap a radian angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _compare_one(spec: ParamSpec, row: pd.Series, sigma_scale: float) -> dict:
    """Build the comparison record for one parameter of one system.

    All *_cmp fields are expressed in `spec.space` (dex / linear / rad), so
    cp_halfwidth_cmp and bayes_halfwidth_cmp share an axis and their ratio is
    unit-free.
    """
    pred = float(row[spec.pred_col])
    tab = float(row[spec.tab_col])
    cp_hw = float(row[spec.cp_hw_col])
    cat_sigma_phys = _symmetric_sigma(row.get(spec.cat_err1_col), row.get(spec.cat_err2_col))

    if spec.space == "log10":
        # x -> log10(x); dlog10(x) ~= dx / (x ln10).
        pred_c, tab_c = math.log10(pred), math.log10(tab)
        cat_sigma_cmp = cat_sigma_phys / (abs(tab) * LN10) if tab != 0 else float("nan")
        residual = tab_c - pred_c
    elif spec.space == "linear":
        pred_c, tab_c = pred, tab
        cat_sigma_cmp = cat_sigma_phys
        residual = tab_c - pred_c
    elif spec.space == "angle":
        pred_c, tab_c = pred, tab
        cat_sigma_cmp = math.radians(cat_sigma_phys)
        residual = _wrap_to_pi(tab_c - pred_c)
    else:  # pragma: no cover - guarded by PARAM_SPECS
        raise ValueError(f"unknown space {spec.space!r}")

    bayes_hw_cmp = sigma_scale * cat_sigma_cmp
    width_ratio = cp_hw / bayes_hw_cmp if bayes_hw_cmp not in (0.0,) and np.isfinite(bayes_hw_cmp) else float("nan")

    return {
        "host": row.get("host", ""),
        "pl_name": row.get("pl_name", ""),
        "split": row.get("split", ""),
        "param": spec.name,
        "space": spec.space,
        "unit": spec.unit,
        "pred_phys": pred,
        "tab_phys": tab,
        "cat_sigma_phys": cat_sigma_phys,
        "cp_halfwidth_cmp": cp_hw,
        "bayes_halfwidth_cmp": bayes_hw_cmp,
        "width_ratio_cp_over_bayes": width_ratio,
        # coverage-style sanity checks (not a formal coverage guarantee):
        "cp_covers_tab": bool(abs(residual) <= cp_hw) if np.isfinite(cp_hw) else False,
        "bayes_covers_pred": bool(abs(residual) <= bayes_hw_cmp) if np.isfinite(bayes_hw_cmp) else False,
        "omega_near_vacuous": bool(spec.space == "angle" and cp_hw >= math.pi),
    }


def build_comparison(cp_df: pd.DataFrame, labels: pd.DataFrame, sigma_scale: float) -> pd.DataFrame:
    """Join CP predictions to catalog uncertainties and compare per parameter.

    Join key is `pl_name` (falls back to `host`->`hostname` for rows without a
    planet-name match). Returns a tidy frame with one row per (system, param).
    """
    cat_cols = [c for spec in PARAM_SPECS for c in (spec.cat_val_col, spec.cat_err1_col, spec.cat_err2_col)]
    keep = ["pl_name", "hostname", *dict.fromkeys(cat_cols)]
    keep = [c for c in keep if c in labels.columns]
    lab = labels[keep].drop_duplicates(subset=["pl_name"])

    merged = cp_df.merge(lab, on="pl_name", how="left", suffixes=("", "_lab"))
    # Fallback join on host name for any planet that didn't match by pl_name.
    missing = merged[PARAM_SPECS[0].cat_val_col].isna()
    if missing.any() and "hostname" in labels.columns:
        by_host = labels[keep].drop_duplicates(subset=["hostname"]).set_index("hostname")
        for i in merged.index[missing]:
            host = merged.at[i, "host"]
            if host in by_host.index:
                for c in keep:
                    if c not in ("pl_name", "hostname"):
                        merged.at[i, c] = by_host.at[host, c]

    records = [_compare_one(spec, row, sigma_scale)
               for _, row in merged.iterrows()
               for spec in PARAM_SPECS]
    return pd.DataFrame.from_records(records)


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter medians and coverage fractions across systems."""
    rows = []
    for name in [s.name for s in PARAM_SPECS]:
        sub = comp[comp["param"] == name]
        finite = sub[np.isfinite(sub["width_ratio_cp_over_bayes"])]
        rows.append({
            "param": name,
            "n": int(len(sub)),
            "n_with_catalog_sigma": int(len(finite)),
            "median_cp_halfwidth": float(np.nanmedian(sub["cp_halfwidth_cmp"])),
            "median_bayes_halfwidth": float(np.nanmedian(sub["bayes_halfwidth_cmp"])),
            "median_width_ratio": float(np.nanmedian(finite["width_ratio_cp_over_bayes"])) if len(finite) else float("nan"),
            "cp_covers_tab_frac": float(sub["cp_covers_tab"].mean()),
            "bayes_covers_pred_frac": float(sub["bayes_covers_pred"].mean()),
        })
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, out_path: Path, sigma_scale: float) -> None:
    """Emit a compact per-parameter summary table for Overleaf."""
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Param & CP half-width & Bayes half-width & ratio & CP covers tab. \\",
        r"\hline",
    ]
    unit = {s.name: s.unit for s in PARAM_SPECS}
    symbol = {"P": "P", "K": "K", "e": "e", "omega": r"\omega"}
    for _, r in summary.iterrows():
        u = unit[r["param"]]
        usuffix = f" {u}" if u != "-" else ""
        lines.append(
            f"${symbol.get(r['param'], r['param'])}$ & {r['median_cp_halfwidth']:.3g}{usuffix} & "
            f"{r['median_bayes_halfwidth']:.3g}{usuffix} & "
            f"{r['median_width_ratio']:.2f} & "
            f"{r['cp_covers_tab_frac']:.0%} \\\\"
        )
    lines += [
        r"\hline",
        rf"\multicolumn{{5}}{{l}}{{\footnotesize Median over held-out hosts. Bayesian half-width "
        rf"= {sigma_scale:g}$\times$ tabulated 1$\sigma$ (NASA Exoplanet Archive). "
        rf"$P$, $K$ in $\log_{{10}}$ dex; $\omega$ in rad.}} \\",
        r"\end{tabular}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def plot_comparison(comp: pd.DataFrame, out_path: Path, sigma_scale: float) -> None:
    """Scatter CP half-width (y) vs Bayesian half-width (x), one panel per parameter."""
    specs = PARAM_SPECS
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.0))
    for ax, spec in zip(np.atleast_1d(axes), specs):
        sub = comp[comp["param"] == spec.name]
        x = sub["bayes_halfwidth_cmp"].to_numpy(dtype=float)
        y = sub["cp_halfwidth_cmp"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=36, color="tab:blue", edgecolor="k", linewidth=0.4, zorder=3)
        if ok.any():
            lo = float(np.nanmin(np.concatenate([x[ok], y[ok]])))
            hi = float(np.nanmax(np.concatenate([x[ok], y[ok]])))
            pad = 0.1 * (hi - lo + 1e-9)
            line = np.array([lo - pad, hi + pad])
            ax.plot(line, line, ls="--", color="grey", lw=1.0, zorder=1, label="CP = Bayes")
            if spec.space == "log10":
                ax.set_xscale("log")
                ax.set_yscale("log")
        ax.set_title(f"${spec.name}$")
        ax.set_xlabel(f"Bayesian half-width [{spec.unit}]")
        ax.set_ylabel(f"CP half-width [{spec.unit}]")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        f"Conformal vs tabulated ({sigma_scale:g}x 1$\\sigma$) intervals, held-out hosts",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cp-csv", type=Path, default=DEFAULT_CP_CSV,
                   help="CP predictions + half-widths (default: figures/paper/earthlike_top10.csv)")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                   help="catalog labels with err1/err2 (default: data/labels.csv)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="where to write the csv/tex/png (default: figures/paper)")
    p.add_argument("--sigma-scale", type=float, default=1.6449,
                   help="multiply tabulated 1-sigma by this to form the Bayesian interval "
                        "(default 1.6449 = two-sided 90%%, matching alpha=0.1; pass 1.0 for raw 1-sigma)")
    args = p.parse_args()

    if not args.cp_csv.exists():
        raise SystemExit(f"CP csv not found: {args.cp_csv} (run scripts/paper_rv_figures.py first)")
    if not args.labels.exists():
        raise SystemExit(f"labels csv not found: {args.labels} (run parse_and_label.py first)")

    cp_df = pd.read_csv(args.cp_csv)
    labels = pd.read_csv(args.labels)
    comp = build_comparison(cp_df, labels, args.sigma_scale)
    summary = summarize(comp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comp_path = args.out_dir / "bayesian_interval_comparison.csv"
    tex_path = args.out_dir / "bayesian_interval_comparison.tex"
    png_path = args.out_dir / "bayesian_interval_comparison.png"
    comp.to_csv(comp_path, index=False)
    write_latex(summary, tex_path, args.sigma_scale)
    plot_comparison(comp, png_path, args.sigma_scale)

    pd.set_option("display.width", 120)
    print(f"[bayes] {len(cp_df)} systems x {len(PARAM_SPECS)} params -> {len(comp)} comparisons")
    print(summary.to_string(index=False))
    print(f"[done] {comp_path}")
    print(f"[done] {tex_path}")
    print(f"[done] {png_path}")


if __name__ == "__main__":
    main()
