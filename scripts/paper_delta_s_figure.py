"""Surrogate-label gap Delta s per coordinate, from a run's metrics JSON.

Replaces figures/paper/delta_s_gap_distribution.svg, which was hand-authored:
no script produced it, so it could not be regenerated when the numbers changed,
and its own aria-label still describes "four orbital coordinates" from when
omega was reported. This makes the figure reproducible for the first time.

What it shows: for each reported coordinate, the median, 90th percentile and
maximum of the surrogate-label gap over the tuning set. The argument the figure
carries is the spread -- the median sits three orders of magnitude below the
max, so Delta_c taken as the max is fixed by a single extreme draw, which is
what motivates the conformal quantile of --gap-beta.

HONEST LIMITATION: conformal_shift.py persists only these three statistics
(`naive_adjustment`), not the full gap array, so this is a three-point summary
rather than a distribution. Worth persisting the full array in a future run if
the paper wants a real histogram.

Design notes:
  - A dot plot, not bars. Bar length encodes magnitude from zero and a log axis
    has no zero, so bars here would be actively misleading.
  - Marker SHAPE carries the statistic and the only colours are greys, so the
    figure survives greyscale printing and colour-vision deficiency with no
    reliance on hue at all.
  - Median and max are direct-labelled because they are the comparison the
    figure exists to make; p90 is not, to avoid a number on every point.

Usage
-----
    python scripts/paper_delta_s_figure.py \
        --metrics synthetic_generation/regression/.../conformal_shift_metrics.json \
        --out figures/paper/delta_s_gap_distribution.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRETTY = {"log10_P": r"$\log_{10} P$", "log10_K": r"$\log_{10} K$", "e": r"$e$"}
# (key, label, marker, facecolor) -- shape carries the statistic, not hue.
STATS = [
    ("median", "median", "o", "0.35"),
    ("p90", "90th pct", "s", "0.60"),
    ("used_max", r"max (used as $\Delta_c$)", "D", "white"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/paper/delta_s_gap_distribution.png"))
    args = ap.parse_args()

    d = json.loads(args.metrics.read_text())
    adj = d.get("naive_adjustment")
    if not adj:
        raise SystemExit(f"no 'naive_adjustment' block in {args.metrics}")
    coords = [c for c in d.get("coords", list(adj)) if c in adj]

    fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(coords) + 1.05))
    ys = list(range(len(coords)))[::-1]

    for y, c in zip(ys, coords):
        vals = [float(adj[c][k]) for k, *_ in STATS]
        # Range spine first, so markers sit on top of it.
        ax.plot([min(vals), max(vals)], [y, y], color="0.75", lw=1.4, zorder=1,
                solid_capstyle="round")
        for (key, _lab, mk, fc) in STATS:
            ax.plot(float(adj[c][key]), y, marker=mk, ms=8, mfc=fc, mec="0.15",
                    mew=1.2, ls="none", zorder=3)
        ax.annotate(f"{float(adj[c]['median']):.1e}",
                    (float(adj[c]["median"]), y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, color="0.30")
        ax.annotate(f"{float(adj[c]['used_max']):.3g}",
                    (float(adj[c]["used_max"]), y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, color="0.15")

    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(c, c) for c in coords])
    ax.set_xscale("log")
    ax.set_xlabel(r"surrogate-label gap $\Delta s$  (coordinate units, log scale)")
    ax.grid(axis="x", which="major", color="0.88", lw=0.7, zorder=0)
    ax.grid(axis="x", which="minor", color="0.94", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(coords) - 0.4)

    handles = [plt.Line2D([], [], marker=mk, ms=8, mfc=fc, mec="0.15", mew=1.2,
                          ls="none", label=lab) for _k, lab, mk, fc in STATS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, handletextpad=0.4,
              columnspacing=1.6)

    # The caveat goes in the LaTeX caption, not baked into the PNG: any
    # in-figure placement collides with the xlabel once bbox_inches="tight"
    # recomputes the extent, and a caption is where a paper states it anyway.
    n_tune = d.get("n_tune")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print("CAPTION NOTE (put this in the LaTeX caption, it is not in the PNG): "
          "three summary statistics per coordinate over the tuning set"
          + (f" (n = {n_tune})" if n_tune else "") + ", not a full distribution.")
    for c in coords:
        a = adj[c]
        print(f"  {c:9s} median={a['median']:.3e}  p90={a['p90']:.3e}  "
              f"max={a['used_max']:.4g}  (max/median = "
              f"{a['used_max'] / max(a['median'], 1e-12):.3g}x)")


if __name__ == "__main__":
    main()
