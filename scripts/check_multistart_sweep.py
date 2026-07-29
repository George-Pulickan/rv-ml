"""Does multi-start GD fix the TAIL of the conformity score?

The 2026-07-28 ladder showed that refining psi with GD alone makes the score
worse: psi(y) starts in theta*'s basin on only 2.5-8% of curves, so descending
just fits the wrong alias better. Refinement sharpens whichever basin it is
handed; it cannot change basins.

It also showed where the interval width actually comes from. Median
|psi(y) - theta*| on log10 P is 0.18 dex, but the reported half-width is 1.6 dex
-- so the alpha = 0.10 quantile is set entirely by the tail of curves where psi
picked the wrong alias. Median accuracy is irrelevant to the width; only the
tail matters.

This sweeps K (GD steps) x M (periodogram restarts) and reports the QUANTILE of
the score, not the median, because the quantile is what becomes the half-width.
It deliberately does NOT run the CP pipeline: the question is whether the tail
moves at all, and a full run per configuration would cost ~11 h instead of ~2.

If the 0.90 quantile does not fall with M, the whole refinement route is dead
and we should say so before spending days on it.

Usage
-----
    python scripts/check_multistart_sweep.py \
        --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
        --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
        --n-syn 300 --ks 20,50 --ms 0,3,5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_generation"))

from conformal import make_real, make_synthetic  # noqa: E402
from conformal_shift import (  # noqa: E402
    _load_mlp_psi,
    multistart_fit_gd,
    surrogate_fit_gd,
)
from feature_columns import TARGET_COLUMNS  # noqa: E402
from models.kepler_torch import KeplerDecoder  # noqa: E402

COORDS3 = ["log10_P", "log10_K", "e"]


def feat_matrix(systems: list, feature_cols: list[str]) -> np.ndarray:
    def _val(fr: dict, lsp: np.ndarray, c: str) -> float:
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)
    return np.asarray(
        [[_val(s["feat_row"], s["lsp"], c) for c in feature_cols] for s in systems],
        dtype=float)


def score_stats(refined: list, theta_star: np.ndarray, alphas=(0.10, 0.32)) -> dict:
    """Per-coordinate score |psi' - theta*|: median and upper quantiles.

    The 1-alpha quantile is (up to normalisation and the Bonferroni divisor)
    what becomes the reported half-width, so it is the number to watch.
    """
    d = np.abs(np.asarray(refined, dtype=float)[:, :3] - theta_star[:, :3])
    out = {"median": {}, "same_basin_all3_tol0.05": float(np.mean((d <= 0.05).all(axis=1)))}
    for k, c in enumerate(COORDS3):
        out["median"][c] = float(np.median(d[:, k]))
    for a in alphas:
        q = 1.0 - a / 3.0  # Bonferroni divisor d = 3, as the paper uses
        out[f"q{a:.2f}"] = {c: float(np.quantile(d[:, k], q))
                            for k, c in enumerate(COORDS3)}
    return out


def run_domain(name, systems, theta_ref, psi_y, decoder, ks, ms, gd_steps, gd_lr):
    print(f"\n=== {name}: {len(systems)} systems ===")
    print(f"theta* : GD {gd_steps} steps from reference ...")
    t0 = time.time()
    theta_star = np.stack(surrogate_fit_gd(decoder, list(theta_ref), systems,
                                           gd_steps, gd_lr))
    print(f"  done in {time.time() - t0:.0f}s")

    res = {}
    base = score_stats(psi_y, theta_star)
    res["K0_M0"] = base
    print(f"  {'config':<12} {'med P':>8} {'q.90 P':>8} {'q.90 K':>8} {'q.90 e':>8} {'basin':>7}")
    print(f"  {'K=0 M=0':<12} {base['median']['log10_P']:>8.4f} "
          f"{base['q0.10']['log10_P']:>8.4f} {base['q0.10']['log10_K']:>8.4f} "
          f"{base['q0.10']['e']:>8.4f} {base['same_basin_all3_tol0.05']:>7.3f}")

    for K in ks:
        for M in ms:
            t0 = time.time()
            refined = multistart_fit_gd(decoder, list(psi_y), systems, K, gd_lr, M)
            st = score_stats(refined, theta_star)
            st["seconds"] = round(time.time() - t0, 1)
            res[f"K{K}_M{M}"] = st
            print(f"  {'K=%d M=%d' % (K, M):<12} {st['median']['log10_P']:>8.4f} "
                  f"{st['q0.10']['log10_P']:>8.4f} {st['q0.10']['log10_K']:>8.4f} "
                  f"{st['q0.10']['e']:>8.4f} {st['same_basin_all3_tol0.05']:>7.3f}"
                  f"   [{st['seconds']:.0f}s]")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n-syn", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ks", default="20,50")
    ap.add_argument("--ms", default="0,3,5")
    ap.add_argument("--gd-steps", type=int, default=200)
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures" / "paper" / "multistart_sweep.json")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ms = [int(x) for x in args.ms.split(",") if x.strip()]
    decoder = KeplerDecoder().eval()
    psi_predict, _ = _load_mlp_psi(args.checkpoint, torch.device("cpu"))
    feature_cols = [c for c in pd.read_csv(args.csv, nrows=1).columns
                    if c not in TARGET_COLUMNS]

    report = {"checkpoint": str(args.checkpoint), "ks": ks, "ms": ms,
              "gd_steps": args.gd_steps, "gd_lr": args.gd_lr, "domains": {}}

    syn = make_synthetic(args.n_syn, args.seed)
    psi_syn = psi_predict(feat_matrix(syn, feature_cols))
    report["domains"]["synthetic"] = run_domain(
        "synthetic", syn, [s["theta5"] for s in syn], psi_syn,
        decoder, ks, ms, args.gd_steps, args.gd_lr)

    real = make_real(args.real_split, 0.1, 100.0)
    psi_real = psi_predict(feat_matrix(real, feature_cols))
    report["domains"]["real"] = run_domain(
        f"real/{args.real_split}", real, [s["theta5"] for s in real], psi_real,
        decoder, ks, ms, args.gd_steps, args.gd_lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
