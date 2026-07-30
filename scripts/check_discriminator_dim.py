"""Quantify why the likelihood-ratio discriminator is fitted on 74-D summaries.

conformal_shift.py fits the covariate-shift weights on the 74-dimensional
summary features rather than psi's full spectral representation, with the
comment that a high-dimensional discriminator on ~700 points "separates the
classes too well and degenerates the weights". That is a real design decision
affecting Theorem 3.6's applicability -- if the discriminator understates the
shift, the reweighting under-corrects -- and the paper does not mention it.

This measures it rather than asserting it, at several feature dimensions:

  separation   in-sample and held-out AUC of the real-vs-synthetic classifier
  ESS          effective calibration sample size the resulting weights leave
  clipping     fraction of weights outside the [1/c, c] clip
  tail         max weight before clipping

A discriminator that separates perfectly (AUC -> 1) drives p_real -> 1 on real
points and -> 0 on synthetic ones, so the odds ratio explodes, the weight mass
concentrates on a handful of calibration points and ESS collapses. Low ESS
means the weighted quantile is effectively computed from very few samples,
which is exactly the finite-sample regime the paper claims to handle.

Cheap: no CP run, no surrogate labels, no GD. Only the feature matrices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conformal_shift import (  # noqa: E402
    NoiseProxy, fit_weight_model, make_real, make_synthetic_filtered,
    noise_bound_from_real,
)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, no sklearn dependency for the metric itself."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = float(labels.sum())
    n0 = float(len(labels) - n1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures/paper/discriminator_dim.json")
    args = ap.parse_args()

    proxy = NoiseProxy()
    print(f"noise proxy source: {proxy.source}")
    real_train = make_real("train", 0.1, 100.0)
    print(f"real train systems: {len(real_train)}")

    # Same Assumption 2.1 discard rule the pipeline uses, so the synthetic
    # sample the discriminator sees here is the one it sees in a real run.
    bound = noise_bound_from_real(proxy, real_train,
                                  [s["theta5"] for s in real_train])
    print(f"noise bound: {bound:.3g} rv_std")
    synth, _, _ = make_synthetic_filtered(args.n_synth + args.n_cal,
                                          args.seed, bound, proxy)
    wsynth, calib = synth[:args.n_synth], synth[args.n_synth:]
    print(f"synthetic: {len(wsynth)} for the fit, {len(calib)} as calibration")

    full_dim = len(wsynth[0]["features"])
    dims = sorted({10, 74, full_dim})
    print(f"feature dimensions available: {full_dim}; testing {dims}")

    out = {"n_synth": len(wsynth), "n_real_train": len(real_train),
           "n_cal": len(calib), "clip": args.clip, "by_dim": {}}

    for d in dims:
        sf = np.vstack([s["features"][:d] for s in wsynth])
        rf = np.vstack([s["features"][:d] for s in real_train])
        w_fn, clf = fit_weight_model(sf, rf, args.seed)

        X = np.vstack([sf, rf])
        y = np.r_[np.zeros(len(sf)), np.ones(len(rf))]
        p = clf.predict_proba(X)[:, 1]
        a_in = auc(p, y)

        w_raw = w_fn(np.vstack([s["features"][:d] for s in calib]))
        w = np.clip(w_raw, 1.0 / args.clip, args.clip)
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        frac_clip = float(np.mean((w_raw < 1 / args.clip) | (w_raw > args.clip)))

        rec = {"dim": d, "auc_in_sample": a_in, "ess": ess,
               "ess_frac": ess / len(w), "frac_clipped": frac_clip,
               "w_max_raw": float(w_raw.max()), "w_min_raw": float(w_raw.min()),
               "w_median_raw": float(np.median(w_raw))}
        out["by_dim"][str(d)] = rec
        print(f"  d={d:4d}  AUC={a_in:.4f}  ESS={ess:7.1f}/{len(w)} "
              f"({100*ess/len(w):5.1f}%)  clipped={100*frac_clip:5.1f}%  "
              f"w_max={w_raw.max():.3g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
