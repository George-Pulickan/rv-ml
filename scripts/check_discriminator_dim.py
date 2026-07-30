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


def fit_weight_model_C(synth_feats: np.ndarray, real_feats: np.ndarray,
                       seed: int, C: float):
    """`conformal_shift.fit_weight_model` with the L2 strength exposed.

    That function hard-codes C=1.0. The ESS collapse at high dimension is a
    variance problem -- an under-regularised logistic fit spreads the logits, the
    odds ratio explodes and the weight mass concentrates -- so C is the natural
    dial to test against. Kept byte-compatible with the original otherwise, so
    C=1.0 here reproduces it exactly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([synth_feats, real_feats])
    y = np.r_[np.zeros(len(synth_feats)), np.ones(len(real_feats))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=C, random_state=seed))
    clf.fit(X, y)
    n0, n1 = float(len(synth_feats)), float(len(real_feats))

    def w(feats: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(feats)[:, 1].clip(1e-6, 1 - 1e-6)
        return (p / (1.0 - p)) * (n0 / n1)

    return w, clf


def heldout_auc(sf: np.ndarray, rf: np.ndarray, seed: int, C: float,
                folds: int = 5) -> float:
    """Cross-fitted AUC: the separation that generalises.

    The in-sample AUC is not comparable across dimensions -- a 74-parameter fit
    on ~640 points overfits more than a 10-parameter one, so 74-D scoring higher
    in sample says nothing about whether it captures more real covariate shift.
    Only an out-of-fold AUC answers that, and it is the number the module
    docstring always claimed to report.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([sf, rf])
    y = np.r_[np.zeros(len(sf)), np.ones(len(rf))]
    idx = rng.permutation(len(X))
    oof = np.zeros(len(X))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        _, clf = fit_weight_model_C(X[tr][y[tr] == 0], X[tr][y[tr] == 1], seed, C)
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return auc(oof, y)


def weight_stats(w_raw: np.ndarray, clip: float) -> dict:
    w = np.clip(w_raw, 1.0 / clip, clip)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return {"ess": ess, "ess_frac": ess / len(w),
            "frac_clipped": float(np.mean((w_raw < 1 / clip) | (w_raw > clip))),
            "w_max_raw": float(w_raw.max()), "w_min_raw": float(w_raw.min()),
            "w_median_raw": float(np.median(w_raw))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dims", type=int, nargs="*", default=None,
                    help="feature dimensions to test; default 5 10 20 40 and "
                         "the full width. NOTE the summary features are only 74-D, "
                         "so there is no dimension above 74 to test -- the "
                         "'full spectral representation' the conformal_shift.py "
                         "comment appeals to is not reachable from s['features'].")
    ap.add_argument("--c-grid", type=float, nargs="*",
                    default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                    help="L2 strengths to sweep at the full dimension")
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
    dims = args.dims or sorted({5, 10, 20, 40, full_dim})
    dims = [d for d in dims if d <= full_dim]
    print(f"feature dimensions available: {full_dim}; testing {dims}")
    if full_dim <= 74:
        print("NOTE s['features'] is only "
              f"{full_dim}-D, so no dimension above {full_dim} is reachable here; "
              "the 'separates too well at high dimension' claim in "
              "conformal_shift.py cannot be tested against a wider "
              "representation by this script.")

    cal_feats_full = np.vstack([s["features"] for s in calib])
    out = {"n_synth": len(wsynth), "n_real_train": len(real_train),
           "n_cal": len(calib), "clip": args.clip, "full_dim": full_dim,
           "by_dim": {}, "by_C_at_full_dim": {}}

    print("\n=== separation vs weight degeneracy by dimension (C=1.0) ===")
    print(f"  {'d':>4}  {'AUC_in':>7} {'AUC_oof':>8}  {'ESS':>15}  "
          f"{'clip%':>6}  {'w_max':>8}")
    for d in dims:
        sf = np.vstack([s["features"][:d] for s in wsynth])
        rf = np.vstack([s["features"][:d] for s in real_train])
        w_fn, clf = fit_weight_model(sf, rf, args.seed)

        X = np.vstack([sf, rf])
        y = np.r_[np.zeros(len(sf)), np.ones(len(rf))]
        a_in = auc(clf.predict_proba(X)[:, 1], y)
        a_oof = heldout_auc(sf, rf, args.seed, C=1.0)

        w_raw = w_fn(cal_feats_full[:, :d])
        rec = {"dim": d, "auc_in_sample": a_in, "auc_out_of_fold": a_oof,
               **weight_stats(w_raw, args.clip)}
        out["by_dim"][str(d)] = rec
        print(f"  {d:4d}  {a_in:7.4f} {a_oof:8.4f}  {rec['ess']:7.1f}/{len(w_raw)} "
              f"({100*rec['ess_frac']:4.1f}%)  {100*rec['frac_clipped']:5.1f}%  "
              f"{rec['w_max_raw']:8.3g}")

    print(f"\n=== L2 strength at d={full_dim}: can regularisation buy back ESS "
          f"without losing real separation? ===")
    print(f"  {'C':>6}  {'AUC_in':>7} {'AUC_oof':>8}  {'ESS':>15}  "
          f"{'clip%':>6}  {'w_max':>8}")
    sf = np.vstack([s["features"] for s in wsynth])
    rf = np.vstack([s["features"] for s in real_train])
    X = np.vstack([sf, rf])
    y = np.r_[np.zeros(len(sf)), np.ones(len(rf))]
    for C in args.c_grid:
        w_fn, clf = fit_weight_model_C(sf, rf, args.seed, C)
        a_in = auc(clf.predict_proba(X)[:, 1], y)
        a_oof = heldout_auc(sf, rf, args.seed, C=C)
        w_raw = w_fn(cal_feats_full)
        rec = {"C": C, "auc_in_sample": a_in, "auc_out_of_fold": a_oof,
               **weight_stats(w_raw, args.clip)}
        out["by_C_at_full_dim"][str(C)] = rec
        print(f"  {C:6.3g}  {a_in:7.4f} {a_oof:8.4f}  {rec['ess']:7.1f}/{len(w_raw)} "
              f"({100*rec['ess_frac']:4.1f}%)  {100*rec['frac_clipped']:5.1f}%  "
              f"{rec['w_max_raw']:8.3g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
