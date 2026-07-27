# NEXT SESSION — the cluster run is done; write the paper

Working note for whoever (or whatever) picks this up next. Sanitised on purpose:
this file is tracked and **`main` is public**, so no hostnames, accounts, or
credentials live here. Those are in `handover.md`, which is gitignored and kept
in a secret gist — get it from there and save it to the repo root before doing
anything cluster-related.

## State as of 2026-07-27

**The 2026-07-26 cluster job finished cleanly: `exit=0` at 07:04:30 BST on
2026-07-27**, ~9¼ h wall. All six CP runs and both Hessian sweeps completed, no
tracebacks. Code work remains done (`main` @ `72a7b83` or later); tests green.

**Nothing is waiting on compute any more. The remaining work is writing.**

## Results

All figures below are surrogate + papernorm at α = 0.10, joint over the reported
coordinates, from `synthetic_generation/regression/`.

| Run | n_real | syn cov | real cov | real cov (LR-wt) | vacuous | ESS |
|---|---|---|---|---|---|---|
| `pke_20260726/gamma_syn` | 51 | 0.9450 | 0.9804 | 0.9804 | 0.0 | 288.3 |
| `pke_20260726/gamma_real_val` | 51 | 0.9400 | 0.9216 | 0.9216 | 0.0 | 288.3 |
| `pke_20260726/gamma_real_val_1perhost` | **34** | 0.9400 | **0.9706** | 0.9706 | 0.0 | 261.9 |
| `pke_20260726/gamma_real_val_nofilter` | 57 | 0.9400 | 0.9298 | 0.9298 | 0.0 | 289.7 |
| `pke_20260726/gamma_real_val_gapbeta02` | 51 | 0.9400 | 0.9216 | 0.9216 | 0.0 | 288.3 |
| `pkew_20260726` | 51 | 0.9475 | 0.9804 | 0.9608 | 0.0 | 288.3 |

Median half-widths at α = 0.10 (`gamma_real_val`): log10_P **1.608 dex**,
log10_K **1.089 dex**, e **0.902**. The 1-per-host variant is tighter in P
(1.437) and wider in e (0.942).

**`papernorm_weight` w = 1.00 for the surrogate score in all six runs** —
at n_cal = 400, not just smoke. This is the confirmation Nicolò was waiting on:
δ_y contributes nothing once δ_c is available. (Two incidental exceptions on
other score types: `naive` picks 0.75 in `1perhost`, `naive_adj` picks 0.75 in
`pkew`.)

**Vacuous fraction is 0.0 in every run**, confirming the float32 time-quantisation
fix holds at full scale (it was 1.8% before).

### Sanity bars — three met, one not

| Bar | Outcome |
|---|---|
| `papernorm_weight` w = 1.00 at n_cal = 400 | ✅ all six runs |
| joint coverage ≈ 0.90 at α = 0.1 | ✅ 0.94 syn / 0.92 real — conservative, the expected direction under Bonferroni |
| one series per host = 34 independent points | ✅ 34 (vs 51 filtered, 57 unfiltered) |
| PD fraction rises ≈ 0.64 → 0.76 at 200 steps | ❌ **0.88 → 0.88, no rise** |

The failed bar failed *favourably*: both coordinate sets sit at 0.88, far above
the predicted values, so the expected gain from dropping ω could not show up in
this metric — PKew was already fine. Dropping ω still helps by every other
measure: median λ_min 29.7 vs 5.9, κ_median 7.4k vs 31.3k.

### Assumption 3.2 / Theorem 3.6 — the old "vacuous" claim is superseded

`figures/paper/assumption32/hessian_PKe.json`:

| steps | PD fraction | λ_min median | √(C_H·ε) |
|---|---|---|---|
| 200 | 0.88 | 29.705 | **0.454** |
| 1000 | 0.84 | 30.943 | 0.444 |
| 4000 | 0.88 | 32.227 | 0.435 |

with ε = 6.111 (`noise_filter.bound_rv_std`). **The previously quoted
√(C_H·ε) = 6.71 — "larger than the support of e, so Theorem 3.6 is vacuous" —
no longer holds.** At 0.45 the bound constrains e to under half its range, which
is meaningful.

The real limitation is different: **the assumption fails outright on 12% of
draws** (λ_min goes negative, min −0.18 at 200 steps). State it that way, not as
vacuity.

⚠️ **√(C_H·ε) is a derived quantity — no pipeline output reports it.** It was
computed as `sqrt(eps / lambda_min_median)` across two files. Re-derive before
quoting it in the paper.

## What to do next

1. Fill the `[RERUN]` placeholders in `paper/end_to_end_pipeline.tex` (5 markers)
   from the table above. The numbers there now are smoke-scale and not quotable.
2. Send Nicolò the follow-up with the real numbers. **Compose it fresh** — the
   07-26 email already went out and both reply drafts were deleted when it did,
   so there is no draft to fill in.
3. Regenerate `figures/paper/earthlike_top10.*` from the new ψ, then commit.
4. **Write §4.** It currently contains only the perturbed-pendulum toy; the
   exoplanet experiment the Abstract and Intro promise is absent from the paper
   entirely. This is the real remaining work.
5. Paste the 18 corrections in `paper_corrections.md` (gitignored) into Overleaf.

## Getting the results off the cluster

Step 5 of the sbatch tried to push to `rhul-results` and **failed as expected** —
that branch carries the 07-25 results, is based on an older commit, and has
diverged from `main`, so the push is rejected non-fast-forward. This is not a
PAT problem and is non-fatal: **the artifacts are intact on cluster disk.**
Push them to a fresh dated branch or copy them off by hand.

## Re-running it (only if needed)

```bash
git checkout main && git pull
sed "s/^\([[:space:]]*\)srun /\1/" slurm/nicolo_20260726.sbatch | bash
```

**The `sed` is not optional, and do not simplify it to `s/^srun //`** — two of
the eight `srun` calls are indented inside `for` loops, and a column-0 anchor
leaves them, which under `set -e` kills the job at the first seed with nothing
produced. Preconditions: campus network or CIM VPN, and a valid GitHub PAT
(**expires 2026-08-13**). Launch with `at`, not `nohup` — see `handover.md`.

Measured timings: ψ retrain **~65 s/seed** (~7 min for six); each CP run **~1 h**;
the whole job **~9¼ h**.

**A quiet log does not mean a dead job.** `conformal_shift.py` emits nothing
during calibration — gaps of 10–40 min are normal. Check accumulated CPU time
(`ps -o etime,time -p <pid>`), not log mtime. And note `%CPU` in `ps` is a
lifetime average; diff CPU against wall across two samples for the real rate.

## Two things not to trip over

- `checkpoints/regression_mlp_74.pt` in the repo is a **15-epoch smoke artifact**
  that predicts near the mean. Never generate paper artifacts with it. Retraining
  a real one takes ~65 s on a CIM ts-node
  (`python regression.py --feature-set 74 --epochs 200`); pass `--checkpoint`
  explicitly everywhere.
- **The 07-26 results use a different JSON schema.**
  `regression/paper_20260726-0946/` records `{"gamma0", "mix"}` and has **no
  `papernorm_weight` key at all**; this run records `papernorm_weight` with `w`.
  Same quantity, different name — a script reading `papernorm_weight` finds
  nothing in the older files and may report a default rather than failing.
