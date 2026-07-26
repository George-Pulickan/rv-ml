# NEXT SESSION — run the cluster job

Working note for whoever (or whatever) picks this up next, including on the
Windows machine. Sanitised on purpose: this file is tracked and **`main` is
public**, so no hostnames, accounts, or credentials live here. Those are in
`handover.md`, which is gitignored and kept in a secret gist — get it from
there and save it to the repo root before doing anything cluster-related.

## State as of 2026-07-26

Everything Nicolò asked for in his 2026-07-26 reply is **implemented and
committed** on branch `nicolo-20260726`. No code work is outstanding.
51/51 tests green.

**The only remaining task is compute, and it has to run on the RHUL machines.**
Nothing on this list can be done on a laptop — see the house rule in
`handover.md` (smoke runs local, full scale on the cluster).

## What to run

```bash
git checkout nicolo-20260726
sed "s/^srun //" slurm/nicolo_20260726.sbatch | bash
```

**The `sed` is not optional** — there is no Slurm on the machines we can reach,
so `srun` has to be stripped. The `#SBATCH` headers are kept only so the file
still works if a real scheduler ever appears. Read the script's header comment
before launching; it explains each of the five steps.

Two preconditions must hold or the job fails silently. Both are written up in
`handover.md` under *Cluster access*: a **fresh GitHub PAT** in the cluster
clone's git config (the previous one expired and the auto-push to the results
branch fails without a word), and **network access** to the cluster.

Expect a long run. The 4000-step Hessian sweep is the slow part.

## What it produces, and the bar each output must clear

| Output | Sanity bar |
|---|---|
| `regression/pke_20260726/gamma_real_val/` | joint coverage near 0.90 at α = 0.1 over (P, K, e) |
| `.../gamma_real_val_1perhost/` | **the headline number** — one series per star, 34 independent points |
| `.../gamma_real_val_nofilter/` | quantifies what the min-nights filter changed |
| `.../gamma_real_val_gapbeta02/` | Δ_c as a conformal quantile with a stated budget |
| `pkew_20260726/` | ω-included control; shows what dropping ω bought |
| `figures/paper/assumption32/hessian_{PKe,PKew}.json` | PD fraction should rise ≈ 0.64 → 0.76 at 200 steps |
| `papernorm_weight` in the metrics JSON | smoke picks w = 1.00 — **confirming this at n_cal = 400 is what Nicolò is waiting on** |

Check **mtimes, not existence**. Several older helper scripts have
precondition checks that pass happily on stale artifacts, so a dead job can
look like a successful one.

## Only after the results land

1. Fill the `[RERUN]` placeholders in `paper/end_to_end_pipeline.tex`. They are
   marked in comments specifically so they cannot be pasted into Overleaf
   early — the numbers there now are smoke-scale and not quotable.
2. Fill the `[rerun]` markers in the reply draft (gitignored) and send it.
3. Regenerate `figures/paper/earthlike_top10.*` from the **new** ψ, then commit
   those artifacts. They are deliberately untracked right now.

## Two things not to trip over

- `checkpoints/regression_mlp_74.pt` in the repo is a **15-epoch smoke
  artifact** that predicts near the mean. Never generate paper artifacts with
  it. Retraining a real one takes ~14 s
  (`python regression.py --feature-set 74 --epochs 200`); pass `--checkpoint`
  explicitly everywhere.
- Coverage numbers quoted before commit `1517033` came from a tree that no
  longer exists (three fixes had been reported as done but were never
  committed). They will change on this rerun. Do not reconcile against them.
