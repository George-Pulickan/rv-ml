# NEXT SESSION — run the cluster job

Working note for whoever (or whatever) picks this up next, including on the
Windows machine. Sanitised on purpose: this file is tracked and **`main` is
public**, so no hostnames, accounts, or credentials live here. Those are in
`handover.md`, which is gitignored and kept in a secret gist — get it from
there and save it to the repo root before doing anything cluster-related.

## State as of 2026-07-26

Everything Nicolò asked for in his 2026-07-26 reply is **implemented and merged
into `main`** (tip `f997191`). No code work is outstanding. Tests green.

**The only remaining task is compute, and it has to run on the RHUL machines.**
Nothing on this list can be done on a laptop — see the house rule in
`handover.md` (smoke runs local, full scale on the cluster).

**This job was launched 2026-07-26 21:42 BST** and was still running at the time
of writing — see the 07-26 session note in `handover.md` for the cluster-state
problems that had to be cleared first, and for how to check on it.

## What to run

```bash
git checkout main && git pull        # f997191 or later
sed "s/^\([[:space:]]*\)srun /\1/" slurm/nicolo_20260726.sbatch | bash
```

**The `sed` is not optional, and do not simplify it to `s/^srun //`** — there is
no Slurm on the machines we can reach, so `srun` has to be stripped, and two of
the eight calls are indented inside `for` loops. A column-0 anchor leaves those
two, which under `set -e` kills the job at the first seed with nothing
produced. The `#SBATCH` headers are kept only so the file still works if a real
scheduler ever appears. Read the script's header comment before launching; it
explains each of the five steps.

Preconditions, both written up in `handover.md` under *Cluster access*:
**network access** (campus or CIM VPN — SSH times out off-campus), and a valid
**GitHub PAT** in the cluster clone's git config. The PAT was re-verified on
2026-07-26 and **expires 2026-08-13**; after that date reissue it before
running, or step 5 cannot push.

Note that step 5 also fails for a reason the PAT cannot fix: `rhul-results`
already exists on GitHub carrying the 2026-07-25 results, and has diverged from
`main`, so the push is rejected non-fast-forward. This is non-fatal — artifacts
stay on cluster disk — but they need pushing to a fresh branch or rsyncing off
by hand.

### How long it takes

Measured on the CIM ts-nodes 2026-07-26, and extrapolated from the 07-25 run:

| Step | Time |
|---|---|
| 1 — ψ retrain, 6 seeds | **~65 s/seed, ~7 min total** (measured) |
| 2 — six CP runs at n_cal = 400 | ~1 h each, **~6 h** (extrapolated: 07-25 ran two in ~2 h) |
| 3 — PKew control | included in the six above |
| 4 — Hessian sweep, both coord sets | ~1.5–2 h; the 4000-step draws dominate |

**Budget 8–10 h.** Steps 2–4 are extrapolations, not measurements — trust the
log over this table. Every earlier timing estimate in these docs ran ~4–5×
optimistic, so treat any unsourced number here with suspicion.

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
  it. Retraining a real one takes **~65 s** on a CIM ts-node
  (`python regression.py --feature-set 74 --epochs 200`, measured 2026-07-26 —
  the "~14 s" quoted here previously was wrong); pass `--checkpoint`
  explicitly everywhere.
- Coverage numbers quoted before commit `1517033` came from a tree that no
  longer exists (three fixes had been reported as done but were never
  committed). They will change on this rerun. Do not reconcile against them.
