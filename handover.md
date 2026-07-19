# Handover — RV-ML Project

## TL;DR for the next Claude

You're continuing summer research with George Pulickan (Notre Dame CS+Stats freshman) and advisor Nicolò Colombo (Royal Holloway). Goal: publish a paper — ML system that predicts Kepler orbital parameters from RV time series, with conformal vs Bayesian UQ.

**This session's focus:** synthetic dataset diagnostics for a team meeting today. Jovie reported an eccentricity-distribution mismatch between her synthetic samples (300 systems generated with `synthetic_rv.py`) and NASA data. The professor's next steps were: (i) overlay the exact Keplerian curve on synthetic observations and (ii) train a binary classifier to discriminate real vs synthetic. **Both done. Also discovered and fixed two real bugs in `synthetic_dataset.py`.**

George is terse and technical — match that. No over-explaining.

> **NOTE (updated 2026-07-15):** Read **"SESSION 2026-07-15"** first (on-arrival RHUL triage plan), then 07-14 / 07-12 / 07-11 / 07-06 / 07-04 / 07-02 / 07-01 / 06-30 in order. Everything below that is older context.

---

## SESSION 2026-07-15 — no rhul-results yet; on-arrival triage plan

As of 07-15: **`rhul-results` branch still absent** on GitHub (~24h after the 07-14 12:20 BST
launch; job 1 est. 6–14h) and linux.cim SSH times out off-campus. Job 1 may be slow or dead —
unknowable until George is on campus (or gets CIM VPN via the new UROP setup, see below).

**DO NOT blind-fire the campus follow-up one-liner. Diagnose first:**

```
ssh VPAC005@linux.cim.rhul.ac.uk 'hostname; cd ~/rv-ml; echo == procs ==; pgrep -fl "run_jobs_direct|gp_residual|conformal_shift|regression"; echo == runner.log ==; tail -25 slurm/logs/runner.log; echo == log mtimes ==; ls -lt slurm/logs | head -8; for f in $(ls -t slurm/logs/direct-*.log 2>/dev/null | head -2); do echo == tail $f ==; tail -12 "$f"; done; echo == freshness ==; ls -l models/gp_residual_svgp.pt synthetic_generation/datasets/*.csv; git log --oneline -2'
```

Then branch on what it shows (logs are NFS-shared, so mtimes are node-independent truth;
pgrep is NOT — linux.cim fronts 3 ts-nodes and shows only the node you landed on):

1. **pgrep sees the runner** → original plan holds: fire the 07-14 part-7 one-liner from
   that same session (its wait-loop watches the same node).
2. **pgrep empty but direct-*.log mtime fresh (<~20 min)** → runner is alive on a *sibling
   node*. Firing the one-liner here would merge origin/main UNDER the running job (its
   pgrep wait exits immediately). Instead ssh again until you land on the right node
   (or ssh cim-ts-node-01/02/03 directly), then treat as case 1.
3. **log mtimes stale + no completion marker in runner.log** → job died. Diagnose from the
   log tails (OOM / admin kill / crash). **Don't run the follow-up**: run_cp_rerun's
   existence preconditions pass on STALE artifacts (old checkpoint is committed; LSP CSV
   was rsynced), and run_campus_followup would then run the benchmark on the stale
   7-feature checkpoint → plausible-looking wrong numbers. Verify freshness by MTIME
   (checkpoint + CSVs must postdate 07-14 12:20), rerun job 1 from where it failed
   (`sed "s/^srun //" slurm/gp_conformal.sbatch | bash` after the merge), then follow up.

**UROP/Run:AI update (from Nicolò's helpdesk ticket 0016212, fwd'd by George 07-15):**
a shared GPU JupyterLab/VS Code environment now exists (rhul.run.ai, 24GB pods, storage
`/mnt/urop-2026`, CIM VPN required; Francesco emailing students setup instructions). This is
irrelevant to the current CPU jobs on the ts-nodes, but is (a) the future home for
`train_encoder.sbatch` (needs CUDA — ts-nodes have none) and (b) once George has an RHUL
account + CIM VPN, off-campus checking of the ts-node jobs becomes possible.

PAT in the RHUL clone's .git/config expires ~07-21 — still valid; revoke after results merge.

---

## SESSION 2026-07-14 — e-head zero-inflation counters (Nicolò's 07-12 open item) @ `7a7e3b2`

Closes the "resample/balance or classify-then-regress" open item by implementing **both**
behind opt-in flags (defaults bit-identical to before — verified: same RNG draws, same loss
graph, `predict()` reproduces the old inline denorm):

1. **`--e-balance`** — `theta_loss.e_balance_weights`: inverse-frequency reweighting of the
   e-dim loss. e==0 point mass is its own category + 20 bins over (0,1]; weights capped at
   10× the most-populated category, normalized to mean 1 over train; val weights use
   train-fitted bins (consistent early-stopping objective). Fit on has_ecc rows only.
2. **`--e-head hurdle`** — classify-then-regress: 6th output = e>0 logit (BCE, has_ecc-masked,
   `--hurdle-bce-weight` default 1.0); e-regression dim masked to e>0 rows; at predict
   e=0 where P(e>0)<0.5 (combine only in physical units — skipped when denorm_targets=False,
   so the raw-output saturation diagnostic still sees the raw head). Checkpoints carry
   `out_dim`/`e_head` in norm_stats; both rebuild sites (`load_checkpoint_and_predict_val`,
   diagnostics `_load_model`) read `out_dim` with default 5 → old checkpoints keep working.
3. **`slurm/regression_benchmark.sbatch` step 6** — full-scale comparison on 109-D:
   balance / hurdle / hurdle+balance → `figures/regression_synthetic/e_head_*/metrics.json`;
   baseline = gate_b_109_oracle in benchmark.json. New `e_zero_classifier` metrics block
   (acc / recall_zero / precision_zero) when hurdle.
4. **Bugfix (pre-existing, hit in smoke):** main's real-transfer path crashed for feature-set
   74 — `real_df.get("has_t_peri", pd.Series(0.0))` is length-1 when the column is missing →
   boolean-mask IndexError. Now full-length zeros fallback.
5. Guarded by `tests/test_e_head.py` (8 tests: balance-weight properties + train/round-trip
   both modes). Full suite 16/16 green. NB `unittest discover` still clobbers
   `figures/synthetic_plots/` (test_generate_300) — checked out post-run as usual.

**Upstream pulled first (George's ask):** `9c735a6..d872a7b` — Shuaib/Daksh centralised
feature columns into new `feature_columns.py`; regression.py now imports from it. No conflicts.

Smoke run (15 ep, 74-D, hurdle+balance): pipeline end-to-end OK; classifier frac_true_zero
0.2395 matches the known 23.6%; recall_zero ~0 at that scale is expected (74-D has little e
signal — the real test is 109-D full-scale on the cluster). **Numbers not meaningful — don't quote.**

**Still pending:** both RHUL jobs (gp_conformal FIRST, then regression_benchmark — now
includes the e-head comparison); RHUL access checklist unchanged from 07-12.

### Same session, part 2 — paper alignment audit + paper-spec CP additions @ `bdb212c`

Nicolò's draft ("Simulation-based Conformal Prediction for Parameter Estimation") audited
against the repo. **The draft = conformal_shift.py's spec; conformal.py's unsupervised
reconstruction-score CP (E1/E2, --profile) is NOT in it** — Nicolò must decide if that gets a
section or drops (determines whether the full-scale profiled run stays queued). Full gap list
is in the 2026-07-14 message drafted for Nicolò (see chat; George forwards).

Implemented (all default-on, flow into gp_conformal.sbatch automatically since it runs bare
`python conformal_shift.py`):
1. **papernorm** — the draft's eqs 18–24 literally: delta_c = RF of the re-encode residual
   |psi_c(h(psi(y))) − psi_c(y)| (recon keeps the observation's time grid + sigmas, normalized
   by its own std; `reencode_features`), delta_y = RF of mean_t|y−h(psi(y))| in rv_std units;
   fit on a fresh supplementary synthetic set (`dnorm`, n_tune-sized, the paper's D_theta/D_y);
   4th norm variant alongside raw/vnorm/v2norm, gamma tuned identically.
2. **naive_adj strategy** — eq 41 quantile adjustment: naive calibration scores shifted by
   Delta_c = max over tune of |theta_bar_c − theta*_c| (shift of scores = shift of raw
   quantile); gap median/p90/max reported so the max choice can be revisited. Smoke: median
   gap = 0 (GD returns theta_bar when it can't improve), max 0.19–0.55.
3. **Assumption 2.1 filter** — synthetic draws with max_t|y−kepler(theta_bar)| > bound
   discarded (bound = max over real TRAIN of max_t|y−kepler(psi(y))|, rv_std units;
   `--no-noise-filter` to disable). **Smoke: bound≈6.0, rejects ~26% of draws** — it's cutting
   the GP's Student-t heavy-tail realizations real data never shows (scale-mixture finding
   again); NB this truncates the calibration distribution vs all previous runs — flag when
   comparing to old numbers.
4. **Assumption 2.3 constants** — kappa(H) via FD Hessian of the L2 recon loss (5-dim decoder
   parameterization, e clipped at FD boundary) + ||grad h|| via autograd Jacobian spectral
   norm, on --n-constants prior draws (default 25). Smoke: kappa(H)~1e6 (theory bound very
   loose — tell Nicolò), ||grad h|| med~380.

Smoke (n_cal=24 etc., 166s) end-to-end OK; inf=1.00 = known Bonferroni triviality at n<39.
Suite 16/16; synthetic_plots clobber restored again. Committed artifacts in
synthetic_generation/regression/ untouched (smoke wrote to scratchpad).

**Open per Nicolò's answers:** psi trained on surrogate labels (cluster-scale, add to sbatch
if he wants it); paper-side text fixes (L1 eq 4 vs L2 eq 9, p(t|theta) → bootstrap, 512-bin
LSP, RV-only).

### Same session, part 7 — Daksh's PR #5 reviewed + merged @ `357632c`

PR #5 (daksh/regression-head, properly rebased onto be9df5a this time): predicted-fold
two-step (`--stage2-fold {oracle,predicted,jitter}`, `--period-source {mlp74,lsp_peak,hybrid}`,
default now predicted+lsp_peak — SEMANTICS CHANGE for bare `--two-step`), ω scored only on
e>0.1 everywhere, `--period-tolerance` curve (ω MAE vs fold-period error), `--e-head-ablate`
(runs my 4 e-head variants + comparison.json — replaces my 3 explicit sbatch runs),
`--max-rows` debug flag, ω-vs-e + parameter-pair diagnostics, 2 new test files (25 total).
Reviewed clean; smoke-tested both modes; merged with a PR comment noting the one soft spot
(in-sample stage-1 preds feed train-fold periods → optimistic for mlp74 source; latent under
lsp_peak default). two_step_metrics.json keys renamed (stage2_109_train, two_step, note).

**Operational note:** the RUNNING RHUL job 2 is @ be9df5a → produces the OLD e_head_* dirs,
not e_head_ablate, and no period-tolerance/predicted-fold runs.

**THE campus one-liner (@ `0cb4c4b`, replaces run_cp_rerun-only launch):** waits for the
original runner, merges origin/main with `-X ours` (resolves the binary PNG conflict between
the job's artifact commits and PR #5's refreshed figures), then runs
`slurm/run_campus_followup.sh` = CP rerun (3 invocations) + PR #5 benchmark rerun, each
pushing to rhul-results:
```
ssh VPAC005@linux.cim.rhul.ac.uk 'cd ~/rv-ml && git fetch origin && nohup bash -c "while pgrep -f run_jobs_direct.sh >/dev/null; do sleep 600; done; git merge --no-edit -X ours origin/main && ./slurm/run_campus_followup.sh" > slurm/logs/followup-nohup.log 2>&1 & echo started'
```
Safe to fire immediately on arrival even if the original job is still running (it waits, and
does NOT pull until the job is done — pulling mid-job would swap code under job 2).

### Same session, part 6 — Nicolò's answers implemented @ `08eb329`

All six 07-14 questions answered (full text in chat). Decisions + what changed:
1. **papernorm deltas now POINTWISE** (his observation: h, psi deterministic → eqs 18/19 need
   no model). RF delta models + the dnorm supplementary set deleted; delta_c/delta_y computed
   per curve on every set. He expects v2norm (his Slack spec) to win — both still run.
2. **Reconstruction-score CP (conformal.py) OUT of the paper** — "let's focus on |psi(y)−theta*|".
   The full-scale profiled Keomega run is permanently descoped. conformal.py stays in repo.
3. **gamma on real val: approved** ("size-only validation, let's try") — already runs as the
   second CP invocation in the job; quote `gamma_real_val/` as primary once he confirms.
4. **Empirical Delta_c approved**; theoretical-bound implementation can stay out (constants
   reported anyway, fine).
5. **Filter histogram figure** added (`filter_param_histograms.png`: real tabulated vs
   accepted vs rejected synthetic per coord) for his figure-caption discussion.
6. **psi* ablation implemented**: `--psi-labels star` (replay all CSV rows + L1 GD init at
   theta_bar, cached npz next to CSV) + third CP invocation in gp_conformal.sbatch.

**IMPORTANT VERSION NOTE:** the RUNNING RHUL job is @ `be9df5a` — it predates these changes.
Its conformal_shift outputs use the RF-based papernorm and lack the histogram figure + psi*
run. The CP-step rerun is TURNKEY @ `5eba5b6` (`slurm/run_cp_rerun.sh`, pushed to origin):
waits for the running job, checks preconditions, runs the three CP invocations
(default / gamma_real_val / psi_star), commits+pushes to rhul-results. George's campus
one-liner:
```
ssh VPAC005@linux.cim.rhul.ac.uk 'cd ~/rv-ml && git pull --no-edit && nohup ./slurm/run_cp_rerun.sh > slurm/logs/cp-rerun-nohup.log 2>&1 & echo started'
```
Fire-and-forget: safe to run even while the original job is still going (it waits).

### Same session, part 5 — JOBS LAUNCHED at RHUL (direct-run, no Slurm)

**Reality check on "the RHUL cluster": there is no Slurm.** George's account (VPAC005 — a
Paccanaro-lab visitor account) reaches `linux.cim.rhul.ac.uk` = CIM terminal servers
(cim-ts-node-01/02/03, 24 cores / 188 GB each, idle). No sbatch/sinfo; `vulcan.cim` exists
but is firewalled; `/rmt/nas/paccanaro-shared` denies (lab group pending — asked Nicolò).
Homes are NFS (`/home/cim/misc/vpac005`), shared across CIM hosts.

Setup done (all from George's laptop via SSH key `~/.ssh/id_ed25519`, passwordless):
repo rsynced to `~/rv-ml` @ be9df5a (data + GP ckpt verified; pretrain caches/checkpoints
excluded), venv built (torch 2.12), tests + conformal_shift smoke passed on the box,
GitHub fine-grained PAT wired into origin URL (plaintext in .git/config; 7-day expiry —
**revoke after results merged**), push to `rhul-results` verified then branch deleted so the
job's push fast-forwards.

**Launched 2026-07-14 12:20 BST:** `nohup ~/rv-ml/run_jobs_direct.sh` (pid 44884; runs both
sbatch scripts with `srun` stripped, sequentially, job 2 only if job 1 exits 0;
OMP_NUM_THREADS=16). Logs: `~/rv-ml/slurm/logs/runner.log` + `direct-gp-cp-*.log` +
`direct-reg-bench-*.log`. Results auto-commit+push to **rhul-results** at the end of each job.

**Next session:** check the `rhul-results` branch on GitHub (works from anywhere). If absent
after ~20h, the job died — George must ssh from campus or ask Nicolò. Sanity bars on arrival:
GP std log-corr >> 0, cov68 vs 0.59/0.66, naive/naive_adj/surrogate × 4 norms tables,
e-head trio vs gate_b_109_oracle, benchmark.json finally quotable.

### Same session, part 4 — RHUL visit prep @ `be9df5a`

George is going to RHUL IN PERSON (2026-07-14) to submit both jobs. Both sbatch files now end
with a **push-to-branch step**: commit artifacts on the current branch, `git push origin
HEAD:refs/heads/rhul-results` (George-authored identity, no trailer; push failure never fails
the job). Chain: `jid=$(sbatch --parsable slurm/gp_conformal.sbatch); sbatch
--dependency=afterok:$jid slurm/regression_benchmark.sbatch`. On-site checklist: venv + pip,
fix --partition (ADJUST lines), **set up push credentials (PAT or SSH deploy key) and verify
with a test push before leaving** — retrieval is otherwise impossible off-site (no VPN).
Excluded from push (large/regenerable): LSP-512 CSV, checkpoints/. Est. wall: job1 6–14h,
job2 3–6h — results land on rhul-results overnight; merge to main after review.
**NOTE George's push rule:** the branch push is George-configured automation he requested —
merging rhul-results into main + pushing main remains his call.

### Same session, part 3 — --gamma-tune-on flag @ `eca644a`

`conformal_shift.py --gamma-tune-on {synthetic,real-val}` (default synthetic = old behavior;
real-val = the paper's D_val — legal because tune_gamma only measures widths, no labels;
warns if it overlaps --real-split). `gp_conformal.sbatch` step 3 now runs the CP step TWICE
(default + real-val → `regression/gamma_real_val/`), so Nicolò's gamma answer can't
invalidate the job. Both paths smoke-tested (gammas differ as expected).
**Answer-dependence of the RHUL job after this:** only "drop the Assumption 2.1 filter" or
"use p90 instead of max for naive_adj" would force a rerun — and only of step 3 (~1h CP
step), never the SVGP retrain. Q1 (v2norm vs papernorm) and Q2 (conformal.py scope) don't
affect the job; Q6 (surrogate-label psi) would only add a step.

---

## SESSION 2026-07-12 — Daksh force-push triaged; RHUL access checklist

- **Daksh force-pushed `daksh/regression-head`** (`2865cff` → `b8f4f98`): rewrote history into
  granular commits with his own variants of the 07-11 review fixes. **Verified functionally
  equivalent to main** (his loss, as called with the redundant `w_batch = wb * dim_w`, is
  numerically identical — the extra factor cancels in his per-dim normalization; replay uses a
  module-level params cache + bounds check, both fine). Diffs vs main: dead double-multiply,
  `now` docstring typo, dropped the CSV-independent replay test, and an unflagged semantics
  change in the non-circular MSE path (per-dim means vs global weighted mean; only affects
  `--no-circular-omega`). **Nothing to change on main.** Asked him (PR #3 comment, 4951567937)
  to rebase onto main before his next PR to avoid conflict noise.
- **Nicolò's Slack thread** (e/ω): tracked CSV has **23.6% e==0, 36.3% e≤0.05, 50.3% e>0.1**
  — his zero-inflation suspicion confirmed. ω-loss mask handles ω training; the e head still
  trains on the zero-inflated prior (open item: resample/balance or classify-then-regress).
  On "predict single ω": the merged cos/sin + unit-circle projection + 1−cos(Δω) loss already
  IS single-ω prediction without the 0/2π wraparound — push back if he insists on raw ω.
- **RHUL in person**: eduroam via ND credentials works for wifi, but George needs (1) a cluster
  account (Nicolò requests it), (2) login-node hostname, (3) whether eduroam reaches the login
  node or VPN/jump-host is still required. Simpler path remains Nicolò running both sbatch
  jobs from a plain clone (order: gp_conformal → regression_benchmark).

## SESSION 2026-07-11 — Daksh's PR #3 reviewed, fixed, merged; benchmark regen job for RHUL

### PR #3 (daksh/regression-head): phase-fold e/ω features + 74/35/109-D regression
Reviewed on GitHub (request-changes posted), then fixed + merged to main @ `0b246d2`
(fixes verified locally with smoke runs + new unit test; all 8 tests pass):

1. **Replay RNG-prefix trap** — `replay_synthetic_sample(i, seed)` drew only `i+1` params
   from the shared stream, but the CSV drew all 10k at once → every predicted-P refold
   (`recompute_phasefold_block`) folded a *different system's* curve. Now takes `n_samples`
   (+ `corpus_orbital_params` helper, params drawn once per batch). Guarded by
   `tests/test_replay_synthetic_sample.py` (round-trips vs `generate_rows` and the tracked CSV).
2. **`--feature-set 109/35` crash** — real systems lack catalog `t_peri` → all phase-fold
   features NaN → every real row dropped → `plot_combined_scatter` hit empty arrays. Real
   transfer now skipped with a note in metrics.json when `n_real == 0`.
3. **Loss weights no-op / double-applied** — dim weights cancelled in the circular-ω per-dim
   normalization AND were applied twice in the MSE path (train_model pre-multiplied).
   `regression_theta_loss` now takes raw sample weights, applies `dim_weight` exactly once
   (circular path: dim-weighted mean over per-dim terms). Default all-ones loss bit-identical.
4. Minor: `--fold-period predicted` embedded ndarrays in metrics.json (stripped now).

### RHUL: TWO jobs pending submission (still VPN-blocked; Nicolò from plain clone)
1. `sbatch slurm/gp_conformal.sbatch` — unchanged from 07-04/07-06, run FIRST.
2. `sbatch slurm/regression_benchmark.sbatch` — NEW @ `9c735a6`: regenerates the phasefold
   CSV (self-consistent with whatever GP checkpoint is on the cluster), runs the replay
   round-trip test as a guard, then Gates A/B/C + ablation → `benchmark.json`, two-step
   pipeline, and 109-D diagnostics. **The committed
   `figures/regression_synthetic/benchmark.json` + Daksh's Gate C / two-step numbers are
   INVALID (computed with the broken replay) — never quote them until this job reruns.**
   Rsync back + commit: `figures/regression_synthetic/` + regenerated phasefold CSV.

---

## SESSION 2026-07-06 — Nicolò's answers implemented (σ-conditioning, GD surrogate, two-factor norm)

### Nicolò's Slack answers (all three 07-04 questions closed)
1. **`s/(γ+v)` confirmed** — "Yes, sorry, I meant s' = s/(gamma + v)". Caveat removed from
   `conformal_shift.py` docstring.
2. **Cadence bootstrap: keep** — "Everything we can learn from the training data is OK with me."
   Quote that in the methods section.
3. **σ-conditioning: GO, as an extra feature** (not the r/σ label variant).

### His follow-ups added new spec (same thread), both implemented
4. **Surrogate labels by gradient descent**: θ* = argmin_θ E_t|y_t − kepler(θ,t)| (note **L1**),
   GD init at the **tabulated / data-generating** values. Replaces the coordinate-descent
   surrogate (which warm-started at ψ(y)). Implemented as batched Adam through the
   differentiable `KeplerDecoder` (`surrogate_fit_gd` / `_gd_batch` in `conformal_shift.py`;
   t_peri/γ refit is detached — envelope-style, verified `fit_t_peri` re-enables grad
   internally). CLI: `--gd-steps` (200), `--gd-lr` (0.02); `--sweeps` removed.
5. **Two-factor normalization** `s_c' = s_c/(γ + v_y + v_c)`: v_y = existing SVGP proxy,
   v_c = per-coordinate RF model of the surrogate-label error E|θ̄_c − θ*_c|
   (`fit_vk_models`), trained on the synthetic **tuning set** (θ̄ known there), features =
   74-dim summaries (same non-degeneracy argument as the weight discriminator). New norm
   variant `v2norm` alongside `raw`/`vnorm`; γ tuned per variant; `evaluate`/`tune_gamma`
   generalized to per-coordinate denominators.
6. He is **writing the full-pipeline section in Overleaf** — treat it as the reference spec
   when ready. He also said **git pull first** — done, origin had nothing new.

### Defaults chosen without asking (flag to Nicolò if he objects)
- v_c regresses on the 74-dim summary features of y (not ψ's 586-dim set).
- L1 taken literally as the GD objective (mean |·| in rv_std units).
- On real curves the GD init would be the tabulated θ (surrogates are currently only computed
  on synthetic cal/tune sets, so this is latent).

### σ-conditioning (`gp_residual_model.py` + consumers)
- `log10_sigma` appended as **8th feature** (FEATURE_NAMES, build_split, docstring); cache key
  **fv4 → fv5**; partial-dependence plot gains a log10_sigma panel.
- `synthetic_dataset._gp_residual_features(sigma=...)` appends the column **only when the
  loaded checkpoint's `feature_names` contains `log10_sigma`** — old 7-feature checkpoint keeps
  working (verified both directions). `_inject_noise` passes σ through.
- `conformal_shift.NoiseProxy` likewise (sig in curve dict is in rv_std units → ×rv_std for m/s).
- Smoke retrain: even at --smoke scale std log-corr went ~0 → 0.4–0.6 (the metric σ-conditioning
  targets; full-scale value TBD on cluster).

### Verified locally (smoke only, artifacts restored after)
- `conformal_shift.py --n-cal 24 ... --gd-steps 40` end-to-end with the committed 7-feature
  checkpoint: runs, v2norm reported, v_c medians sensible (1e-3–5e-2). inf=1.00 at n_cal=24 is
  the known Bonferroni triviality (needs n_cal ≥ 39), not a bug.
- `gp_residual_model.py --smoke`: 8-feature pipeline trains/evals/saves.
- 8-feature checkpoint consumed by `generate_one` (noise_mode=gp_residual_svgp) and NoiseProxy
  (wants_sigma=True).
- `tests.test_time_series_features` green. Clobbered `models/`, `figures/gp_residual/`,
  `conformal_shift_{report,metrics}` + figures restored via git checkout.

### Next action — RHUL cluster job, now runs from a plain clone (UPDATE 07-06 later)
George can't reach RHUL; **Nicolò will run it**. To make that possible the job's data inputs
were **force-added to git** @ `7d462d8` (~10 MB: `data/rv_raw/` 1072 .tbl, labels/splits/
stats/residuals_index, gp_fits fallback, rv_index, simbad_cache — caches stay gitignored;
NB `data/pretrain_cache_v3.pt` DOES exist locally now, 2.9 GB). Nicolò's flow: clone →
venv + `pip install -r requirements.txt` → set `--partition` (+`--account`) in
`slurm/gp_conformal.sbatch` from `sinfo` → `sbatch` from repo root. Full step-by-step incl.
sanity checks was given to George on 07-06 to forward. Only prereq: repo access for Nicolò.
After the run, sanity-check: GP std log-corr should
now be well above 0 (σ-conditioning), cov68 vs the old 0.59/0.66 bar, and the naive-vs-surrogate
+ raw/vnorm/v2norm width tables are the paper's comparison.

---

## SESSION 2026-07-04 — Nicolò's 2026-07 spec implemented; everything pushed @ `761fe2f`; cluster run pending

### THE immediate next action — RHUL cluster job (blocked on VPN)

The committed GP checkpoint and both `synthetic_generation/datasets/*.csv` **predate** the LS-γ
and train-only-H changes below (intentional — George's rule: **no CPU/GPU-exhaustive tasks on
the laptop**, smoke runs only). SSH to RHUL needs their VPN; no RHUL host in `~/.ssh/config` —
get host/user/path from George when he's connected. Then:

```bash
rsync -avz --exclude .git --exclude .venv --exclude data/gp_residual_cache \
    --exclude data/pretrain_cache_v3.pt --exclude checkpoints \
    ~/rv-ml/ <user>@<host>:~/rv-ml/
# on the cluster, first time: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# check `sinfo`, fix --partition in slurm/gp_conformal.sbatch if needed, then from repo root:
sbatch slurm/gp_conformal.sbatch
```

Job = SVGP retrain (LS-γ) → regenerate both regression CSVs (train-only H + fresh checkpoint)
→ `conformal_shift.py` full scale (n_cal=400). Rsync back + commit: `models/gp_residual_svgp.pt`,
`models/gp_residual_metrics.json`, `figures/gp_residual/`, both CSVs,
`synthetic_generation/regression/conformal_shift_{report.txt,metrics.json}` + figures.
Retrain sanity bar: previous full-scale val cov68 ≈ 0.59 / test ≈ 0.66. The committed
n_cal=30 smoke report is trivially conservative — never quote it.

### What was implemented (Nicolò's Slack decisions, all pushed)

1. **H train-only** — P/K/e priors + (already) cadence/σ bootstrap filter `split=="train"`
   (`synthetic_dataset.py`). Real val/test reserved for testing CP intervals.
2. **`conformal_shift.py`** — split-CP calibrated on fake only, tested on real test split:
   naive `|ψ(y)−θ̄|` (ground-truth θ̄) vs surrogate (θ̄ → argmin‖y−Kepler(θ)‖, coord descent
   warm-started at ψ(y)); surrogate reweighted by `p_real/p_fake` (logistic discriminator fit
   on real TRAIN vs fresh synth; Tibshirani 2019 weighted quantile, mass at +∞, clipping+ESS).
   ψ trains on the **512-bin raw-LSP CSV by default** (feature cols follow `--csv`); the weight
   discriminator deliberately stays on the 74-dim summaries (586-dim would degenerate weights).
3. **Noise-normalized score `s/(γ_reg+v)`** — v = SVGP predictive-std proxy, γ_reg tuned on a
   synthetic tuning set. **Nicolò literally wrote `s/(γ+s)` — monotone in s, changes no CP
   set; the `v` reading is documented at `conformal_shift.py:31` but NOT yet confirmed.**
4. **LS-γ offset** in `gp_residual_model.py` (`_ls_gamma`, cache key **fv4**), replaces the
   first-obs anchor.
5. Merged from GitHub: daksh's `regression.py` (MLP head — reviewed clean) + the `scripts/`
   reorg (compat wrapper keeps root `parse_and_label` imports working). Fixed
   `tests/test_generate_300.py` hanging `unittest discover` (`plt.show()` under macosx →
   Agg + `plt.close()`). NB: `tests/test_parser.py` + `test_generate_300.py` are import-time
   smoke scripts, not TestCases; only `test_time_series_features` has real assertions.

### Waiting on Nicolò (email drafted 2026-07-04 — ask George if sent)

1. Confirm the `s/(γ+v)` reading (point 3 above).
2. Cadence bootstrap (paired real (cadence,σ) profiles vs uniform) — keep? Never answered.
3. σ-conditioning of the residual GP — go/no-go (the std-ratio-1.76 amplitude fix).

### After the cluster run

- Quote full-scale naive-vs-surrogate coverage/width tables (the paper's comparison).
- Full-scale profiled-CP run for `conformal.py` (`--profile Keomega`) still open.
- Encoder run needs `data/pretrain_cache_v3.pt` regenerated **after** the new checkpoint
  (priors + noise changed).

### House rules (George) — unchanged but restated

- Never push unless he says so. No `Co-Authored-By` trailer.
- Smoke runs local (`--smoke`, small `--n-cal`); full-scale → RHUL only.
- If a smoke run overwrites committed artifacts (`models/`, `figures/`), `git checkout` them
  back before committing. (Happened twice this session: gp_residual smoke artifacts + the
  `figures/synthetic_plots/` PNGs clobbered by the 300-system test.)

---

## SESSION 2026-07-02 — Profiled conformity score + slurm encoder script (committed `3c8aadb`, `a96be97`)

Committed work that was sitting uncommitted in the tree (done outside a Claude session or in an unrecorded one):

### `conformal.py` — profiled conformity score (the top tightening lever from 07-01)
- **`profiled_min()`**: batched coordinate-descent minimisation of the reconstruction score over a chosen nuisance set at each swept value of the tested coord, warm-started from θ̂; the incumbent is kept as a candidate so profiling can never do worse than the pinned score. `profile_coords=()` reduces bit-for-bit to the pinned baseline.
- **Calibration is profiled identically** (pin c at its own predicted value θ̂_c, profile the rest) → exchangeability preserved, but the quantile q is now **per-coordinate**.
- CLI: `--profile {none,K,Keomega}` (default **K**), `--profile-grid` (default 33), `--sweeps` (default 2), `--chi2` (the χ² variant is now **opt-in**, no longer always run).
- **Quick-run result (n_cal=40, n_test=40 — NOT the full n=400): profiling K left median widths unchanged** (log10_P=3.35, K=1.21, e=0.95, ω=5.86) at valid coverage. So at n=40, prof_K is not enough — consistent with the 07-01 diagnosis that e/ω nuisance (RF is weakest there) matters most. **Open: full-scale `--profile Keomega` run at default n=400** (cost: Keomega profiling is ~grid×sweeps×3 more score evals — budget accordingly or run on the cluster).
- `synthetic_generation/figures/keomega/` holds stashed E1/E2 figures from a K+e+ω profiling run (old `_rv_std` suffix naming — predates the config-label suffixes). The committed report/metrics only cover `rv_std` and `rv_std+prof_K`.

### `slurm/train_encoder.sbatch` — encoder training on the RHUL GPU cluster
Two-phase `train.py` run (pretrain 300 ep → finetune 100 ep, resnet, batch 128). Preflight: asserts CUDA visible to torch, requires **`data/pretrain_cache_v3.pt`** (a 500k-sample cache — does NOT exist yet locally; v2 at `data/pretrain_cache_v2.pt` is 20k) and real `data/` present on the cluster. Partition/module names are ADJUST placeholders — set to whatever RHUL's `module avail` / partition list shows. Submit from repo root: `sbatch slurm/train_encoder.sbatch`.

### Docs
README updated: `slurm/` in the tree + module tables, conformal row/current-state/next-steps reflect the profiled score. **Not pushed — George pushes himself.**

---

## SESSION 2026-07-01 (cont.) — Step 6 Unsupervised CP, encoder training, ω NN-vs-RF

### The project's 6-step pipeline (Nicolò's canonical plan) and where it stands
1. Data preprocessing — ✅ `download_rv` / `parse_and_label` / `kepler_check` (validation overlays) / `cache_residuals` (stores r = obs − RV(θ)).
2. Infer distributions H + GP noise — ✅ `synthetic_dataset.py` (empirical histograms) + `gp_residual_model.py` (`noiseGP`, `models/gp_residual_svgp.pt`).
3. Synthetic data generation — ✅ `synthetic_dataset.py` + `generate_synthetic_regression_csv.py`.
4. **RV encoder = the spline+power-spectrum FEATURE extractor** — ✅ `time_series_features.py` (this is Nicolò's "simple version": UnivariateSpline → power spectrum). *Not* the neural net.
5. Regression θ from features — ✅ `synthetic_generation/train_regression_models.py` (RF).
6. Conformal Prediction — ✅ **now built this session: `conformal.py`** (was the only gap).

**The neural `RVEncoder`+`KeplerDecoder` is a parallel/optional track, not part of this 6-step plan** (which uses spline-features + RF). Don't conflate.

### Relationship to Baragatti et al. 2026 (the reference Nicolò flagged)
Baragatti [2] = ABC + NN(MC-dropout) + **supervised** conformal (needs a ground-truth-θ calibration set). **Our novelty = *unsupervised* CP** (Overleaf §2.2.1): the conformity score is the reconstruction residual, needs no true θ, so it calibrates on **real** curves and converts an unestimable conditional label shift into an estimable covariate shift (which the real-vs-synthetic classifier measures). Almost every component already existed in the repo (decoder = `kepler_torch`, noise = `gp_residual_model`, classifier = `validate_synthetic_dataset`); the gap was the CP assembly.

### `conformal.py` — Step 6, unsupervised CP (committed `508cc81`, `a5ede10`; George pushed)
- **Point predictor** φ = RF(features(y)) (the Step-5 model). **Score** `s(θ,y)=‖Kepler(θ)−y‖` via `KeplerDecoder` (refits t_peri/γ internally). Split-conformal, **Bonferroni over d=4 physical coords** (log10_P, log10_K, e, ω; cos/sin ω → the single angle). **All calibration draws + grids come from the empirical histograms H** (period mixture, e-histogram, K range; ω uniform because the corpus has no preferred periastron) — per George's instruction to justify assumptions via H, not ad-hoc priors.
- **E1 (coverage): the guarantee HOLDS — coverage ≥ nominal on synthetic AND real** (conservative). e.g. synthetic joint 0.94/0.90 at nominal 0.95/0.90; real joint ~0.88. This is the paper's core claim, demonstrated, and it transfers to real (Baragatti's can't).
- **E2 (monotonicity, Assumption 2.3): HOLDS for all 4 coords including ω** (clean V, min at truth). **This revised my earlier prediction that ω would break it** — the score reads the *raw curve* so it resolves ω, even though the regressor's phase-blind features can't *predict* ω. So CP can bound ω where regression can't.
- **σ-normalized (χ²) score variant — DOES NOT tighten the sets.** Widths unchanged (log10_P=3.35, e=0.95, ω=6.13 ≈ full histogram support; K slightly worse). Coverage stays valid. **So noise scale is NOT the bottleneck.**
- **Diagnosis of the wide (valid-but-uninformative) sets:** the univariate CP (eq 9) fixes nuisance coords at the RF point estimate θ̂; the RF is weak on K/e/ω, so with wrong nuisance the reconstruction goes flat in the swept coord → near-full-support sets. E2 (nuisance = truth) is sharp, proving this. log10_P is also inflated by **period aliasing** (max−min width spans aliases).
- **Next levers (NOT noise normalization):** (1) **profiled conformity score** — minimise over nuisance instead of fixing at θ̂ (the highest-value tightening lever); (2) stronger point predictor; (3) report accepted-*measure* not max−min width. Figures: `conformal_{e1_coverage,e2_monotonicity}_{rv_std,chi2}.png`, `conformal_width_comparison.png`; report `regression/conformal_report.txt`.

### Proper encoder training (the parallel NN track)
Trained a real resnet encoder (20k current-priors cache → 30 pretrain + 80 finetune ep; pretrain loss 0.86→0.15, best val 0.87). New checkpoint `checkpoints/resnet_finetune_best.pt` (the older `checkpoints/*.pt` were **epoch-1 smoke artifacts**). Cache regenerated at `data/pretrain_cache_v2.pt` (current priors).

### ω recovery: NN vs RF — RF wins, ω unrecoverable either way (`eval_omega_nn_vs_rf.py`)
On matched real test systems: **NN recovers ω *worse* than the RF** (NN cos/sin R²=−0.43/−1.02 vs RF −0.13/−0.25). The NN is *confidently wrong*; the RF hedges to the mean. NN also loses on K (0.04 vs ~0.8) and P (0.56 vs 0.82) — this was a modest run, but ω is an *identifiability* problem, not capacity. The encoder DOES work (validates the eval: NN log10_P R²=0.56 on real). Refutes the "NN sees phase → recovers ω" hypothesis.

### Housekeeping
- **Commits are George-only now** — the `Co-Authored-By: Claude` trailer is stripped (see [[feedback_no_coauthor]]). All session commits pushed to `origin/main` individually.
- Large/regenerable artifacts gitignored: `data/`, `checkpoints/`, `synthetic_generation/datasets/synthetic_lsp_regression_10000.csv`.

---

## SESSION 2026-07-01 — RF regression baseline + PCA true-vs-fake (Nicolò's spec)

Context: in the Slack thread, Nicolò asked for (a) an input→output regression model where **input = power spectrum + Shuaib's summary features**, **output = true generating params**; RF *or* NN both fine — and (b) a **2D PCA of real vs synthetic, white=true / black=fake dots**. Jovie had done the NN side (RVEncoder, joint-vs-separate). This session built the **RF side + the PCA**, both against the existing `synthetic_generation/synthetic_regression_10000.csv` (10k rows, 74-D input = 64 spectral-power bins + 10 summaries → 5 targets). Two new scripts, both `--help`-documented and house-style:

- **`synthetic_generation/train_regression_models.py`** — RF regression. Joint multi-output RF (targets standardized so no target dominates the split criterion) vs per-target separate RFs; feature-block ablation {summary / spectral / both}; 5-fold CV; held-out test; and a synthetic-trained→real transfer test. Writes `regression/regression_{metrics.json,report.txt,feature_importances.csv}` and figures `regression_true_vs_pred_{joint,separate,real_transfer}.png`, `regression_feature_importance.png`.
- **`synthetic_generation/pca_real_vs_synthetic.py`** — Nicolò's PCA. z-scores the pooled 74-D features, fits PCA(2), plots PC1 vs PC2 (**real = white, synthetic = black**), plus a feature-block ablation panel. Writes `regression/pca_{coords,summary}_both.*` and figures `pca_real_vs_synthetic_{both,feature_blocks}.png`.

Run config for the headline numbers: 300 trees, 5-fold CV, seed 0, real split = all (357 real single-planet systems after the σ∈[0.1,100] filter).

### RF results — the findings (all stable, tight CV std ~0.01)

**Cross-validated R² (separate|both, mean±std over 5 folds):**

| Target | CV R² | Synthetic→real transfer R² |
|---|---|---|
| log10_P | **0.672 ± 0.012** | **0.816** |
| log10_K | **0.794 ± 0.005** | **0.842** |
| e | 0.160 ± 0.017 | 0.115 |
| cos_omega | −0.029 ± 0.005 | −0.586 |
| sin_omega | −0.030 ± 0.006 | −0.109 |

1. **The 64-bin power spectrum ALONE is useless** — R² negative for every target (spectral-only ≈ −0.16 across the board), for both joint and separate. Directly answers Nicolò's "did the power spectra work as inputs?": **not at 64-bin resolution.** The coarse normalized bins lose the sharp LSP peak that encodes P.
2. **The 10 summaries carry essentially all the signal**, and **adding the spectrum on top barely moves R²** (summary ≈ both). So in this representation the spectrum is redundant given the summaries.
3. **joint ≈ separate** everywhere (within ~0.01–0.02 R²) — matches Jovie's NN observation that neither clearly wins.
4. **P and K are well recovered; e is weak; ω is unrecoverable** (R²≈0 on synthetic, negative on real — the model predicts ~mean because ω is barely constrained; consistent with the encoder refitting ω/t_peri analytically, so not alarming).
5. **Synthetic→real transfer is excellent and is the headline:** a model trained only on 10k synthetic and tested on the 357 real systems gets **log10_P R²=0.82, log10_K R²=0.84 — better than on synthetic held-out.** This validates the synthetic corpus as a pretraining set, and **explains why the earlier `random_forest_regressor.py` got R²=−0.16**: that script trained on only 247 real systems AND used the useless raw spectrum. Lesson: use the summaries, and pretrain on synthetic.
6. **Feature importances are physical:** log10_P ← `lsp_peak_period_d` (0.55; LSP peak ≈ P), log10_K ← `rv_std_ms` (0.75; RV scatter ≈ K). Spectral bins rank last everywhere.

### PCA results
- Feature space = same 74-D. **PC1 = 82.5% var** (collective spectral-power axis — the 64 bins are strongly collinear, they sum to ≤1), **PC2 = 3.6%** (cadence/σ axis: baseline_d, p90_gap_d, median_sigma_ms). Real overlaps the synthetic cloud but sits off-center (real PC1 mean +0.23 vs synth ~0) — consistent with the ~0.5–0.65 classifier: mostly overlapping, mild separation.
- **Finding to flag to Shuaib:** the spectral-only PCA panel exposed a handful of **synthetic systems with near-delta spectra** (all power in one bin, PCA coords many σ out) that **real data never shows** — likely very clean single-planet signals; a possible generator artifact worth checking.

### Notes / caveats
- **Dimensionality answer for Nicolò:** this RF uses the compact **74-D** representation (64 spectral + 10 summary). The RVEncoder NN Jovie used consumes **~1500-D** (512-bin LSP + a (4,256) summary tensor). Different encodings of the same series — metrics not directly comparable. **The 512-bin follow-up is now done — see the next subsection.**
- `train_regression_models.py --save-models` writes a ~1.3 GB joblib (300 trees × 5 targets) — **not committed, deleted after the run.** Regenerate on demand.
- Both scripts run with `MPLBACKEND=Agg`; full run ≈ 22 min on the M3 (all cores, `n_jobs=-1`). Quick check: `--n-estimators 40 --cv-folds 2` (~90 s), numbers essentially identical.

### 512-bin LSP resolution experiment — does the full power spectrum beat 64 bins? (added same session)

Directly answers Nicolò's "did the power spectra work as inputs, at what dimensionality?". Two new scripts:
- **`generate_lsp_regression_csv.py`** — regenerates the 10k dataset on the **same seeds** as `synthetic_regression_10000.csv` but stores the **full 512-bin LSP** (`lsp_power_001..512`) alongside the 64 spectral bins and 10 summaries (591 cols, ~130 MB, **gitignored — regenerable in ~30 s**). Output: `datasets/synthetic_lsp_regression_10000.csv`.
- **`lsp_resolution_experiment.py`** — RF (separate + joint) across 5 feature sets {summary, spectral64, lsp512, spectral64+summary, lsp512+summary}, 5-fold CV + holdout + synthetic→real transfer. Single config `max_features="sqrt"` across all sets so 64-vs-512 is fair and the 512-D fits stay tractable. Also emits the **joint-vs-separate true-vs-pred** plots on `lsp512+summary` (George's ask — directly comparable to Jovie's NN joint-vs-separate). Added a `max_features` knob to `train_regression_models.py` (default None = unchanged) that this reuses.

**CV R² by feature set (separate RFs, 300 trees, 5-fold):**

| feature set | log10_P | log10_K | e | ω |
|---|---|---|---|---|
| summary (10-D) | **0.672** | **0.799** | 0.134 | ~0 |
| spectral64 (64-D) | −0.211 | −0.201 | −0.203 | neg |
| **lsp512 (512-D)** | **0.531** | 0.118 | −0.026 | ~0 |
| spectral64+summary (74-D) | 0.616 | 0.773 | 0.087 | ~0 |
| lsp512+summary (522-D) | 0.621 | 0.676 | 0.041 | ~0 |

**Findings (the meeting answer):**
1. **Resolution matters — a lot.** Raw **512-bin LSP recovers log10_P (R²=0.53)**; the **64-bin** version is useless (R²=−0.21). The coarse sum-normalized binning smears out the periodogram peak that encodes P; the full-resolution LSP keeps it. So "the spectrum is uninformative" was a **64-bin artifact, not a property of the power spectrum.**
2. **The LSP encodes period, not amplitude.** lsp512 alone gets P (0.53) but **not K** (0.12) — the LSP power is normalized, so it locates the peak (→ P) but not the signal size (→ K, which lives in `rv_std_ms`). Transfer to real: lsp512 → log10_P R²=0.72, log10_K R²≈0.
3. **But for a random forest, the extracted scalar still wins.** `summary` alone (which contains `lsp_peak_period_d`) beats `lsp512` and even `lsp512+summary` on P — adding 512 raw bins **dilutes** the tree (curse of dimensionality) rather than helping. So an RF gets period more cheaply from the pre-extracted peak.
4. **Implication for model choice:** the raw high-resolution LSP is exactly where a **CNN/NN (Jovie's RVEncoder) can extract structure an RF cannot** — a clean, defensible justification for the NN over the RF on the spectral input. The RF is the right baseline; the NN earns its keep on the raw 512-bin spectrum.
5. **joint ≈ separate again** on `lsp512+summary` (log10_K: separate 0.678 vs joint 0.618; rest within ~0.01) — consistent with Jovie's NN and the 74-D RF.

Figures: `regression_true_vs_pred_lsp_{joint,separate}.png`, `lsp_resolution_r2_by_featureset.png`. Report: `regression/lsp_resolution_{report.txt,metrics.json}`.

---

## LATEST (2026-06-30) — GP residual noise wired into the generator, real-cadence bootstrapping, regression CSV workflow

Seven commits landed (`8aa1637`..`d96f29a`), authors **Shuaib** and **Jovie**. Pulled to `main` @ `d96f29a`. Working tree clean.

### `8aa1637` (Shuaib) — GP residual SVGP is now the primary synthetic noise source
`synthetic_dataset.py._inject_noise` now tries, in order: **(1)** the trained global SVGP+Student-t from `models/gp_residual_svgp.pt` (the `gp_residual_model.py` deliverable, see section below), **(2)** the older per-system `GPNoiseLibrary` (`data/gp_fits.json`), **(3)** i.i.d. `N(0, σ²)` white Gaussian.
- New helpers: `_load_gp_residual_sampler` (loads+caches the checkpoint, rebuilds SVGP via `_make_svgp`, restores StudentT likelihood + standardizer), `_gp_residual_features` (builds the 7-D feature rows — phase, log10 P, log10 K, e, cos ω, sin ω, y_rel — from the **dominant planet's clean RV**), `_sample_gp_residual_noise`.
- `generate_one` now tracks per-planet clean RV parts so the dominant part can feed the GP, and records **`noise_mode`** in the info dict (`gp_residual_svgp` / `GPNoiseLibrary` / `white_gaussian_fallback`).
- `get_noise_model_status()` reports which backend is live. **The checkpoint is committed (`f8c173e`), so generation uses the SVGP path by default now.**

### `be793a7` (Shuaib) — real observation-profile bootstrapping + GP amplitude scale
- **Cadence + σ now bootstrapped from real training-split `.tbl` files**, not the heuristic samplers. `_load_real_observation_profiles` / `_sample_observation_profile` draw a **paired (time grid, per-obs σ)** from a real train system (single-planet, median σ ∈ [0.1, 100], n≥10, train-only to avoid leakage). Falls back to `_sample_time_grid` + `_sample_sigma` only if profiles can't load. This preserves real within-system σ spread and real cadence/gaps — directly attacks the `rv_std_ms` / `n_obs` / cadence discriminators.
- `_gp_residual_scale()`: multiplies GP noise samples by env `RVML_GP_RESIDUAL_SCALE` (**default 0.85**) for validation amplitude sweeps; recorded in info.

### Current classifier numbers (GP-residual noise + real-cadence bootstrap + tuning — NEW HEADLINE, supersedes all tables below)

| Real split | Balanced acc | Top individual discriminator | Top feature group |
|---|---|---|---|
| train | **0.599 ± 0.019** | `lsp_peak_power` | `rv_std_ms` |
| test  | **0.498 ± 0.023** | `rv_std_ms` | `n_obs` |
| val   | **0.522 ± 0.052** | `lsp_peak_power` | `rv_iqr_ms` |
| all   | **0.650 ± 0.021** | `lsp_peak_power` | `rv_std_ms` |

- **Test is now ≈0.5 (indistinguishable)** and **val is no longer the degenerate 0.500 ± 0.000** — the earlier "suspicious val" flag is resolved; it now has real variance (±0.052).
- The top discriminator moved **`log10_K` → `lsp_peak_power` / `rv_std_ms`** — i.e. the orbital-parameter priors are now well-matched and the residual signal is **noise-amplitude / periodogram-power / cadence**, exactly the things the GP-residual noise + real-profile bootstrapping target next. `median_sigma_ms` is no longer a top discriminator (real-σ bootstrapping worked).

### Jovie — `random_forest_regressor.py` (`47eb33a`, `6aebead`, `0c0300f`, `73ac222`)
Standalone sklearn `RandomForestRegressor` predicting **log10_P from 64 spectral features** of **real** RV data (`RVDataset` → `parse_tbl` → `time_series_features.spectral_features`). Trains on train split, evaluates MAE/MSE/R² on test, scatter-plots true vs predicted. A first supervised baseline for the encoder task on real data (not synthetic). Script is top-level exec (no `main()`), `create_dataset(split)` drops invalid systems. `train` default arg was removed from `create_dataset` last (`73ac222`).

**Result (ran 2026-07-01, default 100-tree RF, train=247 valid systems):** MAE **0.877**, MSE **1.089**, **R² = −0.161**. The model is **worse than predicting the mean**. **RESOLVED (see SESSION 2026-07-01):** the cause is now understood — (a) only 247 real systems, and (b) it uses the raw 64-bin spectrum, which carries no recoverable parameter signal at that resolution. Training the same feature family on the **10k synthetic CSV** and testing on real gives **log10_P R²=0.82** (vs −0.16 here). So this script is superseded as a baseline by `train_regression_models.py`; keep it only as the cautionary "real-only + raw-spectrum" data point. `plt.show()` blocks — run with `MPLBACKEND=Agg`.

### Shuaib — `synthetic_generation/` regression CSV workflow (`d96f29a`)
Self-contained folder (core generator stays in root `synthetic_dataset.py`):
- `generate_synthetic_regression_csv.py` — emits an input→output CSV: **targets** = 5 Kepler params (log10_P, log10_K, e, cos ω, sin ω) from `generate_one`'s dominant theta; **features** = 64 spectral power bins (`spectral_features`, grid 1024) + 10 summary features (n_obs, baseline, rv_std/iqr, median/iqr σ, LSP peak period/power, median/p90 gap). Default 10k rows, seed 123, `--f-multi 0.0`.
- `datasets/synthetic_regression_10000.csv` (10k×79), plus `validate_…` (all 24 checks PASS) and `plot_…` (real-vs-synthetic comparison figures) scripts + reports under `validation/` and `figures/`.
- Note this samples orbital params via `_sample_orbital_params` and reconstructs features from the **generated x tensor** (un-normalises with `rv_std_ms` / `t_span_days`), independent of `random_forest_regressor.py`'s real-data pipeline. Two parallel regression datasets exist now — reconcile which the encoder baseline should use.

---

## NEW (2026-06-12): GP fit to real residuals — Nicolò's spec — `gp_residual_model.py`

Nicolò specified the GP noise model must be fit to **real-data residuals** with a specific
feature/label design (different from the per-system celerite2 model in `gp_noise_model.py`).
Implemented in the new module **`gp_residual_model.py`**. **Committed, and as of 2026-06-30 the trained checkpoint `models/gp_residual_svgp.pt` is the live synthetic-noise backend** — see the LATEST section at the top.

**Residual definition (his spec, literal):** for each single-planet system, integrate the
*tabulated* catalog Kepler params to get the noiseless curve `y(t)=rv_keplerian(P,K,e,ω,t_peri)`,
anchor the vertical offset to the **first observation** (`γ = ŷ(t₀) − y(t₀)`), and take
`r(t) = y(t) − ŷ(t)`. `t_peri` is catalog when available (124 train systems) else analytically
phase-fit (106). Phase-fold `t mod T`, `T=P`.
- **Features (7-D):** `phase=t%T/T`, `log10 P`, `log10 K`, `e`, `cos ω`, `sin ω`, `y_rel`.
- **Label:** `r(t)`.
- **`y_rel = y(t) − y(t₀)`**, NOT raw `y(t)` — critical fix: raw model RV carries each star's
  arbitrary systemic velocity (±tens of km/s) which swamps the orbital signal after
  standardization. `y_rel` (RV change since first obs) is the DC-free predictor. The label `r`
  is already DC-free (γ cancels).
- **Augmentation (his spec):** resample `ŷ ~ U(ŷ−σ, ŷ+σ)`, re-anchor, recompute `r`
  (`--n-aug`, default 20). Train only; val/test use nominal ŷ.

**Quality filters (a noise model must see noise, not junk):** median σ ∈ [0.1, 100] m/s (drops
`11 Com` absolute-RV at ~5 km/s etc.) + residual RMS/σ ≤ 30 (drops gross catalog mismatch).
Train: 230 systems kept, 5 cut on σ, 12 on rms.

**Model:** gpytorch **SVGP** (512 inducing pts, ARD Matérn-5/2) with a **Student-t likelihood**
(default; RV residuals are leptokurtic, kurtosis~30 — Gaussian under-covers the 2σ tail).
sklearn exact GP cross-check on a subsample. Eval on host-grouped val/test (no leakage).

**Results (Student-t, test split):** RMSE/std=1.03, NLL=4.58, cov68=0.66, cov95=0.91.
(Gaussian was NLL=6.18, cov95=0.84 — Student-t roughly halves NLL and fixes the tail.)
- **RMSE/std ≈ 1 is the correct, honest outcome, not a failure:** the held-out residual at a
  given (phase, orbit, y_rel) is **not point-predictable across stars** — stellar jitter is an
  independent per-system realization. The **exact GP also sits at RMSE≈std**, so this is the
  data's truth, not an SVGP limitation. The deliverable's value is a **calibrated probabilistic
  noise model** (feature-dependent predictive variance), confirmed by cov68≈nominal.

**Outputs:** `models/gp_residual_svgp.pt`, `models/gp_residual_metrics.json`,
`figures/gp_residual/{pred_vs_true,phase_residual,partial_dependence,calibration,svgp_vs_exact}.png`.
Residual build is cached in `data/gp_residual_cache/` (~13 min to build, instant on rerun).
New deps: `gpytorch==1.15.2`, `linear_operator==0.6.1` (added to requirements.txt).

**Flag to Nicolò:** offset `γ` anchored to the single (noisy) first observation per his literal
wording — its uncertainty is absorbed by the augmentation re-anchoring. If he meant a
least-squares `γ`, it's a one-line change in `_system_residual`.

**Generative validation (added same day, `generative_validation()` in the module):** sample the
posterior predictive at each held-out system's real feature rows; compare per-system std,
excess kurtosis, pooled self-standardized shape, and scatter-vs-phase against real residuals.
Results (test, 55 systems): **std ratio median 1.76, std log-corr ≈ 0** — the GP generates a
near-constant noise amplitude regardless of the system's true amplitude (real per-system std
spans ~1–100+ m/s). **Per-system kurtosis: real ≈ −0.3 (Gaussian-ish!), GP ≈ 2.0.**
Figures: `generative_validation_{test,val}.png`.

**The scientific finding:** the famous heavy tails of pooled RV residuals (kurtosis ~30) are a
**scale mixture across systems**, not heavy tails within systems — each system is roughly
Gaussian at its own amplitude; pooling different amplitudes manufactures the leptokurtosis.
The Student-t likelihood gets good pooled NLL/coverage by hedging with one global heavy tail,
but it is the wrong generative decomposition. Orbit features (P, K, e, ω, phase, y_rel)
**cannot predict noise amplitude** — amplitude is set by instrument precision + stellar
activity, not orbital geometry.

**Recommended fix (needs Nicolò sign-off — extends his literal feature spec):** condition on
the measurement uncertainty, which IS known at generation time (tabulated per obs; the
synthetic generators sample σ anyway). Either add `log10 σ(t)` as an 8th feature, or model the
normalized label `r/σ`. Expect std log-corr to jump and the Student-t df to grow (per-system
residuals are near-Gaussian once scale is known).

---

## Incoming changes (Shuaib, Jun 3–4 — already committed)

Three commits on top of `9a43898`, all by Shuaib, all pushed:

### `288560d` — empirical eccentricity prior (replaces the Beta fix)
- `synthetic_dataset.py`: e is **no longer Beta(0.867, 3.03)**. New `_sample_eccentricity` draws from a **zero-preserving empirical histogram** of real catalog eccentricities (30 bins over (0, 0.99] plus an explicit `p_zero` point mass at e=0). This reproduces the exact-zero pile-up that no smooth Beta can match — i.e. it directly attacks the `e` discriminator from the old classifier. Beta(0.867, 3.03) is now only the **fallback** when the corpus file is missing.
- Default validation output dir moved `data/synthetic_validation/` → **`figures/synthetic_validation/`**.

### `ab94e10` — tune priors against the validation corpus
- **Period prior**: `LogUniform(1, 3000)` → **3-component Gaussian mixture in log10(P/d)** (`_sample_period`). Hardcoded weights `[0.377, 0.242, 0.381]`, means `[0.51, 1.55, 2.80]`, stds `[0.178, 0.490, 0.335]`, clipped to [1, 3000] d. Modes ≈ 3.3 d, 35 d, 638 d.
- **K prior**: `LogUniform(1, 300)` → **`LogUniform(8, 400)` m/s** (`_K_MIN_MS=8`, `_K_MAX_MS=400`).
- Eccentricity prior source switched from `labels.csv` → **`splits.csv`** (filtered to `has_ecc==True & n_planets==1`), with `labels.csv` then Beta as fallbacks.
- His reported all-split classifier balanced accuracy: **0.787 → 0.734**.

### `10db715` — split-aware validation
- `validate_synthetic_dataset.py`: `collect_real(real_split=...)` + `--real-split {all,train,val,test}` CLI flag. Output auto-routes to `figures/synthetic_validation/real_<split>/` when a split is named.
- `make_classifier_report` now **returns a metrics dict**; balanced accuracy, top feature, full feature importances, and counts are persisted into `generation_mode_summary.json` and the README.
- Added per-split diagnostic dirs `real_train/`, `real_val/`, `real_test/` (full plot + CSV + JSON sets each).

### Classifier numbers at this point (split-aware, tuned priors) — SUPERSEDED by the LATEST section above

> These are the Jun 3–4 numbers, before GP-residual noise + real-cadence bootstrapping. Kept for the trend record. Use the **LATEST (2026-06-30)** table for anything current.

| Real split | Balanced acc | Top discriminator | n_real / n_synth |
|---|---|---|---|
| train | 0.668 ± 0.036 | `log10_K` | 242 / 400 |
| test  | 0.568 ± 0.023 | `log10_K` | 57 / 400 |
| val   | 0.500 ± 0.000 | `log10_P` | 58 / 400 |

- `e` is **no longer a top discriminator** (importance ~0.07–0.08 everywhere) — the empirical-histogram prior worked. At this stage the remaining signal was **`log10_K`**; by Jun 30 (real-σ bootstrapping) it has moved to `lsp_peak_power` / `rv_std_ms`.
- The suspicious `val = 0.500 ± 0.000` seen here is **resolved** in the Jun 30 run (now 0.522 ± 0.052 with real variance) — it was the degenerate small-n CV, and adding real-cadence/σ variety broke the degeneracy.

---

## What was completed in the earlier session (George + Claude, Jun 3)

### Bugs fixed in `synthetic_dataset.py`

1. **Eccentricity prior typo** (line 101): was `rng.beta(2, 5, ...)` cited as Kipping 2013, but Kipping (2013) MNRAS 434 L51 Table 1 fits **Beta(0.867, 3.03)** — a J-shaped distribution peaked at e=0. Beta(2,5) is peaked at e=0.2. Changed to the actual Kipping values. This is what Jovie was observing: synthetic e was too high because the prior was wrong.

2. **σ sampling collapsed per-system median** (`_sample_sigma`): was drawing σ **independently per observation** from LogN(μ, σ_pop), so the *median σ within a system* collapsed to ~4.62 m/s regardless of system identity. Real corpus has wide per-system spread (HARPS 0.5 m/s vs HIRES 3 m/s vs older surveys 10 m/s). Refactored to hierarchical draw:
   ```python
   log_sys = rng.normal(_SIGMA_LOG_MEAN, _SIGMA_LOG_STD)   # one draw per system
   log_obs = log_sys + rng.normal(0.0, 0.10, size=n_obs)   # 10% per-obs jitter
   ```
   `_SIGMA_OBS_JITTER_LOG_STD = 0.10` is a new module constant.

3. **`generate_one` info dict** now includes `t_peri` and `rv_med_ms` so downstream plots can reconstruct and overlay the exact Keplerian curve. Added without breaking existing consumers.

### New features in `synthetic_rv.py`

- `plot_examples(out_dir, n_examples)`: loads `manifest.csv`, computes per-system SNR, plots a stacked grid of highest- and lowest-SNR systems with the exact Keplerian curve overlaid on the noisy observations. CLI flag `--plot --plot-n N`.
- `train_real_vs_synthetic_classifier(real_labels, synth_dir, out_dir)`: sklearn RandomForest on (log10_P, log10_K, e) features; 5-fold balanced-accuracy CV. CLI flag `--classify`.

### New features in `validate_synthetic_dataset.py`

- `_overlay_exact_curve(ax, theta, info)`: helper used in `make_examples_pdf` to overlay the noiseless Keplerian curve on each synthetic example. Only fires when `t_peri` is present in info (real-data plots unaffected).
- `make_classifier_report(real, synth, out)`: RandomForest on 11 tabular features; saves feature-importance bar chart.
- `collect_real()` now takes `sigma_min=0.1`, `sigma_max=100.0` and rejects systems outside that σ range — same filter that `synthetic_rv.build_noise_pool` already uses. Removes 7 junk `.tbl` files (11 Com absolute-RV in m/s, HD 185269 σ=0.01 placeholder, etc.) from the comparison set.

### Classifier results (the headline numbers for the meeting)

| Pipeline | Balanced accuracy | Interpretation |
|---|---|---|
| `synthetic_rv.py` vs NASA catalog | **0.493 ± 0.003** | Indistinguishable by design (samples directly from catalog rows). ✓ |
| `synthetic_dataset.py` vs real RV corpus (before σ fix) | 0.875 ± 0.012 | Mostly driven by `median_sigma_ms` collapsed distribution |
| `synthetic_dataset.py` vs real RV corpus (after σ fix) | **0.806 ± 0.013** | Top discriminator now `e` (delta at 0 in real data); rest is intentional broad-prior pretraining mismatch |

### Outputs

All plots moved to `figures/synthetic_validation/`:

- `examples_with_exact_curve.png` — `synthetic_rv.py` grid, exact curve overlay
- `classifier_real_vs_synthetic.png` — `synthetic_rv.py` classifier
- `examples_single_planet.pdf` — `synthetic_dataset.py` per-sample RV+LSP with exact-curve overlay
- `real_vs_synthetic_parameters.png` — P/K/e/SNR/std/LSP-peak histograms
- `real_vs_synthetic_cadence.png` — n_obs, baseline, gap diagnostics
- `real_vs_synthetic_noise.png` — σ/std comparison (post-fix overlap is clean)
- `classifier_feature_importance.png` — `synthetic_dataset.py` classifier
- `lsp_examples.png` — synthetic LSP gallery with true-period marker

---

## Key takeaways for the meeting

> Items 1 and 3 below are partly superseded by Shuaib's work above: the e prior is now an **empirical histogram** (not any Beta), and the classifier numbers are the split-aware 0.668/0.568/0.500.

1. **Jovie's observation was real and traceable to a typo.** Beta(2,5) doesn't match Kipping (2013); Beta(0.867, 3.03) does. Fix is one line. Synthetic e was *too high* because (2,5) is peaked at 0.2 not 0. **(Shuaib then went further — replaced the Beta entirely with a zero-preserving empirical histogram, which also captures the e=0 pile-up.)**
2. **`synthetic_rv.py` is statistically sound.** Classifier acc = 0.493 confirms catalog resampling is working. The "exact curve" diagnostic revealed two unrelated issues: (a) catalog contains stellar-binary K > 9000 m/s entries that should be filtered, (b) low-K systems (K < 1 m/s) produce featureless plots because the signal is below the noise floor.
3. **`synthetic_dataset.py` is now much better but still 0.806 distinguishable** — and this is mostly **by design**. The pretraining priors are intentionally broader than the catalog (LogUniform P, K) to give the encoder a wide parameter space; the residual gap from `e` is the delta-at-zero in real data that no smooth prior can match.
4. **Don't read 0.806 as "synthetic is bad."** Read it as: "the encoder will see a broader distribution during pretraining than at fine-tune time, and the only structural mismatch we can't smoothly fix is real-data NaN-imputation."

---

## Project goals (unchanged)

1. ✅ RV literature
2. ✅ Data + validation pipeline
3. ✅ GP noise model, preprocess.py, encoder stack, synthetic dataset pipeline
4. ⏳ **NEXT** — Uncertainty quantification (conformal + Bayesian)

**Nicolò's autoencoder framing (do not deviate):**
- Encoder φ(RV) → orbital state X = (P, K, e, ω) [T_peri refitted analytically]
- Decoder = Kepler integrator (fixed, no learned weights)
- Loss = ‖RV − Kepler(φ(RV))‖

---

## Repository state

- Local: `~/rv-ml`, GitHub: `George-Pulickan/rv-ml` (private)
- Python 3.13 venv at `.venv`. `sklearn==1.9.0` installed this session.
- **George pushes himself** — commit but do not auto-push.
- `handover.md` is gitignored — local-only working doc.

### Files touched this session

| File | Change |
|---|---|
| `synthetic_dataset.py` | Beta prior fix, hierarchical `_sample_sigma`, `t_peri`/`rv_med_ms` in info dict |
| `synthetic_rv.py` | Added `plot_examples`, `train_real_vs_synthetic_classifier`, `--plot`/`--classify` CLI flags |
| `validate_synthetic_dataset.py` | `_overlay_exact_curve`, `make_classifier_report`, σ filter in `collect_real` |
| `figures/synthetic_validation/` | 8 new diagnostic plots (PNG + PDF) |

Earlier-session files were committed in `9a43898`. **Shuaib's three follow-up commits (`288560d`, `ab94e10`, `10db715`) are also already committed and pushed** — see the incoming-changes section. They further touch `synthetic_dataset.py` (empirical e prior, period mixture, K range) and `validate_synthetic_dataset.py` (split-aware, classifier metrics persisted), and add `figures/synthetic_validation/real_{train,val,test}/`.

---

## Technical pitfalls (additions from this session)

- **`generate_one` info dict adds `t_peri` and `rv_med_ms`** — these are required to overlay the exact curve in unit-correct un-normalised m/s.
- **`_sample_sigma` is now hierarchical** — if anyone changes back to per-obs draws, the per-system median σ will collapse and break the σ distribution match. The two-level structure is intentional.
- **`collect_real` sigma_min/sigma_max** — set narrower for a stricter validation set; loosen if you want to see junk files. Default mirrors `build_noise_pool`.
- **`synthetic_rv.py` does NOT filter K** — the manifest contains stellar-binary K > 9000 m/s entries straight from the catalog. Open issue: should add `1 ≤ K ≤ 1000 m/s` filter to `sample_params` if these systems aren't wanted for pretraining.
- **Classifier balanced accuracy 0.5 ≠ "calibration target"** for `synthetic_dataset.py`. It's the goal for `synthetic_rv.py` (catalog-faithful) but not for the encoder pretraining set (intentionally broader).
- **e prior is now empirical (Shuaib).** `_sample_eccentricity` reads `splits.csv` (`has_ecc & n_planets==1`), then `labels.csv`, then falls back to Beta(0.867, 3.03). Result is cached in `_ECC_CACHE` for the process lifetime — if you edit the corpus mid-run, the stale histogram persists.
- **Period prior is now a hardcoded 3-component log10 Gaussian mixture** (`_P_LOG10_*` constants), not LogUniform. **K is `LogUniform(8, 400)`**, not (1, 300). If you re-derive these from a different corpus, update the module constants.
- **Kipping (2013) actual values** are Beta(0.867, 3.03) — keep this consistent as the *fallback*. Other commonly cited values: Beta(0.697, 3.27) ("all RV" MLE) and Beta(1.12, 3.09) ("long-period"). Don't blindly write Beta(2, 5) — that's a different prior.

---

## Other unchanged technical pitfalls (carry-over from previous handover)

- `validate_one` takes `labels` as required positional arg.
- `preprocess.py` loads raw RV, not residuals.
- T_peri excluded from θ — analytically refit in decoder.
- Normalisation stats from train split only — `data/dataset_stats.json`.
- `encoder_loss` requires `stats` kwarg for ω gate.
- `fit_t_peri` mask parameter is required.
- n_obs < 10 → valid=False.
- MPS available on M3 but data generation is the CPU bottleneck.
- `--arch` flag must match checkpoint.
- Companion injection label = dominant planet (highest K).

---

## Next steps

1. **`lsp_peak_power` / `rv_std_ms` are the new top discriminators** (train 0.599, all 0.650). Now that params + σ + cadence are matched, the remaining gap is periodogram-power / residual-amplitude structure — tune `RVML_GP_RESIDUAL_SCALE` (default 0.85) and/or revisit the GP-residual amplitude decomposition (see the amplitude-conditioning fix in the GP-residual section — the GP can't predict per-system noise amplitude from orbit features; conditioning on σ is the recommended fix and would also help here).
2. **✅ DONE — 512-bin LSP resolution experiment** (see SESSION 2026-07-01 subsection). Verdict: resolution matters (512-bin LSP recovers P at R²=0.53, 64-bin fails); the LSP encodes period not K; but the extracted `lsp_peak_period_d` scalar still beats the raw spectrum for an RF, so the raw high-res LSP is where the NN adds value. **Follow-up:** confirm this on the NN side — does Jovie's RVEncoder extract *more* from the raw 512-bin LSP than the RF does (i.e., beat summary-only on P)? If yes, that's the concrete argument for the NN over the RF.
   - (Also resolved: the two regression datasets are reconciled — `random_forest_regressor.py` is the cautionary real-only/raw-spectrum baseline; `synthetic_generation/train_regression_models.py` on the 10k CSV is the real baseline, transfer R²=0.82/0.84 for P/K.)
3. **Decide on the e prior with Nicolò** (unchanged): empirical histogram vs parametric Kipping Beta — the empirical prior couples pretraining to the current catalog and won't generalise beyond it.
4. **GP-residual amplitude finding still needs Nicolò sign-off** — the SVGP now drives synthetic noise, but the generative-validation finding (std ratio ~1.76, per-system amplitude not predictable from orbit features) means the noise amplitude may be miscalibrated per-system. Conditioning on σ (8th feature or model `r/σ`) is the pending fix.
5. **`synthetic_rv.py` K filter still open** — that file was NOT touched; the unphysical 9000 m/s catalog entries remain. (`synthetic_dataset.py` is capped at 400 m/s, so this only affects `synthetic_rv.py`.)
6. **Task 4 — uncertainty quantification:** still pending. Encoder training (or post-training) needed first.
7. **Regenerate the pretrain cache** with the new pipeline (empirical e, period mixture, K∈[8,400], **GP-residual noise, real-cadence/σ bootstrapping**) before any fine-tuning. Old cache at `data/pretrain_cache.pt` (~3.1 GB) predates all of this.

---

## References (added this session)

- Kipping, D.M. 2013, MNRAS 434, L51 — **actual** values: Beta(0.867, 3.03) for RV eccentricity prior (Table 1).
