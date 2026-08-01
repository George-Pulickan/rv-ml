# rv-ml

Predicting Keplerian orbital parameters from radial-velocity (RV) time series,
with **conformal prediction intervals that are calibrated on simulated data and
remain valid on real stars**.

The method is simulation-based split conformal prediction under covariate shift:
a point predictor ψ is trained on synthetic RV curves drawn from priors fitted to
the real training split; the conformity scores are calibrated on synthetic curves
only; and the calibration quantile is reweighted by a real-vs-synthetic
likelihood ratio (Tibshirani et al. 2019) so the coverage guarantee transfers to
real observations that were never used for calibration.

**Headline result** (`pke_20260726/gamma_real_val_1perhost`, ψ =
`checkpoints/psi_20260726/mlp74_s42.pt`, surrogate score + papernorm — see
[Verifying the results](#verifying-the-results)): at nominal 90% joint coverage
over `(log10_P, log10_K, e)`, the intervals achieve **0.93 on held-out synthetic**
and **1.00 on the 34 held-out real hosts**, with **no** vacuous intervals. The
intervals are honest but wide, and on real data conservative — that trade-off,
and its cause, is documented in [Limitations](#limitations).

> ⚠️ The numbers **committed under `synthetic_generation/regression/mlp_psi/`
> are superseded** (0.89/0.97, 1.8% vacuous, d = 4). They came from an earlier ψ
> checkpoint that was never reproducible from `main`, and their 1.8% vacuous rate
> was a float32 time-quantisation bug, now fixed. Do not mix pre- and
> post-2026-07-27 numbers; see [Reproducibility status](#reproducibility-status).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests     # 51 tests, ~1 s
python kepler_check.py                   # end-to-end pipeline validation on 51 Peg b
```

`data/` is committed (~10 MB: 1071 raw `.tbl` RV curves, labels, splits,
normalisation stats), so both commands work in a fresh clone. Large regenerable
artifacts — `checkpoints/`, the pretrain caches, the 512-bin LSP CSV — are not;
see [Reproducibility status](#reproducibility-status).

> Running `python -m unittest discover -s tests` overwrites the tracked PNGs in
> `figures/synthetic_plots/` (one test is an import-time plotting script).
> `git checkout figures/synthetic_plots/` afterwards.

---

## The pipeline

```
 raw RV (NASA Exoplanet Archive)
   └─ scripts/data/download_rv.py ────────────────► data/rv_raw/*.tbl
   └─ scripts/data/parse_and_label.py ────────────► data/labels.csv, data/rv_index.csv
        │   (TAP query + SIMBAD alias resolution)
        ▼
 validation & splits
   └─ kepler_check.py ────────────────────────────► forward-model check vs catalog
   └─ cache_residuals.py ─────────────────────────► data/residuals*.{npz,csv}
   └─ preprocess.py ──────────────────────────────► data/splits.csv, data/dataset_stats.json
        ▼
 noise model (fit to REAL residuals)
   └─ gp_residual_model.py ───────────────────────► models/gp_residual_svgp.pt
        ▼
 synthetic corpus (priors H fitted on the TRAIN split only)
   └─ synthetic_dataset.py ───────────────────────► pretrain cache / on-the-fly samples
   └─ synthetic_generation/generate_synthetic_regression_csv.py ──► regression CSVs
        ▼
 point predictor ψ
   └─ regression.py (MLP)  /  synthetic_generation/train_regression_models.py (RF baseline)
        ▼
 uncertainty quantification  ◄── the paper's contribution
   └─ conformal_shift.py ─────────────────────────► coverage/width tables, per-system widths
   └─ scripts/paper_rv_figures.py ────────────────► Fig 1, Fig 2, Earth-like table
   └─ scripts/paper_tables.py ────────────────────► cp_ablation.tex, cp_assumptions.tex
   └─ scripts/diagnostics/assumption_hessian.py ──► Assumption 3.2 (lambda_min, PD fraction)
   └─ scripts/bayesian_interval_comparison.py ────► CP vs catalog "Bayesian" intervals
```

**Held-out discipline.** `data/splits.csv` is a host-grouped 70/15/15 split
(seed 42), so no star appears in two splits. Everything that could leak is fitted
on `train` only: the normalisation stats, the empirical parameter priors H, the
cadence/σ bootstrap pool, the GP noise model, and the real-vs-synthetic
discriminator. Real `val`/`test` systems are used **only** to test intervals.

**Parameter vector.** θ = `[log10_P, log10_K, e, cos ω, sin ω]`. Periastron time
`t_peri` is deliberately excluded — it is an epoch-dependent phase offset that is
refit analytically inside the decoder, and including it would restrict the corpus
to the ~43% of systems with a catalog value. CP operates on the *physical*
coordinates `(log10_P, log10_K, e, ω)`, since `(cos ω, sin ω)` redundantly encode
one angle.

**ω is estimated but no longer reported** (Nicolò, 2026-07-26), so the paper's
reported dimension is **d = 3**: `(log10_P, log10_K, e)`. It is not identifiable
at this SNR: on the d = 4 runs its half-width ran from 3.9 rad up to 8.7 rad
against a 2π support, i.e. at or past the point where the interval wraps the
whole circle and says nothing. It
stays inside the model because e is parameterised through `(k, h) = (e cos ω,
e sin ω)`, and pinning an unidentifiable angle would bias the two coordinates
that *are* reported. `conformal_shift.py` still computes and prints all four;
the PKe-vs-PKew control is in `slurm/nicolo_20260726.sbatch`.

---

## Repository map

### Data acquisition and labelling

| File | Purpose |
|---|---|
| `scripts/data/download_rv.py` | Bulk-download RV time series from the NASA Exoplanet Archive → `data/rv_raw/` |
| `scripts/data/parse_and_label.py` | IPAC `.tbl` parser + TAP label query + SIMBAD alias matching → `labels.csv`, `rv_index.csv` |
| `parse_and_label.py` | Compatibility shim re-exporting the above (keeps root-level imports working) |
| `kepler_check.py` | Forward-models the RV curve from tabulated parameters and compares to observations; also the shared numpy Kepler solver |
| `cache_residuals.py` | Runs `kepler_check.validate_one` over the corpus, caches `(t, residual, σ)` per system |

### Preprocessing and features

| File | Purpose |
|---|---|
| `preprocess.py` | `RVDataset`, host-grouped splits, normalisation stats, 512-bin GLS periodogram (`compute_lsp`) |
| `time_series_features.py` | Fixed-length features for unevenly sampled series: spline→FFT spectral bins, phase-fold bins (`t_peri`-anchored or epoch-free) |
| `feature_columns.py` | **Canonical** column names for the 74-D / 35-D / 109-D feature sets — import from here, never redefine |

Feature sets: **74-D** = 64 spectral power bins + 10 observation summaries ·
**35-D** = 32 phase-fold RV bins + 3 shape scalars · **109-D** = 74 + 35.

### Noise model

| File | Purpose |
|---|---|
| `gp_residual_model.py` | **Canonical.** Global SVGP + Student-t fit to *real* residuals → `models/gp_residual_svgp.pt` |
| `gp_noise_model.py` | Per-system celerite2 GP (5 kernels, BIC selection, KS/Ljung-Box diagnostics) — *legacy fallback* |
| `scripts/gp/gp_corpus_fit.py`, `gp_sensitivity.py`, `gp_demo.py` | Corpus-wide celerite2 fit, threshold sensitivity, 3-system demo |

### Synthetic generation and validation

| File | Purpose |
|---|---|
| `synthetic_dataset.py` | **Canonical** generator for pretraining/CP: empirical priors, GP-residual noise, real-cadence bootstrap |
| `synthetic_generation/generate_synthetic_regression_csv.py` | Builds the 74-D (and 109-D with `--with-phasefold`) regression CSV; also `replay_synthetic_sample` for exact row replay |
| `synthetic_generation/generate_lsp_regression_csv.py` | Same corpus, storing the full 512-bin LSP (591 cols, gitignored — regenerate in ~30 s) |
| `validate_synthetic_dataset.py` | Real-vs-synthetic validation: classifier, histograms, cadence/noise diagnostics |
| `synthetic_generation/validate_synthetic_regression_csv.py` | Structural/physical sanity checks on a generated CSV |
| `synthetic_generation/plot_synthetic_regression_csv.py` | Real-vs-synthetic comparison plots; `collect_real_summary` builds real feature rows |
| `synthetic_generation/pca_real_vs_synthetic.py` | 2-D PCA of real (white) vs synthetic (black) systems |
| `scripts/legacy/synthetic_rv.py` | Separate catalog-resampling generator (300-system diagnostic sets) — **not** the pretraining source |

### Models and point predictors

| File | Purpose |
|---|---|
| `models/kepler_torch.py` | Differentiable Kepler decoder (Newton solve + analytic `t_peri`/γ refit). **No learned weights** |
| `models/encoder.py` | `RVEncoder` zoo — 7 dual-branch architectures (resnet/deep/tcn/inception/lstm/transformer/nolsp) |
| `theta_loss.py` | Shared losses: circular ω loss, ω gating on low-e, e-balance weights, θ↔h/k conversion |
| `regression.py` | **The paper's ψ.** MLP on 74/35/109-D features → θ or h/k. e-head variants, two-step pipeline, gates, ablations |
| `synthetic_generation/train_regression_models.py` | Random-forest baseline (joint vs separate, feature-block ablation, CV, synthetic→real transfer) |
| `train.py` | Two-phase encoder training (pretrain on synthetic → finetune on real) |
| `injection_recovery.py` | Injection-recovery benchmark on a (period × SNR) grid, classical-LS or encoder mode |

### Uncertainty quantification — the contribution

| File | Purpose |
|---|---|
| `conformal_shift.py` | **The paper's method.** Split-CP calibrated on synthetic, tested on real. Three score strategies × four normalisations, likelihood-ratio reweighting, Bonferroni over 4 coordinates |
| `conformal.py` | *Unsupervised* CP via the reconstruction residual ‖Kepler(θ) − y‖ (E1 coverage, E2 monotonicity, profiled scores). **Descoped from the paper** (Nicolò, 2026-07-14) but still runnable; it also supplies the shared `Scorer` / `make_real` / `make_synthetic` / `histogram_grids` helpers |
| `scripts/paper_rv_figures.py` | All paper deliverables, selectable with `--only`: `phasefold`, `predtrue`, `table`, plus three added 2026-07-25 on Nicolò's request — `mse` (per-planet reconstruction MSE, tabulated vs ours, with the GD θ\* floor), `boxes` (2-D conformal vs catalog regions), `trajectories` (unfolded RV time series). `--only boxes` needs no checkpoint; the rest do |
| `scripts/bayesian_interval_comparison.py` | CP half-widths vs the tabulated NASA `*err1/err2` intervals for held-out hosts |
| `scripts/paper_tables.py` | Emits `cp_ablation.tex` (strategy × normalisation, `naive`+`raw` as the plain split-CP baseline) and `cp_assumptions.tex` from a `conformal_shift_metrics.json`, so the LaTeX cannot drift from the run behind it |
| `scripts/diagnostics/assumption_hessian.py` | Assumption 3.2 diagnostic: is θ\* a strict local minimum of eq (2)? Hessian λ_min, PD fraction and κ(H) over prior draws, in **physical** coordinates (`--coords PKew` or `PKe`) |

### Diagnostics and tests

| File | Purpose |
|---|---|
| `synthetic_generation/regression_diagnostics.py` | SNR slicing, P-vs-baseline, LSP-vs-MLP period recovery, e-prior histogram, ω-vs-e pair plots, sanity JSON |
| `synthetic_generation/lsp_resolution_experiment.py` | 64-bin vs 512-bin power-spectrum recovery comparison |
| `synthetic_generation/eval_omega_nn_vs_rf.py` | ω recovery: trained encoder vs RF on matched real systems |
| `scripts/diagnostics/diagnostics.py`, `init_experiment.py` | Corpus-level plots; least-squares corrections to tabulated parameters |
| `scripts/legacy/random_forest_regressor.py` | Real-only, raw-spectrum RF (R² = −0.16) — kept as a cautionary baseline |
| `tests/` | 51 unit tests (see [Verifying the results](#verifying-the-results)) |

### Write-up and vendored material

| Path | Purpose |
|---|---|
| `paper/methodology_as_implemented.md` | What the pipeline actually does, written against the code rather than the draft. Where it disagrees with the Overleaf draft, **the draft is what needs changing** |
| `paper/end_to_end_pipeline.tex` | Draft §-subsection on the end-to-end pipeline. Numbers marked `[RERUN]` are placeholders — do not paste those into Overleaf |
| `paper/pendulum_simulated_experiment{,_supplement}.tex` | Draft write-up of the pendulum experiment |
| `pendulum/` | **Snapshot, not a fork to edit.** Nicolò's perturbed-pendulum toy experiment ([Exoplanets_2026](https://github.com/nicoloRHUL/Exoplanets_2026) @ `7db4c62`, taken 2026-07-27), plus the forced-pendulum and Bayesian variants added here. Re-sync by re-copying over the top; there is no shared history. **Its committed `Data/` matches neither committed script** (28 embedding columns implies `n_components = 10`; the scripts use 20 and 40), so the data and plots predate both and are not reproducible from the snapshot — see `pendulum/README.md` |

---

## Canonical choices

Several files do similar things. These are the live ones; the rest are baselines
or ablations kept for the record.

**Noise — `gp_residual_model.py`.** A single global sparse variational GP (512
inducing points, ARD Matérn-5/2) with a **Student-t** likelihood, fitted to real
single-planet residuals. Features are
`(phase, log10 P, log10 K, e, cos ω, sin ω, y_rel, log10 σ)`; the label is the
residual `r(t) = y(t) − ŷ(t)` against the catalog Keplerian with a
least-squares systemic offset γ. `y_rel` (RV *change* since the first
observation) is used rather than raw model RV, which would carry each star's
arbitrary systemic velocity. Quality cuts: median σ ∈ [0.1, 100] m/s and
residual RMS/σ ≤ 30. Training-set augmentation resamples ŷ ~ U(ŷ−σ, ŷ+σ) 20×;
val/test use the nominal residual.

`synthetic_dataset._inject_noise` prefers this SVGP, then falls back to the
per-system `GPNoiseLibrary` (`data/gp_fits.json`), then to i.i.d. Gaussian.
Query the live backend with `synthetic_dataset.get_noise_model_status()`.
Sample amplitude is scaled by `RVML_GP_RESIDUAL_SCALE` (default **0.85**).

**Generator — `synthetic_dataset.py`.** Priors, all fitted on the **train split**
of the real corpus:

| Quantity | Prior | Fallback if the corpus files are absent |
|---|---|---|
| P | empirical 40-bin histogram in log10(P/d) | 3-component log10 Gaussian mixture (modes ≈ 3.3, 35, 638 d) |
| K | empirical 40-bin histogram in log10(K) | LogUniform(8, 400) m/s |
| e | zero-preserving 30-bin histogram over (0, 0.99] **plus an explicit point mass at e = 0** | Beta(0.867, 3.03) (Kipping 2013) |
| ω | Uniform(0, 2π) for e > 0.05; **ω ≡ 0 for near-circular orbits** (degenerate) | — |
| cadence + σ | paired `(time grid, per-obs σ)` bootstrapped from real train `.tbl` files | seasonal Poisson-gap model + hierarchical log-normal σ |

Companions are injected with probability `f_multi = 0.30` (1 companion w.p. ¾,
2 w.p. ¼); the label is always the dominant planet (highest K), matching
`preprocess._usable_systems`. One noise realisation is shared across planets.

**Point predictor ψ — `regression.py`,** an MLP (hidden 128→64, ReLU, AdamW,
early stopping) on the 74-D feature set for the paper runs. ω is masked out of
the loss below e = 0.05 (degenerate) and scored only on e > 0.1. The
zero-inflated e prior (~24% exact zeros) has three opt-in counters: `--e-balance`
(inverse-frequency reweighting), `--e-head hurdle` (shared trunk + e>0
classifier), `--e-head dual` (separate circular/eccentric MLPs + gate).
`--targets hk` swaps `(e, cos ω, sin ω)` for `(k = e cos ω, h = e sin ω)`.

**UQ — `conformal_shift.py`.** Calibration uses synthetic curves only. Two score
strategies plus one adjustment:

- `naive` — `s_c = |ψ(y)_c − θ̄_c|` against the data-generating θ̄ (synthetic only);
- `surrogate` — `s_c = |ψ(y)_c − θ*_c|`, where θ* minimises the **squared**
  reconstruction error `Σ_t (y_t − Kepler(θ, t))²` — eq (2), the same loss whose
  Hessian Assumption 3.2 talks about — by Adam through the differentiable
  decoder, initialised at θ̄ (synthetic) / tabulated (real). Computable on real
  curves, hence usable under shift. An earlier L1 objective was inconsistent
  with eq (2); switching to squared error did **not** tighten the intervals;
- `naive_adj` — `naive` with the quantile shifted by the worst observed
  surrogate gap Δ_c (paper eq. 41).

Four score normalisations `s' = s/(γ + ·)`, with γ tuned per variant to minimise
support-normalised median width: `raw`, `vnorm` (GP predictive std), `v2norm`
(+ per-coordinate surrogate-label error model), `papernorm` (re-encode residual
δ_c and reconstruction residual δ_y, computed pointwise and combined by eq (17)'s
**convex** combination — the tuner selects mix = 1.0, i.e. δ_y contributes
nothing at the current settings). Coverage under shift uses
the Tibshirani et al. (2019) weighted quantile with likelihood ratios from a
logistic real-vs-synthetic discriminator on the 74-D summaries (deliberately not
ψ's 586-D set, which separates the classes too well and degenerates the weights);
weights are clipped to [1/20, 20] and the effective sample size is reported.

Two assumptions from the draft are checked empirically: **3.1** (bounded noise) —
synthetic draws exceeding the real-train reconstruction bound are discarded; and
**3.2** (θ\* is a strict local minimum of eq 2) — λ_min, the positive-definite
fraction and κ(H) are estimated on prior draws by
`scripts/diagnostics/assumption_hessian.py`.

> ⚠️ Two numbering traps. **The draft renumbered these 2.1 → 3.1 and 2.3 → 3.2;
> `conformal_shift.py` still prints the old numbers** in its report. And **ε
> means three different things** — `noise_filter.bound_rv_std` (a pointwise sup
> in rv_std units, the only ε any output records, ≈ 6.1) is *not* the loss-scale
> ε ≤ ‖y − h(θ)‖² that Theorem 3.6 and Lemma 3.3 need (≈ 504). Using the sup-norm
> value makes the theorem look fine; on the loss scale √(C_H·ε) ≈ 4.1, several
> times the log₁₀P half-width and beyond the support of e, so **Theorem 3.6 is
> vacuous at the measured constants.** Nothing reports √(C_H·ε) — re-derive it,
> on the loss scale.

The Hessian diagnostic must be run in the **physical** coordinates. In the
5-vector the decoder sees ω only through `atan2(sin ω, cos ω)`, so `(cos ω,
sin ω)` has an exact null direction along its radius and H is singular by
construction — an earlier κ(H) ~ 1e6 figure was that artefact, not physics.

---

## Verifying the results

Everything below runs from a fresh clone unless marked otherwise.

**1. The pipeline reproduces published RV curves.**

```bash
python kepler_check.py            # 51 Peg b: chi2_red = 1.31, RMS/sigma = 1.19
python kepler_check.py --all      # corpus-wide -> data/validation_summary.csv
```

51 Peg b is the gold-standard single-system check: the Kepler model traces the
data to within the measurement noise with **zero** free physical parameters (γ is
anchored at the first observation).

`--all` takes ~40 min over the full corpus and writes `data/validation_summary.csv`.
Measured 2026-07-25 on the committed data: 535 of 1071 files validate cleanly;
after the quality filter (`n_obs ≥ 10`, median σ ∈ [0.1, 100] m/s) **440 systems
have a median RMS/σ of 3.67** (median residual 18.1 m/s; 42.7% below 3, 62.3%
below 5). That excess over 1 is the stellar-activity floor that catalog
uncertainties do not include — not a pipeline error. The status breakdown
accounts for the rest: 260 files whose planets have no catalog `t_peri` (usable
only with `--fit-tperi`), 198 with no label match, 69 with neither K nor M sin i,
9 with no period.

The host join matches **857 of 1071** files (766 by direct identifier, 91
recovered via SIMBAD aliases); the 214 unmatched are predominantly 2MASS-designated
survey candidates absent from the confirmed-planet table.

**2. The synthetic corpus is hard to distinguish from real data.**

```bash
python validate_synthetic_dataset.py --real-split test
```

Balanced accuracy of a random-forest real-vs-synthetic classifier, last measured
2026-06-30: **0.498 on test**, 0.522 on val, 0.599 on train, 0.650 pooled — i.e.
indistinguishable on held-out data. (Train/pooled sit above 0.5 by design: the
pretraining priors are deliberately broader than the catalog.) These predate the
switch to empirical P/K histogram priors, so a fresh run will not match to three
decimals; the held-out figure should still sit near 0.5.

**3. The conformal intervals cover.** The paper's runs live under
`synthetic_generation/regression/pke_20260726/`, all at n_cal = 400, ψ =
`checkpoints/psi_20260726/mlp74_s42.pt`, coords `(log10_P, log10_K, e)`,
surrogate score + papernorm (tuner selects mix = 1.0), γ tuned on real val.
Joint coverage at nominal 0.90, and median half-widths at the same level:

| Real test set | synthetic | real | real (LR-weighted) | %inf | ESS | half-widths (real, weighted) |
|---|---|---|---|---|---|---|
| 51 series (`gamma_real_val`) | 0.922 | 0.980 | 0.961 | 0% | 289/400 | P 2.10 dex · K 0.97 dex · e 0.86 |
| **34 hosts, one series each** (`…_1perhost`) | 0.925 | 1.000 | 1.000 | 0% | 263/400 | P 1.82 dex · K 1.04 dex · e 0.82 |

The one-series-per-host variant is the headline: the corpus stores one row per RV
file, so the 51 surviving test series come from only 34 distinct stars (HD 179949
contributes four, 51 Peg three), and treating correlated repeats of one star as
independent test points overstates the evidence.

The Assumption-3.1 filter rejected 48% of synthetic draws (bound 6.29 rv_std
units) — it is cutting the Student-t heavy-tail realisations that real data never
shows, and it truncates the calibration distribution relative to pre-2026-07-14
runs.

> ⚠️ **Those output directories are not on `main`** — they are on branch
> `rhul-results-20260727-mlp`. What is committed here is the **superseded**
> `mlp_psi/` run from the older checkpoint (0.890 / 0.965, 1.8% vacuous, ESS
> 203/400, d = 4). Kept for the record, structurally identical, but **not
> comparable** to the table above and not to be quoted.

```bash
git show origin/rhul-results-20260727-mlp:synthetic_generation/regression/pke_20260726/gamma_real_val_1perhost/conformal_shift_report.txt
```

Two traps when re-running. `conformal_shift.py` writes to
`synthetic_generation/regression/` **by default**, so a local smoke run silently
creates a low-`n_cal` directory that looks like a real result — always check
`n_cal`, `psi` and `checkpoint` in the report header and metrics JSON before
quoting anything. And the report still prints the draft's **old** assumption
numbers (2.1, 2.3) where the text above uses 3.1 and 3.2.

**4. CP vs the tabulated "Bayesian" intervals.**

```bash
python scripts/bayesian_interval_comparison.py
```

CP intervals are **two to three orders of magnitude wider than the catalog's**
for P, and roughly one order wider for K and e.

> ⚠️ **Do not quote a ratio from the committed files without re-running.**
> `figures/paper/` holds two sets, from two *different* and now-superseded ψ
> trainings — neither is `mlp74_s42.pt`, the checkpoint behind every reported CP
> number:
>
> | Source | P | K | e | ω |
> |---|---|---|---|---|
> | `figures/paper/*` (pre-07-26 ψ) | 549× | 15.7× | 6.5× | 3.1× |
> | `figures/paper/regenerated_pinned/*` (07-26 ψ, since retracted) | 808× | 12.1× | 9.8× | 6.5× |
>
> Re-run the script against `checkpoints/psi_20260726/mlp74_s42.pt` to get the
> number the paper should carry. The ω row is moot either way now that d = 3.

The script prints its own caveats: the CP half-width is nearly constant across
systems (CV < 0.1), so the ratio reflects a fixed near-vacuous width rather than
per-system adaptivity, and the point predictor is one-sided on K and e. It also
reports ψ **systematically over-predicting e on 100% of held-out hosts** —
noticed 2026-07-26, still unexamined.

**5. Unit tests.**

```bash
python -m unittest discover -s tests    # 51 tests
```

Notable guards: `test_replay_synthetic_sample.py` pins the RNG-prefix contract
that makes CSV row replay exact (a bug here silently folded the wrong system);
`test_bayesian_interval_comparison.py` covers the CP-vs-catalog join on synthetic
frames, independent of any generated CSV.

---

## Limitations

State these plainly in the paper; they are properties of the problem, not bugs.

- **ω is not recovered without a periastron epoch.** Real systems lack catalog
  `t_peri`, and the epoch-free phase-fold anchor did not restore absolute ω
  (e R² ≈ 0.08 epoch-free vs ≈ 0.50 with oracle `t_peri`). The CP interval for ω
  was correspondingly vacuous — 3.9 to 8.7 rad against a 2π support — which is
  why ω is estimated but **not reported** as a CP coordinate.
- **The intervals are valid but wide.** For `log10_P` the ~1.8–2.2 dex half-width
  spans most of the prior support. The width is set by the weak nuisance point
  estimate the univariate CP conditions on, and by period aliasing — not by the
  noise scale. A σ-normalised (χ²) score did *not* tighten them, and profiling
  the nuisance coordinates left median widths unchanged at n = 40.
- **CP half-widths are nearly system-independent** (CV < 0.1 across held-out
  hosts), so the CP-vs-Bayes ratio should not be read as per-system UQ.
- **The residual GP cannot predict per-system noise amplitude from orbital
  geometry.** Generative validation found a std ratio ≈ 1.76 with ≈ 0 std
  log-correlation, which motivated adding `log10 σ` as a conditioning feature.
  The related scientific finding: the famous heavy tails of *pooled* RV residuals
  (excess kurtosis ~30) are a **scale mixture across systems**, not heavy tails
  within systems — each system is roughly Gaussian at its own amplitude.
- **The empirical priors couple the model to the current catalog** and will not
  generalise beyond it. This is a deliberate choice (all distributional
  assumptions come from H, not ad-hoc priors) with a real cost.
- **The surrogate label's contribution is currently unevidenced**, on three
  fronts: real coverage (n = 57 cannot resolve a one-system difference), a
  mean-structure misspecification test (`--test-f-multi`, confounded — companion
  injection drifts θ\* from θ̄ by construction, 1.5–3.5×), and a correlated-noise
  test (`--test-noise-frac`/`--test-noise-tau`), where naive and surrogate
  degrade identically. Per the paper's framing the argument is that the surrogate
  makes the shift *estimable*, not that it improves coverage — so coverage may
  never have been the right evidence.
- **Assumption 3.2 does not hold everywhere.** The Hessian at θ\* is
  positive-definite on ~88% of prior draws, and the measured rate is currently
  **inconsistent between runs** (an earlier WIP tree reported 0.64/0.76,
  degrading with GD steps; the full-scale sweep reports 0.88 flat across
  200/1000/4000 steps with λ_min median ~30). `assumption_hessian.py` never
  touches ψ, so this is a code-version difference, not the checkpoint. Resolve
  which is right before either number appears in the paper.

---

## Reproducibility status

**Committed and directly checkable:** `data/` (raw curves, labels, splits,
stats), `models/gp_residual_svgp.pt`, the 74-D and phase-fold regression CSVs,
**the six ψ checkpoints in `checkpoints/psi_20260726/`**, the (superseded) CP
outputs in `synthetic_generation/regression/mlp_psi/`, and `figures/paper/`.

⚠️ `figures/paper/` is a **mix of three ψ generations** — the top-level
`earthlike_top10.*` is current (d = 3, from `mlp74_s42.pt`), the top-level
`bayesian_interval_comparison.*` is the oldest, and `regenerated_pinned/` is the
retracted 07-26 training. Check each file's provenance before quoting it.

**Not committed** (gitignored, regenerable): the rest of `checkpoints/`, the
pretrain caches,
`synthetic_generation/datasets/synthetic_lsp_regression_10000.csv`.

Traps for anyone re-running things:

> ✅ **ψ is committed** (2026-07-27), so Table 1 is reproducible from a clone
> alone. `checkpoints/psi_20260726/` holds the six seeds trained by
> `slurm/nicolo_20260726.sbatch` step 1 (~76 KB each, 454 KB total, deliberately
> excepted from the `*.pt` ignore rule). **Seed 42 is the paper's**; the spread
> across seeds is what gets reported, and no single seed is presented as the
> model. `figures/paper/psi_checkpoint_provenance.txt` records every sha256, the
> training command, the commit (`d323631`), and the source-CSV hash.
> `regression.py` is deterministic per seed, so retraining at that commit
> reproduces the files. This closes
> [issue #10](https://github.com/George-Pulickan/rv-ml/issues/10), still open on
> GitHub and closable against that file.
>
> Two earlier pins from 2026-07-26 exist **outside** the repo and describe
> different trainings (sha `4c09059a…`, `30f40259…`). An earlier provenance note
> named the first of them as the paper checkpoint; that was wrong, and no
> reported number came from it.

> ⚠️ **Identify a checkpoint by sha256 and pass `--checkpoint` explicitly.**
> `checkpoints/regression_mlp_74.pt` — same naming, different file — is a
> 15-epoch smoke artifact that is a degenerate near-mean predictor (R² +0.055 /
> +0.112 / +0.002 on `log10_P`/`log10_K`/`e`, predictions varying at 5–29% of the
> spread of the truth, overall MSE 0.536 against a predict-the-mean baseline of
> 0.467). It is not on `main` — it lives on the `laptop-sync` transport branch —
> but on any checkout where it does exist, a `--psi mlp` run that omitted
> `--checkpoint` used to calibrate against it and still exit 0.
> `conformal_shift.py` now **requires** `--checkpoint` with `--psi mlp`, and every
> CP run records the path it used in `conformal_shift_metrics.json` — that record,
> not any prose, is the authority on what produced a given number.

> ⚠️ **The committed GP checkpoint is stale.** `models/gp_residual_svgp.pt` on
> `main` (1077253 bytes) predates the LS-γ, σ-conditioning and train-only-H
> changes, so the committed CSVs were drawn from full-corpus priors H rather than
> train-only H. The retrained σ-conditioned SVGP (1081541 bytes) lives on branch
> `cluster-wip-20260726`. `conformal_shift.py` loads this sampler for the CP noise
> draws, so **which one sits in `models/` changes the results** — and a
> `git reset --hard origin/main` silently reinstates the stale one. Check the
> byte size during preflight, not merely that the file exists.

> ⚠️ **Verify artifact freshness by mtime, not existence.** Several helper
> scripts guard themselves with existence checks that pass happily on stale
> outputs. Check what a file *is*, not that it is there.

Two further artifacts were **deleted** rather than left to mislead, and are
regenerated by the cluster jobs below:

- `synthetic_generation/regression/conformal_shift_{metrics.json,report.txt}` —
  an n_cal = 30 smoke run from 2026-07-04 with the RF ψ, trivially conservative
  and easily mistaken for the paper's result. The paper's numbers are one
  directory down in **`mlp_psi/`**; `conformal_shift.py` rewrites the root copy
  on its next run.
- `figures/regression_synthetic/benchmark.json` — computed before the replay fix
  with a bug that folded a different system per row, so its Gate C and two-step
  numbers were invalid. `slurm/regression_benchmark.sbatch` regenerates it.

Checkpoints record only normalisation arrays and dimensions in `norm_stats` — no
seed, epoch count, or source CSV — so two checkpoints cannot be told apart from
the files alone. `psi_checkpoint_provenance.txt` patches this by hand for the one
checkpoint that matters; stamping the training config at save time is still the
durable fix.

One more schema trap: `papernorm` has been recorded under two different key
layouts. Runs from the 2026-07-26 WIP tree write `{"gamma0": …, "mix": 1.0}`;
current runs write `papernorm_weight` with `selected`/`tuned`/`grid`. Same
substance (mix = 1.0 ≡ selected 1.0), different keys — anything parsing
`papernorm_weight` finds **nothing** in the older files and may silently fall
back to a default. Do not diff old against new runs key-by-key.

---

## Cluster jobs

Heavy runs go to the RHUL cluster, never a laptop. Submit from the repo root.

| Script | What it does | Est. wall |
|---|---|---|
| `slurm/gp_conformal.sbatch` | SVGP retrain (LS-γ + σ-conditioning) → regenerate both regression CSVs → `conformal_shift.py` at n_cal = 400 | 6–14 h, CPU |
| `slurm/regression_benchmark.sbatch` | Regenerate phase-fold CSV → replay guard → Gates A/B/C + ablations → two-step → 109-D diagnostics → e-head ablation | 3–6 h, CPU |
| `slurm/train_encoder.sbatch` | Two-phase encoder training (needs `data/pretrain_cache_v3.pt` and CUDA) | ~8 h, GPU |
| `slurm/nicolo_20260726.sbatch` | Everything from Nicolò's 2026-07-26 reply: multi-seed ψ retrain (skip with `RVML_SKIP_PSI=1` to reuse the committed checkpoints), full-scale CP on the PKe coordinates (four ways: γ on synthetic vs real-val × series-level vs one-series-per-host), the PKew control, and the Assumption-3.2 Hessian sweep at 200/1000/4000 GD steps | ~24 h, CPU |
| `slurm/run_cp_rerun.sh` | CP step only — assumes the GP checkpoint and CSVs are already regenerated on the box. Waits for any active runner, so it can be launched and left | ~2 h, CPU |
| `slurm/run_campus_followup.sh` | One-shot driver: `run_cp_rerun.sh`, then `regression_benchmark` with `srun` stripped | ~4 h, CPU |

Run `gp_conformal` **before** `regression_benchmark` — the latter consumes the
refreshed checkpoint. Set `--partition` (and `--account` if required) from
`sinfo` first; those files mark the lines `ADJUST`.

**There is no Slurm on the boxes actually reachable**, so the `.sbatch` files are
run with `srun` stripped. The `#SBATCH` headers are kept for the day a real
scheduler appears. The sed must handle indentation — some `srun` calls sit inside
`for` loops, and a column-0 anchor leaves them in place, which is exit 127 and,
under `set -e`, a job that dies at the first seed:

```bash
sed "s/^\([[:space:]]*\)srun /\1/" slurm/nicolo_20260726.sbatch | bash
```

Note `nohup` does **not** survive logout on those boxes; use an `at now` pattern.
Cluster host, account and access details are deliberately **not** in this public
repo — see `handover.md`.

Status: the 07-25, 07-26 and 07-27 jobs are **done** (all exit 0); the 07-27
seed sweep produced the committed ψ checkpoints and the `pke_20260726/` results
quoted above, on branch `rhul-results-20260727-mlp`. Still outstanding: a full-scale
encoder run evaluated with `injection_recovery.py` — it needs a GPU allocation
separate from the CPU jobs and is not on the paper's critical path.

---

## Coordination

- Project log (who is doing what): <https://docs.google.com/document/d/1OZliqxJH3tyKIoUy9zpJO3d2aDqwG9eJZJ9lcB3FvqU/edit>
- Overleaf draft: <https://www.overleaf.com/8188483955gysdcwmjrwhq#ac30a1>
- **Session-to-session working state, current results, open decisions with
  Nicolò, traps, and cluster access: `handover.md`.** It is the single source of
  truth for anything not derivable from the code. Deliberately **not** in this
  repo — `main` is public and it carries cluster access details. It is kept in a
  private gist; ask George for the link. Fetch it to the repo root before doing
  anything cluster-related.

Data sources: NASA Exoplanet Archive bulk RV download (1071 curves) and the
Planetary Systems table via TAP.

### References

- Kipping, D. M. 2013, MNRAS 434, L51 — eccentricity prior Beta(0.867, 3.03)
- Zechmeister & Kürster 2009, A&A 496, 577 — generalised Lomb–Scargle
- Foreman-Mackey et al. 2017, AJ 154, 220 — celerite
- Titsias 2009; Hensman et al. 2013, 2015 — sparse variational GPs
- Tibshirani, Barber, Candès & Ramdas 2019 — conformal prediction under covariate shift
- Howard et al. 2010, Science 330, 653 — RV multiplicity fraction
