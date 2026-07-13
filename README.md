# rv-ml

ML pipeline for predicting exoplanet orbital parameters from radial velocity
time series. Encoder → embedding → continuous-output decoder, with conformal
prediction intervals.

## 📋 Project log & coordination — read before starting work

Current steps, task assignments, and status updates live in the shared project log:

**https://docs.google.com/document/d/1OZliqxJH3tyKIoUy9zpJO3d2aDqwG9eJZJ9lcB3FvqU/edit**

Check it (and post an update) before you start a task, so work isn't duplicated. The
"Current state" section below tracks the canonical code/models; the Google Doc tracks the
*who/what/next* of the ongoing work. Session-to-session working state (immediate next
actions, pending decisions) lives in `handover.md`.

## Repository structure

The pipeline flows: **download real RV → parse/label → validate against Kepler →
fit a noise model + priors → generate synthetic training data → train the
encoder (pretrain on synthetic, finetune on real) → quantify uncertainty (CP).**

```
rv-ml/
├── data/                  # (gitignored) real RV .tbl files, labels, splits, stats, GP fits
│   ├── rv_raw/            #   raw NASA Exoplanet Archive RV time series (.tbl)
│   ├── labels.csv         #   tabulated Keplerian parameters per system
│   ├── splits.csv         #   train/val/test assignment (single-planet aware)
│   ├── dataset_stats.json #   train-split normalisation stats (used everywhere)
│   └── gp_fits.json       #   per-system GP noise fits
├── models/                # model code + the trained noise-model checkpoint
│   ├── encoder.py         #   RVEncoder architectures (resnet/deep/tcn/lstm/transformer/…)
│   ├── kepler_torch.py    #   KeplerDecoder — differentiable Kepler RV integrator (fixed decoder)
│   └── gp_residual_svgp.pt#   trained global SVGP residual noise model (committed)
├── checkpoints/           # (gitignored) trained encoder checkpoints (*.pt)
├── figures/               # diagnostic + real-vs-synthetic validation plots
├── slurm/                 # sbatch scripts for cluster (RHUL GPU) training
└── synthetic_generation/  # regression-baseline & analysis sub-project (see below)
```

### Root modules by pipeline stage

**Data acquisition & labelling**
| File | Purpose |
|---|---|
| `scripts/data/download_rv.py` | Download all RV time series from the NASA Exoplanet Archive → `data/rv_raw/` |
| `scripts/data/parse_and_label.py` | Parse raw `.tbl` → ML-ready `(X, y)`; SIMBAD alias matching; writes `labels.csv`/`splits.csv` |
| `kepler_check.py` | Pipeline validator: forward-model Keplerian RV from tabulated params vs observations (51 Peg b canonical) |

**Preprocessing & features**
| File | Purpose |
|---|---|
| `preprocess.py` | `RVDataset`: normalised `(x, lsp, theta)` tensors, Lomb–Scargle periodogram, splits, `dataset_stats.json` |
| `time_series_features.py` | Fixed-length spectral + summary features for unevenly-sampled RV |
| `feature_columns.py` | Canonical target/input column names for 74-D, 35-D phase-fold, and 109-D regression feature sets |

**Noise model (Gaussian Processes)**
| File | Purpose |
|---|---|
| `gp_residual_model.py` | Global SVGP + Student-t fit to real residuals (Nicolò's spec; least-squares systemic offset γ, σ-conditioned via a log10 σ feature) → `models/gp_residual_svgp.pt` |
| `gp_noise_model.py` | Per-system celerite2 GP noise model |
| `scripts/gp/gp_corpus_fit.py` / `scripts/gp/gp_sensitivity.py` / `scripts/gp/gp_demo.py` | Corpus-wide GP fit + kernel selection, threshold sensitivity, 3-system demo |
| `cache_residuals.py` | Cache `(t, residual, sigma)` per system for the residual GP |

**Synthetic data generation & validation**
| File | Purpose |
|---|---|
| `synthetic_dataset.py` | Synthetic RV generator for encoder pretraining (empirical priors, GP-residual noise, real-cadence bootstrap); `SyntheticRVDataset`, `generate_cache` |
| `scripts/legacy/synthetic_rv.py` | Catalog-resampling generator (300-system sets) + example/classifier plots |
| `validate_synthetic_dataset.py` | Real-vs-synthetic validation: classifier, histograms, split-aware diagnostics → `figures/synthetic_validation/` |

**Model & training**
| File | Purpose |
|---|---|
| `train.py` | Two-phase encoder training: pretrain on synthetic → finetune on real → `checkpoints/` |
| `slurm/train_encoder.sbatch` | RHUL GPU batch job wrapping `train.py` (pretrain 300 ep on the synthetic cache → finetune 100 ep on real); submit from repo root with `sbatch slurm/train_encoder.sbatch` |
| `slurm/gp_conformal.sbatch` | RHUL CPU batch job: full-scale SVGP retrain (LS-γ offset) → full-scale `conformal_shift.py` (n_cal=400); submit from repo root with `sbatch slurm/gp_conformal.sbatch` |
| `injection_recovery.py` | Injection-recovery benchmark for a trained encoder |
| `regression.py` | MLP regression on 74-dim encoder features → 5 Kepler params (default: 74-D CSV). For **e / ω**, use `--feature-set 109` with the phasefold CSV (`--with-phasefold` in `generate_synthetic_regression_csv.py`). Run **`--diagnose`** for SNR/P-baseline/LSP/sanity diagnostics before changing targets. |

**Uncertainty quantification (Step 6)**
| File | Purpose |
|---|---|
| `conformal.py` | Unsupervised conformal prediction: turns the Step-5 regressor's point predictions into prediction sets via the reconstruction-residual score `‖Kepler(θ)−y‖` (no ground-truth θ). Runs coverage (E1) + monotonicity (E2). Score variants: profiled over nuisance coords (`--profile {none,K,Keomega}`, default K) and σ-normalized χ² (`--chi2`, opt-in) |
| `conformal_shift.py` | Split-CP calibrated on fake data, tested on real (Nicolò's 2026-07 spec): naive score `\|ψ(y)−θ̄\|` (ground-truth θ̄) vs surrogate score (θ* = argmin of the L1 reconstruction error by gradient descent, init at θ̄/tabulated), likelihood-ratio reweighting `p_real/p_fake` via a real-vs-fake discriminator (Tibshirani et al. 2019 weighted quantile), and the normalized scores `s/(γ+v_y)` and two-factor `s_c/(γ+v_y+v_c)` (v_c = surrogate-label-error model) with tuned γ. ψ trains on the 512-bin raw-LSP dataset by default (feature columns follow `--csv`) |

**Diagnostics & misc**
| File | Purpose |
|---|---|
| `scripts/diagnostics/diagnostics.py` | Corpus-level diagnostic plots (RMS vs params, galleries, parameter histograms) |
| `scripts/diagnostics/init_experiment.py` | Quantify least-squares corrections to tabulated params (Nicolò's request) |
| `scripts/legacy/random_forest_regressor.py` | Standalone RF (log10_P from 64 spectral features on real data) — cautionary baseline |
| `tests/test_*.py` | Unit tests and smoke checks (parser, time-series features, 300-system generation) |

### `synthetic_generation/` — regression baselines & real-vs-synthetic analysis

A self-contained sub-project (its own `README.md`). Builds an input→output regression
CSV (power spectrum + summary features → true Keplerian params) and analyses it.

| File | Purpose |
|---|---|
| `generate_synthetic_regression_csv.py` | Build the 74-D regression CSV (64 spectral bins + 10 summaries → 5 targets) |
| `generate_lsp_regression_csv.py` | Variant storing the full 512-bin Lomb–Scargle spectrum (resolution experiment) |
| `validate_synthetic_regression_csv.py` | Structural/physical sanity checks on a CSV |
| `plot_synthetic_regression_csv.py` | Real-vs-synthetic comparison plots (+ `collect_real_summary`) |
| `train_regression_models.py` | RF regression baseline: joint vs separate, feature-block ablation, CV, synthetic→real transfer |
| `regression_diagnostics.py` | Automated MLP diagnostics: SNR-sliced errors, P/baseline identifiability, LSP vs MLP period recovery, e-prior histogram, raw-output saturation check, sanity JSON |
| `pca_real_vs_synthetic.py` | 2D PCA of real (white) vs synthetic (black) systems |
| `lsp_resolution_experiment.py` | 64-bin vs 512-bin power-spectrum recovery comparison |
| `eval_omega_nn_vs_rf.py` | ω recovery: trained NN encoder vs RF, on matched real systems |
| `datasets/`, `figures/`, `regression/`, `validation/` | Generated CSVs, figures, metrics/reports |

> **Note:** `data/` and `checkpoints/` are gitignored — regenerate them with `download_rv.py`
> / `parse_and_label.py` / `preprocess.py` and `train.py`. The trained residual noise model
> (`models/gp_residual_svgp.pt`) *is* committed.

## Current state — canonical models & configuration

This project has several files that do similar things; the list below is the **currently
canonical** choice for each stage, so collaborators build on the live path, not a legacy one.
(Query the live noise backend at runtime with `synthetic_dataset.get_noise_model_status()`.)

**Noise model — global SVGP + Student-t residual GP.**
`gp_residual_model.py` → checkpoint `models/gp_residual_svgp.pt` (512 inducing points, ARD
Matérn-5/2, Student-t likelihood; fit to real single-system residuals, with a least-squares
systemic offset γ and a **log10 σ conditioning feature** — the per-obs measurement uncertainty,
so the GP can track per-system noise amplitude; both per Nicolò 2026-07. The *committed*
checkpoint predates these changes (7 features, first-obs γ) and is
refreshed by `slurm/gp_conformal.sbatch`; consumers detect the feature set from the
checkpoint's `feature_names`). It is the **primary**
backend in `synthetic_dataset._inject_noise`, which falls back in order to: the per-system
celerite2 `GPNoiseLibrary` (`data/gp_fits.json`, produced by `gp_noise_model.py` /
`gp_corpus_fit.py` — *legacy/fallback*) → i.i.d. white Gaussian. GP-sample amplitude is scaled
by env `RVML_GP_RESIDUAL_SCALE` (default **0.85**). The σ feature targets the known
per-system-amplitude miscalibration (generative-validation std ratio 1.76 with ~zero
std log-correlation) — verify std log-corr on the retrained checkpoint.

**Synthetic generator — `synthetic_dataset.py`** (canonical for encoder pretraining). Current
priors, all bootstrapped from the **train split** of the real corpus (Nicolò, 2026-07: H is
fit train-only; real val/test are held out for testing the CP intervals):
- **Eccentricity:** zero-preserving empirical histogram (30 bins over (0, 0.99] + explicit point
  mass at e=0) from `data/splits.csv` (`has_ecc` single-planet); Beta(0.867, 3.03) fallback.
- **Period:** 3-component log10 Gaussian mixture (modes ≈ 3.3, 35, 638 d).
- **Semi-amplitude K:** LogUniform(8, 400) m/s.
- **Cadence + σ:** bootstrapped paired `(time grid, per-obs σ)` from real training `.tbl` files.
- `synthetic_rv.py` is a **separate** catalog-resampling generator (300-system diagnostic sets),
  *not* the pretraining source — don't confuse the two.

**Encoder — `models/encoder.py`, default arch `resnet`** (registry: resnet/deep/tcn/inception/
lstm/transformer/nolsp). Decoder is the fixed `models/kepler_torch.py:KeplerDecoder` (no learned
weights; refits `t_peri` analytically). Train with `train.py` (pretrain on synthetic → finetune
on real); checkpoints save as `checkpoints/<arch>_finetune_best.pt`. **Regenerate the pretrain
cache** (e.g. `generate_cache(...)`) before a real run — an old `data/pretrain_cache.pt` predates
the current priors.

**Regression baseline — `synthetic_generation/train_regression_models.py`** on
`datasets/synthetic_regression_10000.csv` (74-D: 64 spectral bins + 10 summaries → 5 targets).
Baseline for the encoder task; key result — summaries recover P/K well, the raw power spectrum
only helps at full 512-bin resolution (`lsp_resolution_experiment.py`).

**Regression diagnostics — run before changing targets or priors:**

```powershell
.venv\Scripts\python.exe regression.py --diagnose --feature-set 109 `
  --csv synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv `
  --checkpoint checkpoints/regression_mlp_109.pt
```

Outputs land in `figures/regression_synthetic/diagnostics/`:

| Artifact | What it answers |
|---|---|
| `metrics_by_snr.json`, `pred_vs_true_*_by_snr*.png` | Is error dominated by low-SNR systems? |
| `p_baseline_metrics.json` | Does P error blow up when P > baseline? |
| `period_recovery.json` | Does Lomb–Scargle argmax beat the MLP on period? |
| `raw_output_hist.png` | Are ±1 pileups from unit-circle projection vs saturation? |
| `e_prior_train_hist.png` | Is e banding from the discrete histogram prior? |
| `sanity_report.json` | Train/val gap, residual vs covariates, ω MAE vs e, leakage notes |

**Uncertainty quantification (the paper's main contribution) — implemented in `conformal.py`.**
*Unsupervised* conformal prediction: the conformity score is the reconstruction residual
`‖Kepler(θ) − y‖` (evaluated via the fixed `KeplerDecoder`, no ground-truth θ), with split-conformal
calibration + Bonferroni over the four physical coordinates (log10_P, log10_K, e, ω). All
calibration draws and parameter search grids come from the empirical corpus histograms H, not
ad-hoc priors. **Result: coverage is valid (≥ nominal) on synthetic AND real data** (the guarantee
transfers — the point vs Baragatti's supervised calibration). The sets are currently *valid but
wide*: a σ-normalized (χ²) score did **not** tighten them, because the width is limited by the
weak nuisance point-estimate the univariate CP conditions on (+ period aliasing), not the noise
scale. A *profiled* conformity score (minimise over nuisance coords instead of fixing at θ̂;
`--profile K` / `--profile Keomega`) is now implemented; in a quick run (n=40) profiling K left
the median widths unchanged — a full-scale run and/or a stronger point predictor is the open
question. See the Overleaf draft (§2.2.1) linked at the bottom.

**Nicolò's 2026-07 CP spec is implemented in `conformal_shift.py`** — the paper's comparison is
now: split-CP calibrated *only on fake data* and tested on real, (i) naive score `|ψ(y)−θ̄|`
with the ground-truth generating parameter θ̄ vs (ii) the surrogate-label strategy — per his
Slack follow-up, θ* = argmin_θ E_t|y_t−Kepler(θ,t)| solved by Adam gradient descent through the
differentiable decoder, initialized at the data-generating (synthetic) / tabulated (real)
values — the latter reweighted by the likelihood ratio `p_real/p_fake`
(estimated by a real-vs-fake logistic discriminator; weighted quantile per Tibshirani et al.
2019). Two normalized-score variants (γ tuned per variant on a synthetic tuning set):
`s/(γ+v_y)` (v_y = SVGP predictive-std proxy; Nicolò confirmed this reading of his `s/(γ+s)`)
and the two-factor `s_c/(γ+v_y+v_c)` with v_c a per-coordinate RF model of the surrogate-label
error E|θ̄_c−θ*_c| fit on the tuning set. ψ defaults to the 512-bin raw-LSP feature set
(Nicolò OK'd more Fourier bins); the weight discriminator and the v_c model deliberately stay
on the 74-dim summary features to keep the likelihood-ratio weights non-degenerate.

**Next steps:** (1) full-scale SVGP retrain + `conformal_shift.py` run on the RHUL cluster
(`slurm/gp_conformal.sbatch`; the committed checkpoint/CSVs still predate the LS-γ,
σ-conditioning and train-only-H changes — heavy jobs run on the cluster, not locally);
(2) full-scale profiled-CP
run (`--profile Keomega`, default n=400) + a stronger point predictor to tighten the sets;
(3) a full-scale encoder training
run (`slurm/train_encoder.sbatch` on the RHUL GPU cluster; needs the regenerated
`data/pretrain_cache_v3.pt`), evaluated with `injection_recovery.py`. Nicolò is writing the
full-pipeline section in the Overleaf draft — treat it as the reference spec when it lands.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python scripts/data/download_rv.py    --out data/rv_raw
    python scripts/data/parse_and_label.py --rv-dir data/rv_raw --out data/labels.csv

## Data sources

- NASA Exoplanet Archive bulk RV download (1,072 RV curves)
- NASA Exoplanet Archive Planetary Systems table via TAP service

## Validation

The pipeline is validated end-to-end by forward-modeling the radial velocity
signal from each host's tabulated Keplerian parameters and comparing to the
published observations:

    python kepler_check.py            # canonical test (51 Peg b)
    python kepler_check.py --all      # corpus-wide summary

51 Peg b is the gold-standard test (χ²_reduced = 1.31, RMS/σ = 1.19); the
Kepler model traces the data within measurement noise. Across the full
corpus, 432 quality-filtered systems validate with a median RMS/σ of 3.7,
consistent with the stellar-activity floor that catalog uncertainties do
not include. The pipeline matches 857 of 1,071 files to known planet
hosts (766 by direct identifier matching, plus 91 recovered via SIMBAD
alias resolution); the remaining 214 unmatched files are predominantly
2MASS-designated survey candidates that don't appear in NASA's confirmed-
planet table.

## Overleaf draft

A work-in-progress draft about the methodology and related work is here: https://www.overleaf.com/8188483955gysdcwmjrwhq#ac30a1
