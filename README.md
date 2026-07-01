# rv-ml

ML pipeline for predicting exoplanet orbital parameters from radial velocity
time series. Encoder → embedding → continuous-output decoder, with conformal
prediction intervals.

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
└── synthetic_generation/  # regression-baseline & analysis sub-project (see below)
```

### Root modules by pipeline stage

**Data acquisition & labelling**
| File | Purpose |
|---|---|
| `download_rv.py` | Download all RV time series from the NASA Exoplanet Archive → `data/rv_raw/` |
| `parse_and_label.py` | Parse raw `.tbl` → ML-ready `(X, y)`; SIMBAD alias matching; writes `labels.csv`/`splits.csv` |
| `kepler_check.py` | Pipeline validator: forward-model Keplerian RV from tabulated params vs observations (51 Peg b canonical) |

**Preprocessing & features**
| File | Purpose |
|---|---|
| `preprocess.py` | `RVDataset`: normalised `(x, lsp, theta)` tensors, Lomb–Scargle periodogram, splits, `dataset_stats.json` |
| `time_series_features.py` | Fixed-length spectral + summary features for unevenly-sampled RV |

**Noise model (Gaussian Processes)**
| File | Purpose |
|---|---|
| `gp_residual_model.py` | Global SVGP + Student-t fit to real residuals (Nicolò's spec) → `models/gp_residual_svgp.pt` |
| `gp_noise_model.py` | Per-system celerite2 GP noise model |
| `gp_corpus_fit.py` / `gp_sensitivity.py` / `gp_demo.py` | Corpus-wide GP fit + kernel selection, threshold sensitivity, 3-system demo |
| `cache_residuals.py` | Cache `(t, residual, sigma)` per system for the residual GP |

**Synthetic data generation & validation**
| File | Purpose |
|---|---|
| `synthetic_dataset.py` | Synthetic RV generator for encoder pretraining (empirical priors, GP-residual noise, real-cadence bootstrap); `SyntheticRVDataset`, `generate_cache` |
| `synthetic_rv.py` | Catalog-resampling generator (300-system sets) + example/classifier plots |
| `validate_synthetic_dataset.py` | Real-vs-synthetic validation: classifier, histograms, split-aware diagnostics → `figures/synthetic_validation/` |

**Model & training**
| File | Purpose |
|---|---|
| `train.py` | Two-phase encoder training: pretrain on synthetic → finetune on real → `checkpoints/` |
| `injection_recovery.py` | Injection-recovery benchmark for a trained encoder |

**Diagnostics & misc**
| File | Purpose |
|---|---|
| `diagnostics.py` | Corpus-level diagnostic plots (RMS vs params, galleries, parameter histograms) |
| `init_experiment.py` | Quantify least-squares corrections to tabulated params (Nicolò's request) |
| `random_forest_regressor.py` | Standalone RF (log10_P from 64 spectral features on real data) — cautionary baseline |
| `test_*.py` | Unit tests (parser, time-series features, 300-system generation) |

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
| `pca_real_vs_synthetic.py` | 2D PCA of real (white) vs synthetic (black) systems |
| `lsp_resolution_experiment.py` | 64-bin vs 512-bin power-spectrum recovery comparison |
| `eval_omega_nn_vs_rf.py` | ω recovery: trained NN encoder vs RF, on matched real systems |
| `datasets/`, `figures/`, `regression/`, `validation/` | Generated CSVs, figures, metrics/reports |

> **Note:** `data/` and `checkpoints/` are gitignored — regenerate them with `download_rv.py`
> / `parse_and_label.py` / `preprocess.py` and `train.py`. The trained residual noise model
> (`models/gp_residual_svgp.pt`) *is* committed.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python download_rv.py    --out data/rv_raw
    python parse_and_label.py --rv-dir data/rv_raw --out data/labels.csv

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
