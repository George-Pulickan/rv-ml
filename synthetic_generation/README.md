# Synthetic Generation Outputs

This folder contains the synthetic regression CSV workflow.

- `generate_synthetic_regression_csv.py` creates an input-output CSV from the main synthetic RV generator.
- `datasets/` stores generated CSV datasets.
- `validate_synthetic_regression_csv.py` runs structural and physical sanity checks on a generated CSV.
- `plot_synthetic_regression_csv.py` creates real-vs-synthetic comparison plots for the CSV.
- `train_regression_models.py` fits the random-forest regression baseline (joint multi-output vs
  per-target separate models) on the CSV, ablates the feature blocks (summary / spectral / both),
  cross-validates, and measures synthetic-trained -> real transfer. Writes metrics + report to
  `regression/` and diagnostic figures to `figures/synthetic_regression_10000/`.
- `pca_real_vs_synthetic.py` computes a 2D PCA of the shared 74-D feature space and plots real
  (white) vs synthetic (black) systems, plus a feature-block ablation. Writes to `regression/`
  (coords + summary) and `figures/synthetic_regression_10000/`.
- `validation/` stores validation reports for the generated CSVs.
- `regression/` stores regression metrics, the PCA summary, and (with `--save-models`) fitted models.
- `figures/` stores real-vs-synthetic comparison figures.

The core generator remains in `synthetic_dataset.py` at the repository root because it is already imported by training,
validation, and cache-generation code elsewhere in the project.
