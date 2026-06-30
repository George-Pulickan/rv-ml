# Synthetic Generation Outputs

This folder contains the synthetic regression CSV workflow.

- `generate_synthetic_regression_csv.py` creates an input-output CSV from the main synthetic RV generator.
- `datasets/` stores generated CSV datasets.
- `validate_synthetic_regression_csv.py` runs structural and physical sanity checks on a generated CSV.
- `plot_synthetic_regression_csv.py` creates real-vs-synthetic comparison plots for the CSV.
- `validation/` stores validation reports for the generated CSVs.
- `figures/` stores real-vs-synthetic comparison figures.

The core generator remains in `synthetic_dataset.py` at the repository root because it is already imported by training,
validation, and cache-generation code elsewhere in the project.
