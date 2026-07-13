"""Canonical feature and target column definitions.

Keep model-facing column names in one lightweight module so the synthetic CSV
builder, regression baselines, and validation diagnostics do not drift apart.
"""

from __future__ import annotations

from time_series_features import phase_fold_feature_names, spectral_feature_names


SPECTRAL_DIM = 64
SPECTRAL_GRID_SIZE = 1024
PHASE_FOLD_N_BINS = 32

TARGET_COLUMNS = [
    "log10_P",
    "log10_K",
    "e",
    "cos_omega",
    "sin_omega",
]

SPECTRAL_COLUMNS = spectral_feature_names(SPECTRAL_DIM)
PHASE_FOLD_COLUMNS = phase_fold_feature_names(PHASE_FOLD_N_BINS)

SUMMARY_COLUMNS = [
    "n_obs",
    "baseline_d",
    "rv_std_ms",
    "rv_iqr_ms",
    "median_sigma_ms",
    "sigma_iqr_ms",
    "lsp_peak_period_d",
    "lsp_peak_power",
    "median_gap_d",
    "p90_gap_d",
]

# Backwards-compatible name used by validate_synthetic_dataset.py.
OBSERVATION_SUMMARY_FEATURES = SUMMARY_COLUMNS

BASE_74_COLUMNS = [*SPECTRAL_COLUMNS, *SUMMARY_COLUMNS]
PHASE_35_COLUMNS = PHASE_FOLD_COLUMNS
PHASE_109_COLUMNS = [*BASE_74_COLUMNS, *PHASE_FOLD_COLUMNS]

FEATURE_SET_COLUMNS: dict[str, list[str]] = {
    "74": BASE_74_COLUMNS,
    "35": PHASE_35_COLUMNS,
    "109": PHASE_109_COLUMNS,
}

CSV_COLUMNS = [*TARGET_COLUMNS, *BASE_74_COLUMNS]
CSV_COLUMNS_PHASEFOLD = [*CSV_COLUMNS, *PHASE_FOLD_COLUMNS, "has_t_peri"]


def feature_columns(feature_set: str) -> list[str]:
    """Return input column names for a named regression feature set."""
    if feature_set not in FEATURE_SET_COLUMNS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SET_COLUMNS)}")
    return FEATURE_SET_COLUMNS[feature_set]
