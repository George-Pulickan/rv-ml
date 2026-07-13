"""
Validate a synthetic RV regression CSV.

The checks here are intentionally lightweight and structural. They confirm that
the generated CSV has the expected columns, sensible numeric values, and basic
internal consistency before deeper scientific validation plots are produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_columns import (
    CSV_COLUMNS,
    SPECTRAL_COLUMNS,
    SUMMARY_COLUMNS,
)


def _record(checks: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _finite(values: pd.Series) -> pd.Series:
    return np.isfinite(values.to_numpy(dtype=float))


def _range_check(df: pd.DataFrame, column: str, lo: float, hi: float) -> tuple[bool, str]:
    values = df[column].to_numpy(dtype=float)
    ok = np.isfinite(values) & (values >= lo) & (values <= hi)
    detail = (
        f"{int(ok.sum())}/{len(values)} values within [{lo}, {hi}], "
        f"min={np.nanmin(values):.6g}, max={np.nanmax(values):.6g}"
    )
    return bool(ok.all()), detail


def validate_csv(path: Path, expected_rows: int | None = 10_000) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path)
    checks: list[dict[str, object]] = []

    _record(
        checks,
        "expected_columns",
        list(df.columns) == CSV_COLUMNS,
        f"found {list(df.columns)}",
    )
    if expected_rows is not None:
        _record(
            checks,
            "expected_row_count",
            len(df) == expected_rows,
            f"found {len(df)} rows, expected {expected_rows}",
        )

    missing_by_col = df.isna().sum().to_dict()
    _record(
        checks,
        "no_missing_values",
        int(df.isna().sum().sum()) == 0,
        f"missing values by column: {missing_by_col}",
    )

    numeric_ok = {}
    for col in CSV_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_ok[col] = bool(_finite(numeric).all())
    _record(
        checks,
        "all_columns_finite_numeric",
        all(numeric_ok.values()),
        f"finite numeric columns: {numeric_ok}",
    )

    ranges = [
        ("log10_P", -1.0, 4.0),
        ("log10_K", 0.0, 4.0),
        ("e", 0.0, 0.99),
        ("cos_omega", -1.0, 1.0),
        ("sin_omega", -1.0, 1.0),
        ("n_obs", 10.0, 1_000.0),
        ("baseline_d", 0.0, 10_000.0),
        ("rv_std_ms", 0.0, 10_000.0),
        ("rv_iqr_ms", 0.0, 10_000.0),
        ("median_sigma_ms", 0.0, 1_000.0),
        ("sigma_iqr_ms", 0.0, 1_000.0),
        ("lsp_peak_period_d", 0.5, 5000.0),
        ("lsp_peak_power", 0.0, 1.0),
        ("median_gap_d", 0.0, 10_000.0),
        ("p90_gap_d", 0.0, 10_000.0),
    ]
    for col, lo, hi in ranges:
        passed, detail = _range_check(df, col, lo, hi)
        _record(checks, f"{col}_range", passed, detail)

    n_obs = df["n_obs"].to_numpy(dtype=float)
    n_obs_integer = np.isclose(n_obs, np.round(n_obs))
    _record(
        checks,
        "n_obs_integer",
        bool(n_obs_integer.all()),
        f"{int(n_obs_integer.sum())}/{len(n_obs)} values are integer-like",
    )

    unit_radius = df["cos_omega"].to_numpy(dtype=float) ** 2 + df["sin_omega"].to_numpy(dtype=float) ** 2
    unit_error = np.abs(unit_radius - 1.0)
    _record(
        checks,
        "omega_unit_circle",
        bool((unit_error <= 1e-5).all()),
        f"max |cos^2 + sin^2 - 1| = {float(unit_error.max()):.6g}",
    )

    spectral = df[SPECTRAL_COLUMNS].to_numpy(dtype=float)
    spectral_nonnegative = np.isfinite(spectral) & (spectral >= 0.0)
    _record(
        checks,
        "spectral_power_nonnegative",
        bool(spectral_nonnegative.all()),
        f"{int(spectral_nonnegative.sum())}/{spectral.size} spectral values are finite and non-negative",
    )

    spectral_sums = spectral.sum(axis=1)
    _record(
        checks,
        "spectral_power_sum_leq_one",
        bool((spectral_sums <= 1.000001).all()),
        f"min row sum={float(spectral_sums.min()):.6g}, max row sum={float(spectral_sums.max()):.6g}",
    )

    summary_numeric = df[SUMMARY_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    _record(
        checks,
        "summary_features_complete",
        bool(summary_numeric.all()),
        f"{int(summary_numeric.sum())}/{len(summary_numeric)} rows have complete summary inputs",
    )

    summary = df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    failed = [check for check in checks if not check["passed"]]
    report = {
        "csv_path": str(path),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "passed": len(failed) == 0,
        "n_failed_checks": len(failed),
        "checks": checks,
    }
    return summary, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("synthetic_generation") / "datasets" / "synthetic_regression_10000.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("synthetic_generation") / "validation",
    )
    p.add_argument("--expected-rows", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary, report = validate_csv(args.csv, expected_rows=args.expected_rows)
    stem = args.csv.stem
    summary_path = args.out_dir / f"{stem}_column_summary.csv"
    json_path = args.out_dir / f"{stem}_validation.json"
    text_path = args.out_dir / f"{stem}_validation.txt"

    summary.to_csv(summary_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"CSV validation report: {args.csv}",
        f"Rows: {report['n_rows']}",
        f"Columns: {report['n_columns']}",
        f"Passed: {report['passed']}",
        "",
        "Checks:",
    ]
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {status}: {check['name']} ({check['detail']})")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote validation report to {json_path}")
    print(f"wrote column summary to {summary_path}")
    print(f"overall passed: {report['passed']}")


if __name__ == "__main__":
    main()
