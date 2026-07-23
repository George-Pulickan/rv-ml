"""Unit tests for the CP-vs-catalog (Bayesian) interval comparison scaffold."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bayesian_interval_comparison import (  # noqa: E402
    PARAM_SPECS,
    _symmetric_sigma,
    _wrap_to_pi,
    build_comparison,
    caveat_notes,
    summarize,
)


def _cp_row(**over) -> dict:
    base = dict(
        host="HD 1", pl_name="HD 1 b", split="test",
        P_pred_d=100.0, P_tab_d=100.0,
        K_pred_ms=10.0, K_tab_ms=10.0,
        e_pred=0.20, e_tab=0.20,
        omega_pred_rad=0.0, omega_tab_rad=0.0,
        halfwidth_log10_P_a01=0.30, halfwidth_log10_K_a01=0.30,
        halfwidth_e_a01=0.10, halfwidth_omega_a01=3.0,
    )
    base.update(over)
    return base


def _labels_row(**over) -> dict:
    base = dict(
        pl_name="HD 1 b", hostname="HD 1",
        pl_orbper=100.0, pl_orbpererr1=1.0, pl_orbpererr2=-1.0,
        pl_rvamp=10.0, pl_rvamperr1=0.5, pl_rvamperr2=-0.5,
        pl_orbeccen=0.20, pl_orbeccenerr1=0.05, pl_orbeccenerr2=-0.05,
        pl_orblper=0.0, pl_orblpererr1=20.0, pl_orblpererr2=-20.0,
    )
    base.update(over)
    return base


class TestHelpers(unittest.TestCase):
    def test_symmetric_sigma_averages_abs(self):
        self.assertAlmostEqual(_symmetric_sigma(2.0, -4.0), 3.0)

    def test_symmetric_sigma_one_sided(self):
        self.assertAlmostEqual(_symmetric_sigma(2.0, float("nan")), 2.0)

    def test_symmetric_sigma_missing_is_nan(self):
        self.assertTrue(math.isnan(_symmetric_sigma(float("nan"), None)))

    def test_wrap_to_pi(self):
        # Boundary maps to -pi (the (-pi, pi] edge); only the magnitude is used downstream.
        self.assertAlmostEqual(abs(_wrap_to_pi(3 * math.pi)), math.pi, places=6)
        self.assertAlmostEqual(_wrap_to_pi(0.5 * math.pi), 0.5 * math.pi, places=6)
        self.assertAlmostEqual(_wrap_to_pi(2 * math.pi + 0.3), 0.3, places=6)


class TestComparison(unittest.TestCase):
    def setUp(self):
        self.cp = pd.DataFrame([_cp_row()])
        self.labels = pd.DataFrame([_labels_row()])

    def test_shape_and_columns(self):
        comp = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        self.assertEqual(len(comp), len(PARAM_SPECS))
        for col in ("cp_halfwidth_cmp", "bayes_halfwidth_cmp", "width_ratio_cp_over_bayes",
                    "cp_covers_tab", "bayes_covers_pred"):
            self.assertIn(col, comp.columns)

    def test_log10_sigma_conversion(self):
        # sigma_dex = sigma_phys / (x * ln10). For P: 1.0 / (100 * ln10).
        comp = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        p = comp[comp["param"] == "P"].iloc[0]
        expected = 1.0 / (100.0 * math.log(10.0))
        self.assertAlmostEqual(p["bayes_halfwidth_cmp"], expected, places=9)

    def test_omega_degrees_to_radians(self):
        comp = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        w = comp[comp["param"] == "omega"].iloc[0]
        self.assertAlmostEqual(w["bayes_halfwidth_cmp"], math.radians(20.0), places=9)
        self.assertFalse(bool(w["omega_near_vacuous"]))  # 3.0 rad < pi

    def test_omega_near_vacuous_flag(self):
        # A half-width >= pi covers the whole wrapped circle -> flagged vacuous.
        cp = pd.DataFrame([_cp_row(halfwidth_omega_a01=3.2)])
        comp = build_comparison(cp, self.labels, sigma_scale=1.0)
        w = comp[comp["param"] == "omega"].iloc[0]
        self.assertTrue(bool(w["omega_near_vacuous"]))

    def test_sigma_scale_multiplies(self):
        c1 = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        c2 = build_comparison(self.cp, self.labels, sigma_scale=2.0)
        r1 = c1[c1["param"] == "e"].iloc[0]["bayes_halfwidth_cmp"]
        r2 = c2[c2["param"] == "e"].iloc[0]["bayes_halfwidth_cmp"]
        self.assertAlmostEqual(r2, 2.0 * r1, places=9)

    def test_coverage_when_pred_equals_tab(self):
        # pred == tab everywhere, so any positive half-width covers.
        comp = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        self.assertTrue(comp["cp_covers_tab"].all())

    def test_missing_catalog_sigma_is_nan_not_crash(self):
        labels = pd.DataFrame([_labels_row(pl_orbeccenerr1=float("nan"), pl_orbeccenerr2=float("nan"))])
        comp = build_comparison(self.cp, labels, sigma_scale=1.0)
        e = comp[comp["param"] == "e"].iloc[0]
        self.assertTrue(math.isnan(e["bayes_halfwidth_cmp"]))
        self.assertFalse(bool(e["bayes_covers_pred"]))

    def test_host_fallback_join(self):
        # Planet name mismatch, but hostname matches -> catalog still attached.
        cp = pd.DataFrame([_cp_row(pl_name="unmatched name")])
        labels = pd.DataFrame([_labels_row(pl_name="different b", hostname="HD 1")])
        comp = build_comparison(cp, labels, sigma_scale=1.0)
        p = comp[comp["param"] == "P"].iloc[0]
        self.assertTrue(np.isfinite(p["bayes_halfwidth_cmp"]))

    def test_summary_fractions_in_range(self):
        comp = build_comparison(self.cp, self.labels, sigma_scale=1.0)
        summary = summarize(comp)
        self.assertEqual(len(summary), len(PARAM_SPECS))
        self.assertTrue(((summary["cp_covers_tab_frac"] >= 0) & (summary["cp_covers_tab_frac"] <= 1)).all())

    def test_signed_error_sign(self):
        # pred > tab in log space -> positive pred_minus_tab_cmp for P.
        cp = pd.DataFrame([_cp_row(P_pred_d=200.0, P_tab_d=100.0)])
        comp = build_comparison(cp, self.labels, sigma_scale=1.0)
        p = comp[comp["param"] == "P"].iloc[0]
        self.assertAlmostEqual(p["pred_minus_tab_cmp"], math.log10(2.0), places=9)

    def test_diagnostic_columns_present(self):
        summary = summarize(build_comparison(self.cp, self.labels, sigma_scale=1.0))
        for col in ("cp_halfwidth_cv", "median_pred_minus_tab", "frac_pred_over_tab"):
            self.assertIn(col, summary.columns)

    def test_caveat_flags_constant_width(self):
        # Two systems with identical CP widths -> CV 0 -> "marginal" flag fires.
        cp = pd.DataFrame([_cp_row(host="A", pl_name="A b"), _cp_row(host="B", pl_name="B b")])
        labels = pd.DataFrame([_labels_row(pl_name="A b", hostname="A"),
                               _labels_row(pl_name="B b", hostname="B")])
        notes = caveat_notes(summarize(build_comparison(cp, labels, sigma_scale=1.0)))
        self.assertTrue(any("~constant across systems" in n for n in notes))

    def test_caveat_flags_one_sided_bias(self):
        # Every host over-predicts e -> one-sided bias flag fires.
        rows = [_cp_row(host=f"H{i}", pl_name=f"H{i} b", e_pred=0.5, e_tab=0.1) for i in range(5)]
        labs = [_labels_row(pl_name=f"H{i} b", hostname=f"H{i}") for i in range(5)]
        notes = caveat_notes(summarize(build_comparison(pd.DataFrame(rows), pd.DataFrame(labs), sigma_scale=1.0)))
        self.assertTrue(any("over-predicts e" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
