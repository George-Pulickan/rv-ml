"""Tests for period fold helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression import (  # noqa: E402
    DatasetBundle,
    apply_log10_p_jitter,
    lsp_peak_log10_P,
    resolve_fold_log10_P,
)


class TestResolveFoldLog10P(unittest.TestCase):
    def setUp(self):
        self.true = np.array([1.0, 2.0, 3.0])
        self.pred = np.array([1.01, 2.2, 2.95])
        self.lsp = np.array([1.0, 2.01, 3.5])

    def test_oracle(self):
        out = resolve_fold_log10_P("oracle", true_log10_P=self.true)
        np.testing.assert_allclose(out, self.true)

    def test_mlp74(self):
        out = resolve_fold_log10_P("mlp74", pred_log10_P=self.pred)
        np.testing.assert_allclose(out, self.pred)

    def test_lsp_peak(self):
        out = resolve_fold_log10_P("lsp_peak", lsp_log10_P=self.lsp)
        np.testing.assert_allclose(out, self.lsp)

    def test_hybrid_uses_lsp_when_close(self):
        out = resolve_fold_log10_P(
            "hybrid", pred_log10_P=self.pred, lsp_log10_P=self.lsp
        )
        # close -> lsp, far -> mlp
        np.testing.assert_allclose(out, np.array([1.0, 2.2, 2.95]))


class TestJitterAndLspColumn(unittest.TestCase):
    def test_jitter_shape(self):
        base = np.linspace(0.5, 2.5, 20)
        resid = np.array([-0.1, 0.05, 0.2])
        out = apply_log10_p_jitter(base, resid, np.random.default_rng(0))
        self.assertEqual(out.shape, base.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_lsp_peak_from_bundle(self):
        df = pd.DataFrame({"lsp_peak_period_d": [10.0, 100.0, 1000.0]})
        bundle = DatasetBundle(
            np.zeros((3, 2)),
            np.zeros((3, 5)),
            row_idx=np.array([0, 2]),
            e=np.zeros(2),
            has_t_peri=np.ones(2),
            has_ecc=np.ones(2, dtype=bool),
            df=df,
        )
        out = lsp_peak_log10_P(bundle)
        np.testing.assert_allclose(out, np.log10([10.0, 1000.0]))


if __name__ == "__main__":
    unittest.main()
