"""Omega scatter panels ignore near-circular rows (e <= 0.1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression import OMEGA_EVAL_E_MIN, _omega_eval_mask, _omega_panel_arrays  # noqa: E402


class TestOmegaEvalFilter(unittest.TestCase):
    def test_mask_uses_e_gt_0_1(self):
        y = np.column_stack([
            np.zeros(4),
            np.zeros(4),
            np.array([0.0, 0.05, 0.1, 0.2]),
            np.ones(4),
            np.zeros(4),
        ])
        m = _omega_eval_mask(y)
        np.testing.assert_array_equal(m, [False, False, False, True])

    def test_omega_panels_drop_low_e(self):
        n = 100
        e = np.concatenate([np.full(40, 0.0), np.full(60, 0.3)])
        cos = np.concatenate([np.ones(40), np.linspace(-1, 1, 60)])
        sin = np.concatenate([np.zeros(40), np.linspace(-1, 1, 60)])
        y_true = np.column_stack([
            np.zeros(n),
            np.zeros(n),
            e,
            cos,
            sin,
        ])
        y_pred = y_true.copy()
        yt, yp, r2, n_keep = _omega_panel_arrays(y_true, y_pred, "cos_omega")
        self.assertEqual(n_keep, 60)
        self.assertEqual(len(yt), 60)
        self.assertAlmostEqual(r2, 1.0)

    def test_non_omega_targets_keep_all(self):
        y_true = np.random.default_rng(0).normal(size=(50, 5))
        y_true[:, 2] = np.abs(y_true[:, 2])
        y_pred = y_true.copy()
        yt, yp, _, n_keep = _omega_panel_arrays(y_true, y_pred, "e")
        self.assertEqual(n_keep, 50)
        self.assertEqual(len(yt), 50)


if __name__ == "__main__":
    unittest.main()
