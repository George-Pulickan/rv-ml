"""Unit tests for h/k target encode/decode and --targets hk training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression import DatasetBundle, train_model  # noqa: E402
from theta_loss import (  # noqa: E402
    apply_hk_constraints,
    hk_to_theta,
    theta_to_hk,
)


def _toy_bundle(n: int = 400, in_dim: int = 8, seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim))
    e = np.where(rng.random(n) < 0.3, 0.0, rng.beta(0.867, 3.03, size=n))
    omega = rng.uniform(0, 2 * np.pi, size=n)
    y = np.column_stack(
        [
            rng.normal(1.5, 0.8, size=n),
            rng.normal(1.2, 0.5, size=n),
            e,
            np.cos(omega),
            np.sin(omega),
        ]
    )
    return DatasetBundle(
        X,
        y,
        row_idx=np.arange(n),
        e=e,
        has_t_peri=np.ones(n),
        has_ecc=np.ones(n, dtype=bool),
        df=pd.DataFrame({"median_sigma_ms": np.full(n, 1.0)}),
    )


class TestHKConvert(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        e = rng.uniform(0.05, 0.9, size=200)
        omega = rng.uniform(0, 2 * np.pi, size=200)
        y = np.column_stack(
            [rng.normal(size=200), rng.normal(size=200), e, np.cos(omega), np.sin(omega)]
        )
        hk = theta_to_hk(y)
        back = hk_to_theta(hk)
        np.testing.assert_allclose(back[:, :3], y[:, :3], atol=1e-10)
        np.testing.assert_allclose(back[:, 3:], y[:, 3:], atol=1e-10)

    def test_e_zero_maps_to_omega_zero(self):
        y = np.array([[1.0, 2.0, 0.0, 0.3, 0.4]])
        hk = theta_to_hk(y)
        np.testing.assert_allclose(hk[0, 2:], [0.0, 0.0])
        back = hk_to_theta(hk)
        self.assertAlmostEqual(back[0, 2], 0.0)
        self.assertAlmostEqual(back[0, 3], 1.0)
        self.assertAlmostEqual(back[0, 4], 0.0)

    def test_apply_hk_constraints_clips_e(self):
        hk = np.array([[0.0, 0.0, 2.0, 2.0]])
        clipped = apply_hk_constraints(hk, e_max=0.99)
        e = np.sqrt(clipped[0, 2] ** 2 + clipped[0, 3] ** 2)
        self.assertAlmostEqual(e, 0.99, places=6)


class TestHKTraining(unittest.TestCase):
    def test_train_hk_direct_returns_theta(self):
        bundle = _toy_bundle()
        _, preds, metrics = train_model(
            bundle,
            feature_set="74",
            epochs=3,
            batch_size=64,
            lr=1e-3,
            val_frac=0.2,
            seed=0,
            device=torch.device("cpu"),
            patience=5,
            targets="hk",
            checkpoint_path=None,
        )
        self.assertEqual(metrics["targets"], "hk")
        self.assertEqual(preds["y_true"].shape[1], 5)
        self.assertEqual(preds["y_pred"].shape[1], 5)
        self.assertIn("stratified_omega", metrics)
        self.assertIn("by_e_band", metrics["stratified_omega"])

    def test_train_hk_dual(self):
        bundle = _toy_bundle(n=500)
        _, preds, metrics = train_model(
            bundle,
            feature_set="74",
            epochs=3,
            batch_size=64,
            lr=1e-3,
            val_frac=0.2,
            seed=1,
            device=torch.device("cpu"),
            patience=5,
            targets="hk",
            e_head="dual",
            checkpoint_path=None,
        )
        self.assertEqual(metrics["e_head"], "dual")
        self.assertEqual(metrics["norm_stats"]["out_dim"], 4)
        self.assertEqual(preds["y_pred"].shape[1], 5)


if __name__ == "__main__":
    unittest.main()
