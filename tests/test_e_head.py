"""Unit tests for the zero-inflated e-target counters (--e-balance, --e-head hurdle/dual)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression import DatasetBundle, load_checkpoint_and_predict_val, train_model  # noqa: E402
from theta_loss import e_balance_weights  # noqa: E402


def _toy_bundle(n: int = 400, in_dim: int = 8, seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim))
    e = np.where(rng.random(n) < 0.3, 0.0, rng.beta(0.867, 3.03, size=n))
    omega = rng.uniform(0, 2 * np.pi, size=n)
    y = np.column_stack([
        rng.normal(1.5, 0.8, size=n),
        rng.normal(1.2, 0.5, size=n),
        e,
        np.cos(omega),
        np.sin(omega),
    ])
    return DatasetBundle(
        X,
        y,
        row_idx=np.arange(n),
        e=e,
        has_t_peri=np.ones(n),
        has_ecc=np.ones(n, dtype=bool),
        df=pd.DataFrame(),
    )


class TestEBalanceWeights(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 2000
        self.e = np.where(rng.random(n) < 0.3, 0.0, rng.beta(0.867, 3.03, size=n))

    def test_mean_one_over_train(self):
        w = e_balance_weights(self.e)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=10)
        self.assertTrue((w > 0).all())

    def test_zero_mass_downweighted_vs_rare_bins(self):
        w = e_balance_weights(self.e, np.array([0.0, 0.85]))
        self.assertLess(w[0], w[1])

    def test_cap_bounds_weight_ratio(self):
        w = e_balance_weights(self.e, max_ratio=10.0)
        self.assertLessEqual(float(w.max() / w.min()), 10.0 + 1e-9)

    def test_query_unseen_bin_gets_finite_weight(self):
        e_train = np.concatenate([np.zeros(50), np.full(50, 0.1)])
        w = e_balance_weights(e_train, np.array([0.95]))
        self.assertTrue(np.isfinite(w).all())


class TestEHeadTraining(unittest.TestCase):
    def _train(self, **kw):
        bundle = _toy_bundle()
        return bundle, train_model(
            bundle,
            feature_set="74",
            epochs=3,
            batch_size=64,
            lr=1e-3,
            val_frac=0.2,
            seed=0,
            device=torch.device("cpu"),
            patience=10,
            **kw,
        )

    def test_direct_default_unchanged(self):
        _, (model, preds, metrics) = self._train()
        self.assertEqual(metrics["e_head"], "direct")
        self.assertEqual(metrics["norm_stats"]["out_dim"], 5)
        self.assertEqual(preds["y_pred"].shape[1], 5)

    def test_balance_runs_and_predicts(self):
        _, (model, preds, metrics) = self._train(e_balance=True)
        self.assertTrue(metrics["e_balance"])
        self.assertTrue(np.isfinite(preds["y_pred"]).all())

    def test_hurdle_shapes_metrics_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "hurdle.pt"
            bundle = _toy_bundle()
            model, preds, metrics = train_model(
                bundle,
                feature_set="74",
                epochs=3,
                batch_size=64,
                lr=1e-3,
                val_frac=0.2,
                seed=0,
                device=torch.device("cpu"),
                patience=10,
                checkpoint_path=ckpt,
                e_head="hurdle",
                e_balance=True,
            )
            self.assertEqual(metrics["norm_stats"]["e_head"], "hurdle")
            self.assertEqual(metrics["norm_stats"]["out_dim"], 6)
            self.assertEqual(preds["y_pred"].shape[1], 5)
            e_pred = preds["y_pred"][:, 2]
            self.assertTrue(((e_pred >= 0.0) & (e_pred <= 0.99)).all())
            clf = metrics["e_zero_classifier"]
            self.assertEqual(clf["n"], len(preds["y_pred"]))
            self.assertGreater(clf["frac_true_zero"], 0.0)

            y_val, y_pred, _ = load_checkpoint_and_predict_val(
                bundle, ckpt, val_frac=0.2, seed=0, device=torch.device("cpu")
            )
            np.testing.assert_allclose(y_pred, preds["y_pred"], rtol=1e-5, atol=1e-6)

    def test_hurdle_rejects_unknown_head(self):
        with self.assertRaises(ValueError):
            self._train(e_head="bogus")

    def test_dual_shapes_metrics_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "dual.pt"
            bundle = _toy_bundle(n=500)
            model, preds, metrics = train_model(
                bundle,
                feature_set="74",
                epochs=3,
                batch_size=64,
                lr=1e-3,
                val_frac=0.2,
                seed=0,
                device=torch.device("cpu"),
                patience=10,
                checkpoint_path=ckpt,
                e_head="dual",
            )
            self.assertEqual(metrics["e_head"], "dual")
            self.assertEqual(metrics["norm_stats"]["e_head"], "dual")
            self.assertIn("gate_threshold", metrics["norm_stats"])
            self.assertEqual(preds["y_pred"].shape[1], 5)
            self.assertGreater(metrics["n_train_circ"], 0)
            self.assertGreater(metrics["n_train_ecc"], 0)
            e_pred = preds["y_pred"][:, 2]
            self.assertTrue(((e_pred >= 0.0) & (e_pred <= 0.99)).all())
            clf = metrics["e_zero_classifier"]
            self.assertEqual(clf["n"], len(preds["y_pred"]))
            self.assertIn("f1_zero", clf)
            self.assertIn("e_report", metrics)
            self.assertIn("e_gt_0", metrics["e_report"])

            y_val, y_pred, loaded = load_checkpoint_and_predict_val(
                bundle, ckpt, val_frac=0.2, seed=0, device=torch.device("cpu")
            )
            np.testing.assert_allclose(y_pred, preds["y_pred"], rtol=1e-5, atol=1e-6)
            self.assertIn("e_report", loaded)
            self.assertTrue(hasattr(model, "gate") and hasattr(model, "circ") and hasattr(model, "ecc"))

    def test_gate_balanced_weights_equal_mass(self):
        from regression import _gate_balanced_weights

        is_pos = np.array([True, True, True, False])
        has = np.ones(4, dtype=bool)
        w = _gate_balanced_weights(is_pos, has)
        self.assertAlmostEqual(float(w[is_pos].sum()), float(w[~is_pos].sum()), places=6)


if __name__ == "__main__":
    unittest.main()
