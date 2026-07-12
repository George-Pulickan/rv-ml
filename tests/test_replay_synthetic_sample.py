"""replay_synthetic_sample must match rows in the generated CSV corpus."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SYNGEN = ROOT / "synthetic_generation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SYNGEN) not in sys.path:
    sys.path.insert(0, str(SYNGEN))

from generate_synthetic_regression_csv import replay_synthetic_sample  # noqa: E402
from preprocess import THETA_NAMES  # noqa: E402
from theta_loss import regression_theta_loss  # noqa: E402

PHASEFOLD_CSV = SYNGEN / "datasets" / "synthetic_regression_10000_phasefold.csv"
CSV_SEED = 123


@unittest.skipUnless(PHASEFOLD_CSV.exists(), "phasefold CSV not in repo")
class TestReplaySyntheticSample(unittest.TestCase):
    def test_replay_matches_csv_theta(self) -> None:
        df = pd.read_csv(PHASEFOLD_CSV)
        n_samples = len(df)
        for i in (0, 1, 5, 50, 500, 2000, 9999):
            with self.subTest(i=i):
                _, _, theta, _ = replay_synthetic_sample(i, CSV_SEED, n_samples=n_samples)
                expected = df.iloc[i][THETA_NAMES].to_numpy(dtype=np.float64)
                self.assertTrue(
                    np.allclose(theta, expected, rtol=0, atol=1e-4),
                    f"replay={theta} csv={expected}",
                )


class TestLossWeights(unittest.TestCase):
    def test_dim_weights_affect_circular_loss(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(32, 5)
        target = torch.randn(32, 5)
        sw = torch.ones(32, 5)
        y_mean = torch.zeros(5)
        y_std = torch.ones(5)
        w1 = torch.ones(5)
        w5 = torch.tensor([1.0, 1.0, 5.0, 5.0, 5.0])
        loss1 = regression_theta_loss(
            pred, target, sw, w1, y_mean=y_mean, y_std=y_std, circular_omega=True
        )
        loss5 = regression_theta_loss(
            pred, target, sw, w5, y_mean=y_mean, y_std=y_std, circular_omega=True
        )
        self.assertFalse(torch.allclose(loss1, loss5))


if __name__ == "__main__":
    unittest.main()
