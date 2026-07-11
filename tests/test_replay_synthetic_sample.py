"""Replay must reproduce exactly the system stored in a regression CSV row.

Guards the RNG-prefix trap: the shared parameter stream depends on the corpus
size, so replaying with fewer draws silently yields a different system.
"""

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "synthetic_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_synthetic_regression_csv import (
    corpus_orbital_params,
    generate_rows,
    replay_synthetic_sample,
)
from preprocess import THETA_NAMES

PHASEFOLD_CSV = (
    ROOT / "synthetic_generation" / "datasets" / "synthetic_regression_10000_phasefold.csv"
)
CSV_SEED = 123
CSV_N_SAMPLES = 10_000


class ReplaySyntheticSampleTests(unittest.TestCase):
    def test_replay_matches_generate_rows(self):
        seed, n = 7, 6
        rows = generate_rows(n, seed, 0.0, with_phasefold=True)
        params = corpus_orbital_params(seed, n)

        for i in (0, 2, n - 1):
            _, _, theta, _ = replay_synthetic_sample(i, seed, n, params=params)
            expected = np.array([rows[i][name] for name in THETA_NAMES])
            np.testing.assert_allclose(theta, expected, atol=1e-12)

    def test_replay_matches_tracked_csv(self):
        if not PHASEFOLD_CSV.exists():
            self.skipTest(f"missing {PHASEFOLD_CSV}")
        import pandas as pd

        df = pd.read_csv(PHASEFOLD_CSV, usecols=THETA_NAMES)
        params = corpus_orbital_params(CSV_SEED, CSV_N_SAMPLES)

        for i in (0, 1, 137, 4242):
            _, _, theta, _ = replay_synthetic_sample(i, CSV_SEED, CSV_N_SAMPLES, params=params)
            expected = df.iloc[i][list(THETA_NAMES)].to_numpy(dtype=float)
            np.testing.assert_allclose(theta, expected, atol=1e-9, err_msg=f"row {i}")


if __name__ == "__main__":
    unittest.main()
