"""Shared fixtures for the regression-head tests.

``_toy_bundle`` lived in both ``test_e_head`` and ``test_hk_targets`` as
byte-identical copies apart from the ``df`` argument, so a change to the toy
target distribution had to be made twice to keep the two suites comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression import DatasetBundle  # noqa: E402


def toy_bundle(n: int = 400, in_dim: int = 8, seed: int = 0,
               *, with_sigma: bool = False) -> DatasetBundle:
    """Synthetic (X, theta5) bundle with the corpus's e-zero inflation.

    ``with_sigma`` adds the ``median_sigma_ms`` column that the h/k tests need;
    the e-head tests use an empty frame.
    """
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
    df = pd.DataFrame({"median_sigma_ms": np.full(n, 1.0)}) if with_sigma else pd.DataFrame()
    return DatasetBundle(
        X,
        y,
        row_idx=np.arange(n),
        e=e,
        has_t_peri=np.ones(n),
        has_ecc=np.ones(n, dtype=bool),
        df=df,
    )
