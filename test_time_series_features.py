"""Focused checks for the uneven-series spectral encoder."""

import unittest

import numpy as np

from time_series_features import spectral_feature_names, spectral_features


class SpectralFeaturesTests(unittest.TestCase):
    def test_returns_requested_finite_dimension(self):
        t = np.array([0.0, 0.2, 0.55, 1.1, 2.0, 3.4])
        y = np.sin(2 * np.pi * t / 1.7)

        features = spectral_features(t, y, d=32, grid_size=256, smoothing=0)

        self.assertEqual(features.shape, (32,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features.sum(), 0.0)

    def test_is_invariant_to_input_order(self):
        t = np.array([0.0, 0.3, 0.9, 1.7, 2.6, 4.0])
        y = np.cos(t)
        order = np.array([4, 1, 5, 0, 3, 2])

        expected = spectral_features(t, y, smoothing=0)
        shuffled = spectral_features(t[order], y[order], smoothing=0)

        np.testing.assert_allclose(shuffled, expected)

    def test_duplicate_times_are_averaged(self):
        t = np.array([0.0, 0.5, 0.5, 1.0, 1.5])
        y = np.array([0.0, 1.0, 3.0, 0.0, -1.0])

        features = spectral_features(t, y, d=16, smoothing=0)

        self.assertEqual(features.shape, (16,))
        self.assertTrue(np.isfinite(features).all())

    def test_constant_series_returns_zero_power(self):
        features = spectral_features(
            np.linspace(0.0, 1.0, 10),
            np.full(10, 4.2),
        )

        np.testing.assert_array_equal(features, np.zeros(64))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            spectral_features([0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            spectral_features([0.0, np.nan], [1.0, 2.0])
        with self.assertRaises(ValueError):
            spectral_features([1.0, 1.0], [1.0, 2.0])

    def test_feature_names_exclude_dc_bin(self):
        self.assertEqual(
            spectral_feature_names(3),
            ["spectral_power_001", "spectral_power_002", "spectral_power_003"],
        )


if __name__ == "__main__":
    unittest.main()
