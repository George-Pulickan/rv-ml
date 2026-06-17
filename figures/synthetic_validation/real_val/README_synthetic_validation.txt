RV-ML synthetic validation smoke run
===================================

Scope: synthetic generation with f_multi=0.0.
Real comparison split: val
Synthetic samples: 400
Valid real single-planet comparison samples: 58
Real time grids loaded for synthetic cadence bootstrap: 388
GP fits exists: False
GP library loaded: False
Noise mode used by generator: white_gaussian_fallback

Observation-based classifier diagnostic:
- Inputs: 64 normalized spectral power bins plus observation-derived summaries.
- Kepler parameters and K/measurement-uncertainty are excluded from classifier inputs.
- Balanced accuracy: 0.559 +/- 0.033
- Top individual discriminator: sigma_iqr_ms
- Top feature group: lsp_peak_power

Additional classifier diagnostics:
- classifier_probability_histogram.png shows out-of-fold P(real) by class.
- classifier_probability_vs_kepler.png shows out-of-fold P(real) against Kepler diagnostic parameters, which are not classifier inputs.

Important interpretation:
- This is a smoke/diagnostic validation run, not a training cache.
- Because gp_fits.json is absent or unloadable, noise is white Gaussian fallback.
- Next scientific step is to inspect plots and decide whether priors, cadence, or noise need adjustment.
