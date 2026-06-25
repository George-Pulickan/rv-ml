RV-ML synthetic validation smoke run
===================================

Scope: synthetic generation with f_multi=0.0.
Real comparison split: train
Synthetic samples: 400
Valid real single-planet comparison samples: 242
Real time grids loaded for synthetic cadence bootstrap: 388
GP residual checkpoint: models\gp_residual_svgp.pt
GP residual checkpoint exists: True
GP residual model loaded: True
Legacy GP fits path: data\gp_fits.json
Legacy GP fits exists: False
Legacy GP library loaded: False
Noise mode used by generator: gp_residual_svgp

Observation-based classifier diagnostic:
- Inputs: 64 normalized spectral power bins plus observation-derived summaries.
- Kepler parameters and K/measurement-uncertainty are excluded from classifier inputs.
- Balanced accuracy: 0.756 +/- 0.028
- Top individual discriminator: sigma_iqr_ms
- Top feature group: sigma_iqr_ms

Additional classifier diagnostics:
- classifier_probability_histogram.png shows out-of-fold P(real) by class.
- classifier_probability_vs_kepler.png shows out-of-fold P(real) against Kepler diagnostic parameters, which are not classifier inputs.

Important interpretation:
- This is a smoke/diagnostic validation run, not a training cache.
- GP residual SVGP checkpoint loaded successfully.
- Next scientific step is to inspect plots and decide whether priors, cadence, or noise need adjustment.
