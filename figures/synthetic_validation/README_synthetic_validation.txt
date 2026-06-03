RV-ML synthetic validation smoke run
===================================

Scope: synthetic generation with f_multi=0.0.
Synthetic samples: 400
Valid real single-planet comparison samples: 357
Real time grids loaded for synthetic cadence bootstrap: 388
GP fits exists: False
GP library loaded: False
Noise mode used by generator: white_gaussian_fallback

Important interpretation:
- This is a smoke/diagnostic validation run, not a training cache.
- Because gp_fits.json is absent or unloadable, noise is white Gaussian fallback.
- Next scientific step is to inspect plots and decide whether priors, cadence, or noise need adjustment.
