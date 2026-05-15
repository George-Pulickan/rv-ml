# Draft reply to Nicolò

---

Hi Nicolò,

Quick answers to your questions, then results.

---

**Why γ and T_peri as free parameters?**
P, K, e, ω encode the orbit's physical shape and are well-constrained by the catalog from previous long-baseline observations. γ (systemic velocity) is instrument- and epoch-dependent and cannot be taken from the catalog. T_peri is a phase reference that shifts the entire curve — the catalog value refers to a specific observation epoch and cannot be assumed to align with our data window. Both are 1D per-planet parameters that LM refits in milliseconds, so the cost of freeing them is negligible. Freeing all 5–6 parameters per planet would make the initialization problem much harder, which is exactly the motivation for the encoder.

**What is P?**
Orbital period in days.

**Is the scipy approach scalable to the full parameter set?**
No. Finite-difference gradient estimation costs O(n_params) extra forward evaluations per gradient step, and global convergence requires exponentially more restarts as dimensionality grows. The 90.3% same-basin result holds precisely because we only vary T_peri (one parameter, one period's range). Extending random init to 5–6 parameters per planet across multiple planets would need orders of magnitude more restarts to be confident of convergence. This is the argument for the encoder: it maps directly from data → parameters without iterative optimization.

**Am I wrong that autodiff does nearly the same as finite differences?**
Yes — they are fundamentally different. Finite differences estimate the gradient numerically: for n parameters, you need n extra forward passes, each introducing O(ε²) truncation error. Autodiff (backpropagation) computes the exact analytical gradient via the chain rule in a single forward+backward pass, at machine precision, regardless of n_params. For a neural network with millions of parameters, finite differences are completely intractable; autodiff is the only feasible option. scipy.optimize.least_squares uses finite differences, which is fine for our 1–2 free parameters but would not scale.

**Transit integrator — could not find an explicit expression.**
Mandel & Agol 2002 (ApJ 580, L171) give closed-form expressions for limb-darkened transit light curves as a function of (P, R_p/R_*, a/R_*, i, u_1, u_2, T_mid). The standard Python implementation is `batman` (Kreidberg 2015, PASP 127, 1161) — pip-installable, well-documented, and fast. For analytic gradients (needed if we want to backpropagate through the transit decoder), `starry` (Luger et al. 2019) provides these. I would suggest batman for now and switch to starry when we wire up the transit branch of the encoder.

**Architecture / loss.**
Agreed on the formulation. To be precise about what we are building:

    min_φ  ‖RV − Kepler(φ(RV))‖  +  min_ψ  ‖Transit − Tran(ψ(Transit))‖

where Kepler(·) is the existing integrator (fixed, no learned weights) and Tran(·) is the Mandel-Agol integrator (also fixed). φ and ψ are the only learned components. If only one data type is available, we drop the corresponding term. The shared latent space X = (P, e, ω, T_peri, M sin i; plus i and R_p for transits) is exactly the Keplerian orbital state — no additional structure needed.

**"I do not know these works."**
The references I flagged were Neural Posterior Estimation (Cranmer et al. 2020, PNAS) and related simulation-based inference methods. These are general Bayesian parameter estimation approaches that use neural networks trained on simulations — they haven't been applied specifically to Keplerian RV fitting as far as I can tell. If no prior work matches our exact setup (Kepler integrator as decoder, conformal UQ), that strengthens the novelty of the paper. Worth a brief literature check before we write the related work section.

---

**GP noise model — results.**

The GP noise model is complete. I fit a celerite2 GP to the post-Keplerian residuals of 444 quality-filtered systems (RMS/σ < 3, n_obs ≥ 15) from the corpus.

*Kernel selection (BIC):* Matérn-3/2 in 59% of systems, SHO in 39%, composite kernels (rotation, SHO+Matérn) in 2%. The dominance of smooth, stationary kernels suggests the residual noise is primarily instrumental/photon, not stellar-activity driven — consistent with the fact that most systems in the corpus are quiet FGK dwarfs with well-separated planets.

*Goodness of fit:*
- KS test (whitened residuals vs N(0,1)): 98.2% pass at p > 0.05
- Ljung-Box independence test: 90.3% pass at p > 0.05
- Median whitened std = 0.990 (well-calibrated; ideal = 1.0)

*Hyperparameters (Matérn/SHO systems):*
- Median correlation length ρ = 44 days (log ρ = 3.78)
- Median noise amplitude σ = 2.5 m/s
- Median fitted jitter = 1.34 m/s (non-negligible; consistent with Foreman-Mackey 2017 §5)

*Sensitivity:* Statistics are stable across all eight (rms_max, min_obs) threshold combinations tested, so the noise model is not sensitive to the exact quality cut.

*Caveat:* Pure GP samples are Gaussian by construction. The empirical residual pool has kurtosis-excess ≈ 30, driven by outlier-contaminated systems. For pretraining, the GP noise is the right choice for typical systems; if we want to stress-test the encoder against heavy-tailed noise, we can inject bootstrap samples from the empirical pool.

*Deliverables:*
- `gp_noise_model.py` — library (fit, sample, GoF, DCF, GPNoiseLibrary)
- `figures/gp_vs_bootstrap.png` — per-system GP vs bootstrap comparison (51 Peg, HAT-P-11, γ Cep)
- `figures/gp_hyperparams.png` — corpus hyperparameter scatter (444 systems)
- `figures/gp_sensitivity.png` — threshold sensitivity analysis
- `data/gp_fits.json` — full per-system fit records
- `data/gp_fits_summary.csv` — flat table for analysis

Next step on my end: `y(t, pars) = Kepler(t, pars) + gp(t, pars)` is ready to use. Waiting on your direction for whether to proceed to the encoder architecture or whether the noise model needs adjustment first.

Best,
George
