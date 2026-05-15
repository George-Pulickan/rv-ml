# Handover — RV-ML Project

## TL;DR for the next Claude

You're continuing summer research with George Pulickan (Notre Dame CS+Stats freshman) and advisor Nicolò Colombo. Goal: publish a paper — an ML system that predicts Kepler orbital parameters from radial-velocity time series, with conformal-prediction uncertainty quantification vs Bayesian credible intervals.

**Tasks 1–2 are done. The GP noise model (Task 3 prerequisite) is done.** The next work is building the encoder architecture. George is terse and technical — match that. No over-explaining.

---

## Project goals

1. ✅ RV literature (Doppler spectroscopy, Lovis & Fischer, Foreman-Mackey)
2. ✅ Data + validation pipeline — 795 validated systems, median RMS/σ = 1.63, 51 Peg χ²_red = 1.31
3. 🔄 **CURRENT** — GP noise model ✅ done; encoder architecture ⏳ next
4. ⏳ Uncertainty quantification (conformal + Bayesian)

**Nicolò's autoencoder framing (do not deviate from this):**
- Encoder φ(RV) → orbital state X = (P, K, e, ω, T_peri, M sin i)
- Decoder = Kepler integrator (fixed, no learned weights)
- Loss = ‖RV − Kepler(φ(RV))‖
- Multi-modal extension: add ψ(Transit) encoder + Mandel-Agol transit decoder
- Full loss: ‖RV − Kepler(φ(RV))‖ + ‖Transit − Tran(ψ(Transit))‖ (drop missing term)

---

## Repository state

- Local: `~/rv-ml`, GitHub: `github.com/George-Pulickan/rv-ml` (private)
- Python 3.13 venv at `.venv`
- **Always `git push` immediately after committing** (standing rule from George)

### Key files

| File | Status | Purpose |
|---|---|---|
| `kepler_check.py` | ✅ | Validation pipeline. Entry: `validate_one(path, labels, **kwargs)`. Only change: added `tbl_path = Path(tbl_path)` coercion + mod-P T_peri shift reporting. Nothing deleted. |
| `parse_and_label.py` | ✅ | Host matching. `_norm_name` strips hyphens + spaces (fixes CoRoT 3 ↔ CoRoT-3). `match_with_simbad` uses `data/simbad_cache.json`. |
| `gp_noise_model.py` | ✅ | GP library. See below. Smoke-tested. |
| `cache_residuals.py` | ✅ | Runs `validate_one` over corpus → `data/residuals.npz` + `data/residuals_index.csv` |
| `gp_demo.py` | ✅ | 3-system demo (51 Peg, HAT-P-11, γ Cep) → `figures/gp_vs_bootstrap.png` |
| `gp_corpus_fit.py` | ✅ | Corpus-wide GP fit → `data/gp_fits.json`, `figures/gp_hyperparams.png` |
| `gp_sensitivity.py` | ✅ | Threshold sensitivity → `data/gp_sensitivity.csv`, `figures/gp_sensitivity.png` |
| `synthetic_rv.py` | ✅ | Bootstrap noise baseline. `BootstrapNoiseModel`. Pool in `data/noise_pool.npz`. |
| `data/rv_index.csv` | ✅ | 1469 rows (file × planet), joined to NASA ps table |
| `data/labels.csv` | ✅ | Full catalog parameters from NASA ps table |
| `data/simbad_cache.json` | ✅ | 153 host → SIMBAD alias entries. Expand with `resolve_simbad_aliases` in `parse_and_label.py`. |
| `data/residuals.npz` | ✅ | (t, resid, sigma) per system for 795 cached systems |
| `data/gp_fits.json` | ✅ | Full per-system GP fit records (444 systems) |
| `data/init_comparison.csv` | ✅ | Random-init experiment (783 systems, 90.3% same basin) |

---

## GP noise model — what was built and why

### `gp_noise_model.py` (688 lines, fully committed)

**Purpose:** fits a celerite2 GP to per-system post-Keplerian residuals; provides `GPNoiseLibrary` as a drop-in noise sampler for synthetic data generation.

**Design decisions (all justified, all cited):**

| Decision | Citation |
|---|---|
| Fitted `log_jitter` on every kernel | Foreman-Mackey 2017 §5 — prevents log_sigma absorbing unmodeled white noise |
| L-BFGS-B with log-uniform bounds | Foreman-Mackey 2017 §4; Faria et al. 2016 |
| Multi-restart + convergence gap | Gap = ΔlogL between best and 3rd-best restart; indicates multi-modal landscape |
| BIC selection + AIC reported | Pragmatic, defensible; reader can use either |
| Cholesky whitening + KS + Ljung-Box | Rasmussen & Williams 2006 §5.4.2 |
| DCF for time-lag autocorrelation | Edelson & Krolik 1988 |
| Pure GP sampling (no hybrid tail) | Don't fabricate. Kurtosis-30 tails reported honestly. |

**Five kernels:** `sho` (3+1 params), `matern32` (2+1), `rotation` (5+1), `sho+matern32` (5+1), `rotation+matern32` (7+1). All include fitted `log_jitter`.

**API:**
```python
from gp_noise_model import GPNoiseModel, fit_all_kernels, GPNoiseLibrary

best, all_fits = fit_all_kernels(t, residuals, sigma, kernels=('sho','matern32','rotation','sho+matern32'))
gof = best.goodness_of_fit(t, residuals, sigma)   # GoodnessOfFit dataclass
s = best.sample(t, rng=rng)                        # ndarray

lib = GPNoiseLibrary.from_json('data/gp_library.json')
noise = lib.sample(t, rng=rng)                     # draws random system from library
```

**Corpus results (444 systems, RMS/σ < 3, n_obs ≥ 15):**
- Matérn-3/2: 59%, SHO: 39%, composites: 2%
- KS normality pass: 98.2%, Ljung-Box independence pass: 90.3%
- Median whitened std = 0.990 (ideal: 1.0)
- Median correlation length ρ = 44 d, noise amplitude σ = 2.5 m/s, jitter = 1.34 m/s
- Sensitivity: stable across all 8 (rms_max, min_obs) threshold combinations

**Caveat:** GP samples are Gaussian; empirical pool has kurtosis-excess ≈ 30. For stress-testing the encoder, inject bootstrap samples from `data/noise_pool.npz`.

---

## Corpus pipeline — failure breakdown

795/1071 files cached. 276 failures:
- **193** — host not in NASA confirmed planet labels (genuinely absent, or discovered post-archive download)
- **21** — host unresolvable in simbad_cache (Pr 211, ChaHA 8 genuinely unresolvable; others had wrong name format)
- **66** — `no_planets:no_K_and_no_msini` — host in labels but planet K amplitude missing
- **9** — `no_planets:no_period` — host in labels but orbital period missing

The 276 is a hard ceiling given the current `labels.csv`. No further matching improvements available without a new catalog download.

---

## Nicolò's open questions (answered in `nicolo_reply_draft.md`)

All answered and reply drafted. Key points:

- **Why γ and T_peri free:** catalog P/K/e/ω are well-constrained from prior observations; γ is instrument/epoch-dependent, T_peri is a phase reference tied to the observation window. Freeing all params would make init intractable — that's the encoder's job.
- **Autodiff vs finite differences:** fundamentally different. FD: O(n) extra forward passes, O(ε²) error. Autodiff: exact gradient via chain rule in one forward+backward pass, O(1) in n_params. scipy uses FD, which is fine for 1–2 params but intractable for millions.
- **Transit integrator:** Mandel & Agol 2002 (ApJ 580, L171). Python: `batman` (Kreidberg 2015) for now; `starry` (Luger et al. 2019) for analytic gradients when wiring transit branch.
- **Architecture:** Nicolò's formulation is correct and matches the plan exactly.
- **"I do not know these works":** NPE/SBI (Cranmer et al. 2020) — not previously applied to Keplerian RV; brief lit check needed before writing related work section.

---

## Next steps — Task 3 encoder

Waiting on Nicolò's direction after the GP results. When ready:

1. **`preprocess.py`** — `data/rv_index.csv` + synthetic manifest → PyTorch tensors. Host-grouped 70/15/15 split (no leakage). Fixed length 256 with mask channel. Log10 normalization of P/K/M sin i.
2. **`models/kepler_torch.py`** — differentiable PyTorch port of Kepler integrator (the decoder).
3. **`models/encoder.py`** — φ(RV) → (P, K, e, ω, T_peri, M sin i, existence_logit) per planet. 1D CNN or Transformer → MLP head. Variable planet count via max-K slots + existence gate.
4. **`train.py`** — reconstruction loss ‖RV − Kepler(φ(RV))‖ + optional supervised auxiliary on synthetic. Pretrain on synthetic (GP noise), fine-tune on real.
5. **Multi-modal (later):** `batman`/`starry` transit decoder + ψ(Transit) encoder.

---

## Technical pitfalls

- **`validate_one` takes `labels` as a required positional arg** (not a kwarg). Signature: `validate_one(path, labels, mode, auto_sign, fit_tperi, trend_order, return_residuals, plot, verbose, simbad_cache)`.
- **`_extract` in gp_demo/cache_residuals uses `next()` not `or`** for dict key selection — `or` on numpy arrays raises ValueError.
- **`bootstrap_chunk` clamps** `lo = min(5, hi-1)` to avoid `low >= high` at series end.
- **`GPNoiseModel.fit()` returns a single `GPFit`**, not a tuple.
- **`GoodnessOfFit` is a dataclass** — access fields as attributes (`gof.ks_pvalue`), not dict keys.
- **`gp_fits_summary.csv` param columns** are named `p_log_sigma`, `p_log_rho`, etc. (prefixed with `p_`).
- **Cholesky whitening is O(N³)** — guarded to N ≤ 2500.
- **simbad_cache uses exact string keys** — host name must match exactly. `match_with_simbad` does `alias_cache.get(host, [])`.

---

## References

- Foreman-Mackey, D. et al. 2017, AJ 154, 220 — celerite (GP framework)
- Faria, J. et al. 2016, A&A 588, A31 — kima (prior conventions)
- Haywood, R. et al. 2014, MNRAS 443, 2517 — CoRoT-7 RV+GP methodology
- Rasmussen & Williams 2006 — GP textbook (whitening, GoF)
- Edelson & Krolik 1988, ApJ 333, 646 — DCF
- Mandel & Agol 2002, ApJ 580, L171 — transit light curves
- Kreidberg 2015, PASP 127, 1161 — batman transit code
- Luger et al. 2019 — starry (analytic transit gradients)
- Cranmer et al. 2020, PNAS — Neural Posterior Estimation / SBI
- Rajpaul et al. 2015, MNRAS 452, 2269 — multi-output GPs for RV
