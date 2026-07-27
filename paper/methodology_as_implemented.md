# Methodology as implemented

A precise statement of what the pipeline actually does, written against the code
rather than against the draft. Intended as raw material for §3–§4 and as a
cross-check: where this disagrees with the paper, the paper is what needs
changing (the discrepancies are catalogued in `paper_corrections.md`).

---

## 1. The problem

Given a radial-velocity time series `y = {(t_i, v_i, σ_i)}` for a star, produce
**intervals** for the Keplerian orbital parameters that are valid — i.e. cover
the truth at the stated rate — including on real stars, where the calibration
data are synthetic and the test data are not.

Reported coordinates are **θ = (log₁₀P, log₁₀K, e)**, so d = 3. The argument of
periastron ω is estimated by the model but **no longer reported**: its interval
half-width exceeds 2π, so it wraps the full circle and is not identifiable at
this SNR. It stays in the model because e is parameterised through
(h, k) = (e cos ω, e sin ω), and pinning an unidentifiable angle would bias the
two quantities that *are* reported. Time of periastron `T_peri` and the systemic
offset are excluded from θ entirely — both enter the decoder linearly given the
rest, and are refit analytically at each evaluation, removing two nuisance
directions from the optimisation.

## 2. Data

Radial-velocity time series retrieved in bulk from the **NASA Exoplanet
Archive**, with tabulated orbital parameters from its Planetary Systems table —
1071 curves, ground-based spectrographs (predominantly Keck/HIRES, Lick, AAT,
HARPS). **Not Kepler**, which is a transit-photometry mission; the draft says
otherwise and is wrong.

Split train/val/test by system. Curves with fewer than 10 observations are
marked invalid. Two further filters are applied at test time:

- **Minimum baseline** — at least five distinct observing nights. Deliberately
  **label-free**: it reads only the observing pattern, never the tabulated
  period, so test-set selection never consults the values we are scored against.
  It removes 6 of 57 test series. A label-aware version (baseline ≥ one period)
  removes exactly the same six.
- **One series per host** (optional, and now the headline variant). The corpus
  stores one row per RV file, so the 51 surviving test series come from only
  **34 distinct stars** — HD 179949 contributes four, 51 Peg three. Treating
  correlated repeats of one star as independent test points overstates the
  evidence. This option keeps the best-sampled series per star.

## 3. The simulator

Synthetic curves are drawn as `y = h(θ̄) + noise`, where `h` is a differentiable
Keplerian decoder and θ̄ is the data-generating parameter vector.

- **Priors** on P and K are empirical log-histograms **fitted on the train split
  only**. (The 3-component Gaussian mixture and LogUniform(8, 400) that appear in
  older documentation are fallbacks, used only when the corpus files are absent.)
- **Eccentricity** follows Kipping (2013), Beta(0.867, 3.03).
- **ω** is set to 0 for near-circular orbits (e ≤ 0.05), where it is not
  identifiable.
- **Noise** comes from a GP-residual model (sparse variational GP) fitted to
  pooled real RV residuals, conditioned on the reported σ. Observing cadence and
  σ are bootstrapped from real curves, so synthetic sampling patterns match real
  ones.
- **Assumption 2.1 noise filter.** Draws whose `max_t |y_t − h(θ̄, t)|` exceeds a
  bound estimated on real *train* curves are discarded at generation time. This
  rejects **~48%** of draws and truncates the calibration distribution relative to
  any run predating 2026-07-14.

The scientific finding behind the noise model: pooled RV residual heavy tails
(kurtosis ~30) are a **scale mixture across systems**, not heavy tails within
systems.

## 4. The point predictor ψ

A 74-dimensional summary featurisation of each curve (Lomb–Scargle **power**
spectrum features, cadence and amplitude statistics — note: power, not Fourier
coefficients) feeding an MLP that outputs (log₁₀P, log₁₀K, e, cos ω, sin ω).

ψ is trained **only on synthetic curves**, against the data-generating θ̄.
Catalogued values never serve as a training target. They enter in four places,
and in the first three only via the train split: fitting the priors; the
(cadence, σ) bootstrap; the GP residual fit; and at test time as the initialiser
for the gradient-descent fit below.

Cost: featurisation 6.4 ms per system, forward pass 1.0 µs — 6.4 ms end to end.
Training ψ is ~65 s once for the whole corpus.

## 5. Surrogate labels θ*

The central construct. For an observation `y`, define

> **θ\*(y) = argmin_θ ‖y − h(θ)‖²**

computed by batched Adam gradient descent through the differentiable decoder,
initialised at the data-generating value (synthetic) or the tabulated value
(real). 200 steps by default; ~4.0 s per system.

Two properties matter:

1. **θ\* depends on y alone.** It is computable on real curves, where θ̄ does not
   exist. This is what makes the method applicable under shift.
2. The objective is **squared error**, matching eq (2) and the Hessian used in
   Assumption 3.2. (It was L1 until 2026-07-25; that was a genuine mismatch.)

⚠️ **Caveat now known:** the GD output is not in general a strict local minimum —
measured directly, λ_min(H\*) is not reliably positive, and it degrades with more
optimisation steps. So eq (2)'s definition as an argmin does not describe what
the algorithm returns. The definition should probably be weakened to "the output
of procedure (3)". Open with Nicolò.

## 6. Conformity scores and normalisation

The base score for coordinate c is `s_c = |ψ_c(y) − θ_c|`, with the ω coordinate
(when used) measured as circular distance.

Four normalisations are implemented, `s'_c = s_c / v_c(y)`:

| name | v_c(y) |
|---|---|
| `raw` | 1 |
| `vnorm` | δ_y(y) |
| `v2norm` | δ_y(y) + δ_c(y) |
| `papernorm` | γ₀ + w·δ_c(y) + (1−w)·δ_y(y) — eq (17) |

where

- **δ_c(y) = \|ψ_c(h(ψ(y))) − ψ_c(y)\|** — the *re-encoding residual*. Take the
  prediction, render the noiseless curve it implies on the observation's own time
  grid and σ, push that back through ψ, and measure the disagreement. It measures
  self-consistency of the encoder on that curve and is computable at test time
  without any label.
- **δ_y(y) = (1/T) Σ_t \|y_t − h(ψ(y), t)\|**, in units of the curve's RV
  standard deviation.

Both are **pointwise**: h and ψ are deterministic, so no supplementary dataset
`D_δ` is generated and no auxiliary model is fitted. The paper's eq (13)
describes a conditional expectation that, for a deterministic map, collapses to
the score itself — making eq (15) normalise the score by itself. The code does
the re-encoding residual above; the text needs to change to match.

**w is tuned, not fixed.** It is grid-searched over [0,1] jointly with γ₀ on the
tuning set. It selects **w = 1.0** — δ_y contributes nothing once δ_c is present.
This is an empirical result, and it means the old implicit 1:1 sum was strictly
worse than the tuned combination.

## 7. Conformal calibration under covariate shift

Split conformal, following Tibshirani et al. (2019). Calibration scores come
from synthetic draws (n_cal = 400); a separate tuning set (n_tune = 100) selects
γ₀ and w.

Because the test distribution is real curves and the calibration distribution is
synthetic, quantiles are taken with **likelihood-ratio weights** w(y) estimated
by a discriminator between the two, with a class-balance correction. The weighted
quantile normalises by `Σw + w_test` and carries the Tibshirani point mass at
+∞, returning an infinite interval when the level falls in the test atom — this
is how "vacuous" intervals arise, and the implementation is correct even though
the paper's eq (10) omits the normalisation.

Joint coverage over the d reported coordinates uses a **Bonferroni** correction,
α′ = α/d, so d = 3 now that ω is not reported.

Effective sample size of the weights is reported (≈ 288 of 400), as a check that
the reweighting is not degenerate.

## 8. The gap Δs, and what we do about it

The naive strategy calibrates against θ̄ — which exists only synthetically. The
surrogate strategy calibrates against θ*. The difference between the two scores
is the gap

> **Δs = \|θ̄ − θ\*\|**

**The reported intervals set Δs = 0.** The `naive + Δ_c` variant instead inflates
the calibration quantile by Δ_c, the **maximum** gap over the tuning set.

Measured (see `figures/paper/delta_s_gap_distribution.svg`): the median gap is
essentially zero (5×10⁻⁴ dex in log₁₀P) but the maximum is three orders of
magnitude larger (0.54 dex). Because Δ_c takes the max, the naive + Δ_c variant
pins at 57/57 coverage on every normalisation — that is conservatism from an
extreme tail, not robustness, and it is the evidence that max is too aggressive
a choice.

**Status: contested.** Treating Δ_c as a conformal quantile (the k-th order
statistic, k = ⌈(1−β)(n+1)⌉, giving P(gap > Δ_c) ≤ β out of sample) was
implemented as `--gap-beta`. Nicolò's objection is that s_k depends on the GD
initialisation, and that reweighting p(y) is insufficient — the argument would
need the ratio of p(y, θ̄). His proposed alternative is to assume |θ̄ − θ*| < ε
and give a heuristic for estimating ε rather than claim a finite-sample
guarantee. **This is unresolved.**

## 9. What is being claimed

Per Nicolò (2026-07-26): **the goal is validity, not efficiency.** The naive
strategy is *not a baseline* — on real data true labels do not exist, so it is a
synthetic-calibration artefact. Validity can only be defined against the
surrogate labels.

Consequently:

- Coverage comparisons between naive and surrogate should **not** appear as a
  headline result. Two controlled misspecification studies (mean-structure and
  correlated-noise) found no difference, which under this framing is the expected
  outcome rather than a negative result.
- The mean-structure study is additionally **confounded by construction**:
  injecting a companion moves θ* toward a blend while coverage is still scored
  against θ̄, so |θ̄ − θ*| grows 1.5–3.5× across the sweep. It cannot test the
  claim in either direction and has been set aside.
- The application-side evidence Nicolò asked for — showing that θ* often
  coincides with the Bayesian/catalogue values the community trusts — **is not
  yet built.**

**On the theory.** Theorem 3.6's constant is vacuous at the measured values:
√(C_H·ε) ≈ 4.1 on the loss scale, several times the log₁₀P half-width and beyond
the entire support of e. Assumption 3.2 additionally fails on a substantial
fraction of draws. The reported intervals do not depend on this — they are
calibrated by the conformal construction, which needs no curvature constant. The
honest presentation is to report the empirical constants, state the bound is
vacuous at those values, and rest validity on the conformal argument.

## 10. Evaluation

Coverage on real systems is measured **against the catalogued values** — the
numbers an astronomer would look up — not against our own optimiser's output.
θ* appears only in the conformity score on the synthetic calibration set, where
θ̄ is known.

Reported at α = 0.10 unless stated. Synthetic test n = 400; real test n = 34
independent hosts (51 series before host-deduplication, 57 before the baseline
filter).

---

### Known-good vs provisional

| Component | Status |
|---|---|
| Simulator, priors, noise model, filters | stable |
| ψ architecture and training | stable; pinned checkpoint with provenance |
| θ* / squared-error objective | stable since 2026-07-25 |
| papernorm eq (17), tuned w | stable; selects w = 1.0 |
| Conformal machinery, weights, Bonferroni | stable |
| **Full-scale coverage numbers** | **provisional** — the 2026-07-26 cluster run used the RandomForest fallback predictor, not the MLP, because the job script omitted `--psi mlp`. Rerun required. |
| Assumption 3.2 constants | **contradictory between runs** — 0.64/0.76 vs 0.88/0.88 PD fraction. Unresolved. |
| Δ_c treatment | **contested** — see §8. |
