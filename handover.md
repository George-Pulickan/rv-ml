# Handover — RV-ML Project

## TL;DR for the next Claude

You're continuing summer research with George Pulickan (Notre Dame CS+Stats freshman) and advisor Nicolò Colombo (Royal Holloway University London). Goal: publish a paper — an ML system that predicts Kepler orbital parameters from radial-velocity time series, with conformal-prediction uncertainty quantification vs Bayesian credible intervals.

**Tasks 1–3 are done. All five encoder-stack modules are built, paper-grade reviewed, and committed. Seven encoder architectures exist and a 500K pretrain cache is on disk.** The next work is Task 4: uncertainty quantification (conformal + Bayesian).

George is terse and technical — match that. No over-explaining.

---

## What was completed in the most recent session

Full paper-grade refactor of all five modules. Everything smoke-tested and committed.

- **`models/kepler_torch.py`:** Differentiable Kepler decoder. Proper Newton-Raphson with exact second derivative (double backward). `fit_t_peri` takes explicit `mask` to avoid padding zeros corrupting γ/χ². 64-point coarse phase grid.
- **`models/encoder.py`:** Seven encoder architectures in `ENCODER_REGISTRY` (see Architecture summary). All share Branch B (GLS periodogram ResNet → 64-d) and the same MLP head. Baseline `resnet` is dual-branch 5-block ResNet ~452K params. Circular-orbit ω gate uses physical eccentricity (not normalised), sigmoid((e−0.05)×40).
- **`synthetic_dataset.py`:** Bootstraps real cadences from training split `.tbl` files. Beta(2,5) prior (Kipping 2013). GP noise via celerite2 + white-noise fallback on NaN. **Companion injection:** with probability `f_multi=0.30` (Howard et al. 2010), 1–2 companion planets are added; label is always the dominant planet (highest K). Returns 4-tuple (x, lsp, θ, info) matching RVDataset.
- **`preprocess.py`:** GLS periodogram (Zechmeister & Kürster 2009) added as second encoder input. LSP_N=512 log-spaced frequencies 0.5–5000 d. RVDataset now returns 4-tuple (x, lsp, θ, info). Systems with n_obs < 10 marked valid=False (single-obs inputs caused NaN encoder output via degenerate LSP/BatchNorm).
- **`train.py`:** AdamW + WarmupCosineSchedule (per-step warmup). 300 pretrain + 100 finetune epochs. Gradient clip 5.0. Collate handles (x, lsp, θ, info) 4-tuples. Reconstruction loss in normalised units.
- **`injection_recovery.py`:** 40-point phase grid; inverse-variance γ; e restart range [0, 0.99]; encoder mode builds LSP and refits T_peri for fair comparison.
- **`requirements.txt`:** Added `torch==2.12.0` and `celerite2==0.3.2` (were missing).

**Training smoke-tested:** `--no-pretrain --finetune-epochs 2` → clean finite loss. `--pretrain-epochs 2 --pretrain-n 500` → clean. MPS (Apple M3) confirmed working.

**Full pretrain was started on MPS then cancelled** — too slow (~45s/epoch) because the bottleneck is CPU-side data generation (LSP + GP per sample), not GPU. George's Royal Holloway colleagues will run it on their cluster instead.

**Pretrain cache generated:** `data/pretrain_cache.pt` — 500K samples, seed=42, `f_multi=0.30`. Multi-planet mix: 70.1% single, 22.4% one companion, 7.5% two companions. File is ~3.1 GB and git-ignored. Use `--pretrain-cache data/pretrain_cache.pt` on the cluster to bypass on-the-fly generation entirely.

---

## Project goals

1. ✅ RV literature (Doppler spectroscopy, Lovis & Fischer, Foreman-Mackey)
2. ✅ Data + validation pipeline — 795 validated systems, median RMS/σ = 1.63, 51 Peg χ²_red = 1.31
3. ✅ **GP noise model, preprocess.py, encoder stack (all five modules)**
4. ⏳ **NEXT** — Uncertainty quantification (conformal + Bayesian)

**Nicolò's autoencoder framing (do not deviate from this):**
- Encoder φ(RV) → orbital state X = (P, K, e, ω)  [T_peri refitted analytically per forward pass]
- Decoder = Kepler integrator (fixed, no learned weights)
- Loss = ‖RV − Kepler(φ(RV))‖
- Multi-modal extension: add ψ(Transit) encoder + Mandel-Agol transit decoder
- Full loss: ‖RV − Kepler(φ(RV))‖ + ‖Transit − Tran(ψ(Transit))‖ (drop missing term)

---

## Repository state

- Local: `~/rv-ml`, GitHub: `github.com/George-Pulickan/rv-ml` (private)
- Python 3.13 venv at `.venv`
- **George pushes himself** — commit but do not auto-push

### Key files

| File | Status | Purpose |
|---|---|---|
| `kepler_check.py` | ✅ | Validation pipeline. Entry: `validate_one(path, labels, **kwargs)`. |
| `parse_and_label.py` | ✅ | Host matching. `_norm_name` strips hyphens + spaces. `match_with_simbad` uses `data/simbad_cache.json`. |
| `gp_noise_model.py` | ✅ | GP library. `GPNoiseLibrary` drop-in noise sampler. |
| `cache_residuals.py` | ✅ | Runs `validate_one` over corpus → `data/residuals.npz` + `data/residuals_index.csv` |
| `gp_corpus_fit.py` | ✅ | Corpus-wide GP fit → `data/gp_fits.json`, `figures/gp_hyperparams.png` |
| `preprocess.py` | ✅ | RVDataset. 631 systems total (444 single-planet), 5-dim θ, host-grouped split, GLS periodogram. Returns (x, lsp, θ, info). |
| `synthetic_dataset.py` | ✅ | Synthetic RV generator with companion injection. SyntheticRVDataset, PregenSyntheticDataset, generate_one, generate_cache. |
| `models/kepler_torch.py` | ✅ | Differentiable Kepler decoder. KeplerDecoder, fit_t_peri, rv_keplerian. |
| `models/encoder.py` | ✅ | 7-arch encoder zoo φ(RV, LSP) → θ_norm. build_encoder, ENCODER_REGISTRY, encoder_loss, normalise_theta. |
| `injection_recovery.py` | ✅ | Injection-recovery benchmark. Decoder (classical LS) + encoder mode. |
| `train.py` | ✅ | Two-phase training. Phase 1: synthetic pretrain. Phase 2: real fine-tune. |
| `data/splits.csv` | ✅ | 631-row manifest: 437 train / 96 val / 98 test all systems; 292 / 78 / 74 after single-planet filter (host-grouped, seed=42) |
| `data/pretrain_cache.pt` | ✅ | 500K synthetic samples, seed=42, f_multi=0.30. ~3.1 GB, git-ignored. |
| `data/dataset_stats.json` | ✅ | Per-parameter normalisation constants (train split only) |
| `data/gp_fits.json` | ✅ | Per-system GP fit records (444 systems) |

---

## Running training on Royal Holloway cluster

The training bottleneck is CPU-side synthetic data generation (~9 ms/sample for LSP + GP). A100 GPU recommended.

**The pretrain cache (`data/pretrain_cache.pt`) eliminates the CPU bottleneck.** Copy it to the cluster and use `--pretrain-cache`. Epoch time drops from ~45s to ~5s.

```bash
# Install
pip install -r requirements.txt

# Recommended: use pretrain cache, run all 7 architectures
python train.py --arch resnet      --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch tcn         --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch inception   --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch lstm        --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch transformer --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch deep        --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4
python train.py --arch nolsp       --pretrain-cache data/pretrain_cache.pt --device cuda --workers 4

# Two-phase manually (no cache)
python train.py --pretrain-epochs 300 --finetune-epochs 0 --device cuda --workers 8
python train.py --finetune-only --resume checkpoints/resnet_pretrain_best.pt --device cuda --workers 8
```

Checkpoint names are arch-prefixed: `{arch}_pretrain_best.pt`, `{arch}_finetune_best.pt`.

**Speed tip:** with the cache, workers=4 is sufficient (just feeds tensors). Without it, max out physical CPU cores for LSP+GP generation.

---

## Architecture summary

### Encoder φ(RV, LSP) → θ — 7 architectures

All share **Branch B** (GLS periodogram ResNet → 64-d) and the same MLP head. Branch A varies.

**Branch B — GLS periodogram** (512 → 64-d, shared across all archs):
- Stem: Conv1d(1→16, k=11) + BN + ReLU
- 3 pre-activation ResBlocks: 2×stride-2 → spatial /4
- Global avg pool

**Head (shared):** cat(Branch A, 64) → Linear → ReLU → Linear(→5)

**Output:** θ_norm = [log10_P, log10_K, e, cos_ω, sin_ω] in N(0,1) space

| Key | Class | Branch A | Params | Notes |
|---|---|---|---|---|
| `resnet` | `RVEncoder` | 5-block ResNet, 2×stride-2, → 128-d | ~452K | baseline |
| `deep` | `RVEncoderDeep` | 7-block ResNet, 3×stride-2, → 256-d | ~1.05M | depth ablation |
| `tcn` | `RVEncoderTCN` | 6-layer dilated TCN, d=[1,2,4,8,16,32], RF≈T_MAX, → 128-d | ~295K | Bai et al. 2018 |
| `inception` | `RVEncoderInception` | 2-block InceptionTime k=11/21/41, → 128-d | ~255K | Fawaz et al. 2020 |
| `lstm` | `RVEncoderLSTM` | 2-layer BiLSTM hidden=64, → 128-d | ~196K | pack_padded_sequence |
| `transformer` | `RVEncoderTransformer` | 4-layer Transformer d=128 h=8, t_norm as feature, → 128-d | ~900K | irregular cadence |
| `nolsp` | `RVEncoderNoLSP` | 5-block ResNet only (no Branch B) | ~387K | ablation: no LSP |

Use `build_encoder(arch)` to instantiate by name.

### Decoder (KeplerDecoder)

Given θ_phys and (t_norm, t_span, t_min, rv_obs, rv_std, mask):
1. Un-normalise θ_norm → θ_phys = (P, K, e, ω)
2. Refit T_peri analytically: 64-point phase grid coarse search + 8 Newton-Raphson refinement steps (proper second-derivative, differentiable)
3. Compute RV curve via `rv_keplerian` (differentiable Kepler equation solve)
4. Return rv_pred in normalised units

### Loss (fine-tune)

`total = encoder_loss(θ̂_norm, θ_norm) + λ_rec × reconstruction_loss`

- `encoder_loss`: per-dim MSE with circular-orbit gate on cos_ω/sin_ω dims (gate ≈ 0 for e < 0.05, ≈ 1 for e > 0.15)
- `reconstruction_loss`: ‖rv_norm − rv_pred‖² averaged over real observations (dimensionless)
- Default λ_rec = 0.1 (pretrain uses λ_rec = 0)

---

## preprocess.py — design decisions

**631 usable systems** (from 795 cached): requires pl_rvamp (K) and pl_orbper. 338 unique hosts, split 437/96/98 train/val/test by host (no leakage). Systems with n_obs < 10 excluded (valid=False) — cannot constrain 5 params. After the `single_planet=True` filter (default in training): **292 train / 78 val / 74 test**. Normalisation stats (`data/dataset_stats.json`) computed on single-planet + known-eccentricity systems only.

**Theta (5-dim):** `[log10(P), log10(K), e, cos(ω), sin(ω)]`
- T_peri excluded — epoch-dependent; analytically refittable given (P,K,e,ω).
- (cos ω, sin ω) circular encoding avoids 2π discontinuity.
- e=0 for systems missing eccentricity (circular orbit prior). ω=0 for e=0.

**Input tensor (4×256 float32):**
- Row 0: `t_norm = (t - t_min) / t_span` ∈ [0,1]
- Row 1: `rv_norm = (rv - median(rv)) / std(rv)` — median subtraction removes γ
- Row 2: `sig_norm = sigma / std(rv)`
- Row 3: `mask` = 1.0 for real obs, 0.0 for padding

**GLS periodogram (512 float32):** log-spaced frequencies 1/5000 to 1/0.5 d⁻¹. Computed via `astropy.timeseries.LombScargle(normalization='standard', fit_mean=True)`. Clipped to [0,1].

**API:**
```python
from preprocess import RVDataset, THETA_NAMES, LSP_N, compute_lsp
# THETA_NAMES = ['log10_P', 'log10_K', 'e', 'cos_omega', 'sin_omega']
# LSP_N = 512

ds = RVDataset('train')
x, lsp, theta, info = ds[0]          # (4,256), (512,), (5,), dict
# info keys: host, file, n_obs, n_planets, valid, t_span_days, t_min_days, rv_std_ms
```

---

## GP noise model

**`gp_noise_model.py`** — fits celerite2 GP to per-system post-Keplerian residuals.

**Corpus results (444 systems):** Matérn-3/2: 59%, SHO: 39%, composites: 2%. KS normality pass: 98.2%, Ljung-Box: 90.3%. Median ρ = 44 d, σ = 2.5 m/s, jitter = 1.34 m/s.

**API:**
```python
from gp_noise_model import GPNoiseLibrary
lib = GPNoiseLibrary.from_json('data/gp_fits.json')
noise = lib.sample(t, rng=rng)   # ndarray, same shape as t
```

---

## Next steps — Task 4: uncertainty quantification

The encoder is trained (or will be once Royal Holloway runs the full training). Next:

1. **Run full training on Royal Holloway cluster** → `checkpoints/pretrain_best.pt`, `checkpoints/finetune_best.pt`
2. **Conformal prediction:** split-conformal on val set. Calibrate per-parameter residuals → coverage-guaranteed prediction intervals. Library: `MAPIE` or implement directly.
3. **Bayesian baseline:** HMC/NUTS posterior on (P,K,e,ω) per system (e.g. via `NumPyro` or `Stan`). Compare credible intervals to conformal intervals on test set.
4. **Evaluation:** coverage (empirical vs nominal), interval width, calibration curves per parameter. Compare encoder + conformal vs classical Bayesian on test set.
5. **Multi-modal (later):** `batman`/`starry` transit decoder + ψ(Transit) encoder.

---

## Technical pitfalls

- **`validate_one` takes `labels` as required positional arg** — signature: `validate_one(path, labels, mode, ...)`.
- **`preprocess.py` loads raw RV, not residuals** — `residuals.npz` has no orbital signal.
- **T_peri excluded from θ** — 56.6% of systems have no catalog T_peri; analytically refit in decoder.
- **Normalisation stats from train split only** — `data/dataset_stats.json`. Apply same to val/test.
- **`encoder_loss` requires `stats` kwarg** for the circular-orbit ω gate to work correctly. Passing `stats=None` disables the gate.
- **`fit_t_peri` mask parameter is required** — do not pass pre-masked rv; pass full rv_obs and mask separately. Otherwise padding zeros corrupt the γ/χ² computation.
- **n_obs < 10 → valid=False** — single-observation systems cause NaN encoder output (degenerate LSP, degenerate BatchNorm). Already guarded in `RVDataset`.
- **MPS is available on M3** but data generation is the bottleneck, not GPU. Use `--device mps` but don't expect linear speedup over CPU.
- **DataLoader workers each re-load real time grids** — the "loaded 388 real time grids" message prints once per worker (expected, harmless).
- **`--arch` flag must match checkpoint** — train.py reads `arch` from the checkpoint dict on resume and overrides `--arch` to prevent mismatch. Checkpoint names are arch-prefixed (`{arch}_pretrain_best.pt` etc.).
- **Companion injection label is the dominant planet (highest K)** — not the primary (first-drawn) planet. If a companion is drawn with higher K it becomes the label. This matches `preprocess._usable_systems`.

---

## References

- Foreman-Mackey, D. et al. 2017, AJ 154, 220 — celerite (GP framework)
- Faria, J. et al. 2016, A&A 588, A31 — kima (prior conventions)
- Haywood, R. et al. 2014, MNRAS 443, 2517 — CoRoT-7 RV+GP methodology
- Rasmussen & Williams 2006 — GP textbook (whitening, GoF)
- Edelson & Krolik 1988, ApJ 333, 646 — DCF
- Zechmeister & Kürster 2009, A&A 496, 577 — Generalised Lomb-Scargle periodogram
- He et al. 2016, CVPR — Identity mappings in deep residual networks (pre-activation ResBlocks)
- Kipping 2013, MNRAS 434, L51 — Beta(2,5) eccentricity prior
- Mandel & Agol 2002, ApJ 580, L171 — transit light curves
- Kreidberg 2015, PASP 127, 1161 — batman transit code
- Luger et al. 2019 — starry (analytic transit gradients)
- Cranmer et al. 2020, PNAS — Neural Posterior Estimation / SBI
- Cumming et al. 1999, ApJ 526, 890 — RV detection thresholds (n_obs floor)
- Bai, S. et al. 2018, arXiv:1803.01271 — dilated TCN (encoder arch)
- Fawaz, H.I. et al. 2020, Data Min. Knowl. Disc. 34:1755 — InceptionTime (encoder arch)
- Howard, A.W. et al. 2010, Science 330, 653 — RV multiplicity function (f_multi=0.30)
- Mayor, M. et al. 2011, arXiv:1109.2497 — multi-planet companion count distribution
- Lucy, L.B. & Sweeney, M.A. 1971, AJ 76, 544 — circular-orbit ω degeneracy threshold
