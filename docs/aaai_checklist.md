# AAAI 2027 checklist (exoplanet section)

## Artifacts to paste into Overleaf
- Fig 1: `figures/paper/rv_heldout_phasefold.png` (Γ traces: P frozen at fold period)
- Fig 2: `figures/paper/rv_pred_vs_true.png` (P/K/e only; ω omitted by design)
- Table: `figures/paper/earthlike_top10.tex` (val/test hosts; **per-system papernorm** α=0.1 half-widths)
- CP metrics: `synthetic_generation/regression/mlp_psi/conformal_shift_metrics.json`
  - Per-system widths: `synthetic_generation/regression/mlp_psi/per_system_widths_papernorm.json`
  - Optional gamma-on-real-val: `synthetic_generation/regression/mlp_psi_real_val/`

## Story for reviewers
- ψ = 74-D MLP (dual e-head, θ targets); UQ = `conformal_shift --psi mlp`
- Coverage claim: surrogate/raw real-weighted joint ≈ 0.95 at 90% (n_cal=400)
- Honest limitation: ω intervals near full circle; 74-D has no periapsis epoch
- Prefer hk targets `(e cos ω, e sin ω)` over `(ω, e cos ω)` (singular decode)
- Phase-2 gate: epoch-free phase-fold (max-RV anchor) **did not** restore absolute ω
  (109-D epoch-free: e R²≈0.08, ω MAE≈59°; vs oracle-t_peri: e R²≈0.50, ω MAE≈23°)
- 74-D hk without phase-fold features also failed (e R² negative) — hk needs shape features

## Pre-submit
- [ ] Anonymize author names / repo URLs in the camera-ready / review PDF
- [ ] Confirm page limit under AAAI 2027 style
- [ ] Cite Tibshirani et al. 2019 for weighted conformal under covariate shift (`\cite{tibshirani2019}` in Overleaf)
- [ ] Reproducibility: seed, checkpoint path, CSV, and CLI for the CP run in the supplement
- [ ] Figures in Overleaf match the PNGs above (not older 5-panel ω plot)
