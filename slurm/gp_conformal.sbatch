#!/bin/bash
#SBATCH --job-name=rvml-gp-cp
#SBATCH --partition=compute        # <-- ADJUST: RHUL CPU partition name (no GPU needed; both scripts are CPU-only)
#SBATCH --cpus-per-task=16         # torch/sklearn thread pool
#SBATCH --mem=32G
#SBATCH --time=24:00:00            # SVGP retrain (~180k rows, 400 iters) + n_cal=400 CP run
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --account=YOUR_ACCOUNT    # <-- UNCOMMENT + set if the cluster requires an allocation
##SBATCH --mail-type=END,FAIL      # <-- optional
##SBATCH --mail-user=you@rhul.ac.uk

# -------------------------------------------------------------------------
# Full-scale GP retrain + dataset regen + conformal-shift run (Nicolò's
# 2026-07 spec). Sequential, order matters:
#   1. gp_residual_model.py  — SVGP retrain with the least-squares systemic
#      offset gamma AND the log10_sigma conditioning feature (8-D, fv5 cache;
#      both Nicolò-approved 2026-07). Rebuilds models/gp_residual_svgp.pt +
#      models/gp_residual_metrics.json + figures/gp_residual/*.
#   2. regenerate both synthetic regression CSVs — the committed ones were
#      drawn from FULL-CORPUS priors H; H is now fit train-only (Nicolò's
#      call), and the noise sampler is the fresh step-1 checkpoint. Without
#      this, psi is trained on curves shaped by the real test systems.
#   3. conformal_shift.py    — split-CP calibrated on fake / tested on real,
#      naive / naive_adj (paper eq 41 quantile adjustment) / surrogate
#      (L1 gradient descent init at theta_bar), LR-reweighted, with the
#      raw / vnorm / v2norm / papernorm (paper eqs 18-24) score variants,
#      Assumption 2.1 noise filter + Assumption 2.3 constants, at the full
#      n_cal=400 scale. Uses the checkpoint from step 1 as noise proxy v_y.
#      Run TWICE: gamma tuned on the synthetic tuning set (default) and on
#      the real val split (the paper's D_val) — both of Nicolò's candidate
#      answers land in one job (second run -> regression/gamma_real_val/).
#
# Submit from the repo root:   sbatch slurm/gp_conformal.sbatch
# -------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

# ---- Environment (same venv convention as train_encoder.sbatch) ---------
source .venv/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ---- Preconditions -------------------------------------------------------
test -f data/splits.csv || { echo "MISSING data/splits.csv (rsync data/ over)"; exit 1; }

# ---- 1. Full-scale SVGP retrain (LS-gamma offset) ------------------------
srun python gp_residual_model.py
test -f models/gp_residual_svgp.pt || { echo "retrain did not produce models/gp_residual_svgp.pt"; exit 1; }

# ---- 2. Regenerate synthetic datasets (train-only H + fresh noise model) --
srun python synthetic_generation/generate_synthetic_regression_csv.py
srun python synthetic_generation/generate_lsp_regression_csv.py

# ---- 3. Full-scale conformal shift run (both gamma tuning sets + psi* ablation)
srun python conformal_shift.py
srun python conformal_shift.py --gamma-tune-on real-val \
    --out-dir synthetic_generation/regression/gamma_real_val \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/gamma_real_val
# psi trained on GD surrogate labels theta* instead of theta_bar (Nicolò's
# ablation, 2026-07-14; the theta* replay+GD for 10k rows is cached to an .npz
# next to the CSV, so reruns are cheap)
srun python conformal_shift.py --psi-labels star --gamma-tune-on real-val \
    --out-dir synthetic_generation/regression/psi_star \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/psi_star

# ---- 4. Push artifacts to the rhul-results branch --------------------------
# Commits on the current branch and pushes HEAD to rhul-results, so sequential
# jobs / reruns fast-forward. Needs push credentials (PAT or SSH key) on the
# cluster. A failed push never fails the job — artifacts stay on disk.
# Excluded (large, regenerable): synthetic_lsp_regression_10000.csv (~130 MB).
git add -f \
    models/gp_residual_svgp.pt models/gp_residual_metrics.json \
    figures/gp_residual \
    synthetic_generation/datasets/synthetic_regression_10000.csv \
    synthetic_generation/regression \
    synthetic_generation/figures/synthetic_regression_10000 \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: gp_conformal ($(date +%F), job ${SLURM_JOB_ID:-manual})" || true
git push origin HEAD:refs/heads/rhul-results || \
    echo "WARNING: push failed — fix credentials, then: git push origin HEAD:refs/heads/rhul-results"

echo "Done."
echo "  GP metrics:  models/gp_residual_metrics.json"
echo "  CP report:   synthetic_generation/regression/conformal_shift_report.txt"
echo "  CP metrics:  synthetic_generation/regression/conformal_shift_metrics.json"
echo "  CP (gamma on real val): synthetic_generation/regression/gamma_real_val/"
