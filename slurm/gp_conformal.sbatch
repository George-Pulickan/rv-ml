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
#      offset gamma (fv4 cache). Rebuilds models/gp_residual_svgp.pt +
#      models/gp_residual_metrics.json + figures/gp_residual/*.
#   2. regenerate both synthetic regression CSVs — the committed ones were
#      drawn from FULL-CORPUS priors H; H is now fit train-only (Nicolò's
#      call), and the noise sampler is the fresh step-1 checkpoint. Without
#      this, psi is trained on curves shaped by the real test systems.
#   3. conformal_shift.py    — split-CP calibrated on fake / tested on real,
#      naive vs surrogate score, LR-reweighted, at the full n_cal=400 scale.
#      Uses the checkpoint from step 1 as its noise proxy v.
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

# ---- 3. Full-scale conformal shift run ------------------------------------
srun python conformal_shift.py

echo "Done."
echo "  GP metrics:  models/gp_residual_metrics.json"
echo "  CP report:   synthetic_generation/regression/conformal_shift_report.txt"
echo "  CP metrics:  synthetic_generation/regression/conformal_shift_metrics.json"
