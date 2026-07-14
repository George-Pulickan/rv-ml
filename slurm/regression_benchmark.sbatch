#!/bin/bash
#SBATCH --job-name=rvml-reg-bench
#SBATCH --partition=compute        # <-- ADJUST: RHUL CPU partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --account=YOUR_ACCOUNT    # <-- UNCOMMENT if required

# Regen regression benchmarks after the replay fix (0b246d2).
# Order: CSV -> replay test -> gates -> two-step -> diagnostics -> e-head.
# Submit from repo root:  sbatch slurm/regression_benchmark.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

source .venv/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

test -f data/splits.csv || { echo "MISSING data/splits.csv"; exit 1; }
test -f models/gp_residual_svgp.pt || echo "NOTE: no GP checkpoint — using gp_fits.json fallback"

PHASEFOLD_CSV=synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv

srun python synthetic_generation/generate_synthetic_regression_csv.py \
    --with-phasefold --out "$PHASEFOLD_CSV"

srun python -m unittest tests.test_replay_synthetic_sample -v

if [[ -f checkpoints/regression_mlp_109.pt ]]; then
  srun python regression.py --period-tolerance --feature-set 109 \
      --csv "$PHASEFOLD_CSV" --checkpoint checkpoints/regression_mlp_109.pt --device cpu
fi

srun python regression.py --benchmark-gates --run-optional-ablations --device cpu

srun python regression.py --two-step --device cpu \
    --stage2-fold predicted --period-source lsp_peak

srun python regression.py --two-step --device cpu \
    --stage2-fold oracle --period-source mlp74 \
    --out figures/regression_synthetic/two_step_oracle_mlp74

ADAPTED_CKPT=checkpoints/regression_mlp_109_predicted_lsp_peak.pt
if [[ -f "$ADAPTED_CKPT" ]]; then
  DIAG_CKPT="$ADAPTED_CKPT"
else
  DIAG_CKPT=checkpoints/regression_mlp_109.pt
fi
srun python regression.py --diagnose --feature-set 109 \
    --csv "$PHASEFOLD_CSV" --checkpoint "$DIAG_CKPT" --device cpu

srun python regression.py --e-head-ablate --csv "$PHASEFOLD_CSV" --device cpu \
    --out figures/regression_synthetic

git add -f \
    "$PHASEFOLD_CSV" \
    figures/regression_synthetic \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: regression_benchmark ($(date +%F), job ${SLURM_JOB_ID:-manual})" || true
git push origin HEAD:refs/heads/rhul-results || \
    echo "WARNING: push failed — fix credentials, then: git push origin HEAD:refs/heads/rhul-results"

echo "Done."
echo "  Gates:    figures/regression_synthetic/benchmark.json"
echo "  Two-step: figures/regression_synthetic/two_step_metrics.json"
echo "  Diags:    figures/regression_synthetic/diagnostics/"
echo "  e-head:   figures/regression_synthetic/e_head_ablate/comparison.json"
