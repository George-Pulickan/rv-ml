#!/bin/bash
#SBATCH --job-name=rvml-reg-bench
#SBATCH --partition=compute        # <-- ADJUST: RHUL CPU partition name (CPU-only job)
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00            # CSV regen (10k systems) + ~10 MLP trainings + refolds
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --account=YOUR_ACCOUNT    # <-- UNCOMMENT + set if the cluster requires an allocation

# -------------------------------------------------------------------------
# Regenerate the regression-head benchmark artifacts invalidated by the
# replay bug fixed @ 0b246d2 (predicted-P refolds used a different system's
# curve per row, so the committed benchmark.json / two-step numbers are
# meaningless). Sequential, order matters:
#   1. regenerate the phasefold CSV — makes the job self-consistent with
#      whatever GP noise checkpoint is present (important if
#      gp_conformal.sbatch retrained it; run that job FIRST if both queued)
#   2. replay round-trip test — hard guard that predicted-P refolds will
#      reproduce exactly the systems in the CSV just written
#   3. Gates A/B/C + loss-weight ablation -> figures/regression_synthetic/benchmark.json
#   4. two-step P-then-shape eval -> two_step_metrics.json + checkpoints 74/109
#   5. diagnostics suite on the 109-D checkpoint from step 4
#   6. zero-inflated-e counters on 109-D: --e-balance / --e-head hurdle /
#      both, vs the gate-B 109-D oracle baseline (Nicolo's e==0 pile-up
#      concern; compare per_target.e + e_zero_classifier across variants)
#
# Submit from the repo root:   sbatch slurm/regression_benchmark.sbatch
# Rsync back + commit: figures/regression_synthetic/ (benchmark.json,
# metrics.json, PNGs) and the regenerated phasefold CSV. two_step_metrics.json,
# diagnostics/, and checkpoints/ are gitignored — rsync back for inspection.
# -------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

# ---- Environment (same venv convention as the other sbatch jobs) ---------
source .venv/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ---- Preconditions --------------------------------------------------------
test -f data/splits.csv || { echo "MISSING data/splits.csv (rsync data/ over)"; exit 1; }
test -f models/gp_residual_svgp.pt || echo "NOTE: no models/gp_residual_svgp.pt — noise sampler falls back to gp_fits.json"

PHASEFOLD_CSV=synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv

# ---- 1. Regenerate the phasefold CSV --------------------------------------
srun python synthetic_generation/generate_synthetic_regression_csv.py \
    --with-phasefold --out "$PHASEFOLD_CSV"

# ---- 2. Replay consistency guard ------------------------------------------
srun python -m unittest tests.test_replay_synthetic_sample -v

# ---- 3. Gates A/B/C + ablation --------------------------------------------
srun python regression.py --benchmark-gates --run-optional-ablations --device cpu

# ---- 4. Two-step P-then-shape pipeline -------------------------------------
srun python regression.py --two-step --device cpu

# ---- 5. Diagnostics on the 109-D checkpoint --------------------------------
srun python regression.py --diagnose --feature-set 109 \
    --csv "$PHASEFOLD_CSV" --checkpoint checkpoints/regression_mlp_109.pt --device cpu

# ---- 6. Zero-inflated-e counters on 109-D ----------------------------------
# Baseline for comparison = gate_b_109_oracle in benchmark.json (same
# settings, default e head, no balancing).
srun python regression.py --feature-set 109 --csv "$PHASEFOLD_CSV" --device cpu \
    --e-balance \
    --out figures/regression_synthetic/e_head_balance \
    --checkpoint checkpoints/regression_mlp_109_ebalance.pt
srun python regression.py --feature-set 109 --csv "$PHASEFOLD_CSV" --device cpu \
    --e-head hurdle \
    --out figures/regression_synthetic/e_head_hurdle \
    --checkpoint checkpoints/regression_mlp_109_hurdle.pt
srun python regression.py --feature-set 109 --csv "$PHASEFOLD_CSV" --device cpu \
    --e-head hurdle --e-balance \
    --out figures/regression_synthetic/e_head_hurdle_balance \
    --checkpoint checkpoints/regression_mlp_109_hurdle_ebalance.pt

echo "Done."
echo "  Gates:      figures/regression_synthetic/benchmark.json"
echo "  Two-step:   figures/regression_synthetic/two_step_metrics.json"
echo "  Diagnostics figures/regression_synthetic/diagnostics/"
echo "  e-head:     figures/regression_synthetic/e_head_{balance,hurdle,hurdle_balance}/metrics.json"
