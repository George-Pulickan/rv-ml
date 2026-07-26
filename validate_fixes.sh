#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || { echo "CD FAILED" > "$HOME/validate_cd_failed.log"; exit 1; }
mkdir -p ~/fixvalidate
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/wrapper-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
echo "=== validate_fixes started ($(date)), host=$(hostname), STAMP=$STAMP ==="

# ---- 1. Does the Assumption 3.2 PD failure come from unconverged GD? --------
echo "--- step 1: GD convergence sweep for lambda_min(H*) ---"
cat > gdsweep.py <<'PYEOF'
from conformal_shift import NoiseProxy, estimate_constants
proxy = NoiseProxy()
for steps in (200, 1000, 4000):
    o = estimate_constants(proxy, 12, 41, gd_steps=steps, gd_lr=0.02)
    lm = o["lambda_min_H"]
    print("gd_steps=%5d  PD_frac=%.2f  lam_min med=%10.4g  p10=%10.4g  min=%10.4g  kappa_med=%.4g"
          % (steps, o["frac_positive_definite"], lm["median"], lm["p10"], lm["min"],
             o["kappa_H"]["median"]), flush=True)
PYEOF
python gdsweep.py
echo "step 1 exit=$?"
rm -f gdsweep.py

# ---- 2. Full-scale CP run WITH the assumption constants ---------------------
# psi = MLP on the 74-D CSV, matching the paper's committed configuration.
# Writes to a fresh directory so the paper's mlp_psi/ stays untouched.
echo "--- step 2: full CP (n_cal=400), --psi mlp, --n-constants 25 ---"
python conformal_shift.py \
    --psi mlp \
    --checkpoint checkpoints/regression_mlp_74.pt \
    --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
    --n-constants 25 \
    --out-dir synthetic_generation/regression/mlp_psi_fixed_$STAMP \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/mlp_psi_fixed_$STAMP
echo "step 2 exit=$? ($(date))"

echo "=== validate_fixes done ($(date)) ==="
