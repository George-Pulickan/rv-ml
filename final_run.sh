#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
mkdir -p ~/fixvalidate
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/final-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
echo "=== final run started ($(date)), host=$(hostname), STAMP=$STAMP ==="

python -m unittest discover -s tests 2>&1 | tail -3
git checkout figures/synthetic_plots/ 2>/dev/null

echo "--- full CP, psi=mlp, n_constants=25, post-LSP-fix ---"
python conformal_shift.py \
    --psi mlp \
    --checkpoint checkpoints/regression_mlp_74.pt \
    --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
    --n-constants 25 \
    --out-dir synthetic_generation/regression/mlp_psi_final_$STAMP \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/mlp_psi_final_$STAMP
echo "exit=$? ($(date))"
echo "STAMP=$STAMP"
echo "=== final run done ($(date)) ==="
