#!/bin/bash
# Controlled simulator-misspecification study.
# Calibration/tuning stay at f_multi=0 (the ideal simulator); only the synthetic
# TEST set is perturbed.  Four runs in parallel, 5 threads each on a 24-core node.
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
mkdir -p ~/fixvalidate
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/misspec-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=5

PIN=~/pinned_checkpoints/regression_mlp_74_paper_20260726-0946.pt
CSV=synthetic_generation/datasets/synthetic_regression_10000.csv
echo "=== misspec sweep started ($(date)) host=$(hostname) STAMP=$STAMP ==="
echo "checkpoint: $PIN"

for FM in 0 0.1 0.25 0.5; do
  TAG=$(echo "$FM" | tr -d '.')
  OUT=synthetic_generation/regression/misspec_${TAG}_$STAMP
  (
    python conformal_shift.py \
        --psi mlp --checkpoint "$PIN" --csv "$CSV" \
        --test-f-multi "$FM" \
        --n-constants 0 \
        --out-dir "$OUT" \
        --fig-dir synthetic_generation/figures/synthetic_regression_10000/misspec_${TAG}_$STAMP \
        > "$HOME/fixvalidate/misspec-fm${TAG}-$STAMP.log" 2>&1
    echo "f_multi=$FM exit=$? ($(date))"
  ) &
done
wait
echo "=== all four finished ($(date)) STAMP=$STAMP ==="
for FM in 0 0.1 0.25 0.5; do
  TAG=$(echo "$FM" | tr -d '.')
  echo "  f_multi=$FM -> synthetic_generation/regression/misspec_${TAG}_$STAMP"
done
