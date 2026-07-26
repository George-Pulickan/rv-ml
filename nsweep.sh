#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
mkdir -p ~/fixvalidate
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/noisemis-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=5
PIN=$HOME/pinned_checkpoints/regression_mlp_74_paper_20260726-0946.pt
CSV=synthetic_generation/datasets/synthetic_regression_10000.csv
echo "=== noise-misspecification sweep started ($(date)) host=$(hostname) STAMP=$STAMP ==="
for FR in 0 0.25 0.5 1.0; do
  TAG=$(echo "$FR" | tr -d '.')
  OUT=synthetic_generation/regression/noisemis_${TAG}_$STAMP
  (
    python conformal_shift.py \
        --psi mlp --checkpoint "$PIN" --csv "$CSV" \
        --test-noise-frac "$FR" --test-noise-tau 30 \
        --n-constants 0 \
        --out-dir "$OUT" \
        --fig-dir synthetic_generation/figures/synthetic_regression_10000/noisemis_${TAG}_$STAMP \
        > "$HOME/fixvalidate/noisemis-fr${TAG}-$STAMP.log" 2>&1
    echo "noise_frac=$FR exit=$? ($(date))"
  ) &
done
wait
echo "=== all four finished ($(date)) STAMP=$STAMP ==="
