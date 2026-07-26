#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || { echo "CD FAILED" > "$HOME/fixcheck_cd_failed.log"; exit 1; }
mkdir -p ~/fixcheck
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixcheck/wrapper-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
echo "=== fixcheck smoke started ($(date)), host=$(hostname) ==="

python conformal_shift.py --n-cal 20 --n-test 20 --n-tune 15 --n-weight-synth 50 --n-constants 0 \
  --out-dir "$HOME/fixcheck/cp_out" --fig-dir "$HOME/fixcheck/cp_fig"
rc=$?
echo "conformal_shift smoke exit=$rc ($(date))"
echo "=== fixcheck smoke done ($(date)) ==="
