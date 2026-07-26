#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/figs-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
RUN=synthetic_generation/regression/paper_20260726-0946
PIN=$HOME/pinned_checkpoints/regression_mlp_74_paper_20260726-0946.pt
mkdir -p ~/paper_artifacts/figs_pinned
echo "=== figure regen started ($(date)) host=$(hostname) ==="
# earthlike table must exist first (boxes reads it); regenerate into place then restore.
cp ~/paper_artifacts/earthlike_pinned/earthlike_top10.csv figures/paper/earthlike_top10.csv
for ONLY in phasefold predtrue boxes; do
  echo "--- $ONLY ---"
  python scripts/paper_rv_figures.py --only $ONLY \
      --checkpoint "$PIN" \
      --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
      --metrics $RUN/conformal_shift_metrics.json \
      --widths $RUN/per_system_widths_papernorm.json \
      --device cpu
  echo "$ONLY exit=$?"
done
cp figures/paper/rv_*.png ~/paper_artifacts/figs_pinned/ 2>/dev/null
cp figures/paper/mlp_cp_quantiles.json ~/paper_artifacts/figs_pinned/ 2>/dev/null
git checkout -- figures/paper/
echo "=== restored committed figures; new ones in ~/paper_artifacts/figs_pinned ==="
ls -l ~/paper_artifacts/figs_pinned/
echo "=== done ($(date)) ==="
