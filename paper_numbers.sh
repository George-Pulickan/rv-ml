#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
mkdir -p ~/fixvalidate ~/pinned_checkpoints
STAMP=$(date +%Y%m%d-%H%M)
exec > "$HOME/fixvalidate/paper-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16

COMMIT=$(git rev-parse HEAD)
echo "=== paper_numbers started ($(date)) host=$(hostname) STAMP=$STAMP commit=$COMMIT ==="

# ---- 1. Retrain psi at the CURRENT commit so the checkpoint is reproducible --
echo "--- step 1: retrain 74-D psi at current main ---"
TRAIN_CMD="python regression.py --two-step --device cpu"
echo "command: $TRAIN_CMD"
$TRAIN_CMD
rc=$?
echo "train exit=$rc ($(date))"
if [ $rc -ne 0 ]; then echo "ABORT: training failed"; exit 1; fi

# ---- 2. Pin it, with provenance -------------------------------------------
PIN=~/pinned_checkpoints/regression_mlp_74_paper_$STAMP.pt
cp checkpoints/regression_mlp_74.pt "$PIN"
chmod 444 "$PIN"
SHA=$(sha256sum "$PIN" | cut -d' ' -f1)
PYV=$(python -c 'import sys,torch,numpy;print("python %s | torch %s | numpy %s" % (sys.version.split()[0], torch.__version__, numpy.__version__))')
cat > ~/pinned_checkpoints/PROVENANCE_$STAMP.txt <<EOF
psi checkpoint pinned for the AAAI paper
========================================
file        : $(basename "$PIN")
sha256      : $SHA
created     : $(date -Iseconds)
host        : $(hostname)
git commit  : $COMMIT
command     : $TRAIN_CMD
run from    : $HOME/rv-ml
versions    : $PYV
source CSV  : synthetic_generation/datasets/synthetic_regression_10000.csv
            : sha256 $(sha256sum synthetic_generation/datasets/synthetic_regression_10000.csv | cut -d' ' -f1)
phasefold   : synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv
            : sha256 $(sha256sum synthetic_generation/datasets/synthetic_regression_10000_phasefold.csv 2>/dev/null | cut -d' ' -f1)
notes       : regression.py --two-step trains the 74-D model via train_model(...,
              checkpoint_path=CHECKPOINT_74) with argparse defaults for
              epochs/seed/lr/batch/val_frac/patience.  Re-running the command
              above at this commit reproduces the checkpoint.
              Supersedes the 2026-07-25 19:57 checkpoint, which was trained
              while the cluster was at be9df5a (46 commits behind, regression.py
              differing by 1852 lines) and is therefore NOT reproducible from
              main.
EOF
echo "--- pinned -> $PIN"
cat ~/pinned_checkpoints/PROVENANCE_$STAMP.txt

# ---- 3. Full CP against the pinned checkpoint ------------------------------
echo "--- step 3: full CP (n_cal=400) against the pinned checkpoint ---"
OUT=synthetic_generation/regression/paper_$STAMP
python conformal_shift.py \
    --psi mlp \
    --checkpoint "$PIN" \
    --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
    --n-constants 25 \
    --out-dir "$OUT" \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/paper_$STAMP
echo "cp exit=$? ($(date))"
echo "OUT=$OUT"
echo "STAMP=$STAMP"
echo "=== paper_numbers done ($(date)) ==="
