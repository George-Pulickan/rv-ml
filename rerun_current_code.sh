#!/bin/bash
set -uo pipefail
cd "$HOME/rv-ml" || { echo "CD FAILED" > "$HOME/rerun_cd_failed.log"; exit 1; }
mkdir -p slurm/logs
STAMP=$(date +%Y%m%d-%H%M)
exec > "slurm/logs/rerun-wrapper-$STAMP.log" 2>&1
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
echo "=== rerun started ($(date)), STAMP=$STAMP, host=$(hostname), pwd=$(pwd) ==="

echo "=== conformal_shift.py default (current code) ($(date)) ==="
python conformal_shift.py > "slurm/logs/rerun-cp-default-$STAMP.log" 2>&1
rc=$?
echo "cp-default exit=$rc ($(date))"
if [ $rc -ne 0 ]; then echo "ABORT after cp-default failure"; exit 1; fi

echo "=== conformal_shift.py real-val gamma (current code) ($(date)) ==="
python conformal_shift.py --gamma-tune-on real-val \
    --out-dir synthetic_generation/regression/gamma_real_val \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/gamma_real_val \
    > "slurm/logs/rerun-cp-realval-$STAMP.log" 2>&1
rc=$?
echo "cp-realval exit=$rc ($(date))"
if [ $rc -ne 0 ]; then echo "ABORT after cp-realval failure"; exit 1; fi

echo "=== push job1 refresh (restored GP checkpoint + fresh CP, current code) ($(date)) ==="
git add -f \
    models/gp_residual_svgp.pt models/gp_residual_metrics.json \
    figures/gp_residual \
    synthetic_generation/regression/conformal_shift_metrics.json \
    synthetic_generation/regression/conformal_shift_report.txt \
    synthetic_generation/regression/gamma_real_val \
    synthetic_generation/figures/synthetic_regression_10000 \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: gp_conformal rerun on current main ($(date +%F))" || true
git push origin HEAD:refs/heads/rhul-results || echo "WARNING: push failed"

echo "=== regression_benchmark.sbatch full (current code) ($(date)) ==="
sed "s/^srun //" slurm/regression_benchmark.sbatch | bash > "slurm/logs/rerun-regbench-$STAMP.log" 2>&1
rc=$?
echo "regbench exit=$rc ($(date))"
if [ $rc -ne 0 ]; then echo "ABORT after regbench failure"; exit 1; fi

echo "=== mlp_psi CP run, cluster-trained checkpoint, separate dir ($(date)) ==="
python conformal_shift.py --psi mlp \
    --checkpoint checkpoints/regression_mlp_74.pt \
    --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
    --out-dir synthetic_generation/regression/mlp_psi_cluster_20260725 \
    --fig-dir synthetic_generation/figures/synthetic_regression_10000/mlp_psi_cluster_20260725 \
    > "slurm/logs/rerun-mlp-psi-$STAMP.log" 2>&1
rc=$?
echo "mlp-psi exit=$rc ($(date))"

echo "=== push mlp_psi_cluster result (unofficial checkpoint, separate from paper's mlp_psi/) ($(date)) ==="
git add -f \
    synthetic_generation/regression/mlp_psi_cluster_20260725 \
    synthetic_generation/figures/synthetic_regression_10000/mlp_psi_cluster_20260725 \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: mlp_psi CP run on cluster-trained (unofficial, not issue #10) checkpoint ($(date +%F))" || true
git push origin HEAD:refs/heads/rhul-results || echo "WARNING: push failed"

echo "=== ALL RERUN STEPS DONE ($(date)) ==="
echo "STAMP=$STAMP"
