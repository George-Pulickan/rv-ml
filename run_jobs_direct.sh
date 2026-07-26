#!/bin/bash
# Direct (no-Slurm) runner for the two RHUL jobs on the CIM ts-node.
# Strips srun from the sbatch scripts and runs them sequentially;
# job 2 only starts if job 1 succeeds (afterok equivalent).
set -uo pipefail
cd "$HOME/rv-ml"
mkdir -p slurm/logs
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16     # leave headroom on the shared 24-core box
STAMP=$(date +%Y%m%d-%H%M)
echo "=== job 1: gp_conformal ($(date)) ==="
sed "s/^srun //" slurm/gp_conformal.sbatch | bash > "slurm/logs/direct-gp-cp-$STAMP.log" 2>&1
rc1=$?
echo "job 1 exit=$rc1 ($(date))"
if [ $rc1 -eq 0 ]; then
  echo "=== job 2: regression_benchmark ($(date)) ==="
  sed "s/^srun //" slurm/regression_benchmark.sbatch | bash > "slurm/logs/direct-reg-bench-$STAMP.log" 2>&1
  echo "job 2 exit=$? ($(date))"
else
  echo "job 2 SKIPPED (job 1 failed)"
fi
echo "=== all done ($(date)) ==="
