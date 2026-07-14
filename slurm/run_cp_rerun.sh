#!/bin/bash
# CP-step-only rerun for the RHUL CIM box (no Slurm) — conformal_shift at
# >= 08eb329 (pointwise paper deltas, filter histograms, --psi-labels star).
#
# Assumes the original gp_conformal job's steps 1-2 already produced the fresh
# GP checkpoint + regenerated CSVs on this machine — this script does NOT
# retrain the SVGP. If the original runner (or a previous rerun) is still
# active it WAITS for it, so it can be launched and left unattended:
#
#   ssh VPAC005@linux.cim.rhul.ac.uk 'cd ~/rv-ml && git pull --no-edit && \
#       nohup ./slurm/run_cp_rerun.sh > slurm/logs/cp-rerun-nohup.log 2>&1 & \
#       echo started'
#
# Results are committed and pushed to the rhul-results branch at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "run_jobs_direct.sh|regression.py --benchmark|gp_residual_model.py" > /dev/null; do
    echo "$(date): original runner still active — waiting 10 min"
    sleep 600
done

git merge-base --is-ancestor 08eb3293654b7976c6748496d3884f78a95a94fc HEAD || {
    echo "repo predates 08eb329 — git pull first"; exit 1; }
test -f models/gp_residual_svgp.pt || { echo "missing GP checkpoint"; exit 1; }
test -f synthetic_generation/datasets/synthetic_lsp_regression_10000.csv || {
    echo "missing LSP CSV (job 1 step 2 output)"; exit 1; }

source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
mkdir -p slurm/logs
STAMP=$(date +%Y%m%d-%H%M)
LOG="slurm/logs/cp-rerun-$STAMP.log"
echo "$(date): starting CP rerun @ $(git rev-parse --short HEAD) -> $LOG"

{
    echo "=== CP 1/3: default (gamma on synthetic tune set) ==="
    python conformal_shift.py
    echo "=== CP 2/3: gamma on real val ==="
    python conformal_shift.py --gamma-tune-on real-val \
        --out-dir synthetic_generation/regression/gamma_real_val \
        --fig-dir synthetic_generation/figures/synthetic_regression_10000/gamma_real_val
    echo "=== CP 3/3: psi* ablation ==="
    python conformal_shift.py --psi-labels star --gamma-tune-on real-val \
        --out-dir synthetic_generation/regression/psi_star \
        --fig-dir synthetic_generation/figures/synthetic_regression_10000/psi_star
} > "$LOG" 2>&1
rc=$?
echo "$(date): CP rerun exit=$rc"

git add -f \
    synthetic_generation/regression \
    synthetic_generation/figures/synthetic_regression_10000 \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: CP rerun @ $(git rev-parse --short HEAD) ($STAMP, exit=$rc)" || true
git push origin HEAD:refs/heads/rhul-results || \
    echo "WARNING: push failed — fix credentials, then: git push origin HEAD:refs/heads/rhul-results"
echo "$(date): done"
