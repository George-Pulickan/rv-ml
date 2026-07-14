#!/bin/bash
# One-shot campus follow-up for the RHUL CIM box, to run AFTER git has been
# updated to >= 357632c (the launcher one-liner below handles waiting for the
# original runner and merging origin/main first — see handover.md):
#
#   ssh VPAC005@linux.cim.rhul.ac.uk 'cd ~/rv-ml && git fetch origin && \
#     nohup bash -c "while pgrep -f run_jobs_direct.sh >/dev/null; do sleep 600; done; \
#       git merge --no-edit -X ours origin/main && ./slurm/run_campus_followup.sh" \
#     > slurm/logs/followup-nohup.log 2>&1 & echo started'
#
# (-X ours resolves the binary PNG conflict between the job's artifact commits
# and PR #5's refreshed figures; the reruns regenerate those PNGs anyway.)
#
# Runs, sequentially, each pushing its results to rhul-results:
#   1. slurm/run_cp_rerun.sh — the three conformal_shift invocations at the
#      pointwise-delta code (default / gamma_real_val / psi_star)
#   2. regression_benchmark.sbatch (srun-stripped) — Daksh's PR #5 version:
#      period-tolerance, gates, predicted-fold + oracle two-step, diagnostics,
#      e-head ablate
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

git merge-base --is-ancestor 357632c1 HEAD 2>/dev/null || \
    git merge-base --is-ancestor 08eb3293654b7976c6748496d3884f78a95a94fc HEAD || {
    echo "repo predates the follow-up code — merge origin/main first"; exit 1; }

echo "$(date): follow-up 1/2 — CP rerun"
./slurm/run_cp_rerun.sh
echo "$(date): follow-up 1/2 exit=$?"

echo "$(date): follow-up 2/2 — regression benchmark (PR #5 version)"
STAMP=$(date +%Y%m%d-%H%M)
sed "s/^srun //" slurm/regression_benchmark.sbatch | bash \
    > "slurm/logs/reg-bench-rerun-$STAMP.log" 2>&1
echo "$(date): follow-up 2/2 exit=$?"
echo "$(date): campus follow-up done"
