#!/bin/bash
#SBATCH --job-name=rvml-nicolo-0726
#SBATCH --partition=compute        # <-- ADJUST: RHUL CPU partition name (CPU-only job)
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --account=YOUR_ACCOUNT    # <-- UNCOMMENT + set if the cluster requires an allocation

# -------------------------------------------------------------------------
# Everything from Nicolò's 2026-07-26 reply that needs real compute.
#
# NOTE ON RUNNING THIS: there is no Slurm on the RHUL CIM boxes George can
# reach.  Run it with srun stripped:
#
#     sed "s/^\([[:space:]]*\)srun /\1/" slurm/nicolo_20260726.sbatch | bash
#
# The leading-whitespace group is required: two of the srun calls (steps 1 and
# 4) are indented inside for loops, and a column-0 anchor leaves them in place.
# With no srun on the box that is exit 127, which set -e turns into a dead job
# at the first seed.
#
# The #SBATCH headers are kept so the file still works if a real scheduler
# ever appears.
#
# Steps, in dependency order:
#
#   1. psi retrain.  The checkpoint behind the committed paper table is not in
#      the repo (issue #10) and is not needed: a fresh 200-epoch train of the
#      74-D MLP reproduces its behaviour to well inside the CP intervals
#      (seed-to-seed sd 0.10 dex in log10 P vs a 1.80 dex half-width).  Trained
#      over several seeds so the paper can quote a seed spread instead of one
#      draw's numbers.
#
#   2. Full-scale CP rerun, PKe coordinates (omega dropped as a reported
#      target, per Nicolò), the label-free min-nights real filter, and the
#      eq-(17) papernorm weight tuned rather than fixed at 1:1.  Run four ways:
#      gamma on synthetic vs real-val, and series-level vs one-series-per-host
#      real test sets.
#
#   3. PKew control, so the paper can state exactly what dropping omega bought.
#
#   4. Assumption 3.2 Hessian sweep, both coordinate sets, at 200/1000/4000 GD
#      steps.  This is the convergence sweep Nicolò asked about ("More GD steps
#      bring you to a nearly flat region ... What happens if we ignore omega?").
#      4000 steps x 25 draws is the slow part (~80 s/draw locally).
#
# Prerequisite that is NOT automated: a valid GitHub PAT in the cluster clone's
# .git/config.  Re-verified 2026-07-26; it expires 2026-08-13, after which it
# must be reissued or step 5 cannot push.  (An earlier note here claimed it had
# expired ~2026-07-21 -- that was wrong.)
#
# Step 5 will fail anyway for an unrelated reason: rhul-results already exists
# on GitHub with the 2026-07-25 results and has diverged from main, so the push
# is rejected non-fast-forward.  Non-fatal -- artifacts stay on disk -- but they
# then need pushing to a fresh branch or rsyncing off by hand.
# -------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ---- Preconditions (freshness, not just existence — see handover) ---------
test -f data/splits.csv || { echo "MISSING data/splits.csv (rsync data/ over)"; exit 1; }
test -f synthetic_generation/datasets/synthetic_regression_10000.csv || {
    echo "MISSING synthetic_regression_10000.csv"; exit 1; }
echo "=== inputs ==="
ls -l --time-style=+%F models/gp_residual_svgp.pt \
      synthetic_generation/datasets/synthetic_regression_10000.csv || true
git log --oneline -1

CKPT_DIR=checkpoints/psi_20260726
mkdir -p "$CKPT_DIR"

# ---- 1. psi retrain across seeds -----------------------------------------
# ~65 s each on a CIM ts-node (measured 2026-07-26; the "~15 s" quoted here
# before was wrong).  The spread is what we report, not any single seed.
for SEED in 0 1 2 7 42 123; do
    srun python regression.py --feature-set 74 --epochs 200 --seed "$SEED" \
        --real-split test \
        --out "figures/regression_synthetic/seed_${SEED}" \
        --checkpoint "${CKPT_DIR}/mlp74_s${SEED}.pt"
done
PSI="${CKPT_DIR}/mlp74_s42.pt"
test -f "$PSI" || { echo "psi retrain produced no checkpoint"; exit 1; }

# ---- 2. Full-scale CP, omega dropped (PKe) --------------------------------
# --papernorm-weight omitted on purpose: the eq-(17) weight is tuned on the
# tuning set and reported, which is what Nicolò asked for.
#
# **--psi mlp is mandatory on every call.**  conformal_shift.py defaults --psi
# to "rf", and --checkpoint is READ ONLY when --psi mlp -- argparse accepts it
# either way and the code silently discards it otherwise.  The 2026-07-26 run
# omitted it and spent 9 hours calibrating a RandomForest: exit 0, no warning,
# plausible-looking numbers, wrong predictor.  assert_mlp below checks what each
# run actually RECORDED, so a repeat fails loudly instead of silently.
CP_BASE=synthetic_generation/regression/pke_20260726
FIG_BASE=synthetic_generation/figures/synthetic_regression_10000/pke_20260726

# Fail the job if a completed run did not use the MLP psi.
assert_mlp() {
    local out_dir="$1"
    local m="${out_dir}/conformal_shift_metrics.json"
    test -f "$m" || { echo "FATAL: no metrics written to ${out_dir}"; exit 1; }
    python - "$m" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
psi, ckpt, csv = m.get("psi"), m.get("checkpoint"), str(m.get("csv") or "")
if psi != "mlp" or not ckpt:
    sys.exit(f"FATAL: {sys.argv[1]} recorded psi={psi!r} checkpoint={ckpt!r}; "
             "expected the MLP. Add --psi mlp to this call.")
if "synthetic_regression_10000.csv" not in csv or "lsp" in csv.lower():
    sys.exit(f"FATAL: {sys.argv[1]} recorded csv={csv!r}; the 74-D checkpoint "
             "needs synthetic_regression_10000.csv, not the 512-bin LSP set.")
print(f"  ok: psi={psi} csv={csv.split('/')[-1]}")
PYEOF
}

srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKe \
    --out-dir "${CP_BASE}/gamma_syn" --fig-dir "${FIG_BASE}/gamma_syn"
assert_mlp "${CP_BASE}/gamma_syn"

srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKe \
    --gamma-tune-on real-val \
    --out-dir "${CP_BASE}/gamma_real_val" --fig-dir "${FIG_BASE}/gamma_real_val"
assert_mlp "${CP_BASE}/gamma_real_val"

# One series per star: the corpus stores one row per RV file, so 51 filtered
# test series are only 34 distinct stars (HD 179949 contributes 4).  This
# variant makes the real test points independent across hosts.
srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKe \
    --gamma-tune-on real-val --one-per-host \
    --out-dir "${CP_BASE}/gamma_real_val_1perhost" \
    --fig-dir "${FIG_BASE}/gamma_real_val_1perhost"
assert_mlp "${CP_BASE}/gamma_real_val_1perhost"

# No min-nights filter, to quantify what the filter changed.
srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKe \
    --gamma-tune-on real-val --min-nights 0 \
    --out-dir "${CP_BASE}/gamma_real_val_nofilter" \
    --fig-dir "${FIG_BASE}/gamma_real_val_nofilter"
assert_mlp "${CP_BASE}/gamma_real_val_nofilter"

# Surrogate-gap constant as an explicit conformal quantile rather than the max,
# so Delta_c carries a stated out-of-sample budget (Theorem 3.6 discussion).
srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKe \
    --gamma-tune-on real-val --gap-beta 0.02 \
    --out-dir "${CP_BASE}/gamma_real_val_gapbeta02" \
    --fig-dir "${FIG_BASE}/gamma_real_val_gapbeta02"
assert_mlp "${CP_BASE}/gamma_real_val_gapbeta02"

# ---- 3. PKew control ------------------------------------------------------
srun python conformal_shift.py --psi mlp --csv "$CSV74" --checkpoint "$PSI" --coords PKew \
    --gamma-tune-on real-val \
    --out-dir "synthetic_generation/regression/pkew_20260726" \
    --fig-dir "synthetic_generation/figures/synthetic_regression_10000/pkew_20260726"
assert_mlp "synthetic_generation/regression/pkew_20260726"

# ---- 4. Assumption 3.2 Hessian convergence sweep --------------------------
# ~3 h, and it does NOT depend on psi (assumption_hessian.py never loads a
# predictor -- it works on the decoder loss ||y-h(theta)||^2).  So a rerun that
# only fixes the psi selection can skip it: same code + same --seed gives the
# same numbers.  Set RVML_SKIP_HESSIAN=1 to skip.
if [[ "${RVML_SKIP_HESSIAN:-0}" == "1" ]]; then
    echo "=== step 4 skipped (RVML_SKIP_HESSIAN=1); existing hessian_*.json kept ==="
else
    mkdir -p figures/paper/assumption32
    for C in PKew PKe; do
        srun python -m scripts.diagnostics.assumption_hessian \
            --n-draws 25 --steps 200 1000 4000 --coords "$C" --seed 0 \
            --out "figures/paper/assumption32/hessian_${C}.json"
    done
fi

# ---- 5. Push artifacts to the rhul-results branch -------------------------
# A failed push never fails the job — artifacts stay on disk.
git add -f \
    synthetic_generation/regression/pke_20260726 \
    synthetic_generation/regression/pkew_20260726 \
    synthetic_generation/figures/synthetic_regression_10000/pke_20260726 \
    synthetic_generation/figures/synthetic_regression_10000/pkew_20260726 \
    figures/paper/assumption32 \
    figures/regression_synthetic \
    slurm/logs || true
git -c user.name="George" -c user.email="pulickan06@gmail.com" \
    commit -m "RHUL results: nicolo_20260726 ($(date +%F), job ${SLURM_JOB_ID:-manual})" || true
git push origin HEAD:refs/heads/rhul-results || \
    echo "WARNING: push failed — reissue the PAT, then: git push origin HEAD:refs/heads/rhul-results"

echo "Done."
echo "  CP (PKe, gamma real-val):  ${CP_BASE}/gamma_real_val/conformal_shift_report.txt"
echo "  CP (one series per host):  ${CP_BASE}/gamma_real_val_1perhost/"
echo "  CP (no min-nights filter): ${CP_BASE}/gamma_real_val_nofilter/"
echo "  PKew control:              synthetic_generation/regression/pkew_20260726/"
echo "  Assumption 3.2:            figures/paper/assumption32/hessian_{PKe,PKew}.json"
echo
echo "Sanity bars: papernorm weight w reported in conformal_shift_metrics.json"
echo "  ('papernorm_weight'); joint coverage near 0.90 at alpha=0.1 on PKe;"
echo "  PD fraction should rise from ~0.64 (PKew) to ~0.76 (PKe) at 200 steps."
