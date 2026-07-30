#!/bin/bash
# CP re-run with the likelihood-ratio discriminator at C=0.03 instead of C=1.0.
#
# Why: scripts/check_discriminator_dim.py measures C=1.0 as the wrong operating
# point at d=74 -- out-of-fold log-loss 0.6403 vs 0.6312 at C=0.03, ESS 44.1% vs
# 71.7%, max raw weight 18.7 vs 8.07. Log-loss is a strictly proper score and is
# non-monotone in C (it turns over at 0.03), so this is not just buying ESS by
# under-correcting the shift. What that does to actual coverage and interval
# width has never been measured -- that is this run.
#
# Config is byte-identical to refined_20260729/1perhost_K200M5_uk_thetastar
# except for --weight-c, INCLUDING OMP_NUM_THREADS=8: per Trap 30 the multistart
# winner is a discrete argmin over near-tied losses, so changing thread count
# alone can flip basins and would confound the comparison.
#
# The Kepler tolerance fix (local commit d66c11c) is deliberately NOT deployed
# here, so C is the only variable. The guard below enforces that.
set -uo pipefail
cd "$HOME/rv-ml" || exit 1
LOG="$HOME/rv-ml/slurm/logs/weightc003-$(date +%Y%m%d-%H%M%S).log"
OUT=synthetic_generation/regression/refined_20260730/1perhost_K200M5_uk_C003
BASE=synthetic_generation/regression/refined_20260729/1perhost_K200M5_uk_thetastar
{
echo "=== start $(date) on $(hostname) ==="

B=$(stat -c%s models/gp_residual_svgp.pt)
echo "gp ckpt : $B bytes  [must be 1081541]"
[ "$B" = "1081541" ] || { echo "WRONG GP CHECKPOINT -- aborting (Trap 10)"; exit 1; }

# Keep the A/B clean: the Kepler solver must still be the original 1e-10 form.
if ! grep -q "tol: float = 1e-10" models/kepler_torch.py; then
  echo "ABORT: models/kepler_torch.py is not the original tol=1e-10 version."
  echo "       That would confound the C comparison with the Kepler speedup."
  exit 1
fi
echo "kepler  : original tol=1e-10 (A/B stays clean)"

# --weight-c must actually exist, or argparse would take it as a prefix error
# and this whole run would silently be another C=1.0 job.
.venv/bin/python conformal_shift.py --help 2>&1 | grep -q -- "--weight-c" || {
  echo "ABORT: conformal_shift.py has no --weight-c"; exit 1; }
echo "flag    : --weight-c present"

[ -d "$BASE" ] || echo "WARNING: baseline $BASE missing, comparison will be partial"
mkdir -p "$OUT"

source .venv/bin/activate
export MPLBACKEND=Agg OMP_NUM_THREADS=8

python -u conformal_shift.py --psi mlp \
  --checkpoint checkpoints/psi_20260726/mlp74_s42.pt \
  --csv synthetic_generation/datasets/synthetic_regression_10000.csv \
  --coords PKe --gamma-tune-on real-val --one-per-host \
  --psi-refine 200 --psi-multistart 5 --psi-refit-k --union-regions \
  --weight-c 0.03 \
  --out-dir "$OUT" \
  --fig-dir synthetic_generation/figures/synthetic_regression_10000/refined_20260730/1perhost_K200M5_uk_C003
RC=$?
echo "exit=$RC"
[ "$RC" = "0" ] || { echo "RUN FAILED -- not reporting numbers"; echo "=== end $(date) ==="; exit "$RC"; }

echo
echo "=== C=0.03 vs the C=1.0 baseline ==="
python - "$OUT" "$BASE" <<'PY'
import json, os, statistics as st, sys

def load(p):
    m = json.load(open(os.path.join(p, "conformal_shift_metrics.json")))
    try:
        w = json.load(open(os.path.join(p, "per_system_widths_papernorm.json")))
    except FileNotFoundError:
        w = None
    return m, w

def halfwidths(w, a):
    """The PAPER's half-width: median over systems of per-system `halfwidths`.
    NOT per_coord_median_width, which is ~44% wider on log10_P (Traps 13/26)."""
    if w is None:
        return None
    out = {}
    for c in ("log10_P", "log10_K", "e"):
        v = [s["halfwidths"][a][c] for s in w["systems"] if a in s.get("halfwidths", {})]
        if v:
            out[c] = st.median(v)
    return out

new, wn = load(sys.argv[1])
try:
    old, wo = load(sys.argv[2])
except Exception as e:
    old, wo = None, None
    print(f"(baseline unavailable: {e})")

# Hard assertions -- Trap 11 discipline, plus the one that matters here.
assert new.get("psi") == "mlp", f"psi={new.get('psi')!r} NOT mlp"
assert new.get("checkpoint"), "checkpoint is null"
got_c = new.get("weights", {}).get("C")
assert got_c == 0.03, f"weights.C == {got_c!r}, NOT 0.03 -- the flag did not take"
print(f"ASSERTED psi=mlp  ckpt={new['checkpoint']}  weights.C={got_c}")
print(f"config: refine={new.get('psi_refine')} M={new.get('psi_multistart')} "
      f"refit_k={new.get('psi_refit_k')} n_cal={new.get('n_cal')} "
      f"n_test_real={new.get('n_test_real')}")

def row(label, a, b, fmt="{:.4f}"):
    fa = fmt.format(a) if isinstance(a, float) else str(a)
    fb = fmt.format(b) if isinstance(b, float) else str(b)
    d = ""
    if isinstance(a, float) and isinstance(b, float):
        d = f"   delta {a-b:+.4f}"
    print(f"  {label:38s} C=0.03 {fa:>9s}   C=1.0 {fb:>9s}{d}")

print("\n-- discriminator weights --")
row("ESS", new["weights"]["ess"], old["weights"]["ess"] if old else None, "{:.1f}")
row("frac clipped", new["weights"]["frac_clipped"],
    old["weights"]["frac_clipped"] if old else None)

print("\n-- joint coverage (surrogate/papernorm) --")
rn = new["results"]["surrogate"]["papernorm"]
ro = old["results"]["surrogate"]["papernorm"] if old else {}
for dom in ("synthetic_unweighted", "synthetic_thetastar", "real_weighted"):
    for a in ("0.10", "0.32"):
        if dom in rn:
            row(f"{dom} a={a}", rn[dom][a]["joint_coverage"],
                ro.get(dom, {}).get(a, {}).get("joint_coverage"))

print("\n-- median half-width (per-system export, the paper's definition) --")
for a in ("0.10", "0.32"):
    hn, ho = halfwidths(wn, a), halfwidths(wo, a)
    if hn:
        for c in ("log10_P", "log10_K", "e"):
            if c in hn:
                row(f"a={a} {c}", hn[c], (ho or {}).get(c))

print("\n-- eq (32) gap, synthetic_thetastar --")
for a in ("0.10", "0.32"):
    g = rn.get("synthetic_thetastar", {}).get(a, {}).get("gap")
    gb = (ro.get("synthetic_thetastar", {}) or {}).get(a, {}).get("gap")
    print(f"  a={a}  C=0.03 {g}   C=1.0 {gb}")
PY
echo "=== end $(date) ==="
} > "$LOG" 2>&1
echo "LOG=$LOG"
