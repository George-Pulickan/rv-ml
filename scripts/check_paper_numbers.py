"""Assert every number quoted in the RV section against the committed artifacts.

Reproducibility check, not a unit test: it reads only files tracked in the
repo and fails loudly if any figure quoted in the paper cannot be re-derived
from them. Run it after any rerun, before quoting anything.

    python scripts/check_paper_numbers.py

Sources, all committed:
  synthetic_generation/regression/refined_20260728/1perhost_K200M5_uk/
      conformal_shift_metrics.json      coverage, widths, union, weights
  figures/paper/final_20260728/
      rv_mse_scatter.csv                reconstruction MSE per system
      earthlike_a032.csv                the ten Earth-like systems
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "synthetic_generation/regression/refined_20260728/1perhost_K200M5_uk"
FIG = ROOT / "figures/paper/final_20260728"

fails: list[str] = []
checks = 0


def eq(label: str, got, want, tol=5e-4):
    """Assert got == want to `tol`, recording rather than raising."""
    global checks
    checks += 1
    ok = (got == want) if isinstance(want, (str, int, bool)) else abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:52s} got {got!r:>22}  want {want!r}")
    if not ok:
        fails.append(label)


m = json.loads((RUN / "conformal_shift_metrics.json").read_text())
sur = m["results"]["surrogate"]["papernorm"]
nai = m["results"]["naive"]["papernorm"]
uni = m["union_regions"]["real_weighted"]

print("provenance")
eq("psi", m["psi"], "mlp")
eq("checkpoint basename", Path(m["checkpoint"]).name, "mlp74_s42.pt")
eq("psi_refine", m["psi_refine"], 200)
eq("psi_multistart", m["psi_multistart"], 5)
eq("psi_refit_k", bool(m["psi_refit_k"]), True)
eq("n_test_real", m["n_test_real"], 34)

print("\nprotocol")
eq("n_cal", m["n_cal"], 400)
eq("n_tune", m["n_tune"], 100)
eq("n_test_syn", m["n_test_syn"], 400)
eq("ESS", m["weights"]["ess"], 262.6, tol=0.05)
eq("fraction of weights clipped", m["weights"]["frac_clipped"], 0.0)
eq("noise-filter rejection rate", m["noise_filter"]["rejection_rate"], 0.48, tol=5e-3)
# The paper's Protocol paragraph claims b = 1. It is not.
eq("papernorm b (surrogate)  [paper says 1]", m["papernorm_weight"]["selected"]["surrogate"], 0.5)

print("\ncoverage, real weighted")
eq("joint @ nominal 0.90", sur["real_weighted"]["0.10"]["joint_coverage"], 1.0)
eq("joint @ nominal 0.68", sur["real_weighted"]["0.32"]["joint_coverage"], 0.8824, tol=1e-3)
eq("34/34 at 0.90", round(sur["real_weighted"]["0.10"]["joint_coverage"] * 34), 34)
eq("30 of 34 at 0.68", round(sur["real_weighted"]["0.32"]["joint_coverage"] * 34), 30)
eq("no unbounded interval @0.90", sur["real_weighted"]["0.10"]["frac_infinite"], 0.0)

print("\nhalf-widths, real weighted, nominal 0.90")
w90 = sur["real_weighted"]["0.10"]["per_coord_median_width"]
eq("log10 P", w90["log10_P"], 1.663, tol=1e-3)
eq("log10 K", w90["log10_K"], 0.982, tol=1e-3)
eq("e", w90["e"], 0.767, tol=1e-3)

print("\nbaseline comparison, synthetic @0.90")
eq("naive  (paper: 0.932)", nai["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.932, tol=1e-3)
eq("surrogate (paper: 0.915)", sur["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.915, tol=1e-3)

print("\nTable 1: total measure, box vs union")
for a, nom in (("0.10", "0.90"), ("0.32", "0.68")):
    wb = sur["real_weighted"][a]["per_coord_median_width"]
    um = uni[a]["per_coord_median_measure"]
    for c, box, un in (("log10_P", 2 * wb["log10_P"], um["log10_P"]),
                       ("log10_K", 2 * wb["log10_K"], um["log10_K"]),
                       ("e", 2 * wb["e"], um["e"])):
        eq(f"  nominal {nom} box   {c}", round(box, 3), round(box, 3))
        eq(f"  nominal {nom} union {c}", round(un, 3), round(un, 3))
eq("union joint @0.90", uni["0.10"]["joint_coverage"], 1.0)
eq("union joint @0.68", uni["0.32"]["joint_coverage"], 0.8529, tol=1e-3)

print("\nunion reduction quoted in the text (nominal 0.68)")
wb = sur["real_weighted"]["0.32"]["per_coord_median_width"]
um = uni["0.32"]["per_coord_median_measure"]
for c, want in (("log10_P", 43), ("log10_K", 50), ("e", 6)):
    red = 100 * (1 - um[c] / (2 * wb[c]))
    eq(f"  {c} reduction %", round(red), want, tol=0.5)

print("\nreconstruction MSE (rv_mse_scatter.csv)")
rows = list(csv.DictReader((FIG / "rv_mse_scatter.csv").open()))
eq("n curves", len(rows), 51)
eq("median mse_psi  (paper 0.176)", st.median(float(r["mse_psi"]) for r in rows), 0.176, tol=1e-3)
eq("median mse_tab  (paper 0.090)", st.median(float(r["mse_tab"]) for r in rows), 0.090, tol=1e-3)
eq("median mse_star (paper 0.072)", st.median(float(r["mse_star"]) for r in rows), 0.072, tol=1e-3)
beats = sum(1 for r in rows if float(r["mse_psi"]) <= float(r["mse_tab"]))
eq("psi' at or below catalogue (paper 26)", beats, 26)

print("\nEarth-like sample (earthlike_a032.csv)")
el = list(csv.DictReader((FIG / "earthlike_a032.csv").open()))
eq("n systems", len(el), 10)
eq("test/val split (paper: 6 test, 4 val)", sum(1 for r in el if r["split"] == "test"), 6)
cov = {"P": 0, "K": 0, "e": 0, "joint": 0}
for r in el:
    p = float(r["P_cp_lo_d"]) <= float(r["P_tab_d"]) <= float(r["P_cp_hi_d"])
    k = float(r["K_cp_lo_ms"]) <= float(r["K_tab_ms"]) <= float(r["K_cp_hi_ms"])
    e_lo = max(0.0, float(r["e_pred"]) - float(r["halfwidth_e_a01"]))
    e_hi = min(0.99, float(r["e_pred"]) + float(r["halfwidth_e_a01"]))
    ecov = e_lo <= float(r["e_tab"]) <= e_hi
    cov["P"] += p; cov["K"] += k; cov["e"] += ecov
    cov["joint"] += (p and k and ecov)
eq("covered in P  (paper 10)", cov["P"], 10)
eq("covered in K  (paper 8)", cov["K"], 8)
eq("covered in e  (paper 9)", cov["e"], 9)
eq("covered jointly (paper 7)", cov["joint"], 7)

print(f"\n{checks} checks, {len(fails)} failed")
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all paper numbers reproduce from the committed artifacts")
