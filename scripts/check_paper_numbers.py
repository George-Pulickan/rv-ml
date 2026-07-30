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
# !! THE TWO SLICES DISAGREE AND BOTH ARE ASSERTED HERE ON PURPOSE.
# The submitted paper's table prints per_coord_median_width (1.663/0.982/0.767).
# The per-system export -- which Trap 13 calls the paper's definition, and which
# the supplementary uses -- gives 1.100/0.621/0.585 for THIS SAME RUN, 34%
# narrower on log10 P. Until it is settled which the paper means, both are
# pinned so neither can drift silently. See the note printed at the end.
w90 = sur["real_weighted"]["0.10"]["per_coord_median_width"]
eq("log10 P  [metrics slice, as submitted]", w90["log10_P"], 1.663, tol=1e-3)
eq("log10 K  [metrics slice, as submitted]", w90["log10_K"], 0.982, tol=1e-3)
eq("e        [metrics slice, as submitted]", w90["e"], 0.767, tol=1e-3)
_ps = json.loads((RUN / "per_system_widths_papernorm.json").read_text())
_med = lambda a, c: st.median(s["halfwidths"][a][c] for s in _ps["systems"])
eq("log10 P  [per-system export, supplementary]", _med("0.10", "log10_P"), 1.100, tol=1e-3)
eq("log10 K  [per-system export, supplementary]", _med("0.10", "log10_K"), 0.621, tol=1e-3)
eq("e        [per-system export, supplementary]", _med("0.10", "e"), 0.585, tol=1e-3)

print("\nbaseline comparison, synthetic @0.90")
eq("naive  (paper: 0.932)", nai["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.932, tol=1e-3)
eq("surrogate (paper: 0.915)", sur["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.915, tol=1e-3)

print("\nTable 1: total measure, box vs union")
# NB: these compared a value against itself until 2026-07-30 and so asserted
# nothing. The expected values below are the ones the submitted Table 1 prints.
TABLE1 = {
    "0.10": {"box": {"log10_P": 3.325, "log10_K": 1.965, "e": 1.535},
             "union": {"log10_P": 2.527, "log10_K": 1.135, "e": 1.392}},
    "0.32": {"box": {"log10_P": 1.776, "log10_K": 0.985, "e": 1.161},
             "union": {"log10_P": 1.016, "log10_K": 0.491, "e": 1.087}},
}
for a, nom in (("0.10", "0.90"), ("0.32", "0.68")):
    wb = sur["real_weighted"][a]["per_coord_median_width"]
    um = uni[a]["per_coord_median_measure"]
    for c in ("log10_P", "log10_K", "e"):
        eq(f"  nominal {nom} box   {c}", round(2 * wb[c], 3), TABLE1[a]["box"][c], tol=1e-3)
        eq(f"  nominal {nom} union {c}", round(um[c], 3), TABLE1[a]["union"][c], tol=1e-3)
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

# =====================================================================
# SUPPLEMENTARY (paper_supplementary_20260730.tex)
# Everything below asserts the runs the ~5-page appendix is built on,
# which are NOT the runs checked above.
# =====================================================================
REG = ROOT / "synthetic_generation/regression"
BASE = REG / "refined_20260729/1perhost_K200M5_uk_thetastar"
S51 = REG / "refined_20260729/series51"
S57 = REG / "refined_20260729/nofilter"        # 57 series = min-nights filter off
MIS = {f: REG / f"refined_20260729/misspec_f{f}" for f in ("0.25", "0.5", "1.0")}
C003 = REG / "refined_20260730/1perhost_K200M5_uk_C003"

COORDS = ("log10_P", "log10_K", "e")


def metrics(d):
    return json.loads((d / "conformal_shift_metrics.json").read_text())


def halfwidth(d, a, c):
    """Median over systems of the per-system half-width -- the paper's definition."""
    w = json.loads((d / "per_system_widths_papernorm.json").read_text())
    return st.median(s["halfwidths"][a][c] for s in w["systems"])


b = metrics(BASE)
bsur = b["results"]["surrogate"]["papernorm"]

print("\n\n=== SUPPLEMENTARY ===")
print("\nprovenance, 34-host theta* run")
eq("psi", b["psi"], "mlp")
eq("checkpoint basename", Path(b["checkpoint"]).name, "mlp74_s42.pt")
eq("psi_refine", b["psi_refine"], 200)
eq("psi_multistart", b["psi_multistart"], 5)
eq("n_test_real", b["n_test_real"], 34)
eq("n_cal", b["n_cal"], 400)
eq("ESS", b["weights"]["ess"], 262.6, tol=0.05)
eq("papernorm b (surrogate)", b["papernorm_weight"]["selected"]["surrogate"], 0.5)
eq("gamma (surrogate/papernorm)", b["gamma_reg"]["surrogate"]["papernorm"], 0.002858, tol=1e-6)
eq("noise-filter bound (rv_std)", b["noise_filter"]["bound_rv_std"], 5.99, tol=5e-3)
eq("noise-filter rejection rate", b["noise_filter"]["rejection_rate"], 0.48, tol=5e-3)

print("\nTable: coverage across 7 error levels")
COVER = {  # alpha: (syn vs bar-theta, syn vs theta*, real unweighted, real weighted)
    "0.05": (0.9725, 0.960, 1.000, 1.000),
    "0.10": (0.9150, 0.905, 1.000, 1.000),
    "0.15": (0.8950, 0.8825, 1.000, 0.9706),
    "0.20": (0.8325, 0.850, 0.9706, 0.9412),
    "0.30": (0.7475, 0.770, 0.9706, 0.8824),
    "0.32": (0.7375, 0.7625, 0.9412, 0.8824),
    "0.40": (0.6700, 0.710, 0.8824, 0.8529),
}
for a, (sy, ts, ru, rw) in COVER.items():
    eq(f"  a={a} synthetic vs bar-theta", bsur["synthetic_unweighted"][a]["joint_coverage"], sy, tol=1e-3)
    eq(f"  a={a} synthetic vs theta*", bsur["synthetic_thetastar"][a]["joint_coverage"], ts, tol=1e-3)
    eq(f"  a={a} real unweighted", bsur["real_unweighted"][a]["joint_coverage"], ru, tol=1e-3)
    eq(f"  a={a} real weighted", bsur["real_weighted"][a]["joint_coverage"], rw, tol=1e-3)

print("\ntheta* coverage: the first empirical check of the main theorem")
eq("theta* joint @ nominal 0.90 (text: 0.905)", bsur["synthetic_thetastar"]["0.10"]["joint_coverage"], 0.905, tol=1e-3)
eq("bar-theta joint @ nominal 0.90 (text: 0.915)", bsur["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.915, tol=1e-3)
eq("no unbounded interval, real @0.90", bsur["real_weighted"]["0.10"]["frac_infinite"], 0.0)

print("\nTable: eq (32) band mass, scored vs theta*")
GAP = {"0.10": (0.175, 0.920, 0.7625), "0.32": (0.310, 0.8975, 0.8425)}
for a, want in GAP.items():
    g = bsur["synthetic_thetastar"][a]["gap"]
    for c, v in zip(COORDS, want):
        eq(f"  a={a} {c}", g[c], v, tol=1e-3)

print("\nTable: half-widths, 34 hosts (per-system export)")
WID = {"0.10": (1.100, 0.621, 0.585), "0.32": (0.458, 0.272, 0.405), "0.40": (0.389, 0.233, 0.384)}
for a, want in WID.items():
    for c, v in zip(COORDS, want):
        eq(f"  a={a} {c}", halfwidth(BASE, a, c), v, tol=1e-3)
_hw = [s["halfwidths"]["0.10"]["log10_P"] for s in
       json.loads((BASE / "per_system_widths_papernorm.json").read_text())["systems"]]
eq("per-system log10_P spread, ratio (text: 9.3)", max(_hw) / min(_hw), 9.3, tol=0.05)
eq("per-system log10_P spread, min", min(_hw), 0.275, tol=1e-3)
eq("per-system log10_P spread, max", max(_hw), 2.552, tol=1e-3)

print("\nunion regions, reductions quoted in the text")
buni = b["union_regions"]
for a, nom, want in (("0.10", "0.90", (24, 42, 22)), ("0.32", "0.68", (50, 50, 17))):
    wb = bsur["synthetic_unweighted"][a]["per_coord_median_width"]
    um = buni["synthetic_unweighted"][a]["per_coord_median_measure"]
    for c, v in zip(COORDS, want):
        eq(f"  nominal {nom} {c} reduction %", round(100 * (1 - um[c] / (2 * wb[c]))), v, tol=0.5)
eq("union vs theta* @0.90 (text: 0.933 > box 0.905)",
   buni["synthetic_thetastar"]["0.10"]["joint_coverage"], 0.9325, tol=1e-3)
eq("union real weighted @0.90", buni["real_weighted"]["0.10"]["joint_coverage"], 1.0)

print("\nTable: test-set population, 34 / 51 / 57 at matched K=200")
POP = {  # run: (a=0.10 P/K/e, a=0.32 P/K/e, real wtd 0.10, real wtd 0.32, frac_inf)
    "34 hosts": (BASE, (1.100, 0.621, 0.585), (0.458, 0.272, 0.405), 1.000, 0.8824, 0.0),
    "51 series": (S51, (1.211, 0.594, 0.601), (0.583, 0.268, 0.417), 0.9412, 0.8431, 0.0),
    "57 series": (S57, (1.284, 0.633, 0.639), (0.658, 0.278, 0.438), 0.9298, 0.8070, 0.0175),
}
for name, (d, w10, w32, r10, r32, inf) in POP.items():
    for a, want in (("0.10", w10), ("0.32", w32)):
        for c, v in zip(COORDS, want):
            eq(f"  {name} a={a} {c}", halfwidth(d, a, c), v, tol=1e-3)
    s = metrics(d)["results"]["surrogate"]["papernorm"]["real_weighted"]
    eq(f"  {name} real weighted @0.90", s["0.10"]["joint_coverage"], r10, tol=1e-3)
    eq(f"  {name} real weighted @0.68", s["0.32"]["joint_coverage"], r32, tol=1e-3)
    eq(f"  {name} unbounded fraction", s["0.10"]["frac_infinite"], inf, tol=1e-3)

print("\nTable: misspecification, scored vs bar-theta and vs theta*")
MISSPEC = {  # f: (bar 0.10, star 0.10, bar 0.32, star 0.32)
    "0.25": (0.8975, 0.9025, 0.7050, 0.7500),
    "0.5": (0.8625, 0.8975, 0.6450, 0.7225),
    "1.0": (0.7650, 0.8075, 0.4475, 0.5525),
}
for f, (b10, s10, b32, s32) in MISSPEC.items():
    r = metrics(MIS[f])["results"]["surrogate"]["papernorm"]
    eq(f"  f={f} vs bar-theta @0.90", r["synthetic_unweighted"]["0.10"]["joint_coverage"], b10, tol=1e-3)
    eq(f"  f={f} vs theta*    @0.90", r["synthetic_thetastar"]["0.10"]["joint_coverage"], s10, tol=1e-3)
    eq(f"  f={f} vs bar-theta @0.68", r["synthetic_unweighted"]["0.32"]["joint_coverage"], b32, tol=1e-3)
    eq(f"  f={f} vs theta*    @0.68", r["synthetic_thetastar"]["0.32"]["joint_coverage"], s32, tol=1e-3)
    eq(f"  f={f} real weighted @0.90 (text: 1.000)", r["real_weighted"]["0.10"]["joint_coverage"], 1.0)
# the text claims the theta*-minus-bar-theta margin widens with misspecification
_marg = [MISSPEC[f][1] - MISSPEC[f][0] for f in ("0.25", "0.5", "1.0")]
eq("theta* margin widens @0.90", _marg == sorted(_marg), True)
# and that the three arms were not re-tuned: identical widths
for a in ("0.10", "0.32"):
    for c in COORDS:
        v = {halfwidth(MIS[f], a, c) for f in MIS}
        eq(f"  widths identical across misspec arms, a={a} {c}", len(v), 1)
# the misspec arms are NOT comparable to the clean run as an f=0 column
_q = lambda d: json.loads((d / "per_system_widths_papernorm.json").read_text())["q_normalized"]
eq("misspec quantile differs from clean (text: 1.897 vs 2.121)",
   round(_q(MIS["0.25"])["0.10"]["log10_P"], 3), 1.897, tol=1e-3)
eq("clean quantile", round(_q(BASE)["0.10"]["log10_P"], 3), 2.121, tol=1e-3)

print("\nTable: C=0.03 discriminator re-run (job 497)")
c = metrics(C003)
csur = c["results"]["surrogate"]["papernorm"]
eq("weights.C recorded", c["weights"]["C"], 0.03)
eq("ESS (text: 336.7)", c["weights"]["ess"], 336.7, tol=0.05)
eq("weights clipped", c["weights"]["frac_clipped"], 0.0)
eq("real joint @0.90 unchanged", csur["real_weighted"]["0.10"]["joint_coverage"], 1.0)
eq("real joint @0.68 improves (text: 0.912)", csur["real_weighted"]["0.32"]["joint_coverage"], 0.9118, tol=1e-3)
eq("synthetic vs theta* @0.90 unchanged", csur["synthetic_thetastar"]["0.10"]["joint_coverage"], 0.905, tol=1e-3)
eq("synthetic vs bar-theta @0.90 unchanged", csur["synthetic_unweighted"]["0.10"]["joint_coverage"], 0.915, tol=1e-3)
eq("half-width a=0.10 log10_P (text: 1.160)", halfwidth(C003, "0.10", "log10_P"), 1.160, tol=1e-3)
eq("half-width a=0.32 log10_P (text: 0.614)", halfwidth(C003, "0.32", "log10_P"), 0.614, tol=1e-3)
eq("half-width a=0.32 e (text: 0.420)", halfwidth(C003, "0.32", "e"), 0.420, tol=1e-3)
eq("band mass identical to baseline @0.10",
   csur["synthetic_thetastar"]["0.10"]["gap"] == bsur["synthetic_thetastar"]["0.10"]["gap"], True)

print("\nTable: computational cost (timing.json)")
t = json.loads((ROOT / "figures/paper/timing.json").read_text())
eq("n systems", t["n_systems"], 51)
eq("threads", t["threads"], 1)
eq("featurisation ms", t["per_system_ms"]["featurisation"], 0.018, tol=5e-4)
eq("psi forward ms", t["per_system_ms"]["psi_forward"], 0.028, tol=5e-4)
eq("psi total ms (text: 0.046)", t["per_system_ms"]["psi_total"], 0.046, tol=5e-4)
eq("psi' refinement ms (text: 55913)", t["per_system_ms"]["psi_prime_refine"], 55913.0, tol=1.0)
eq("theta* GD ms (text: 9903)", t["per_system_ms"]["theta_star_gd"], 9903.0, tol=1.0)
eq("theta* cheaper than psi' by (text: 5.6x)", 1 / t["ratios"]["theta_star_over_psi_prime"], 5.6, tol=0.05)

print("\nTable: discriminator diagnostics (discriminator_dim.json)")
dd = json.loads((ROOT / "figures/paper/discriminator_dim.json").read_text())
DIM = {"10": (0.558, 0.6545, 0.008, 0.919, 2.06), "40": (0.558, 0.6546, 0.008, 0.918, 2.07),
       "74": (0.648, 0.6403, 0.029, 0.441, 18.7)}
for d_, (auc, ll, sk, ess, wm) in DIM.items():
    r = dd["by_dim"][d_]
    eq(f"  d={d_} AUC out-of-fold", r["auc_out_of_fold"], auc, tol=1e-3)
    eq(f"  d={d_} log-loss out-of-fold", r["logloss_out_of_fold"], ll, tol=1e-3)
    eq(f"  d={d_} skill", r["logloss_skill"], sk, tol=1e-3)
    eq(f"  d={d_} ESS fraction", r["ess_frac"], ess, tol=1e-3)
    eq(f"  d={d_} max weight", r["w_max_raw"], wm, tol=0.05)
CS = {"0.01": (0.640, 0.6371, 0.034, 0.875, 3.75), "0.03": (0.652, 0.6312, 0.043, 0.717, 8.07),
      "0.1": (0.653, 0.6332, 0.040, 0.542, 14.2), "1.0": (0.648, 0.6403, 0.029, 0.441, 18.7)}
for cc, (auc, ll, sk, ess, wm) in CS.items():
    r = dd["by_C_at_full_dim"][cc]
    eq(f"  C={cc} AUC out-of-fold", r["auc_out_of_fold"], auc, tol=1e-3)
    eq(f"  C={cc} log-loss out-of-fold", r["logloss_out_of_fold"], ll, tol=1e-3)
    eq(f"  C={cc} skill", r["logloss_skill"], sk, tol=1e-3)
    eq(f"  C={cc} ESS fraction", r["ess_frac"], ess, tol=1e-3)
    eq(f"  C={cc} max weight", r["w_max_raw"], wm, tol=0.05)
# the text's claim: C=0.03 beats C=1.0 on every axis at once
_a, _z = dd["by_C_at_full_dim"]["0.03"], dd["by_C_at_full_dim"]["1.0"]
eq("C=0.03 better log-loss", _a["logloss_out_of_fold"] < _z["logloss_out_of_fold"], True)
eq("C=0.03 better AUC", _a["auc_out_of_fold"] > _z["auc_out_of_fold"], True)
eq("C=0.03 better ESS", _a["ess_frac"] > _z["ess_frac"], True)
eq("C=0.03 smaller max weight", _a["w_max_raw"] < _z["w_max_raw"], True)

print("\ntheta* optimality (theta_star_optimality.json)")
to = json.loads((ROOT / "figures/paper/theta_star_optimality.json").read_text())
eq("diagnostic ran at K=50, not 200 (footnoted)", to["refine"], 50)
eq("synthetic: psi' beats theta* on % (text: 23)", 100 * to["domains"]["synthetic"]["frac_psi_better_than_star"], 23, tol=0.5)
eq("synthetic median loss ratio (text: 1.16)", to["domains"]["synthetic"]["median_loss_ratio"], 1.16, tol=5e-3)
eq("real: psi' beats theta* on % (text: 7.8)", 100 * to["domains"]["real"]["frac_psi_better_than_star"], 7.8, tol=0.05)
eq("real median loss ratio (text: 1.59)", to["domains"]["real"]["median_loss_ratio"], 1.59, tol=5e-3)
eq("real median |dlog10P| when better (text: 0.006)", to["domains"]["real"]["when_better"]["median_abs_dlog10P"], 0.006, tol=5e-4)
eq("real max |dlog10P| when better (text: 0.069)", to["domains"]["real"]["when_better"]["max_abs_dlog10P"], 0.069, tol=5e-4)

print("\nassumption constants")
ac = b["assumption_constants"]
eq("n draws", ac["n_draws"], 25)
eq("kappa(H) median (text: 1.62e4)", ac["kappa_H"]["median"], 1.62e4, tol=1e2)
eq("kappa(H) p90 (text: 6.31e6)", ac["kappa_H"]["p90"], 6.31e6, tol=1e4)
eq("kappa(H) max (text: 8.80e7)", ac["kappa_H"]["max"], 8.80e7, tol=1e5)
eq("||grad h|| median (text: 172)", ac["grad_h_spectral_norm"]["median"], 172.0, tol=0.5)

print("\nMSE, the supplementary's phrasing")
eq("theta* at or below catalogue on all 51",
   sum(1 for r in rows if float(r["mse_star"]) <= float(r["mse_tab"])), 51)
eq("psi' at or below catalogue, % (text: 51)", round(100 * beats / len(rows)), 51)

print("\nfigures the supplementary includes")
for fig in ("conformal_shift_coverage.png", "conformal_shift_widths.png",
            "rv_mse_scatter.png", "rv_region_boxes_068.png",
            "rv_trajectories.png", "delta_s_gap_distribution.png"):
    eq(f"  figures/supplementary/{fig}", (ROOT / "figures/supplementary" / fig).exists(), True)

print(f"\n{checks} checks, {len(fails)} failed")
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all paper numbers reproduce from the committed artifacts")
print("\nWARNING -- UNRESOLVED: the submitted table's half-widths (1.663/0.982/0.767) are")
print("    per_coord_median_width; the supplementary's (1.100/0.621/0.585) are the")
print("    per-system export, for the SAME run. Both are pinned above. Decide which")
print("    the paper means before the supplementary goes out.")
