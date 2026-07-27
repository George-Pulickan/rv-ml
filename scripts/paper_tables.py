"""Emit the paper's LaTeX tables from a conformal_shift metrics JSON.

Two tables, both generated rather than hand-copied so they cannot drift from the
run that produced them:

  cp_ablation.tex    — strategy x normalisation.  The ``naive`` + ``raw`` row is
                       the baseline: standard split conformal prediction
                       calibrated on the data-generating parameters, with no
                       surrogate labels and no shift correction.  Everything the
                       method adds is measured against that row.
  cp_assumptions.tex — the empirical status of Assumptions 3.1 and 3.2,
                       including the Lemma 3.4 validity gap.

Usage
-----
    python scripts/paper_tables.py --metrics <dir>/conformal_shift_metrics.json
    python scripts/paper_tables.py --metrics ... --out-dir figures/paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRATEGY_LABEL = {
    "naive": r"naive ($\bar\theta$)",
    "naive_adj": r"naive + $\Delta_c$",
    "surrogate": r"surrogate ($\theta^*$)",
}
NORM_LABEL = {
    "raw": "raw",
    "vnorm": r"$v_y$",
    "v2norm": r"$v_y + v_c$",
    "papernorm": r"$\delta_c,\delta_y$",
}
STRATEGIES = ["naive", "naive_adj", "surrogate"]
NORMS = ["raw", "vnorm", "v2norm", "papernorm"]


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, float) and x != x:      # NaN
        return "--"
    return f"{x:.{nd}f}"


def _sci(x: float | None, nd: int = 2) -> str:
    """LaTeX math-mode scientific notation, safe inside a tabular cell."""
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    if x == 0:
        return "$0$"
    if 1e-3 <= abs(x) < 1e4:
        return f"${x:.{nd}f}$"
    from math import floor, log10
    ex = int(floor(log10(abs(x))))
    mant = x / (10.0 ** ex)
    return rf"${mant:.{nd}f}\times 10^{{{ex}}}$"


def ablation_table(d: dict, alpha: str = "0.10") -> str:
    res = d["results"]
    nominal = 1.0 - float(alpha)
    n_real = d.get("n_test_real", "?")
    n_syn = d.get("n_test_syn", "?")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"strategy & norm & synthetic & real & real & vacuous \\",
        r"         &      &           & (unweighted) & (weighted) & (\%) \\",
        r"\midrule",
    ]
    for si, strat in enumerate(STRATEGIES):
        if strat not in res:
            continue
        for ni, norm in enumerate(NORMS):
            if norm not in res[strat]:
                continue
            r = res[strat][norm]
            syn = r["synthetic_unweighted"][alpha]["joint_coverage"]
            ru = r["real_unweighted"][alpha]["joint_coverage"]
            rw = r["real_weighted"][alpha]["joint_coverage"]
            vac = 100.0 * r["real_weighted"][alpha]["frac_infinite"]
            label = STRATEGY_LABEL.get(strat, strat) if ni == 0 else ""
            row = (f"{label} & {NORM_LABEL.get(norm, norm)} & {_fmt(syn)} & "
                   f"{_fmt(ru)} & {_fmt(rw)} & {vac:.1f}")
            if strat == "naive" and norm == "raw":
                row += r"  \quad$\leftarrow$ baseline"
            lines.append(row + r" \\")
        if si < len(STRATEGIES) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Joint coverage at nominal $1-\alpha = {nominal:.2f}$ "
        r"(Bonferroni-corrected across the $d=4$ coordinates). "
        rf"Synthetic test $n={n_syn}$, real test $n={n_real}$. "
        r"The first row is standard split conformal prediction calibrated on the "
        r"data-generating parameters $\bar\theta$, without surrogate labels or "
        r"covariate-shift reweighting. ``vacuous'' is the fraction of real test "
        r"systems whose interval is unbounded.}",
        r"\label{tab:cp-ablation}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def assumptions_table(d: dict) -> str:
    """Rows are ``(label, value)``; ``value is None`` marks a full-width section
    header, which must NOT be followed by an alignment tab or LaTeX raises
    "Extra alignment tab has been changed to \\cr" and the table fails to
    build."""
    nf = d.get("noise_filter", {}) or {}
    ac = d.get("assumption_constants") or {}
    rows: list[tuple[str, str | None]] = []

    # --- Assumption 3.1 -----------------------------------------------------
    rows.append((r"\multicolumn{2}{l}{\textit{Assumption 3.1 (bounded noise)}}", None))
    if nf.get("enabled"):
        rows.append((r"\quad $\varepsilon_{\sup}=\max_t|y_t-h_t|$ (rv\_std)",
                     _fmt(nf.get("bound_rv_std"), 3)))
        if nf.get("eps_loss_rv_std") is not None:
            rows.append((r"\quad $\varepsilon_{\ell}=\max\|y-h\|^2$ (rv\_std)",
                         _fmt(nf.get("eps_loss_rv_std"), 1)))
        rows.append((r"\quad synthetic draws rejected",
                     f"{100.0 * nf.get('rejection_rate', 0.0):.0f}\\%"))
        rows.append((r"\quad draws generated / kept",
                     f"{nf.get('n_generated','?')} / {nf.get('n_kept','?')}"))
    else:
        rows.append((r"\quad filter", "disabled"))

    # --- Assumption 3.2 -----------------------------------------------------
    rows.append(("", None))
    rows.append((r"\multicolumn{2}{l}{\textit{Assumption 3.2 (non-degenerate labels)}}", None))
    if ac:
        lm = ac.get("lambda_min_H", {})
        rows.append((rf"\quad draws ($H^*$ at $\theta^*$, $d={len(ac.get('coords', []))}$)",
                     str(ac.get("n_draws", "?"))))
        rows.append((r"\quad $\lambda_{\min}(H^*)$ median", _sci(lm.get("median"))))
        rows.append((r"\quad $\lambda_{\min}(H^*)$ $p_{10}$ / min",
                     f"{_sci(lm.get('p10'))} / {_sci(lm.get('min'))}"))
        rows.append((r"\quad \textbf{positive definite}",
                     rf"\textbf{{{100.0 * ac.get('frac_positive_definite', 0.0):.0f}\%}}"))
        ch = ac.get("C_H") or {}
        if ch:
            rows.append((r"\quad $C_H = 1/\lambda_{\min}$ median / $p_{90}$",
                         f"{_sci(ch.get('median'))} / {_sci(ch.get('p90'))}"))
        kap = ac.get("kappa_H") or {}
        if kap:
            rows.append((r"\quad $\kappa(H^*)$ median", _sci(kap.get("median"))))
        gap = ac.get("validity_gap_sqrt_CH_eps") or {}
        if gap:
            rows.append((r"\quad \textbf{Lemma 3.4 gap} $\sqrt{C_H\varepsilon_\ell}$ median",
                         rf"\textbf{{{_fmt(gap.get('median'), 2)}}}"))
    else:
        rows.append((r"\quad \textit{not computed for this run}", "--"))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"quantity & value \\",
        r"\midrule",
    ]
    for label, value in rows:
        if not label and value is None:
            lines.append(r"\addlinespace[4pt]")
        elif value is None:                       # full-width section header
            lines.append(label + r" \\")
        else:
            lines.append(f"{label} & {value}" + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Empirical status of the two assumptions. $\varepsilon_{\sup}$ "
        r"drives the Assumption 3.1 discard rule; $\varepsilon_\ell$ is the same "
        r"discrepancy on the scale of $\ell(\theta)=\|y-h(\theta)\|^2$ and is the "
        r"one that pairs dimensionally with $C_H$. $H^*$ is evaluated at the "
        r"surrogate label $\theta^*$ in the four physical coordinates. Note the "
        r"Lemma 3.4 correction exceeds the support of $e$, so the theoretical "
        r"guarantee is uninformative at these constants; reported intervals are "
        r"calibrated by the empirical $\Delta_c$ instead.}",
        r"\label{tab:cp-assumptions}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, required=True,
                    help="conformal_shift_metrics.json from the run to tabulate")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "paper")
    ap.add_argument("--alpha", default="0.10", help="alpha key to tabulate")
    args = ap.parse_args()

    d = json.loads(args.metrics.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, text in (("cp_ablation.tex", ablation_table(d, args.alpha)),
                       ("cp_assumptions.tex", assumptions_table(d))):
        (args.out_dir / name).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out_dir / name}")
        print(text)
        print()


if __name__ == "__main__":
    main()
