"""
gp_sensitivity.py — threshold sensitivity analysis for the corpus GP fit.

Re-applies different cohort filters to data/gp_fits.json (from
gp_corpus_fit.py) and reports stability of headline statistics across
threshold choices. No re-fitting required.

For each (rms_max, min_obs) combination, reports:
  - Cohort size N
  - Median primary log_sigma and log_rho
  - KS / Ljung-Box pass fractions
  - Kernel selection fractions

Outputs:
  data/gp_sensitivity.csv
  figures/gp_sensitivity.png

Run after gp_corpus_fit.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GP_FITS = ROOT / 'data' / 'gp_fits.json'
RESID_IDX = ROOT / 'data' / 'residuals_index.csv'
OUT_CSV = ROOT / 'data' / 'gp_sensitivity.csv'
OUT_FIG = ROOT / 'figures' / 'gp_sensitivity.png'

THRESHOLDS = [
    dict(rms_max=2.0,    min_obs=10),
    dict(rms_max=2.0,    min_obs=25),
    dict(rms_max=3.0,    min_obs=10),
    dict(rms_max=3.0,    min_obs=15),
    dict(rms_max=3.0,    min_obs=25),
    dict(rms_max=5.0,    min_obs=10),
    dict(rms_max=5.0,    min_obs=25),
    dict(rms_max=np.inf, min_obs=15),
]


def _label(t):
    rm = "inf" if not np.isfinite(t['rms_max']) else f"{t['rms_max']:.0f}"
    return f"R<{rm}, N>={t['min_obs']}"


def main():
    if not GP_FITS.exists():
        print(f"Missing {GP_FITS}. Run gp_corpus_fit.py first.")
        return

    with open(GP_FITS) as f:
        records = json.load(f)

    rows = []
    for r in records:
        best = r['best_kernel']
        bf = r['all_fits'][best]
        row = dict(
            host=r['host'], file=r['file'], n_obs=r['n_obs'],
            best_kernel=best,
            BIC=bf['bic'],
            ks_p=r['goodness_of_fit']['ks_pvalue'],
            lb_p=r['goodness_of_fit']['lb_pvalue'],
        )
        # Primary amplitude and timescale
        s_val = r_val = None
        for pname, pv in bf['params'].items():
            if s_val is None and (pname == 'log_sigma' or pname.endswith('.log_sigma')):
                s_val = pv
            if r_val is None and ('log_rho' in pname or 'log_period' in pname):
                r_val = pv
        row['log_sigma_primary'] = s_val
        row['log_rho_primary'] = r_val
        rows.append(row)
    df = pd.DataFrame(rows)

    if RESID_IDX.exists():
        ridx = pd.read_csv(RESID_IDX)
        if 'rms_over_sigma' in ridx.columns:
            df = df.merge(ridx[['file', 'rms_over_sigma']], on='file', how='left')
    if 'rms_over_sigma' not in df.columns:
        df['rms_over_sigma'] = np.nan

    summary = []
    for thr in THRESHOLDS:
        rms_max, min_obs = thr['rms_max'], thr['min_obs']
        if np.isfinite(rms_max):
            mask = ((df['rms_over_sigma'] < rms_max) | df['rms_over_sigma'].isna()) \
                    & (df['n_obs'] >= min_obs)
        else:
            mask = df['n_obs'] >= min_obs
        sub = df[mask]
        if len(sub) == 0:
            continue

        kernel_fracs = sub['best_kernel'].value_counts(normalize=True).to_dict()
        ks = sub['ks_p'].dropna()
        lb = sub['lb_p'].dropna()

        row = dict(
            rms_max=rms_max, min_obs=min_obs,
            N=len(sub),
            median_log_sigma=sub['log_sigma_primary'].median(),
            median_log_rho=sub['log_rho_primary'].median(),
            ks_pass_frac=float((ks > 0.05).mean()) if len(ks) else float('nan'),
            lb_pass_frac=float((lb > 0.05).mean()) if len(lb) else float('nan'),
        )
        for k, v in kernel_fracs.items():
            row[f'frac_{k}'] = v
        summary.append(row)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")
    print(sdf.to_string(index=False, float_format='%.3f'))

    # --- Figure ---
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    labels = [_label({'rms_max': r, 'min_obs': n})
              for r, n in zip(sdf['rms_max'], sdf['min_obs'])]
    xi = np.arange(len(sdf))

    ax = axes[0, 0]
    ax.bar(xi, sdf['N'], color='steelblue')
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax.set_ylabel("# systems")
    ax.set_title("Cohort size by threshold")

    ax = axes[0, 1]
    ax.plot(xi, sdf['median_log_sigma'], 'o-', label=r'median $\log\sigma$')
    ax.plot(xi, sdf['median_log_rho'], 's-', label=r'median $\log\rho$')
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax.legend()
    ax.set_title("Hyperparameter median stability")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if 'ks_pass_frac' in sdf:
        ax.plot(xi, sdf['ks_pass_frac'], 'o-', label='KS p>0.05 (normality)')
    if 'lb_pass_frac' in sdf:
        ax.plot(xi, sdf['lb_pass_frac'], 's-', label='LB p>0.05 (independence)')
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.legend()
    ax.set_title("GoF pass fraction (best kernel)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    kernel_cols = [c for c in sdf.columns if c.startswith('frac_')]
    bottom = np.zeros(len(sdf))
    colors = ['steelblue', 'C3', 'C2', 'C4', 'C5']
    for i, c in enumerate(kernel_cols):
        vals = sdf[c].fillna(0).values
        ax.bar(xi, vals, bottom=bottom, label=c.replace('frac_', ''),
                color=colors[i % len(colors)])
        bottom = bottom + vals
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title("Kernel selection by BIC")

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130)
    print(f"Saved {OUT_FIG}")


if __name__ == '__main__':
    main()
