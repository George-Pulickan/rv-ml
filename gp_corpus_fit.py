"""
gp_corpus_fit.py — corpus-wide GP fit with kernel selection + GoF.

For each system in the quality-filtered cohort:
  1. Fit each of (sho, matern32, rotation, sho+matern32)
  2. Select by BIC
  3. Compute KS + Ljung-Box goodness-of-fit on best kernel
  4. Save all per-kernel fits and GoF stats

Inputs:
    data/residuals.npz, data/residuals_index.csv  (from cache_residuals.py)

Outputs:
    data/gp_fits.json          — full per-system records (all kernels + GoF)
    data/gp_fits_summary.csv   — flat table (best kernel per system)
    figures/gp_hyperparams.png — 6-panel diagnostic figure

Runtime: ~2-4 hours for ~500 systems.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from gp_noise_model import fit_all_kernels

ROOT = Path(__file__).parent
RESID_NPZ = ROOT / 'data' / 'residuals.npz'
RESID_CSV = ROOT / 'data' / 'residuals_index.csv'
OUT_JSON = ROOT / 'data' / 'gp_fits.json'
OUT_CSV = ROOT / 'data' / 'gp_fits_summary.csv'
OUT_FIG = ROOT / 'figures' / 'gp_hyperparams.png'

KERNELS = ('sho', 'matern32', 'rotation', 'sho+matern32')
RMS_OVER_SIGMA_MAX = 3.0
MIN_OBS = 15


def main():
    if not RESID_NPZ.exists():
        print(f"Missing {RESID_NPZ}. Run cache_residuals.py first.")
        return

    data = np.load(RESID_NPZ, allow_pickle=True)
    idx = pd.read_csv(RESID_CSV)
    print(f"Loaded {len(idx)} cached systems.")

    if 'rms_over_sigma' in idx.columns and idx['rms_over_sigma'].notna().any():
        keep = (idx['rms_over_sigma'] < RMS_OVER_SIGMA_MAX) | idx['rms_over_sigma'].isna()
        idx = idx[keep]
        print(f"  RMS/sigma < {RMS_OVER_SIGMA_MAX}: {len(idx)}")
    idx = idx[idx['n_obs'] >= MIN_OBS]
    print(f"  n_obs >= {MIN_OBS}: {len(idx)}")

    records = []
    failures = 0
    t0 = time()

    for j, (_, row) in enumerate(idx.iterrows()):
        i = int(row['index'])
        host = str(row.get('host'))
        fname = str(row.get('file'))
        try:
            t = data[f't_{i}']
            r = data[f'resid_{i}']
            s = data[f'sigma_{i}']
        except KeyError:
            continue

        try:
            best_model, all_fits = fit_all_kernels(
                t, r, s, host=host, file=fname,
                kernels=KERNELS, verbose=False,
            )
        except Exception:
            failures += 1
            continue

        try:
            gof = best_model.goodness_of_fit(t, r, s).to_dict()
        except Exception:
            gof = dict(ks_statistic=float('nan'), ks_pvalue=float('nan'),
                       lb_statistic=float('nan'), lb_pvalue=float('nan'),
                       lb_df=0, whitened_std=float('nan'),
                       whitened_kurt_excess=float('nan'), n=int(len(t)))

        records.append(dict(
            host=host, file=fname, n_obs=int(len(t)),
            best_kernel=best_model.fit_result.kernel_name,
            selection_method='BIC',
            all_fits={k: f.to_dict() for k, f in all_fits.items()},
            goodness_of_fit=gof,
        ))

        if (j + 1) % 25 == 0:
            elapsed = time() - t0
            rate = (j + 1) / elapsed
            eta = (len(idx) - j - 1) / max(rate, 1e-6)
            print(f"  {j+1}/{len(idx)}  ({rate:.2f}/s, ETA {eta/60:.1f} min)")

    with open(OUT_JSON, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nFit {len(records)} systems ({failures} failed) "
          f"in {(time()-t0)/60:.1f} min")
    print(f"  wrote {OUT_JSON}")

    if not records:
        return

    # Flat summary
    rows = []
    for r in records:
        best = r['best_kernel']
        bf = r['all_fits'][best]
        row = dict(
            host=r['host'], file=r['file'], n_obs=r['n_obs'],
            best_kernel=best,
            logL=bf['log_likelihood'], BIC=bf['bic'], AIC=bf['aic'],
            n_params=bf['n_params'],
            convergence_gap=bf['convergence_gap'],
            **r['goodness_of_fit'],
        )
        for pname, pv in bf['params'].items():
            row[f"p_{pname}"] = pv
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV}")

    # --- Figure ---
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0): SHO hyperparameter scatter
    sho_df = df[df['best_kernel'] == 'sho']
    ax = axes[0, 0]
    if 'p_log_sigma' in sho_df.columns and len(sho_df):
        sc = ax.scatter(sho_df['p_log_sigma'], sho_df['p_log_rho'],
                         c=sho_df['p_log_Q'], cmap='viridis',
                         s=20, alpha=0.75, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=r'$\log Q$')
    ax.set_xlabel(r'$\log\,\sigma_{\rm GP}$')
    ax.set_ylabel(r'$\log\,\rho_{\rm GP}$ (days)')
    ax.set_title(f'SHO-selected ({len(sho_df)} systems)')
    ax.grid(True, alpha=0.3)

    # (0,1): kernel selection bar chart
    ax = axes[0, 1]
    counts = df['best_kernel'].value_counts()
    counts.plot.bar(ax=ax, color='steelblue')
    ax.set_ylabel('count')
    ax.set_title(f'BIC-selected kernel ({len(df)} systems)')
    ax.tick_params(axis='x', rotation=15)

    # (0,2): GoF p-values
    ax = axes[0, 2]
    ks = df['ks_pvalue'].dropna()
    lb = df['lb_pvalue'].dropna()
    if len(ks):
        ax.hist(ks, bins=20, alpha=0.55, label='KS (normality)', color='C0')
    if len(lb):
        ax.hist(lb, bins=20, alpha=0.55, label='Ljung-Box (independence)', color='C3')
    ax.axvline(0.05, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('p-value (whitened residuals)')
    ax.set_ylabel('count')
    ax.set_title('Goodness-of-fit on best kernel')
    ax.legend(fontsize=9)

    # (1,0): primary log_sigma histogram across kernels
    ax = axes[1, 0]
    log_sigmas = []
    for r in records:
        bf = r['all_fits'][r['best_kernel']]
        for pname, pv in bf['params'].items():
            if pname == 'log_sigma' or pname.endswith('.log_sigma'):
                log_sigmas.append(pv); break
    if log_sigmas:
        ax.hist(log_sigmas, bins=40, alpha=0.75, color='steelblue')
        ax.axvline(np.median(log_sigmas), color='k', ls='--', lw=1,
                   label=f"median={np.median(log_sigmas):.2f}")
        ax.legend()
    ax.set_xlabel(r'$\log\,\sigma_{\rm GP}$ (primary)')
    ax.set_ylabel('count')
    ax.set_title('GP amplitude')

    # (1,1): primary timescale histogram
    ax = axes[1, 1]
    log_rhos = []
    for r in records:
        bf = r['all_fits'][r['best_kernel']]
        for pname, pv in bf['params'].items():
            if 'log_rho' in pname or 'log_period' in pname:
                log_rhos.append(pv); break
    if log_rhos:
        ax.hist(log_rhos, bins=40, alpha=0.75, color='C3')
        ax.axvline(np.median(log_rhos), color='k', ls='--', lw=1,
                   label=f"median={np.median(log_rhos):.2f}")
        ax.legend()
    ax.set_xlabel(r'$\log\,\rho_{\rm GP}$ or $\log\,P_{\rm rot}$ (days)')
    ax.set_ylabel('count')
    ax.set_title('GP timescale')

    # (1,2): fitted jitter
    ax = axes[1, 2]
    log_jit = []
    for r in records:
        bf = r['all_fits'][r['best_kernel']]
        if 'log_jitter' in bf['params']:
            log_jit.append(bf['params']['log_jitter'])
    if log_jit:
        ax.hist(log_jit, bins=40, alpha=0.75, color='C2')
        ax.axvline(np.median(log_jit), color='k', ls='--', lw=1,
                   label=f"median={np.median(log_jit):.2f}")
        ax.legend()
    ax.set_xlabel(r'$\log\,\sigma_{\rm jit}$ (m/s)')
    ax.set_ylabel('count')
    ax.set_title('Fitted white jitter')

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130)
    print(f"  wrote {OUT_FIG}")

    print(f"\n=== Corpus summary ===")
    print(f"Kernel selection: {dict(counts)}")
    if len(ks):
        print(f"KS p > 0.05 (normality):   {(ks > 0.05).mean():.1%}")
    if len(lb):
        print(f"LB p > 0.05 (independence): {(lb > 0.05).mean():.1%}")
    print(f"convergence_gap > 1.0 nat: {(df['convergence_gap'] > 1.0).mean():.1%}")


if __name__ == '__main__':
    main()
