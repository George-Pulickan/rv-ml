"""
gp_demo.py — 3-system demo with full kernel comparison and diagnostics.

Targets (with name aliases — searched across host columns):
    51 Peg     (quiet)
    HAT-P-11   (active, rotation-driven noise)
    gam Cep    (long-period post-trend2 residuals)

For each:
  - Fit all 4 kernels (sho, matern32, rotation, sho+matern32)
  - Select best by BIC
  - Compute Cholesky-whitened residuals
  - KS test (normality) + Ljung-Box test (independence)
  - Sample noise, compute time-lag DCF (Edelson & Krolik 1988)
  - Plot 5-row comparison figure

Outputs:
  figures/gp_vs_bootstrap.png
  data/gp_demo_summary.csv
  data/gp_demo_kernel_comparison.csv

Runtime: ~2-5 minutes.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kepler_check import validate_one
from gp_noise_model import fit_all_kernels, time_lag_dcf

FIG_PATH = ROOT / 'figures' / 'gp_vs_bootstrap.png'
SUMMARY_PATH = ROOT / 'data' / 'gp_demo_summary.csv'
KERNEL_TABLE_PATH = ROOT / 'data' / 'gp_demo_kernel_comparison.csv'

# (display_name, list_of_aliases_to_search) — first match wins
TARGETS = [
    ('51 Peg',   ['51 Peg', 'HD 217014', 'HIP 113357']),
    ('HAT-P-11', ['HAT-P-11', 'HIP 97657']),
    ('gam Cep',  ['gam Cep', 'gamma Cep', 'HD 222404', 'HIP 116727', 'Errai']),
]

# Fallback targets if any of the above can't be matched in the corpus.
# Picks well-known systems with different stellar activity / orbit regimes.
FALLBACK_TARGETS = [
    ('55 Cnc',     ['55 Cnc', 'HD 75732', 'HIP 43587']),       # quiet, multi-planet
    ('HD 189733',  ['HD 189733', 'HIP 98505']),                 # active K dwarf
    ('47 UMa',     ['47 UMa', 'HD 95128', 'HIP 53721']),        # long-period multi
    ('HD 209458',  ['HD 209458', 'HIP 108859']),                # transiting hot Jup
    ('Tau Boo',    ['tau Boo', 'HD 120136', 'HIP 67275']),      # active F dwarf
]

KERNELS = ('sho', 'matern32', 'rotation', 'sho+matern32')

VALIDATE_KWARGS = dict(
    mode='fit', auto_sign=True, fit_tperi=True, trend_order=2,
    return_residuals=True, plot=False, verbose=False,
)

# Columns in rv_index.csv to search for host-name matches, in priority order.
HOST_COLUMNS = ['hostname', 'host_in_file', 'hd_name', 'hip_name', 'tic_id']


def _extract(res):
    if isinstance(res, dict):
        if res.get('status', 'ok') != 'ok':
            raise RuntimeError(f"validate_one returned status={res['status']!r}")
        t = next((res[k] for k in ('times', 't') if res.get(k) is not None), None)
        r = next((res[k] for k in ('residuals', 'resid') if res.get(k) is not None), None)
        s = next((res[k] for k in ('sigmas', 'sigma', 'yerr') if res.get(k) is not None), None)
    elif isinstance(res, tuple) and len(res) >= 3:
        t, r, s = res[:3]
    else:
        t = getattr(res, 't', None)
        r = getattr(res, 'residuals', None) or getattr(res, 'resid', None)
        s = getattr(res, 'sigma', None) or getattr(res, 'yerr', None)
    if t is None or r is None or s is None:
        raise RuntimeError("could not extract (t, residuals, sigma) from validate_one return")
    return np.asarray(t, float), np.asarray(r, float), np.asarray(s, float)


def find_target_in_index(idx, aliases):
    """Search across HOST_COLUMNS for any alias; return (matched_rows, column_used, alias_used)."""
    host_cols = [c for c in HOST_COLUMNS if c in idx.columns]
    for alias in aliases:
        for col in host_cols:
            m = idx[idx[col].astype(str).str.contains(alias, case=False, na=False, regex=False)]
            if not m.empty:
                return m, col, alias
    return None, None, None


def get_residuals(idx, labels, aliases, simbad_cache=None):
    rows, col, alias = find_target_in_index(idx, aliases)
    if rows is None:
        return None
    fname = str(rows.iloc[0]['file'])
    matched_value = rows.iloc[0][col]
    print(f"    matched via {col!r} = {matched_value!r}  (alias used: {alias!r})")
    print(f"    file: {fname}")
    path = ROOT / 'data' / 'rv_raw' / fname
    kwargs = dict(VALIDATE_KWARGS)
    if simbad_cache is not None:
        kwargs['simbad_cache'] = simbad_cache
    res = validate_one(str(path), labels, **kwargs)
    t, r, s = _extract(res)
    return t, r, s, fname


def bootstrap_chunk(t, pool, rng):
    """Chunk bootstrap (5-30 contiguous points) matching synthetic_rv.py."""
    n = len(t)
    out = np.zeros(n)
    i = 0
    while i < n:
        hi = min(31, n - i + 1)
        lo = min(5, hi - 1)
        L = int(rng.integers(lo, hi))
        if len(pool) > L:
            j = int(rng.integers(0, len(pool) - L))
            out[i:i+L] = pool[j:j+L]
        else:
            out[i:i+L] = rng.choice(pool, L, replace=True)
        i += L
    return out


def load_pool():
    npz = np.load(ROOT / 'data' / 'noise_pool.npz')
    for k in ('residuals', 'pool', 'noise', 'samples'):
        if k in npz.files:
            return npz[k]
    return npz[npz.files[0]]


def resolve_targets(idx):
    """Return up to 3 (display_name, aliases) pairs that are matchable in idx."""
    resolved = []
    failed = []
    for name, aliases in TARGETS:
        rows, _, _ = find_target_in_index(idx, aliases)
        if rows is not None:
            resolved.append((name, aliases))
        else:
            failed.append(name)
    if failed:
        print(f"  primary targets not found in index: {failed}")
        print(f"  trying fallback targets: {[f[0] for f in FALLBACK_TARGETS]}")
        for name, aliases in FALLBACK_TARGETS:
            if len(resolved) >= 3:
                break
            rows, _, _ = find_target_in_index(idx, aliases)
            if rows is not None and name not in [r[0] for r in resolved]:
                resolved.append((name, aliases))
                print(f"    added fallback {name!r}")
    if len(resolved) < 3:
        # Sample some random multi-observation systems from the corpus
        print(f"  still only {len(resolved)} targets; sampling from corpus")
        # Pick highest-N systems for stability
        for col in HOST_COLUMNS:
            if col in idx.columns:
                top = (idx.groupby(col).size().sort_values(ascending=False)
                        .head(20).index.tolist())
                for name in top:
                    if len(resolved) >= 3:
                        break
                    name_str = str(name)
                    if name_str in [r[0] for r in resolved] or name_str == 'nan':
                        continue
                    resolved.append((name_str, [name_str]))
                    print(f"    added corpus-top {name_str!r}")
                break
    return resolved[:3]


def main():
    idx = pd.read_csv(ROOT / 'data' / 'rv_index.csv')
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv')
    print(f"Loaded rv_index ({len(idx)} rows), labels ({len(labels)} rows)\n")

    simbad_cache_path = ROOT / 'data' / 'simbad_cache.json'
    simbad_cache = {}
    if simbad_cache_path.exists():
        try:
            simbad_cache = json.loads(simbad_cache_path.read_text())
            print(f"Loaded simbad_cache ({len(simbad_cache)} entries)")
        except Exception:
            pass

    print("Resolving targets...")
    targets = resolve_targets(idx)
    print(f"Final targets: {[t[0] for t in targets]}\n")
    if not targets:
        print("No usable targets; aborting.")
        return

    pool = load_pool()
    print(f"Bootstrap pool: N={len(pool)}, std={np.std(pool):.2f}, "
          f"kurt_excess={st.kurtosis(pool):.1f}\n")

    n_cols = len(targets)
    fig, axes = plt.subplots(5, n_cols, figsize=(5*n_cols, 14),
                              gridspec_kw=dict(hspace=0.55, wspace=0.32))
    if n_cols == 1:
        axes = axes.reshape(5, 1)

    summary = []
    all_kernel_tables = []
    rng = np.random.default_rng(42)

    for col, (host, aliases) in enumerate(targets):
        print(f"=== {host} ===")
        try:
            got = get_residuals(idx, labels, aliases, simbad_cache=simbad_cache)
            if got is None:
                raise RuntimeError(f"could not match aliases {aliases}")
            t, r, s, fname = got
        except Exception as e:
            print(f"  skipped: {e}")
            for row in range(5):
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"{host}\nfailed:\n{e}",
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            continue

        print(f"  N={len(t)}  std(resid)={np.std(r):.3f}  kurt(resid)={st.kurtosis(r):.2f}")
        print(f"  fitting {len(KERNELS)} kernels...")

        best_model, all_fits = fit_all_kernels(
            t, r, s, host=host, file=fname,
            kernels=KERNELS, verbose=True,
        )
        best_fr = best_model.fit_result
        print(f"  -> BIC-selected: {best_fr.kernel_name}")

        kt = pd.DataFrame([
            dict(host=host, kernel=k,
                 logL=f.log_likelihood, BIC=f.bic, AIC=f.aic,
                 n_params=f.n_params, convergence_gap=f.convergence_gap,
                 selected=(k == best_fr.kernel_name))
            for k, f in all_fits.items()
        ])
        all_kernel_tables.append(kt)

        try:
            gof = best_model.goodness_of_fit(t, r, s)
        except Exception as e:
            print(f"  WARNING goodness_of_fit failed: {e}")
            gof = None
        if gof is not None:
            print(f"  GoF: KS p={gof.ks_pvalue:.3f}, LB p={gof.lb_pvalue:.3f}, "
                  f"std(z)={gof.whitened_std:.2f}, kurt_ex(z)={gof.whitened_kurt_excess:.2f}")

        diag = best_model.diagnose_samples(t, n_draws=200,
                                            rng=np.random.default_rng(7))
        print(f"  GP samples: std={diag['std']:.2f}, kurt_ex={diag['kurt_excess']:.2f}")

        boot = bootstrap_chunk(t, pool, rng)
        gp_draw = best_model.sample(t, rng=rng)
        z = best_model.whiten(t, r, s) if gof is not None else None

        # --- panels ---
        ax = axes[0, col]
        ax.errorbar(t, r, yerr=s, fmt='.', ms=2, lw=0.4, color='k', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_title(f"{host}  [{best_fr.kernel_name}]  N={len(t)}", fontsize=10)
        if col == 0:
            ax.set_ylabel("residuals (m/s)")

        ax = axes[1, col]
        ax.plot(t, boot, '.', ms=2, color='C0', alpha=0.75)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        if col == 0:
            ax.set_ylabel("bootstrap noise")

        ax = axes[2, col]
        ax.plot(t, gp_draw, '.', ms=2, color='C3', alpha=0.75)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        if col == 0:
            ax.set_ylabel("GP noise")

        ax = axes[3, col]
        lr, dr, er, _ = time_lag_dcf(t, r, n_bins=20)
        lb, db, eb, _ = time_lag_dcf(t, boot, n_bins=20)
        lg, dg, eg, _ = time_lag_dcf(t, gp_draw, n_bins=20)
        if len(lr):
            ax.errorbar(lr, dr, yerr=er, fmt='o-', ms=3, color='k',
                        label='residuals', alpha=0.85, capsize=2, lw=0.8)
        if len(lb):
            ax.errorbar(lb, db, yerr=eb, fmt='s-', ms=3, color='C0',
                        label='bootstrap', alpha=0.75, capsize=2, lw=0.8)
        if len(lg):
            ax.errorbar(lg, dg, yerr=eg, fmt='^-', ms=3, color='C3',
                        label='GP', alpha=0.75, capsize=2, lw=0.8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xscale('log')
        ax.set_ylim(-0.6, 1.05)
        ax.set_xlabel("lag (days)")
        if col == 0:
            ax.set_ylabel("DCF")
            ax.legend(fontsize=8, loc='upper right')

        ax = axes[4, col]
        if z is not None and len(z) > 0:
            lo, hi = min(-4, z.min()), max(4, z.max())
            bins = np.linspace(lo, hi, 30)
            ax.hist(z, bins=bins, density=True, alpha=0.6,
                    color='steelblue', edgecolor='none')
            xg = np.linspace(lo, hi, 200)
            ax.plot(xg, st.norm.pdf(xg), 'k-', lw=1.2, label='N(0,1)')
            ax.set_title(f"KS p={gof.ks_pvalue:.3f}  LB p={gof.lb_pvalue:.3f}", fontsize=9)
        ax.set_xlabel("z (whitened)")
        if col == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=8)

        row = dict(
            host=host, file=fname, n_obs=len(t),
            best_kernel=best_fr.kernel_name,
            best_logL=best_fr.log_likelihood,
            best_BIC=best_fr.bic, best_AIC=best_fr.aic,
            convergence_gap=best_fr.convergence_gap,
            std_resid=float(np.std(r)),
            kurt_excess_resid=float(st.kurtosis(r)),
            gp_samples_kurt_excess=diag['kurt_excess'],
            kurt_excess_pool=float(st.kurtosis(pool)),
        )
        if gof is not None:
            row.update(dict(
                ks_p=gof.ks_pvalue, lb_p=gof.lb_pvalue,
                whitened_std=gof.whitened_std,
                whitened_kurt_excess=gof.whitened_kurt_excess,
            ))
        row.update({f"p_{k}": v for k, v in best_fr.params.items()})
        summary.append(row)

    fig.suptitle("RV residuals vs bootstrap noise vs Gaussian Process noise",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(FIG_PATH, dpi=130, bbox_inches='tight')
    print(f"\nSaved {FIG_PATH}")

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(SUMMARY_PATH, index=False)
        print(f"Saved {SUMMARY_PATH}")
        cols = ['host', 'best_kernel', 'best_BIC', 'best_AIC', 'convergence_gap']
        for extra in ('ks_p', 'lb_p', 'whitened_kurt_excess', 'gp_samples_kurt_excess'):
            if extra in df.columns:
                cols.append(extra)
        print("\nSummary:")
        print(df[cols].to_string(index=False, float_format='%.3f'))

        for kt in all_kernel_tables:
            print(f"\n{kt['host'].iloc[0]}  kernel comparison:")
            print(kt[['kernel', 'logL', 'BIC', 'AIC', 'n_params',
                      'convergence_gap', 'selected']].to_string(
                index=False, float_format='%.2f'))
        if all_kernel_tables:
            combined = pd.concat(all_kernel_tables, ignore_index=True)
            combined.to_csv(KERNEL_TABLE_PATH, index=False)
            print(f"\nSaved {KERNEL_TABLE_PATH}")


if __name__ == '__main__':
    main()
