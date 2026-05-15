"""
cache_residuals.py — run validate_one over the corpus once, save
(t, residuals, sigma) per system to data/residuals.npz.

Outputs:
    data/residuals.npz        — t_<i>, resid_<i>, sigma_<i> per system,
                                plus 'hosts', 'files', 'n_obs', 'rms_over_sigma'
    data/residuals_index.csv  — DataFrame of per-system metadata

Defensive extraction handles dict/tuple/object return shapes from
validate_one. Aborts loudly on validate_one signature mismatch.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from kepler_check import validate_one

ROOT = Path(__file__).parent
INDEX_CSV = ROOT / 'data' / 'rv_index.csv'
LABELS_CSV = ROOT / 'data' / 'labels.csv'
RV_DIR = ROOT / 'data' / 'rv_raw'
OUT_NPZ = ROOT / 'data' / 'residuals.npz'
OUT_CSV = ROOT / 'data' / 'residuals_index.csv'

VALIDATE_KWARGS = dict(
    mode='fit',
    auto_sign=True,
    fit_tperi=True,
    trend_order=2,
    return_residuals=True,
    plot=False,
    verbose=False,
)


def _extract(result):
    """Pull (t, resid, sigma, host, rms_over_sigma) from various return shapes."""
    if result is None:
        return None
    t = resid = sigma = host = rms = None
    if isinstance(result, dict):
        if result.get('status', 'ok') != 'ok':
            return None
        t = next((result[k] for k in ('times', 't') if result.get(k) is not None), None)
        resid = next((result[k] for k in ('residuals', 'resid') if result.get(k) is not None), None)
        sigma = next((result[k] for k in ('sigmas', 'sigma', 'yerr') if result.get(k) is not None), None)
        host = result.get('host')
        rms = next((result[k] for k in ('rms_over_sigma', 'rms_sigma') if result.get(k) is not None), None)
    elif isinstance(result, tuple) and len(result) >= 3:
        t, resid, sigma = result[0], result[1], result[2]
    else:
        t = getattr(result, 't', None)
        resid = next((getattr(result, k, None) for k in ('residuals', 'resid') if getattr(result, k, None) is not None), None)
        sigma = next((getattr(result, k, None) for k in ('sigma', 'yerr') if getattr(result, k, None) is not None), None)
        host = getattr(result, 'host', None)
        rms = getattr(result, 'rms_over_sigma', None)
    if t is None or resid is None or sigma is None:
        return None
    return (np.asarray(t, float), np.asarray(resid, float),
            np.asarray(sigma, float), host, rms)


def main():
    idx = pd.read_csv(INDEX_CSV)
    labels = pd.read_csv(LABELS_CSV)
    print(f"Loaded {INDEX_CSV}: {len(idx)} rows")
    print(f"Loaded {LABELS_CSV}: {len(labels)} rows")

    files = sorted(idx['file'].astype(str).unique())
    print(f"Unique files: {len(files)}")

    simbad_cache_path = ROOT / 'data' / 'simbad_cache.json'
    simbad_cache = {}
    if simbad_cache_path.exists():
        try:
            simbad_cache = json.loads(simbad_cache_path.read_text())
            print(f"Loaded simbad_cache ({len(simbad_cache)} entries)")
        except Exception:
            pass

    arrays = {}
    meta = []
    kept = 0
    failed = 0
    fatal_first = True

    for i, fname in enumerate(files):
        path = RV_DIR / fname
        if not path.exists():
            continue
        try:
            result = validate_one(str(path), labels, simbad_cache=simbad_cache, **VALIDATE_KWARGS)
        except TypeError as e:
            if fatal_first:
                print(f"\n[FATAL] validate_one signature still mismatched.")
                print(f"  error: {e}")
                print(f"  Currently calling: validate_one(path, labels, **{VALIDATE_KWARGS})")
                print(f"  Tell Claude the actual signature so we can adjust.")
                return
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [{i}] {fname}: {type(e).__name__}: {e}")
            continue

        extracted = _extract(result)
        if extracted is None:
            failed += 1
            continue

        t, r, s, host, rms = extracted
        arrays[f't_{kept}'] = t
        arrays[f'resid_{kept}'] = r
        arrays[f'sigma_{kept}'] = s
        meta.append(dict(index=kept, file=fname, host=host,
                         n_obs=len(t), rms_over_sigma=rms))
        kept += 1
        if kept % 50 == 0:
            print(f"  cached {kept} / {len(files)} ...")

    df = pd.DataFrame(meta)
    arrays['hosts'] = df['host'].astype(str).to_numpy()
    arrays['files'] = df['file'].astype(str).to_numpy()
    arrays['n_obs'] = df['n_obs'].to_numpy()
    arrays['rms_over_sigma'] = df['rms_over_sigma'].astype(float).to_numpy()

    np.savez(OUT_NPZ, **arrays)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nDone. {kept} cached, {failed} failed.")
    print(f"  wrote {OUT_NPZ}")
    print(f"  wrote {OUT_CSV}")
    if 'rms_over_sigma' in df.columns:
        good = df['rms_over_sigma'].dropna()
        if len(good):
            print(f"  RMS/sigma:  median={good.median():.2f},  "
                  f"frac<3={float((good<3).mean()):.2%}")


if __name__ == '__main__':
    main()