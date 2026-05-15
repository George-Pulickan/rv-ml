"""
parse_and_label.py
------------------
Two utilities to go from raw .tbl files to ML-ready (X, y) records.

(1) `parse_tbl(path)` reads one IPAC ASCII RV table and returns
    (meta: dict, time: ndarray, rv: ndarray, rv_err: ndarray).

(2) `fetch_labels()` issues a single TAP query against the NASA Exoplanet
    Archive's `ps` table (Planetary Systems) and returns a pandas DataFrame of
    Kepler-like orbital + physical parameters for every confirmed exoplanet
    discovered via Radial Velocity.

The labels are the prediction targets: orbital period (pl_orbper), minimum
mass M sin i (pl_bmassj / pl_msinij), eccentricity (pl_orbeccen),
semi-major axis (pl_orbsmax), inclination (pl_orbincl), stellar mass / radius
/ Teff. We keep the published uncertainties (`*err1`, `*err2`) so we can
compare them directly to the Bayesian intervals later in Task 4.

Usage
-----
    python parse_and_label.py --rv-dir data/rv_raw --out data/labels.csv
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns we want for each confirmed RV planet. See the data dictionary:
#   https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
LABEL_COLS = [
    # identifiers
    "pl_name", "hostname", "hd_name", "hip_name", "tic_id",
    "ra", "dec", "sy_dist",
    # planet orbital parameters (our prediction targets)
    "pl_orbper",   "pl_orbpererr1",   "pl_orbpererr2",
    "pl_orbsmax",  "pl_orbsmaxerr1",  "pl_orbsmaxerr2",
    "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2",
    "pl_orbincl",  "pl_orbinclerr1",  "pl_orbinclerr2",
    "pl_orblper",  "pl_orblpererr1",  "pl_orblpererr2",  # argument of periastron
    "pl_orbtper",  "pl_orbtpererr1",  "pl_orbtpererr2",  # time of periastron
    "pl_tranmid",  "pl_tranmiderr1",  "pl_tranmiderr2",  # time of conjunction (T_peri fallback for transit planets)
    # planet mass — note for RV we usually only have m*sin(i)
    "pl_bmassj",   "pl_bmassjerr1",   "pl_bmassjerr2",
    "pl_msinij",   "pl_msinijerr1",   "pl_msinijerr2",
    # RV semi-amplitude (the direct observable)
    "pl_rvamp",    "pl_rvamperr1",    "pl_rvamperr2",
    # stellar parameters (we need M_star to get planet mass from K)
    "st_mass", "st_masserr1", "st_masserr2",
    "st_rad",  "st_raderr1",  "st_raderr2",
    "st_teff", "st_tefferr1", "st_tefferr2",
    # provenance
    "discoverymethod", "disc_year", "default_flag",
]


# ----------------------------------------------------------------------
# (1) IPAC .tbl parser
# ----------------------------------------------------------------------
# Match both `\ key = value` (canonical IPAC) and `# key = value` (rvdb-style).
# Comment lines without an `=` are treated as free-form text and ignored.
_KV_RE = re.compile(r"[\\#]\s*([^=\s]+)\s*=\s*(.*)")


def parse_tbl(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an IPAC ASCII RV table.

    Returns
    -------
    meta : dict
        Keyword-block metadata. Common keys include STAR_ID, TELESCOPE,
        INSTRUMENT, OBSERVATORY, REFERENCE, BIBCODE.
    time : (N,) float64 ndarray
        Observation epochs (BJD).
    rv : (N,) float64 ndarray
        Radial velocity (m/s).
    rv_err : (N,) float64 ndarray
        RV uncertainty (m/s).
    """
    meta: dict[str, str] = {}
    col_names: list[str] = []
    data_lines: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            ln = raw.rstrip("\n")
            if not ln.strip():
                continue
            if ln.startswith(("\\", "#")):
                m = _KV_RE.match(ln)
                if m:
                    meta[m.group(1).strip().upper()] = m.group(2).strip().strip("'\"")
                # else: free-form comment, ignore
            elif ln.startswith("|"):
                # Column-definition rows: names, types, units, nulls — first one
                # gives the column names; subsequent ones are types/units/null-flag.
                if not col_names:
                    col_names = [c.strip().lower() for c in ln.strip("|").split("|") if c.strip()]
            else:
                data_lines.append(ln)

    if not data_lines:
        raise ValueError(f"No data rows found in {path}")

    arr = np.loadtxt(io.StringIO("\n".join(data_lines)))
    if arr.ndim == 1:
        arr = arr[None, :]

    # Standard IPAC RV layout is JD/BJD, RV, RV_err in the first three columns.
    # We trust position rather than column names because the names vary
    # across contributors (jd vs bjd vs date; mnvel vs rv vs vel; etc.).
    time = arr[:, 0]
    rv = arr[:, 1]
    rv_err = arr[:, 2] if arr.shape[1] >= 3 else np.full_like(rv, np.nan)
    return meta, time, rv, rv_err


# ----------------------------------------------------------------------
# (2) TAP label fetcher
# ----------------------------------------------------------------------
def fetch_labels(default_only: bool = True) -> pd.DataFrame:
    """
    Query the Planetary Systems table for every confirmed exoplanet.

    We deliberately do NOT filter on discoverymethod: the bulk RADIAL set
    contains RV time series for every confirmed exoplanet host star,
    regardless of which technique originally discovered the planet
    (transit follow-up like HD 209458 b, direct imaging like HR 8799,
    microlensing, etc. all have RV data in the archive). Filtering to
    'Radial Velocity' here drops ~50% of the available labels for files
    we've already downloaded. The discoverymethod column is still
    returned as metadata.

    Parameters
    ----------
    default_only : bool
        If True, return only the archive-preferred parameter set per planet
        (default_flag = 1). If False, return every published parameter row,
        which is useful for comparing literature values against each other.
    """
    cols = ",".join(LABEL_COLS)
    where = "default_flag=1" if default_only else "1=1"
    query = f"select+{cols}+from+ps+where+{where}"
    url = f"{TAP}?query={query}&format=csv"
    print(f"[tap] GET {url[:120]}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"[tap] {len(df):,} rows, {len(df.columns)} columns")
    return df


# ----------------------------------------------------------------------
# (3) Join: for every downloaded .tbl, attach its host's labels.
# ----------------------------------------------------------------------
def _host_from_nexsci_url(url: str) -> str:
    """Extract the host star from a 'nexsci_url' keyword value, e.g.
    '...?objname=HD%20209458%20b&...' -> 'HD 209458' (strips trailing planet letter)."""
    if not url:
        return ""
    from urllib.parse import parse_qs, unquote, urlparse
    qs = parse_qs(urlparse(url).query)
    name = unquote(qs.get("objname", [""])[0]).strip()
    # Drop the trailing planet-letter component if present ("HD 209458 b" -> "HD 209458")
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[0]
    return name


def _norm_name(s) -> str:
    """Whitespace-, case-, and prefix-insensitive name key.

    SIMBAD returns identifiers with leading sigils (e.g. '* 24 Sex' for
    Flamsteed names, 'V* TW Hya' for variable stars). We strip those so
    they normalize the same way as the NASA archive's hostname.
    """
    s = str(s).strip()
    for pfx in ("V* ", "* ", "NAME "):
        if s.startswith(pfx):
            s = s[len(pfx):].strip()
            break
    return re.sub(r"[\s\-]+", "", s).lower()


def match_host_rows(host: str, labels: pd.DataFrame) -> pd.DataFrame:
    """Return label rows whose hostname OR hd_name OR hip_name OR tic_id
    matches the given host string. Handles cases like a .tbl file saying
    'HIP 108859' while the labels table primary hostname is 'HD 209458'."""
    if not host or not str(host).strip():
        return labels.iloc[0:0]
    key = _norm_name(host)
    mask = pd.Series(False, index=labels.index)
    for col in ("hostname", "hd_name", "hip_name", "tic_id"):
        if col in labels.columns:
            s = labels[col]
            mask = mask | (s.notna() & (s.astype(str).map(_norm_name) == key))
    return labels[mask]


# ----------------------------------------------------------------------
# SIMBAD alias resolution (fallback when direct match fails)
# ----------------------------------------------------------------------
def resolve_simbad_aliases(names: list[str], cache_path: Path) -> dict[str, list[str]]:
    """
    For each name not already cached, query SIMBAD for all known identifiers
    (HD, HIP, TIC, 2MASS, Gaia, Flamsteed, Bayer, Gliese, etc.) and write the
    result to a JSON cache so subsequent runs are free.

    Returns
    -------
    dict mapping every requested name to a list of alias strings; a name
    that SIMBAD doesn't recognize or that errors out maps to an empty list.

    Notes
    -----
    Requires `astroquery` (pip install astroquery). Falls back to the empty
    cache if astroquery is missing or SIMBAD is unreachable, so the rest of
    the pipeline still works.
    """
    import json
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:  # noqa: BLE001 — corrupt cache, start over
            cache = {}

    todo = [n for n in names if n and n not in cache]
    if not todo:
        if names:
            print(f"[simbad] all {len(names)} names already in cache")
        return cache

    try:
        from astroquery.simbad import Simbad
    except ImportError:
        print("[simbad] astroquery not installed (pip install astroquery); "
              "skipping SIMBAD fallback")
        return cache

    import warnings
    warnings.filterwarnings("ignore", module="astroquery")
    warnings.filterwarnings("ignore", module="astropy")

    print(f"[simbad] resolving {len(todo)} unmatched names "
          f"(this takes a few seconds each; cached to {cache_path})...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(todo, 1):
        try:
            result = Simbad.query_objectids(name)
            if result is not None and len(result) > 0:
                # Column name has historically varied; pick the first column
                # if 'ID' isn't present.
                col = "ID" if "ID" in result.colnames else result.colnames[0]
                cache[name] = [str(row[col]).strip() for row in result]
            else:
                cache[name] = []
        except Exception as e:  # noqa: BLE001 — network/parse error, log and move on
            cache[name] = []
            print(f"  [simbad warn] {name!r}: {type(e).__name__}: {e}")

        if i % 25 == 0 or i == len(todo):
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
            print(f"  [simbad] {i}/{len(todo)} done")

    resolved = sum(1 for n in todo if cache.get(n))
    print(f"[simbad] {resolved}/{len(todo)} new names had SIMBAD entries")
    return cache


def match_with_simbad(host: str, labels: pd.DataFrame,
                      alias_cache: dict[str, list[str]]) -> pd.DataFrame:
    """Try direct match first, then each SIMBAD alias in turn."""
    direct = match_host_rows(host, labels)
    if not direct.empty:
        return direct
    for alias in alias_cache.get(host, []):
        m = match_host_rows(alias, labels)
        if not m.empty:
            return m
    return labels.iloc[0:0]


def build_index(rv_dir: Path, labels: pd.DataFrame,
                use_simbad: bool = True,
                simbad_cache: Path = Path("data/simbad_cache.json")) -> pd.DataFrame:
    """
    Walk through downloaded .tbl files, extract each one's host name from its
    metadata block, and join against labels using ANY identifier column
    (hostname, hd_name, hip_name, tic_id). For hosts that don't match directly,
    fall back to SIMBAD alias resolution.
    """
    file_rows = []
    for path in sorted(rv_dir.glob("UID_*_RVC_*.tbl")):
        try:
            meta, t, rv, err = parse_tbl(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not parse {path.name}: {e}")
            continue
        host = (
            meta.get("STAR_ID")
            or meta.get("STARNAME")
            or meta.get("HOSTNAME")
            or meta.get("OBJECT")
            or _host_from_nexsci_url(meta.get("NEXSCI_URL", ""))
            or ""
        ).strip()
        file_rows.append({
            "file": path.name,
            "host_in_file": host,
            "n_obs": len(t),
            "t_baseline_days": float(t.max() - t.min()) if len(t) else 0.0,
            "telescope": meta.get("TELESCOPE", ""),
            "reference": meta.get("REFERENCE", ""),
        })
    idx = pd.DataFrame(file_rows)

    # First pass: direct identifier matching
    direct: dict[str, pd.DataFrame] = {}
    unmatched_hosts: list[str] = []
    for row in idx.itertuples(index=False):
        m = match_host_rows(row.host_in_file, labels)
        if m.empty:
            unmatched_hosts.append(row.host_in_file)
        else:
            direct[row.file] = m

    # Second pass: SIMBAD alias resolution for unmatched
    alias_cache: dict[str, list[str]] = {}
    if use_simbad and unmatched_hosts:
        unique_unmatched = sorted({h for h in unmatched_hosts if h})
        alias_cache = resolve_simbad_aliases(unique_unmatched, simbad_cache)

    via_simbad: dict[str, pd.DataFrame] = {}
    if alias_cache:
        for row in idx.itertuples(index=False):
            if row.file in direct:
                continue
            for alias in alias_cache.get(row.host_in_file, []):
                m = match_host_rows(alias, labels)
                if not m.empty:
                    via_simbad[row.file] = m
                    break

    # Build the final output table
    out = []
    still_unmatched: list[str] = []
    for row in idx.itertuples(index=False):
        m = direct.get(row.file)
        if m is None:
            m = via_simbad.get(row.file)
        if m is None or m.empty:
            empty = {c: None for c in labels.columns}
            out.append({**row._asdict(), **empty})
            still_unmatched.append(row.host_in_file)
        else:
            for _, lbl in m.iterrows():
                out.append({**row._asdict(), **lbl.to_dict()})
    merged = pd.DataFrame(out)

    n_total = len(idx)
    n_direct = len(direct)
    n_simbad = len(via_simbad)
    n_unmatched = len(still_unmatched)
    print(f"[join] direct matches: {n_direct}/{n_total}   "
          f"via SIMBAD: {n_simbad}   still unmatched: {n_unmatched}")
    if still_unmatched:
        sample = sorted(set(still_unmatched))[:15]
        print(f"[join] first {len(sample)} still-unmatched host names: {sample}")
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--all-rows", action="store_true",
                   help="Include every published row, not just default_flag=1")
    p.add_argument("--no-simbad", action="store_true",
                   help="Skip SIMBAD alias-resolution fallback (offline mode)")
    p.add_argument("--simbad-cache", type=Path,
                   default=Path("data/simbad_cache.json"),
                   help="Path to the SIMBAD alias JSON cache")
    args = p.parse_args()

    labels = fetch_labels(default_only=not args.all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f"[done] labels -> {args.out}")

    if args.rv_dir.exists():
        idx = build_index(args.rv_dir, labels,
                          use_simbad=not args.no_simbad,
                          simbad_cache=args.simbad_cache)
        idx_path = args.out.with_name("rv_index.csv")
        idx.to_csv(idx_path, index=False)
        print(f"[done] index -> {idx_path}")


if __name__ == "__main__":
    main()