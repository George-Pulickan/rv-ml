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
    Query the Planetary Systems table for all confirmed RV-discovered planets.

    Parameters
    ----------
    default_only : bool
        If True, return only the archive-preferred parameter set per planet
        (default_flag = 1). If False, return every published parameter row,
        which is useful for comparing literature values against each other.
    """
    cols = ",".join(LABEL_COLS)
    where = "discoverymethod like 'Radial%20Velocity'"
    if default_only:
        where += "+and+default_flag=1"
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


def build_index(rv_dir: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through downloaded .tbl files, extract each one's host name from its
    metadata block, and inner-join against the labels table on hostname.
    """
    rows = []
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
        rows.append(
            {
                "file": path.name,
                "host_in_file": host,
                "n_obs": len(t),
                "t_baseline_days": float(t.max() - t.min()) if len(t) else 0.0,
                "telescope": meta.get("TELESCOPE", ""),
                "reference": meta.get("REFERENCE", ""),
            }
        )
    idx = pd.DataFrame(rows)
    # Loose name match: tolerant of "HD 209458" vs "HD209458"
    norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
    labels = labels.assign(_key=labels["hostname"].map(norm))
    idx = idx.assign(_key=idx["host_in_file"].map(norm))
    merged = idx.merge(labels, on="_key", how="left").drop(columns=["_key"])
    matched = merged["pl_name"].notna().sum()
    print(f"[join] matched {matched}/{len(idx)} RV files to a known planet host")
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--all-rows", action="store_true",
                   help="Include every published row, not just default_flag=1")
    args = p.parse_args()

    labels = fetch_labels(default_only=not args.all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f"[done] labels -> {args.out}")

    if args.rv_dir.exists():
        idx = build_index(args.rv_dir, labels)
        idx_path = args.out.with_name("rv_index.csv")
        idx.to_csv(idx_path, index=False)
        print(f"[done] index -> {idx_path}")


if __name__ == "__main__":
    main()
