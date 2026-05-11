"""Exercise parse_tbl on (a) synthetic canonical IPAC format and (b) real rvdb."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from parse_and_label import parse_tbl

print("=" * 60)
print("CASE A: canonical IPAC format ('\\ key = value', '|cols|')")
print("=" * 60)
meta, t, rv, err = parse_tbl("/home/claude/test_fixture.tbl")
print(f"  n_points     = {len(t)}")
print(f"  STAR_ID      = {meta.get('STAR_ID')!r}")
print(f"  TELESCOPE    = {meta.get('TELESCOPE')!r}")
print(f"  REFERENCE    = {meta.get('REFERENCE')!r}")
print(f"  time range   = {t.min():.4f} .. {t.max():.4f}")
print(f"  rv range     = {rv.min():.2f} .. {rv.max():.2f} m/s")
print(f"  err range    = {err.min():.2f} .. {err.max():.2f} m/s")
print(f"  all meta keys: {sorted(meta.keys())}")

print()
print("=" * 60)
print("CASE B: rvdb '# key = value' variant — real Butler+2006 Keck data")
print("=" * 60)
meta, t, rv, err = parse_tbl("/home/claude/rvdb/data/14Her_1_KECK.vels")
print(f"  n_points     = {len(t)}")
print(f"  TELESCOPE    = {meta.get('TELESCOPE')!r}")
print(f"  INSTRUMENT   = {meta.get('INSTRUMENT')!r}")
print(f"  REFERENCE    = {meta.get('REFERENCE')!r}")
print(f"  BIBCODE      = {meta.get('BIBCODE')!r}")
print(f"  SOURCE       = {meta.get('SOURCE')!r}")
print(f"  time range   = {t.min():.4f} .. {t.max():.4f}  ({t.max()-t.min():.0f} d baseline)")
print(f"  rv range     = {rv.min():.2f} .. {rv.max():.2f} m/s")
print(f"  err range    = {err.min():.2f} .. {err.max():.2f} m/s")

print()
print("=" * 60)
print("CASE C: another real file — HD 209458 ELODIE data (Naef+2004)")
print("=" * 60)
meta, t, rv, err = parse_tbl("/home/claude/rvdb/data/HD209458_1_ELODIE.vels")
print(f"  n_points     = {len(t)}")
print(f"  TELESCOPE    = {meta.get('TELESCOPE')!r}")
print(f"  BIBCODE      = {meta.get('BIBCODE')!r}")
print(f"  time range   = {t.min():.4f} .. {t.max():.4f}  ({t.max()-t.min():.0f} d baseline)")
print(f"  rv range     = {rv.min():.2f} .. {rv.max():.2f} m/s")

print()
print("=" * 60)
print("Host-name extraction (multi-strategy)")
print("=" * 60)
from parse_and_label import _host_from_nexsci_url
# canonical IPAC: STAR_ID keyword
meta, *_ = parse_tbl("/home/claude/test_fixture.tbl")
print(f"  synthetic IPAC -> STAR_ID            = {meta.get('STAR_ID')!r}")
# rvdb: parse out of nexsci_url
meta, *_ = parse_tbl("/home/claude/rvdb/data/14Her_1_KECK.vels")
print(f"  rvdb 14 Her    -> nexsci_url objname = {_host_from_nexsci_url(meta.get('NEXSCI_URL',''))!r}")
meta, *_ = parse_tbl("/home/claude/rvdb/data/HD209458_1_ELODIE.vels")
print(f"  rvdb HD 209458 -> nexsci_url objname = {_host_from_nexsci_url(meta.get('NEXSCI_URL',''))!r}")

print("=" * 60)
import re
# Simulated wget script line (this is exactly the form the .bat uses)
sample = (
    'wget -q -nv -nH --cut-dirs=2 -np "https://exoplanetarchive.ipac.caltech.edu'
    '/data/ExoData/0079/0079248/data/UID_0079248_RVC_002.tbl"\n'
    'wget -q -nv -nH --cut-dirs=2 -np "https://exoplanetarchive.ipac.caltech.edu'
    '/data/ExoData/0108/0108859/data/UID_0108859_RVC_004.tbl"\n'
)
URL_RE = re.compile(r"https?://[^\s'\"]+\.tbl", re.IGNORECASE)
urls = URL_RE.findall(sample)
print(f"  extracted {len(urls)} URLs from sample:")
for u in urls:
    print(f"    {u}")
