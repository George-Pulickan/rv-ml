"""Portable smoke checks for parse_tbl."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse_and_label import _host_from_nexsci_url, parse_tbl


def show_parsed(path: Path, label: str) -> None:
    meta, t, rv, err = parse_tbl(path)
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  path         = {path}")
    print(f"  n_points     = {len(t)}")
    print(f"  STAR_ID      = {meta.get('STAR_ID')!r}")
    print(f"  TELESCOPE    = {meta.get('TELESCOPE')!r}")
    print(f"  REFERENCE    = {meta.get('REFERENCE')!r}")
    print(f"  time range   = {t.min():.4f} .. {t.max():.4f}")
    print(f"  rv range     = {rv.min():.2f} .. {rv.max():.2f} m/s")
    print(f"  err range    = {err.min():.2f} .. {err.max():.2f} m/s")
    print(f"  host from URL = {_host_from_nexsci_url(meta.get('NEXSCI_URL', ''))!r}")


fixture = Path(__file__).with_name("test_fixture.tbl")
show_parsed(fixture, "CASE A: local canonical IPAC fixture")

for path, label in [
    (Path("/home/claude/rvdb/data/14Her_1_KECK.vels"), "CASE B: optional 14 Her rvdb fixture"),
    (Path("/home/claude/rvdb/data/HD209458_1_ELODIE.vels"), "CASE C: optional HD 209458 rvdb fixture"),
]:
    if path.exists():
        show_parsed(path, label)
    else:
        print(f"skipped missing external fixture: {path}")


if __name__ == "__main__":
    print("parse_tbl smoke checks completed")
