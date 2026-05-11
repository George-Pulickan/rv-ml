"""
download_rv.py
--------------
Download all radial-velocity (RV) time series from the NASA Exoplanet Archive.

Source of truth: the official bulk wget script
    https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/wget_RADIAL.bat
which lists ~1,072 files (~3 MB total) for confirmed exoplanet host stars.

We don't actually run wget. We parse URLs out of the script and download each
file with `requests` in a thread pool. This is faster (concurrent), portable
(no external binary), and restartable (skips files already on disk).

Usage
-----
    python download_rv.py --out data/rv_raw --workers 16

Outputs
-------
    data/rv_raw/UID_XXXXXXX_RVC_NNN.tbl      (one file per star/curve)
    data/rv_raw/_manifest.csv                (filename, url, bytes, sha256)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import re
import sys
import time
from pathlib import Path

import requests

BULK_SCRIPT_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/wget_RADIAL.bat"
)
# Files referenced inside the script look like
#   https://exoplanetarchive.ipac.caltech.edu/data/ExoData/0071/0071395/data/UID_0071395_RVC_003.tbl
URL_RE = re.compile(r"https?://[^\s'\"]+\.tbl", re.IGNORECASE)
HEADERS = {"User-Agent": "rv-ml-research/0.1 (+research; contact: george)"}


def fetch_url_list() -> list[str]:
    """Pull the bulk wget script and extract every .tbl URL from it."""
    r = requests.get(BULK_SCRIPT_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    urls = sorted(set(URL_RE.findall(r.text)))
    if not urls:
        raise RuntimeError(
            "No .tbl URLs found in wget_RADIAL.bat — the script format may have "
            "changed. Inspect the file manually:\n  " + BULK_SCRIPT_URL
        )
    return urls


def download_one(
    url: str, out_dir: Path, session: requests.Session, retries: int = 3
) -> tuple[str, int, str]:
    """
    Download a single .tbl. Returns (filename, n_bytes, sha256).
    Skips the file if it already exists with non-zero size (idempotent re-runs).
    """
    name = url.rsplit("/", 1)[-1]
    dst = out_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        data = dst.read_bytes()
    else:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = session.get(url, headers=HEADERS, timeout=60)
                resp.raise_for_status()
                data = resp.content
                dst.write_bytes(data)
                break
            except Exception as e:  # noqa: BLE001 — log and retry any transport error
                last_err = e
                time.sleep(2**attempt)  # exponential backoff
        else:
            raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_err

    return name, len(data), hashlib.sha256(data).hexdigest()


def download_all(out_dir: Path, workers: int = 16) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = fetch_url_list()
    print(f"[manifest] {len(urls)} RV files listed in wget_RADIAL.bat", file=sys.stderr)

    manifest_path = out_dir / "_manifest.csv"
    rows: list[dict] = []
    t0 = time.time()
    with requests.Session() as session, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, u, out_dir, session): u for u in urls}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            url = futures[fut]
            try:
                name, n, sha = fut.result()
                rows.append({"filename": name, "url": url, "bytes": n, "sha256": sha})
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {url}: {e}", file=sys.stderr)
                rows.append({"filename": "", "url": url, "bytes": 0, "sha256": ""})
            if i % 50 == 0 or i == len(urls):
                print(f"  [{i}/{len(urls)}] {time.time() - t0:.1f}s", file=sys.stderr)

    rows.sort(key=lambda r: r["filename"])
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "url", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote manifest -> {manifest_path}", file=sys.stderr)
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=Path("data/rv_raw"), help="output directory"
    )
    p.add_argument("--workers", type=int, default=16, help="parallel downloads")
    args = p.parse_args()
    download_all(args.out, args.workers)


if __name__ == "__main__":
    main()
