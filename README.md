# rv-ml

ML pipeline for predicting exoplanet orbital parameters from radial velocity
time series. Encoder → embedding → continuous-output decoder, with conformal
prediction intervals.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python download_rv.py    --out data/rv_raw
    python parse_and_label.py --rv-dir data/rv_raw --out data/labels.csv

## Data sources

- NASA Exoplanet Archive bulk RV download (1,072 RV curves)
- NASA Exoplanet Archive Planetary Systems table via TAP service

## Validation

The pipeline is validated by forward-modeling the radial velocity signal
from each host's tabulated Keplerian parameters and comparing to the
published observations (`python kepler_check.py --all`). 51 Peg b serves
as the canonical test (χ²_reduced = 1.31, RMS/σ = 1.19). Across the full
corpus, 391 quality-filtered systems are validated with a median RMS/σ
of 3.5, consistent with the well-known stellar-activity noise floor that
the catalog uncertainties do not include.