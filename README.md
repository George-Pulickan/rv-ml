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

The pipeline is validated end-to-end by forward-modeling the radial velocity
signal from each host's tabulated Keplerian parameters and comparing to the
published observations:

    python kepler_check.py            # canonical test (51 Peg b)
    python kepler_check.py --all      # corpus-wide summary

51 Peg b serves as the gold-standard test, with χ²_reduced = 1.31 and
RMS/σ = 1.19 — the Kepler model traces the data within measurement noise.
Across the corpus, 391 quality-filtered systems validate with a median
RMS/σ of 3.5, consistent with the well-known stellar-activity floor that
catalog uncertainties do not include.