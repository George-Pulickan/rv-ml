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

51 Peg b is the gold-standard test (χ²_reduced = 1.31, RMS/σ = 1.19); the
Kepler model traces the data within measurement noise. Across the full
corpus, 432 quality-filtered systems validate with a median RMS/σ of 3.7,
consistent with the stellar-activity floor that catalog uncertainties do
not include. The pipeline matches 857 of 1,071 files to known planet
hosts (766 by direct identifier matching, plus 91 recovered via SIMBAD
alias resolution); the remaining 214 unmatched files are predominantly
2MASS-designated survey candidates that don't appear in NASA's confirmed-
planet table.

## Overleaf draft

A work-in-progress draft about the methodology and related work is here: https://www.overleaf.com/8188483955gysdcwmjrwhq#ac30a1
