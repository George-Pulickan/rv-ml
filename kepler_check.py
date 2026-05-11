"""
kepler_check.py
---------------
Pipeline validator using a Keplerian RV integrator.

For a given RV time series, simulate the expected RV signal from the
tabulated orbital parameters of every known planet around the host star,
then overlay model + data on the same axes. If our download → host-match →
label-join pipeline is correct, the simulation should track the
observations to within the measurement noise.

The Kepler model
----------------
For a single planet, the line-of-sight stellar velocity is

    V_r(t) = K · [cos(ν + ω) + e·cos(ω)] + γ

where ν is the true anomaly at time t (obtained by solving the Kepler
equation M = E - e·sin(E) numerically), ω is the argument of periastron,
e is the eccentricity, K is the RV semi-amplitude, and γ is the
instrument zero-point (a constant nuisance offset). Multi-planet RVs
simply sum.

Per Nicolò's spec, γ is fixed by anchoring the model to pass exactly
through the first observation (t_0, RV_0); this leaves zero free
parameters and turns this into a pure prediction test. A 'fit' mode is
also provided which finds γ by least squares.

Usage
-----
    python kepler_check.py                      # picks a famous example
    python kepler_check.py --host "51 Peg"      # by host star
    python kepler_check.py --file UID_0113357_RVC_001.tbl
    python kepler_check.py --all                # batch summary over the corpus
    python kepler_check.py --host "HD 209458" --mode fit --save fig.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_and_label import (_host_from_nexsci_url, match_host_rows,
                              match_with_simbad, parse_tbl)


# ---------------------------------------------------------------------------
# Kepler integrator
# ---------------------------------------------------------------------------
def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-12, maxiter: int = 50) -> np.ndarray:
    """
    Solve Kepler's equation M = E - e·sin(E) for the eccentric anomaly E.

    Uses Newton-Raphson with a Danby-style starting guess; converges in 3-5
    iterations for typical eccentricities.
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    # Wrap M into [-π, π] for numerical stability
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = M + e * np.sin(M)          # 1st-order guess (exact for e=0)
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def true_anomaly(E: np.ndarray, e: float) -> np.ndarray:
    """Convert eccentric anomaly E to true anomaly ν (numerically robust form)."""
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )


def rv_keplerian(t: np.ndarray, P: float, K: float, e: float,
                 omega: float, t_peri: float) -> np.ndarray:
    """
    RV signal of one Keplerian planet (no γ offset).

    Parameters
    ----------
    t       : observation times [days, any epoch as long as it matches t_peri]
    P       : orbital period [days]
    K       : RV semi-amplitude [m/s]
    e       : eccentricity [0, 1)
    omega   : argument of periastron [radians]
    t_peri  : time of periastron passage [days, same scale as t]
    """
    M = 2.0 * np.pi * (t - t_peri) / P
    E = solve_kepler(M, e)
    nu = true_anomaly(E, e)
    return K * (np.cos(nu + omega) + e * np.cos(omega))


def semi_amplitude(msini_jup: float, period_day: float, e: float,
                   mstar_sun: float) -> float:
    """
    Derive K from M sin i, P, e, M_star when the catalog doesn't list it.

    K [m/s] = 28.4329 · (M_p sin i / M_Jup) · (M_*/M_sun)^(-2/3)
              · (P / yr)^(-1/3) / sqrt(1 - e^2)

    (See e.g. Lovis & Fischer 2010, eq. 14, in the regime M_p << M_*.)
    """
    P_yr = period_day / 365.25
    return (
        28.4329
        * msini_jup
        * (mstar_sun ** (-2.0 / 3.0))
        * (P_yr ** (-1.0 / 3.0))
        / np.sqrt(1.0 - e * e)
    )


# ---------------------------------------------------------------------------
# Catalog row -> planet model parameters
# ---------------------------------------------------------------------------
@dataclass
class Planet:
    name: str
    P: float          # period [days]
    K: float          # semi-amplitude [m/s]
    e: float          # eccentricity
    omega: float      # arg. of periastron [radians]
    t_peri: float     # time of periastron passage [BJD]
    K_source: str     # "catalog" or "derived"


def build_planet(row: pd.Series) -> tuple[Planet | None, str]:
    """Convert one row of the labels DataFrame into a Planet.

    Returns (Planet, "") on success or (None, reason) on failure, where
    `reason` is one of: 'no_period', 'no_tperi_or_tranmid', 'no_K_and_no_msini'.

    For transit-discovered planets the catalog usually tabulates `pl_tranmid`
    (time of conjunction) rather than `pl_orbtper` (time of periastron).
    For low-eccentricity orbits these differ by ~P/4 in a known way; we
    apply the conversion when needed.
    """
    P = row.get("pl_orbper")
    if pd.isna(P):
        return None, "no_period"

    e = float(row["pl_orbeccen"]) if pd.notna(row.get("pl_orbeccen")) else 0.0
    omega_deg = float(row["pl_orblper"]) if pd.notna(row.get("pl_orblper")) else 90.0
    omega = np.radians(omega_deg)

    t_peri = row.get("pl_orbtper")
    if pd.isna(t_peri):
        # Fall back to time of conjunction: convert T_c -> T_peri via the
        # eccentric anomaly at conjunction. f_c = π/2 - ω → E_c → M_c.
        t_c = row.get("pl_tranmid")
        if pd.isna(t_c):
            return None, "no_tperi_or_tranmid"
        f_c = np.pi / 2.0 - omega
        E_c = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(f_c / 2.0),
                                np.sqrt(1.0 + e) * np.cos(f_c / 2.0))
        M_c = E_c - e * np.sin(E_c)
        t_peri = float(t_c) - float(P) * M_c / (2.0 * np.pi)
    else:
        t_peri = float(t_peri)

    K = row.get("pl_rvamp")
    K_source = "catalog"
    if pd.isna(K):
        msini = row.get("pl_msinij")
        mstar = row.get("st_mass")
        if pd.isna(msini) or pd.isna(mstar):
            return None, "no_K_and_no_msini"
        K = semi_amplitude(float(msini), float(P), e, float(mstar))
        K_source = "derived from M sin i, P, M_*"
    return Planet(
        name=row["pl_name"],
        P=float(P),
        K=float(K),
        e=e,
        omega=omega,
        t_peri=float(t_peri),
        K_source=K_source,
    ), ""


def host_name(meta: dict) -> str:
    """Multi-strategy host-name extraction (matches parse_and_label)."""
    return (
        meta.get("STAR_ID")
        or meta.get("STARNAME")
        or meta.get("HOSTNAME")
        or meta.get("OBJECT")
        or _host_from_nexsci_url(meta.get("NEXSCI_URL", ""))
        or ""
    ).strip()


def _norm(s) -> str:
    """Whitespace- and case-insensitive name key."""
    return re.sub(r"\s+", "", str(s)).lower()


# ---------------------------------------------------------------------------
# Validation: simulate, compare, plot
# ---------------------------------------------------------------------------
def evaluate_model(planets: list[Planet], t: np.ndarray) -> np.ndarray:
    return sum((rv_keplerian(t, p.P, p.K, p.e, p.omega, p.t_peri) for p in planets),
               start=np.zeros_like(t))


def validate_one(tbl_path: Path, labels: pd.DataFrame, mode: str = "anchor",
                 plot: bool = True, save: Path | None = None,
                 verbose: bool = True,
                 simbad_cache: dict[str, list[str]] | None = None) -> dict:
    """
    Run the full validation on one RV file. Always returns a dict with a
    'status' field; 'ok' means metrics were computed, anything else means
    the file was skipped for the given reason.

    If `simbad_cache` is supplied, host-name lookups fall back to SIMBAD
    aliases when the direct identifier match fails (matches the behaviour
    of parse_and_label.build_index).
    """
    meta, t, rv, err = parse_tbl(tbl_path)
    base = {"file": tbl_path.name, "n_obs": len(t)}

    host = host_name(meta)
    if not host:
        if verbose: print(f"[{tbl_path.name}] no host found in metadata")
        return {**base, "status": "no_host_in_metadata", "host": ""}

    if simbad_cache is not None:
        rows = match_with_simbad(host, labels, simbad_cache)
    else:
        rows = match_host_rows(host, labels)
    if rows.empty:
        if verbose: print(f"[{tbl_path.name}] no labels for host {host!r}")
        return {**base, "status": "no_labels_for_host", "host": host}

    built = [build_planet(r) for _, r in rows.iterrows()]
    planets = [p for p, _ in built if p is not None]
    if not planets:
        # All catalog rows dropped — report the dominant reason
        reasons = [r for _, r in built if r]
        dominant = max(set(reasons), key=reasons.count) if reasons else "unknown"
        if verbose: print(f"[{tbl_path.name}] {host}: no usable planets ({dominant})")
        return {**base, "status": f"no_planets:{dominant}", "host": host}

    v_model = evaluate_model(planets, t)

    if mode == "anchor":
        gamma = float(rv[0] - v_model[0])
    elif mode == "fit":
        w = 1.0 / np.maximum(err, 1e-6) ** 2
        gamma = float(np.sum(w * (rv - v_model)) / np.sum(w))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    residuals = rv - (v_model + gamma)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    median_err = float(np.median(err))
    chi2_red = float(np.mean((residuals / np.maximum(err, 1e-6)) ** 2))

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"{tbl_path.name}  →  host: {host}  ({len(planets)} planet(s))")
        print(f"{'=' * 70}")
        for p in planets:
            print(f"  {p.name:20s}  P={p.P:>10.4f} d   K={p.K:>7.2f} m/s   "
                  f"e={p.e:.3f}   ω={np.degrees(p.omega):6.1f}°   ({p.K_source})")
        print(f"  N_obs={len(t)}   baseline={t.max() - t.min():.0f} d   "
              f"γ-mode={mode}   γ={gamma:+.2f} m/s")
        print(f"  RMS(residual) = {rms:.2f} m/s   "
              f"median σ_obs = {median_err:.2f} m/s   "
              f"RMS/σ = {rms / median_err:.2f}   "
              f"χ²_red = {chi2_red:.2f}")

    if plot:
        plot_validation(t, rv, err, v_model + gamma, residuals, host, planets,
                        rms, median_err, mode, save, gamma=gamma)

    return {
        "file": tbl_path.name,
        "status": "ok",
        "host": host,
        "n_planets": len(planets),
        "n_obs": len(t),
        "rms_residual_ms": rms,
        "median_sigma_ms": median_err,
        "rms_over_sigma": rms / median_err,
        "chi2_red": chi2_red,
    }


def plot_validation(t, rv, err, model, residuals, host, planets,
                    rms, median_err, mode, save, gamma=0.0):
    """
    Three-panel layout: full time series (top), phase-folded view on the
    shortest-period planet (middle, only if multiple cycles are observed),
    and residuals vs time (bottom).
    """
    # Decide whether to include a phase-folded subplot
    shortest = min(planets, key=lambda p: p.P)
    n_cycles = (t.max() - t.min()) / shortest.P
    show_phase = n_cycles >= 3  # only useful when the planet has wrapped a few times

    if show_phase:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                                 gridspec_kw={"height_ratios": [1.8, 1.8, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                                 gridspec_kw={"height_ratios": [2.2, 1]})

    # ---- (a) full time series ----
    span = t.max() - t.min()
    t_fine = np.linspace(t.min() - 0.02 * span, t.max() + 0.02 * span, 4000)
    v_fine = evaluate_model(planets, t_fine) + gamma
    axes[0].plot(t_fine, v_fine, "C0-", lw=1.0, alpha=0.7,
                 label=f"Kepler model — {len(planets)} planet(s)")
    axes[0].errorbar(t, rv, yerr=err, fmt="ko", ms=3.5, capsize=2, lw=0.7,
                     label="RV data")
    axes[0].set_ylabel("RV  [m/s]")
    axes[0].set_title(
        f"{host}   |   RMS={rms:.1f} m/s, median σ={median_err:.1f} m/s, "
        f"ratio={rms / median_err:.2f}   |   γ-mode: {mode}"
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    # ---- (b) phase-folded on the shortest-period planet ----
    if show_phase:
        # Subtract the contributions of all OTHER planets from the data, so
        # what remains is the signal of the shortest-period planet alone.
        other = [p for p in planets if p is not shortest]
        v_other_at_t = evaluate_model(other, t) if other else np.zeros_like(t)
        rv_iso = rv - gamma - v_other_at_t        # isolated signal of `shortest`
        phase = ((t - shortest.t_peri) / shortest.P) % 1.0

        ph_fine = np.linspace(0, 1, 1000)
        t_for_fine = shortest.t_peri + ph_fine * shortest.P
        v_shortest_fine = rv_keplerian(t_for_fine, shortest.P, shortest.K,
                                        shortest.e, shortest.omega,
                                        shortest.t_peri)

        order = np.argsort(ph_fine)
        axes[1].plot(ph_fine[order], v_shortest_fine[order], "C0-", lw=1.5,
                     label=f"Kepler model — {shortest.name}")
        axes[1].errorbar(phase, rv_iso, yerr=err, fmt="ko", ms=3.5, capsize=2,
                         lw=0.7, label="data (other planets removed)")
        axes[1].set_xlabel(f"Orbital phase (P = {shortest.P:.4f} d)")
        axes[1].set_ylabel("RV  [m/s]")
        axes[1].set_xlim(0, 1)
        axes[1].legend(loc="best", fontsize=9)
        axes[1].grid(alpha=0.3)
        ax_resid = axes[2]
    else:
        ax_resid = axes[1]

    # ---- (c) residuals vs time ----
    ax_resid.errorbar(t, residuals, yerr=err, fmt="ko", ms=3.5, capsize=2, lw=0.7)
    ax_resid.axhline(0, color="C0", alpha=0.6)
    ax_resid.axhspan(-median_err, median_err, color="C0", alpha=0.12,
                     label="±median σ_obs")
    ax_resid.set_xlabel("BJD  [days]")
    ax_resid.set_ylabel("Residual  [m/s]")
    ax_resid.legend(loc="best", fontsize=9)
    ax_resid.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=130, bbox_inches="tight")
        print(f"  → saved figure to {save}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def pick_default_file(rv_dir: Path) -> Path | None:
    """Try to find a canonical demo (51 Peg, then HD 209458, then anything)."""
    idx = pd.read_csv("data/rv_index.csv") if Path("data/rv_index.csv").exists() else None
    if idx is not None:
        for name in ("51 Peg", "HD 209458", "HD 142", "HD 75732"):
            m = idx["host_in_file"].map(_norm) == _norm(name)
            if m.any():
                return rv_dir / idx.loc[m, "file"].iloc[0]
        return rv_dir / idx["file"].iloc[0]
    files = sorted(rv_dir.glob("UID_*_RVC_*.tbl"))
    return files[0] if files else None


def main() -> None:
    import json
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rv-dir", type=Path, default=Path("data/rv_raw"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--simbad-cache", type=Path,
                   default=Path("data/simbad_cache.json"),
                   help="SIMBAD alias JSON written by parse_and_label.py")
    p.add_argument("--file", type=str, help="specific .tbl filename")
    p.add_argument("--host", type=str, help="match by host name")
    p.add_argument("--mode", choices=("anchor", "fit"), default="anchor",
                   help="how to fix γ: anchor at t_0 (Nicolò's spec) or LS fit")
    p.add_argument("--all", action="store_true",
                   help="run over every .tbl and write data/validation_summary.csv")
    p.add_argument("--save", type=Path, default=None,
                   help="save figure here instead of showing interactively")
    args = p.parse_args()

    labels = pd.read_csv(args.labels)

    # Load the SIMBAD alias cache if available so we use the same matching
    # logic as parse_and_label.build_index. Without this, files whose host
    # was only recovered via SIMBAD would be wrongly classified as
    # 'no_labels_for_host' here.
    simbad_cache: dict[str, list[str]] = {}
    if args.simbad_cache.exists():
        try:
            simbad_cache = json.loads(args.simbad_cache.read_text())
            print(f"[simbad] loaded {len(simbad_cache)} cached aliases")
        except Exception as e:  # noqa: BLE001
            print(f"[simbad warn] could not read {args.simbad_cache}: {e}")

    if args.all:
        files = sorted(args.rv_dir.glob("UID_*_RVC_*.tbl"))
        rows = [validate_one(f, labels, mode=args.mode, plot=False, verbose=False,
                             simbad_cache=simbad_cache)
                for f in files]
        df = pd.DataFrame(rows)
        out = Path("data/validation_summary.csv")
        df.to_csv(out, index=False)
        print(f"\nWrote {out}  ({len(df)} rows for {len(files)} files)\n")

        # Status breakdown — why each file ended up where it did
        print("Status breakdown:")
        for status, n in df["status"].value_counts().items():
            print(f"  {n:>5d}  {status}")
        print(f"  {'-' * 5}")
        print(f"  {len(df):>5d}  total\n")

        # Apply quality filters for the headline numbers:
        #  - only files that got an 'ok' status
        #  - n_obs >= 10 (a 1-point 'fit' has zero residual by definition)
        #  - 0.1 <= median σ_obs <= 100 m/s  (excludes corrupt or wrong-unit σ)
        ok = df[df["status"] == "ok"].copy()
        clean = ok[(ok["n_obs"] >= 10)
                   & (ok["median_sigma_ms"].between(0.1, 100.0))].copy()
        clean = clean.sort_values("rms_over_sigma")

        print(f"Quality-filtered summary  (n_obs ≥ 10, 0.1 ≤ σ_obs ≤ 100 m/s):")
        print(f"  N = {len(clean)} of {len(ok)} 'ok' files "
              f"({len(clean) / max(len(ok), 1):.0%})")
        if len(clean) > 0:
            ratios = clean["rms_over_sigma"]
            print(f"  median RMS/σ        = {ratios.median():.2f}")
            print(f"  median RMS residual = {clean['rms_residual_ms'].median():.2f} m/s")
            print(f"  fraction RMS/σ < 3  = {(ratios < 3).mean():.1%}")
            print(f"  fraction RMS/σ < 5  = {(ratios < 5).mean():.1%}")
            print("\n  Best 5 (smallest RMS/σ):")
            print(clean[["file", "host", "n_obs", "median_sigma_ms",
                         "rms_residual_ms", "rms_over_sigma", "chi2_red"]]
                  .head(5).to_string(index=False))
            print("\n  Worst 5 (largest RMS/σ):")
            print(clean[["file", "host", "n_obs", "median_sigma_ms",
                         "rms_residual_ms", "rms_over_sigma", "chi2_red"]]
                  .tail(5).to_string(index=False))
        return

    if args.file:
        tbl = args.rv_dir / args.file
    elif args.host:
        idx = pd.read_csv("data/rv_index.csv")
        mask = idx["host_in_file"].map(_norm) == _norm(args.host)
        if not mask.any():
            raise SystemExit(f"No RV file for host {args.host!r}")
        tbl = args.rv_dir / idx.loc[mask, "file"].iloc[0]
    else:
        tbl = pick_default_file(args.rv_dir)
        if tbl is None:
            raise SystemExit("No RV files found; run download_rv.py first")

    validate_one(tbl, labels, mode=args.mode, plot=True, save=args.save,
                 simbad_cache=simbad_cache)


if __name__ == "__main__":
    main()