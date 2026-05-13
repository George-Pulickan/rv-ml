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
    tperi_known: bool = True   # False if t_peri is a placeholder to be fit


def build_planet(row: pd.Series) -> tuple[Planet | None, str]:
    """Convert one row of the labels DataFrame into a Planet.

    Returns (Planet, "") on success or (None, reason) on failure.

    Failure reasons: 'no_period', 'no_K_and_no_msini'.

    Planets that lack both pl_orbtper and pl_tranmid are still returned
    (with tperi_known=False); validate_one only includes them when
    fit_tperi=True, since otherwise t_peri would be a placeholder.

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

    t_peri_raw = row.get("pl_orbtper")
    tperi_known = True
    if pd.isna(t_peri_raw):
        t_c = row.get("pl_tranmid")
        if pd.notna(t_c):
            # Convert T_c -> T_peri via the eccentric anomaly at conjunction.
            f_c = np.pi / 2.0 - omega
            E_c = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(f_c / 2.0),
                                    np.sqrt(1.0 + e) * np.cos(f_c / 2.0))
            M_c = E_c - e * np.sin(E_c)
            t_peri = float(t_c) - float(P) * M_c / (2.0 * np.pi)
        else:
            # No timing info in the catalog. Keep the planet but flag it;
            # least_squares_refit will grid-search T_peri from the data.
            t_peri = 0.0  # placeholder
            tperi_known = False
    else:
        t_peri = float(t_peri_raw)

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
        tperi_known=tperi_known,
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


def _eval_trend(t: np.ndarray, t_ref: float, coefs: list[float]) -> np.ndarray:
    """Evaluate γ̇(t-t_ref) + γ̈(t-t_ref)² + … for the given coefficients."""
    out = np.zeros_like(t, dtype=float)
    for k, c in enumerate(coefs, start=1):
        out = out + c * (t - t_ref) ** k
    return out


def least_squares_refit(planets: list[Planet], t: np.ndarray, rv: np.ndarray,
                         err: np.ndarray, fit_tperi: bool = False,
                         trend_order: int = 0, auto_sign: bool = False,
                         random_init: bool = False,
                         rng: np.random.Generator | None = None,
                         ) -> tuple[list[Planet], float, list[float], tuple[int, ...], float]:
    """
    Locally refit *nuisance* parameters via Levenberg-Marquardt while keeping
    the physical orbital parameters (P, K, e, ω) at their catalog values.

    Free parameters:
      γ                       always
      γ̇, γ̈, …                 if trend_order >= 1, 2, …
      T_peri for each planet  if fit_tperi=True

    With `auto_sign=True`, the LM solver runs for each 2ⁿ combination of
    ω flips and the best result is returned.

    With `random_init=True`, T_peri for every planet is replaced with a
    uniform sample over [t_min, t_min + P] before optimization, ignoring
    catalog values and disabling the grid search. This is a diagnostic for
    measuring how much the catalog T_peri values actually constrain the
    fit — if LM converges to the same answer regardless of init, the
    catalog isn't providing useful prior information.

    Returns (refit_planets, gamma, trend_coefs, omega_flips, t_ref).
    """
    import dataclasses
    from itertools import product
    from scipy.optimize import least_squares

    n = len(planets)
    t_ref = float(t.mean())
    w = 1.0 / np.maximum(err, 1e-6)
    flip_grid = (list(product((0, 1), repeat=n))
                  if auto_sign and 0 < n <= 8 else [tuple([0] * n)])
    rng = rng or np.random.default_rng()

    def gridsearch_tperi(target_planet, other_planets):
        """For a planet with unknown T_peri, find the best starting value by
        evaluating the residuals on a coarse grid spanning one full period."""
        v_other = evaluate_model(other_planets, t) if other_planets else np.zeros_like(t)
        grid_n = 40
        grid = np.linspace(t.min(), t.min() + target_planet.P, grid_n, endpoint=False)
        best_tp = float(target_planet.t_peri)
        best_rms = float("inf")
        for tp in grid:
            v_test = v_other + rv_keplerian(t, target_planet.P, target_planet.K,
                                             target_planet.e, target_planet.omega, tp)
            gamma_test = float(np.sum((rv - v_test) * w * w) / np.sum(w * w))
            rms = float(np.sqrt(np.mean((rv - v_test - gamma_test) ** 2)))
            if rms < best_rms:
                best_rms = rms
                best_tp = float(tp)
        return best_tp

    best = None
    best_rms = float("inf")
    for flips in flip_grid:
        flipped = [dataclasses.replace(p, omega=p.omega + (np.pi if f else 0.0))
                    for p, f in zip(planets, flips)]
        if random_init:
            # Replace every T_peri with a uniform draw over [t_min, t_min+P).
            # Catalog values are ignored; grid search is skipped.
            flipped = [dataclasses.replace(
                          p,
                          t_peri=float(rng.uniform(t.min(), t.min() + p.P)),
                          tperi_known=True)
                       for p in flipped]
        else:
            # Seed any unknown-T_peri planets via a one-period grid search so LM
            # has a reasonable initial value instead of an arbitrary placeholder.
            for i, p in enumerate(flipped):
                if not p.tperi_known:
                    others = [q for j, q in enumerate(flipped) if j != i]
                    tp_init = gridsearch_tperi(p, others)
                    flipped[i] = dataclasses.replace(p, t_peri=tp_init, tperi_known=True)

        # Initial γ from inverse-variance LS with no trend, no T_peri shift
        v0 = evaluate_model(flipped, t)
        gamma0 = float(np.sum((rv - v0) / np.maximum(err, 1e-6) ** 2)
                        / np.sum(1.0 / np.maximum(err, 1e-6) ** 2))
        p0 = [gamma0] + [0.0] * trend_order
        if fit_tperi:
            p0 += [p.t_peri for p in flipped]
        p0 = np.array(p0, dtype=float)

        def make_model(params):
            i = 1 + trend_order
            v_planet = np.zeros_like(t, dtype=float)
            for j, pl in enumerate(flipped):
                tp = params[i + j] if fit_tperi else pl.t_peri
                v_planet = v_planet + rv_keplerian(t, pl.P, pl.K, pl.e, pl.omega, tp)
            trend = _eval_trend(t, t_ref, list(params[1:1 + trend_order]))
            return params[0] + trend + v_planet

        def residuals_w(params):
            return (rv - make_model(params)) * w

        try:
            res = least_squares(residuals_w, p0, method="lm", max_nfev=200)
        except Exception:  # noqa: BLE001 — degenerate fit, skip
            continue
        rms_here = float(np.sqrt(np.mean((rv - make_model(res.x)) ** 2)))
        if rms_here < best_rms:
            new_planets = list(flipped)
            if fit_tperi:
                istart = 1 + trend_order
                new_planets = [dataclasses.replace(p, t_peri=float(tp))
                                for p, tp in zip(flipped, res.x[istart:])]
            best = (new_planets, float(res.x[0]),
                     list(map(float, res.x[1:1 + trend_order])), flips)
            best_rms = rms_here

    if best is None:
        return planets, 0.0, [0.0] * trend_order, tuple([0] * n), t_ref
    new_planets, gamma, trend_coefs, flips = best
    return new_planets, gamma, trend_coefs, flips, t_ref


def _gamma_for(planets: list[Planet], t: np.ndarray, rv: np.ndarray,
               err: np.ndarray, mode: str) -> tuple[float, np.ndarray]:
    """Compute γ and the full model V_model+γ for a given planet list."""
    v = evaluate_model(planets, t)
    if mode == "anchor":
        gamma = float(rv[0] - v[0])
    elif mode == "fit":
        w = 1.0 / np.maximum(err, 1e-6) ** 2
        gamma = float(np.sum(w * (rv - v)) / np.sum(w))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return gamma, v


def auto_sign_planets(planets: list[Planet], t: np.ndarray, rv: np.ndarray,
                      err: np.ndarray, mode: str = "anchor",
                      ) -> tuple[list[Planet], tuple[int, ...], float]:
    """
    Try every combination of ω and ω+π across the planets and return the
    combination with the smallest residual RMS, the corresponding flip
    pattern, and the resulting γ.

    Astrophysical motivation: the RV formula has the symmetry ω → ω+π
    ⇔ K → −K, so a catalog entry that uses the opposite convention from
    ours produces a perfectly inverted model. With n planets we have 2ⁿ
    combinations, fast for any realistic system (we cap at n ≤ 8).
    """
    import dataclasses
    from itertools import product
    n = len(planets)
    if n == 0 or n > 8:
        gamma, _ = _gamma_for(planets, t, rv, err, mode)
        return planets, tuple([0] * n), gamma

    best = (planets, tuple([0] * n), 0.0, float("inf"))
    for flips in product((0, 1), repeat=n):
        cand = [dataclasses.replace(p, omega=p.omega + (np.pi if f else 0.0))
                for p, f in zip(planets, flips)]
        gamma, v = _gamma_for(cand, t, rv, err, mode)
        rms = float(np.sqrt(np.mean((rv - (v + gamma)) ** 2)))
        if rms < best[3]:
            best = (cand, flips, gamma, rms)
    return best[0], best[1], best[2]


def validate_one(tbl_path: Path, labels: pd.DataFrame, mode: str = "anchor",
                 plot: bool = True, save: Path | None = None,
                 verbose: bool = True,
                 simbad_cache: dict[str, list[str]] | None = None,
                 return_residuals: bool = False,
                 auto_sign: bool = False,
                 fit_tperi: bool = False,
                 trend_order: int = 0,
                 random_init: bool = False,
                 random_seed: int = 0) -> dict:
    """
    Run the full validation on one RV file. Always returns a dict with a
    'status' field; 'ok' means metrics were computed, anything else means
    the file was skipped for the given reason.

    If `simbad_cache` is supplied, host-name lookups fall back to SIMBAD
    aliases when the direct identifier match fails.

    If `auto_sign=True`, for each planet we try both ω and ω+π and pick
    the combination minimizing residual RMS.

    If `fit_tperi=True` or `trend_order>0`, a constrained least-squares
    refit is run: physical orbital parameters (P, K, e, ω) are held at
    catalog values, but T_peri per planet (if fit_tperi) and polynomial
    trend coefficients (γ̇, γ̈, ... up to order `trend_order`) are
    optimized. Diagnostic for separating phase/offset/drift issues from
    real catalog disagreements.

    If `return_residuals=True`, the dict additionally contains 'residuals',
    'times', and 'sigmas' arrays (only for status='ok').
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
    all_planets = [p for p, _ in built if p is not None]
    # Planets without a catalog T_peri are only usable when fit_tperi=True,
    # because their t_peri is a placeholder that must be inferred from data.
    if fit_tperi:
        planets = all_planets
    else:
        planets = [p for p in all_planets if p.tperi_known]
    n_recovered = sum(1 for p in planets if not p.tperi_known)

    if not planets:
        reasons = [r for _, r in built if r]
        # If we have *any* built planets that just lack T_peri, report that;
        # else fall back to whatever other reason dominated.
        any_no_tperi = any((p is not None) and (not p.tperi_known)
                            for p, _ in built)
        if any_no_tperi and not fit_tperi:
            dominant = "no_tperi_use_--fit-tperi"
        else:
            dominant = max(set(reasons), key=reasons.count) if reasons else "unknown"
        if verbose: print(f"[{tbl_path.name}] {host}: no usable planets ({dominant})")
        return {**base, "status": f"no_planets:{dominant}", "host": host}

    # Catalog values (pre-refit) — record initial T_peri to report any shifts
    initial_tperi = [p.t_peri for p in planets]

    do_refit = fit_tperi or trend_order > 0 or random_init
    trend_coefs: list[float] = []
    t_ref = float(t.mean())
    if do_refit:
        rng = np.random.default_rng(random_seed)
        planets, gamma, trend_coefs, omega_flips, t_ref = least_squares_refit(
            planets, t, rv, err,
            fit_tperi=fit_tperi or random_init,
            trend_order=trend_order, auto_sign=auto_sign,
            random_init=random_init, rng=rng,
        )
        v_model = evaluate_model(planets, t) + _eval_trend(t, t_ref, trend_coefs)
    elif auto_sign:
        planets, omega_flips, gamma = auto_sign_planets(planets, t, rv, err, mode)
        v_model = evaluate_model(planets, t)
    else:
        omega_flips = tuple([0] * len(planets))
        gamma, v_model = _gamma_for(planets, t, rv, err, mode)

    residuals = rv - (v_model + gamma)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    median_err = float(np.median(err))
    chi2_red = float(np.mean((residuals / np.maximum(err, 1e-6)) ** 2))

    if verbose:
        print(f"\n{'=' * 70}")
        rec = f"  [{n_recovered} T_peri-recovered]" if n_recovered else ""
        print(f"{tbl_path.name}  →  host: {host}  ({len(planets)} planet(s)){rec}")
        print(f"{'=' * 70}")
        for p, flip, tp0 in zip(planets, omega_flips, initial_tperi):
            tag = "  [ω-flipped]" if flip else ""
            dtp = p.t_peri - tp0
            shift = f"   ΔT_peri={dtp:+.3f}d" if abs(dtp) > 1e-6 else ""
            print(f"  {p.name:20s}  P={p.P:>10.4f} d   K={p.K:>7.2f} m/s   "
                  f"e={p.e:.3f}   ω={np.degrees(p.omega):6.1f}°   "
                  f"({p.K_source}){tag}{shift}")
        mode_label = ("LS-refit" if do_refit else f"γ-mode={mode}")
        if random_init:
            mode_label += " [random-init]"
        trend_str = ""
        if trend_coefs:
            units = ("m/s/d", "m/s/d²", "m/s/d³")
            parts = [f"{c:+.3g} {u}" for c, u in zip(trend_coefs, units)]
            trend_str = f"   trend=({', '.join(parts)})"
        print(f"  N_obs={len(t)}   baseline={t.max() - t.min():.0f} d   "
              f"{mode_label}   γ={gamma:+.2f} m/s{trend_str}"
              + (f"   ω-flips={omega_flips}" if any(omega_flips) else ""))
        print(f"  RMS(residual) = {rms:.2f} m/s   "
              f"median σ_obs = {median_err:.2f} m/s   "
              f"RMS/σ = {rms / median_err:.2f}   "
              f"χ²_red = {chi2_red:.2f}")

    if plot:
        plot_validation(t, rv, err, v_model + gamma, residuals, host, planets,
                        rms, median_err, mode, save, gamma=gamma,
                        trend_coefs=trend_coefs, t_ref=t_ref)

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
        "omega_flips": "".join(str(f) for f in omega_flips),
        "n_tperi_recovered": n_recovered,
        # T_peri shifts are reported modulo one period, mapped into (-P/2, P/2].
        # Raw differences would be cycle-wrap contaminated since T_peri+kP is
        # the same orbital phase for any integer k.
        "tperi_shifts_d": ",".join(
            f"{(((p.t_peri - tp0) + p.P / 2.0) % p.P) - p.P / 2.0:+.4f}"
            for p, tp0 in zip(planets, initial_tperi)
        ),
        "periods_d": ",".join(f"{p.P:.4f}" for p in planets),
        "trend_coefs": ",".join(f"{c:.6g}" for c in trend_coefs),
        **({"residuals": residuals, "times": t, "sigmas": err}
           if return_residuals else {}),
    }


def plot_validation(t, rv, err, model, residuals, host, planets,
                    rms, median_err, mode, save, gamma=0.0,
                    trend_coefs=None, t_ref=0.0):
    """
    Three-panel layout: full time series (top), phase-folded view on the
    shortest-period planet (middle, only if multiple cycles are observed),
    and residuals vs time (bottom).
    """
    trend_coefs = trend_coefs or []
    # Decide whether to include a phase-folded subplot
    shortest = min(planets, key=lambda p: p.P)
    n_cycles = (t.max() - t.min()) / shortest.P
    show_phase = n_cycles >= 3

    if show_phase:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                                 gridspec_kw={"height_ratios": [1.8, 1.8, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                                 gridspec_kw={"height_ratios": [2.2, 1]})

    # ---- (a) full time series ----
    span = t.max() - t.min()
    t_fine = np.linspace(t.min() - 0.02 * span, t.max() + 0.02 * span, 4000)
    trend_fine = _eval_trend(t_fine, t_ref, trend_coefs)
    v_fine = evaluate_model(planets, t_fine) + gamma + trend_fine

    label = f"Kepler model — {len(planets)} planet(s)"
    if trend_coefs:
        label += f" + trend (order {len(trend_coefs)})"
    axes[0].plot(t_fine, v_fine, "C0-", lw=1.0, alpha=0.7, label=label)
    if trend_coefs:
        axes[0].plot(t_fine, gamma + trend_fine, "C3--", lw=1.0, alpha=0.7,
                     label="γ + trend (planet signal removed)")
    axes[0].errorbar(t, rv, yerr=err, fmt="ko", ms=3.5, capsize=2, lw=0.7,
                     label="RV data")
    axes[0].set_ylabel("RV  [m/s]")
    title_mode = "LS-refit" if trend_coefs or any(p.t_peri != tp for p, tp in
                                                    zip(planets, [p.t_peri for p in planets])) else f"γ: {mode}"
    axes[0].set_title(
        f"{host}   |   RMS={rms:.1f} m/s, median σ={median_err:.1f} m/s, "
        f"ratio={rms / median_err:.2f}   |   {title_mode}"
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    # ---- (b) phase-folded on the shortest-period planet ----
    if show_phase:
        other = [p for p in planets if p is not shortest]
        v_other_at_t = evaluate_model(other, t) if other else np.zeros_like(t)
        trend_at_t = _eval_trend(t, t_ref, trend_coefs)
        rv_iso = rv - gamma - v_other_at_t - trend_at_t
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
                         lw=0.7, label="data (other planets & trend removed)")
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
    p.add_argument("--auto-sign", action="store_true",
                   help="for each planet, try ω and ω+π and keep the better fit")
    p.add_argument("--fit-tperi", action="store_true",
                   help="LS-refit T_peri per planet (fixes phase offsets)")
    p.add_argument("--trend", type=int, default=0, metavar="N", choices=(0, 1, 2),
                   help="add polynomial trend of order N=1 (linear) or 2 (quadratic)")
    p.add_argument("--random-init", action="store_true",
                   help="initialize T_peri uniformly over [0, P] instead of catalog "
                        "(diagnostic for measuring catalog informativeness)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed when --random-init is used")
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
                             simbad_cache=simbad_cache, auto_sign=args.auto_sign,
                             fit_tperi=args.fit_tperi, trend_order=args.trend,
                             random_init=args.random_init, random_seed=args.seed)
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
                 simbad_cache=simbad_cache, auto_sign=args.auto_sign,
                 fit_tperi=args.fit_tperi, trend_order=args.trend,
                 random_init=args.random_init, random_seed=args.seed)


if __name__ == "__main__":
    main()