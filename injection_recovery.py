"""
injection_recovery.py — Injection-recovery benchmark

Two modes:

  --mode decoder  (default, no model needed)
      Validates the synthetic data generator + Kepler decoder via classical
      LS fitting.  For each grid cell (P × SNR), generates N synthetic RV
      curves and recovers parameters with scipy L-BFGS-B.  Produces the
      baseline recovery curves the paper reviewers will expect.

      Also implements Nicolò's single-parameter quality check: for each
      realisation, optimise *one* parameter at a time while holding the
      rest at their true values.  The prediction error floor tells us how
      much information each parameter leaves in the data.

  --mode encoder  (requires --checkpoint)
      Same grid, but uses the trained RVEncoder to recover parameters.
      Results are written alongside the decoder baseline so the two can be
      compared directly.

Grid axes
---------
  P_grid   : [3, 10, 30, 100, 300, 1000] d
  snr_grid : [0.5, 1, 2, 5, 10, 20]       K / median(σ_obs)

Per cell
--------
  N = 50 realisations (--n-real to override)
  For each: generate → recover → compute ΔP/P, ΔK/K, Δe
  Period aliases (|log2(P_rec/P_true)| > 0.4) are flagged separately.

Outputs
-------
  data/ir_decoder.csv          — per-realisaton recovery errors (decoder mode)
  data/ir_encoder.csv          — same for encoder mode
  figures/ir_decoder_grid.png  — 6×6 grid heat-map (decoder)
  figures/ir_encoder_grid.png  — same for encoder

Usage
-----
    python injection_recovery.py                          # decoder validation
    python injection_recovery.py --n-real 20 --jobs 4    # faster debug run
    python injection_recovery.py --mode encoder --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kepler_check import rv_keplerian as rv_np
from synthetic_dataset import (
    _inject_noise,
    _sample_sigma,
    _sample_time_grid,
)

# Grid defaults
P_GRID   = np.array([3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
SNR_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

ALIAS_THRESH = 0.4   # |log2(P_rec/P_true)| > this → period alias


# ---------------------------------------------------------------------------
# Classical LS recovery
# ---------------------------------------------------------------------------

def _chi2(params: np.ndarray, t: np.ndarray, rv: np.ndarray,
           sigma: np.ndarray) -> float:
    """χ² / N for a Kepler + γ model (no e < 0 or e > 0.99 guard needed:
    scipy bounds handle it)."""
    log10_P, log10_K, e, cos_w, sin_w = params
    P     = 10.0 ** log10_P
    K     = 10.0 ** log10_K
    omega = np.arctan2(sin_w, cos_w)
    # Phase grid: enough points to sample < 0.05 P spacing in phase.
    # With n=40, each step = P/40; at P=1000 d that is 25 d, appropriate.
    n_tp    = 40
    tp_grid = np.linspace(t.min(), t.min() + P, n_tp + 1)[:-1]
    best_chi2 = np.inf
    best_tp   = tp_grid[0]
    for tp in tp_grid:
        v  = rv_np(t, P, K, e, omega, tp)
        gm = float(np.sum((rv - v) / np.maximum(sigma, 1e-10) ** 2)
                   / np.sum(1.0 / np.maximum(sigma, 1e-10) ** 2))
        r  = rv - v - gm
        c  = float(np.mean((r / np.maximum(sigma, 1e-10)) ** 2))
        if c < best_chi2:
            best_chi2 = c
            best_tp   = tp
    v  = rv_np(t, P, K, e, omega, best_tp)
    gm = float(np.sum((rv - v) / np.maximum(sigma, 1e-10) ** 2)
               / np.sum(1.0 / np.maximum(sigma, 1e-10) ** 2))
    r  = rv - v - gm
    return float(np.mean((r / np.maximum(sigma, 1e-10)) ** 2))


def _recover_classical(
    t: np.ndarray,
    rv: np.ndarray,
    sigma: np.ndarray,
    P_true: float,
    K_true: float,
    e_true: float,
    omega_true: float,
    n_restarts: int = 3,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Recover (P, K, e, ω) via L-BFGS-B with multiple restarts.

    One restart seeds near the true values; the rest are random draws from
    the prior.  Returns the best-χ² result.
    """
    rng = rng or np.random.default_rng()
    bounds = [
        (np.log10(0.5), np.log10(5000.0)),   # log10_P
        (np.log10(0.1), np.log10(2000.0)),   # log10_K
        (0.0, 0.99),                          # e
        (-2.0, 2.0),                          # cos_ω (not unit-constrained; atan2 handles it)
        (-2.0, 2.0),                          # sin_ω
    ]
    true_p0 = np.array([
        np.log10(P_true), np.log10(K_true), e_true,
        np.cos(omega_true), np.sin(omega_true),
    ])

    best_val = np.inf
    best_x   = true_p0.copy()

    for k in range(n_restarts):
        if k == 0:
            # Seed #0: start near truth (perturbation in log-scale for P, K)
            p0 = true_p0 + rng.normal(0, 0.05, size=5)
        else:
            # Random restarts: e uniform on [0, 0.99] (full prior range)
            p0 = np.array([
                rng.uniform(*bounds[0]),
                rng.uniform(*bounds[1]),
                rng.uniform(0.0, 0.99),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
            ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_chi2, p0, args=(t, rv, sigma),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})

        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x

    P_rec = 10.0 ** best_x[0]
    K_rec = 10.0 ** best_x[1]
    e_rec = float(np.clip(best_x[2], 0.0, 0.99))

    dP_rel = abs(P_rec - P_true) / P_true
    dK_rel = abs(K_rec - K_true) / K_true
    de     = abs(e_rec - e_true)
    alias  = int(abs(np.log2(P_rec / P_true)) > ALIAS_THRESH)

    return {
        "P_true": P_true, "K_true": K_true, "e_true": e_true,
        "P_rec": P_rec,   "K_rec": K_rec,   "e_rec": e_rec,
        "dP_rel": dP_rel, "dK_rel": dK_rel, "de": de,
        "chi2": best_val, "alias": alias,
    }


# ---------------------------------------------------------------------------
# Single-parameter optimisation (Nicolò's quality check)
# ---------------------------------------------------------------------------

def _recover_single_param(
    t: np.ndarray,
    rv: np.ndarray,
    sigma: np.ndarray,
    P_true: float,
    K_true: float,
    e_true: float,
    omega_true: float,
    param: str,          # "P", "K", or "e"
    n_grid: int = 64,
) -> tuple[float, float]:
    """
    Recover one parameter via 1-D grid search while holding all others fixed.

    Returns (best_recovered_value, residual_rms).
    """
    if param == "P":
        grid = np.exp(np.linspace(np.log(P_true * 0.1), np.log(P_true * 10.0), n_grid))
    elif param == "K":
        grid = np.exp(np.linspace(np.log(K_true * 0.1), np.log(K_true * 10.0), n_grid))
    else:  # "e"
        grid = np.linspace(0.0, 0.99, n_grid)

    best_val = np.inf
    best_rec = grid[0]

    for val in grid:
        P     = val     if param == "P" else P_true
        K     = val     if param == "K" else K_true
        e     = val     if param == "e" else e_true
        omega = omega_true

        # Phase grid: 40 points per period; at least 40.
        n_tp = 40
        for phase in np.linspace(0, 1, n_tp, endpoint=False):
            tp = t.min() + phase * P
            v  = rv_np(t, P, K, e, omega, tp)
            gm = float(np.sum((rv - v) / np.maximum(sigma, 1e-10) ** 2)
                       / np.sum(1.0 / np.maximum(sigma, 1e-10) ** 2))
            r  = rv - v - gm
            c  = float(np.mean((r / np.maximum(sigma, 1e-10)) ** 2))
            if c < best_val:
                best_val = c
                best_rec = val

    # Prediction error: RMS at the recovered value (inverse-variance gamma)
    P     = best_rec if param == "P" else P_true
    K     = best_rec if param == "K" else K_true
    e     = best_rec if param == "e" else e_true
    n_tp  = 40
    best_rms = np.inf
    for phase in np.linspace(0, 1, n_tp, endpoint=False):
        tp = t.min() + phase * P
        v  = rv_np(t, P, K, e, omega_true, tp)
        gm = float(np.sum((rv - v) / np.maximum(sigma, 1e-10) ** 2)
                   / np.sum(1.0 / np.maximum(sigma, 1e-10) ** 2))
        rms = float(np.sqrt(np.mean((rv - v - gm) ** 2)))
        if rms < best_rms:
            best_rms = rms

    return float(best_rec), best_rms


# ---------------------------------------------------------------------------
# Encoder recovery
# ---------------------------------------------------------------------------

def _recover_encoder(
    x: np.ndarray,
    lsp: np.ndarray,
    encoder,
    stats: dict,
    t: np.ndarray,
    rv: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    """
    Recover orbital parameters from (x, lsp) using the trained encoder,
    then refit T_peri and γ analytically so the comparison with the
    classical LS baseline is fair.

    Returns dict with P_rec, K_rec, e_rec, chi2.
    """
    import torch
    from models.encoder import un_normalise_theta

    x_t   = torch.from_numpy(x).unsqueeze(0).float()
    lsp_t = torch.from_numpy(lsp).unsqueeze(0).float()
    with torch.no_grad():
        theta_norm = encoder(x_t, lsp_t)
    theta_phys = un_normalise_theta(theta_norm, stats).squeeze(0).numpy()

    log10_P, log10_K, e_raw, cos_w, sin_w = theta_phys
    P_rec = float(10.0 ** log10_P)
    K_rec = float(10.0 ** log10_K)
    e_rec = float(np.clip(e_raw, 0.0, 0.99))
    omega_rec = float(np.arctan2(sin_w, cos_w))

    # Refit T_peri analytically (same as classical baseline)
    n_tp = 40
    best_chi2 = np.inf
    best_tp   = t.min()
    for phase in np.linspace(0, 1, n_tp, endpoint=False):
        tp = t.min() + phase * P_rec
        v  = rv_np(t, P_rec, K_rec, e_rec, omega_rec, tp)
        gm = float(np.sum((rv - v) / np.maximum(sigma, 1e-10) ** 2)
                   / np.sum(1.0 / np.maximum(sigma, 1e-10) ** 2))
        r  = rv - v - gm
        c  = float(np.mean((r / np.maximum(sigma, 1e-10)) ** 2))
        if c < best_chi2:
            best_chi2 = c
            best_tp   = tp

    return {
        "P_rec": P_rec, "K_rec": K_rec, "e_rec": e_rec,
        "chi2": best_chi2,
    }


# ---------------------------------------------------------------------------
# Single grid-cell runner
# ---------------------------------------------------------------------------

def _run_cell(
    P: float,
    snr: float,
    n_real: int,
    mode: str,
    rng: np.random.Generator,
    encoder=None,
    stats: dict | None = None,
    n_restarts: int = 3,
) -> list[dict]:
    """Run N realisations for one (P, SNR) grid cell."""
    rows = []
    for _ in range(n_real):
        e     = rng.beta(2, 5) * 0.99
        omega = rng.uniform(0, 2 * np.pi)
        t     = _sample_time_grid(rng)
        sigma = _sample_sigma(rng, len(t))

        # Set K so that K / median(sigma) ≈ SNR
        K      = snr * float(np.median(sigma))
        t_peri = float(t.min()) + rng.uniform() * P

        rv_clean = rv_np(t, P, K, e, omega, t_peri)
        noise    = _inject_noise(t, sigma, rng)
        rv_obs   = rv_clean + noise

        row = {"P_true": P, "snr": snr, "K_true": K,
               "e_true": e, "omega_true": np.degrees(omega)}

        if mode == "decoder":
            rec = _recover_classical(t, rv_obs, sigma, P, K, e, omega,
                                     n_restarts=n_restarts, rng=rng)
            row.update(rec)
        else:
            # encoder mode — need (4, 256) tensor + LSP periodogram
            from preprocess import T_MAX, compute_lsp
            n_real_obs = len(t)
            t_min  = float(t.min())
            t_span = float(t.max() - t.min()) if n_real_obs > 1 else 1.0
            rv_med = float(np.median(rv_obs))
            rv_std = max(float(np.std(rv_obs, ddof=1)) if n_real_obs > 1 else 1.0, 1e-6)

            x = np.zeros((4, T_MAX), dtype=np.float32)
            n = min(n_real_obs, T_MAX)
            x[0, :n] = ((t - t_min) / t_span)[:n]
            x[1, :n] = ((rv_obs - rv_med) / rv_std)[:n]
            x[2, :n] = (sigma / rv_std)[:n]
            x[3, :n] = 1.0

            lsp = compute_lsp(t, rv_obs, sigma)

            enc_rec = _recover_encoder(x, lsp, encoder, stats, t, rv_obs, sigma)
            P_rec = enc_rec["P_rec"]
            K_rec = enc_rec["K_rec"]
            e_rec = enc_rec["e_rec"]
            row.update({
                "P_rec": P_rec, "K_rec": K_rec, "e_rec": e_rec,
                "dP_rel": abs(P_rec - P) / P,
                "dK_rel": abs(K_rec - K) / K,
                "de":     abs(e_rec - e),
                "alias":  int(abs(np.log2(P_rec / P)) > ALIAS_THRESH),
            })

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_grid(df: pd.DataFrame, metric: str, title: str,
               out_path: Path, p_grid: np.ndarray,
               snr_grid: np.ndarray) -> None:
    """Heat-map of median recovery error on (SNR × P) grid."""
    data = np.full((len(snr_grid), len(p_grid)), np.nan)
    for i, snr in enumerate(snr_grid):
        for j, P in enumerate(p_grid):
            sub = df[(np.isclose(df["snr"], snr)) & (np.isclose(df["P_true"], P))]
            if len(sub):
                data[i, j] = sub[metric].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", origin="lower",
                   vmin=0, vmax=np.nanpercentile(data, 90),
                   cmap="RdYlGn_r")
    ax.set_xticks(range(len(p_grid)))
    ax.set_xticklabels([f"{P:.0f}" for P in p_grid])
    ax.set_yticks(range(len(snr_grid)))
    ax.set_yticklabels([f"{s:.1f}" for s in snr_grid])
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("SNR  (K / median σ)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=f"median {metric}")

    # Annotate cells
    for i in range(len(snr_grid)):
        for j in range(len(p_grid)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_alias_rate(df: pd.DataFrame, title: str, out_path: Path,
                     p_grid: np.ndarray, snr_grid: np.ndarray) -> None:
    """Heat-map of period-alias rate."""
    data = np.full((len(snr_grid), len(p_grid)), np.nan)
    for i, snr in enumerate(snr_grid):
        for j, P in enumerate(p_grid):
            sub = df[(np.isclose(df["snr"], snr)) & (np.isclose(df["P_true"], P))]
            if len(sub):
                data[i, j] = sub["alias"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", origin="lower",
                   vmin=0, vmax=1, cmap="Reds")
    ax.set_xticks(range(len(p_grid)))
    ax.set_xticklabels([f"{P:.0f}" for P in p_grid])
    ax.set_yticks(range(len(snr_grid)))
    ax.set_yticklabels([f"{s:.1f}" for s in snr_grid])
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("SNR  (K / median σ)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="alias fraction")
    for i in range(len(snr_grid)):
        for j in range(len(p_grid)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_multi_panel(df: pd.DataFrame, prefix: str,
                      out_dir: Path, p_grid: np.ndarray,
                      snr_grid: np.ndarray) -> None:
    """Three heat-maps (ΔP/P, ΔK/K, Δe) side by side."""
    metrics = [
        ("dP_rel", "median ΔP/P"),
        ("dK_rel", "median ΔK/K"),
        ("de",     "median Δe"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, label) in zip(axes, metrics):
        data = np.full((len(snr_grid), len(p_grid)), np.nan)
        for i, snr in enumerate(snr_grid):
            for j, P in enumerate(p_grid):
                sub = df[(np.isclose(df["snr"], snr)) & (np.isclose(df["P_true"], P))]
                if len(sub):
                    data[i, j] = sub[metric].median()
        vmax = np.nanpercentile(data, 90)
        im = ax.imshow(data, aspect="auto", origin="lower",
                       vmin=0, vmax=vmax, cmap="RdYlGn_r")
        ax.set_xticks(range(len(p_grid)))
        ax.set_xticklabels([f"{P:.0f}" for P in p_grid])
        ax.set_yticks(range(len(snr_grid)))
        ax.set_yticklabels([f"{s:.1f}" for s in snr_grid])
        ax.set_xlabel("Period [d]")
        ax.set_ylabel("SNR")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
        for i in range(len(snr_grid)):
            for j in range(len(p_grid)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

    fig.suptitle(f"{prefix} recovery — {len(df)} realisations", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"{prefix}_grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    mode: str = "decoder",
    n_real: int = 50,
    p_grid: np.ndarray = P_GRID,
    snr_grid: np.ndarray = SNR_GRID,
    checkpoint: Path | None = None,
    jobs: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the injection-recovery benchmark and write outputs.

    Returns the results DataFrame.
    """
    encoder = stats = None
    if mode == "encoder":
        if checkpoint is None or not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Encoder mode requires --checkpoint; got {checkpoint!r}"
            )
        import torch
        from models.encoder import RVEncoder
        stats = json.loads(Path("data/dataset_stats.json").read_text())
        encoder = RVEncoder()
        encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        encoder.eval()
        print(f"Loaded encoder from {checkpoint}")

    total_cells = len(p_grid) * len(snr_grid)
    print(f"\nInjection-recovery ({mode} mode)")
    print(f"  Grid: {len(p_grid)} periods × {len(snr_grid)} SNR values "
          f"= {total_cells} cells × {n_real} realisations "
          f"= {total_cells * n_real} total")

    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []

    cell_idx = 0
    for P in p_grid:
        for snr in snr_grid:
            cell_rng = np.random.default_rng(rng.integers(0, 2**31))
            rows = _run_cell(P, snr, n_real, mode, cell_rng,
                             encoder=encoder, stats=stats)
            all_rows.extend(rows)
            cell_idx += 1
            # Progress indicator every 6 cells (one period row)
            if cell_idx % len(snr_grid) == 0:
                done = cell_idx / total_cells
                print(f"  [{done:5.1%}]  P={P:7.1f} d  done")

    df = pd.DataFrame(all_rows)
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    prefix = f"ir_{mode}"
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    # Summary table
    print("\nMedian recovery errors (all cells):")
    print(f"  ΔP/P  = {df['dP_rel'].median():.3f}")
    print(f"  ΔK/K  = {df['dK_rel'].median():.3f}")
    print(f"  Δe    = {df['de'].median():.3f}")
    if "alias" in df.columns:
        print(f"  alias = {df['alias'].mean():.1%} of realisations")

    # Per-SNR summary (all periods combined)
    print("\nBy SNR (median ΔP/P | ΔK/K | Δe):")
    for snr in snr_grid:
        sub = df[np.isclose(df["snr"], snr)]
        print(f"  SNR={snr:5.1f}  "
              f"ΔP/P={sub['dP_rel'].median():.3f}  "
              f"ΔK/K={sub['dK_rel'].median():.3f}  "
              f"Δe={sub['de'].median():.3f}")

    # Figures
    _plot_multi_panel(df, prefix, fig_dir, p_grid, snr_grid)
    if "alias" in df.columns:
        _plot_alias_rate(df, f"{prefix.replace('ir_', '').capitalize()} alias rate",
                         fig_dir / f"{prefix}_alias.png", p_grid, snr_grid)

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("decoder", "encoder"), default="decoder")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to encoder .pt checkpoint (encoder mode only)")
    p.add_argument("--n-real", type=int, default=50,
                   help="Realisations per grid cell (default 50)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel workers (currently serial; reserved for future use)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-grid", type=float, nargs="+", default=None,
                   help="Override period grid (days)")
    p.add_argument("--snr-grid", type=float, nargs="+", default=None,
                   help="Override SNR grid (K / median σ)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    p_grid   = np.array(args.p_grid)   if args.p_grid   else P_GRID
    snr_grid = np.array(args.snr_grid) if args.snr_grid else SNR_GRID

    run(
        mode=args.mode,
        n_real=args.n_real,
        p_grid=p_grid,
        snr_grid=snr_grid,
        checkpoint=args.checkpoint,
        jobs=args.jobs,
        seed=args.seed,
    )
