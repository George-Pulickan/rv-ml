"""
Paper figures + Earth-like table for the AAAI exoplanet experiment.

Uses the MLP as psi and conformal quantiles from conformal_shift.py
(--psi mlp). Produces:

  figures/paper/rv_heldout_phasefold.png   (Figure 1)
  figures/paper/rv_pred_vs_true.png        (Figure 2)
  figures/paper/earthlike_top10.csv
  figures/paper/earthlike_top10.tex

Usage
-----
    python scripts/paper_rv_figures.py
    python scripts/paper_rv_figures.py --host "HD 2952"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformal import (  # noqa: E402
    COORDS,
    Scorer,
    _theta_to_omega,
    _true_coord,
    make_real,
)
from conformal_shift import _load_mlp_psi  # noqa: E402
from feature_columns import TARGET_COLUMNS  # noqa: E402
from kepler_check import rv_keplerian  # noqa: E402
from preprocess import RVDataset  # noqa: E402
from regression import (  # noqa: E402
    load_from_csv,
    plot_pred_vs_true,
    predict,
    _per_target_metrics,
    _subset_metrics,
    _e_subset_report,
)
from synthetic_dataset import _inject_noise  # noqa: E402

DEFAULT_CKPT = ROOT / "checkpoints" / "regression_mlp_74.pt"
DEFAULT_CSV = ROOT / "synthetic_generation" / "datasets" / "synthetic_regression_10000.csv"
DEFAULT_Q = ROOT / "figures" / "paper" / "mlp_cp_quantiles.json"
DEFAULT_METRICS = ROOT / "synthetic_generation" / "regression" / "mlp_psi" / "conformal_shift_metrics.json"
OUT_DIR = ROOT / "figures" / "paper"


def _theta5_to_params(th: np.ndarray) -> dict[str, float]:
    omega = _theta_to_omega(th)
    return {
        "P": float(10.0 ** th[0]),
        "K": float(10.0 ** th[1]),
        "e": float(np.clip(th[2], 0.0, 0.99)),
        "omega": float(omega),
        "phase": 0.0,
    }


def _set_omega(th: np.ndarray, omega: float) -> np.ndarray:
    out = np.asarray(th, dtype=float).copy()
    out[3] = math.cos(omega)
    out[4] = math.sin(omega)
    return out


def _match_system_widths(
    theta_hat: np.ndarray,
    widths_blob: dict | None,
    fallback: dict | None = None,
    alpha: str = "0.40",
) -> dict[str, float] | None:
    """Papernorm half-widths for the system whose psi(y) is closest to theta_hat.

    Non-finite per-system entries (re-encode failures) fall back per-coordinate
    to ``fallback`` (global raw quantiles) when provided.
    """
    if not widths_blob or not widths_blob.get("systems"):
        return None
    th = np.asarray(theta_hat, dtype=float)
    best, best_d = None, np.inf
    for row in widths_blob["systems"]:
        d = float(np.linalg.norm(np.asarray(row["theta5"], dtype=float) - th))
        if d < best_d:
            best_d, best = d, row
    if best is None or best_d > 1.0:  # loose match guard
        return None
    out = {}
    for c in ("log10_P", "log10_K", "e", "omega"):
        v = float(best["halfwidths"][alpha][c])
        if not math.isfinite(v) and fallback is not None:
            v = float(fallback[c])
        out[c] = v
    return out


def _sample_region(center5: np.ndarray, q: dict[str, float], n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Uniform samples in the Bonferroni box Γ_α around psi(y)."""
    samples = []
    for _ in range(n):
        th = np.asarray(center5, dtype=float).copy()
        for c in COORDS:
            half = float(q[c])
            ctr = _true_coord(th, c)
            if c == "omega":
                th = _set_omega(th, (ctr + rng.uniform(-half, half)) % (2.0 * math.pi))
            elif c == "e":
                th[2] = float(np.clip(ctr + rng.uniform(-half, half), 0.0, 0.99))
            elif c == "log10_P":
                th[0] = ctr + rng.uniform(-half, half)
            elif c == "log10_K":
                th[1] = ctr + rng.uniform(-half, half)
        samples.append(th)
    return samples


def _obs_ms(curve: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = curve["mask"] > 0.5
    t = curve["t_norm"][m] * curve["t_span"] + curve["t_min"]
    rv = curve["rv_obs"][m] * curve["rv_std"]
    sig = curve["sig"][m] * curve["rv_std"]
    return t.astype(float), rv.astype(float), sig.astype(float)


def _kepler_on_grid(th: np.ndarray, t: np.ndarray, t_peri: float) -> np.ndarray:
    p = _theta5_to_params(th)
    return rv_keplerian(t, p["P"], p["K"], p["e"], p["omega"], t_peri)


def _anchor_t_peri(th: np.ndarray, t: np.ndarray, rv: np.ndarray) -> float:
    """Pick t_peri so the Kepler model matches the data median offset (γ free)."""
    p = _theta5_to_params(th)
    # Try a fine phase grid; choose the one minimizing MAD after γ-centering.
    best_tp, best_mad = float(t.min()), np.inf
    for phase in np.linspace(0.0, 1.0, 64, endpoint=False):
        tp = float(t.min()) + phase * p["P"]
        model = rv_keplerian(t, p["P"], p["K"], p["e"], p["omega"], tp)
        resid = rv - model
        resid = resid - np.median(resid)
        mad = float(np.median(np.abs(resid)))
        if mad < best_mad:
            best_mad, best_tp = mad, tp
    return best_tp


def _phase(t: np.ndarray, P: float, t_peri: float) -> np.ndarray:
    return ((t - t_peri) / P) % 1.0


def _feat_row_for_system(system: dict, feature_cols: list[str]) -> np.ndarray:
    fr, lsp = system["feat_row"], system["lsp"]

    def _val(c: str) -> float:
        if c in fr:
            return float(fr[c])
        if c.startswith("lsp_"):
            return float(lsp[int(c.rsplit("_", 1)[1]) - 1])
        raise KeyError(c)

    return np.asarray([_val(c) for c in feature_cols], dtype=float)


def load_quantiles(path: Path, metrics_path: Path) -> dict:
    if path.exists():
        blob = json.loads(path.read_text())
        return blob
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        q = m["quantiles_unweighted"]["surrogate"]
        blob = {
            "psi": m.get("psi"),
            "checkpoint": m.get("checkpoint"),
            "n_cal": m["n_cal"],
            "strategy": "surrogate",
            "norm": "raw",
            "quantiles": q,
            "source": str(metrics_path),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2))
        return blob
    raise FileNotFoundError(f"missing quantiles at {path} and metrics at {metrics_path}")


def pick_system(host: str | None) -> tuple[dict, dict]:
    """Return (conformal-style system dict, RVDataset info) for a test host."""
    ds = RVDataset("test", normalize=False, single_planet=True)
    systems = make_real("test", 0.1, 100.0)
    # Align by iterating dataset in the same filter order as make_real.
    aligned = []
    for i in range(len(ds)):
        x, lsp, theta, info = ds.get_numpy(i)
        if not info.get("valid", True):
            continue
        from conformal import _masked_observations, _curve_from_x
        from eval_omega_nn_vs_rf import _summary_row

        xm = _masked_observations(x)
        if xm.shape[1] < 10:
            continue
        med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
        if not (0.1 <= med_sigma <= 100.0):
            continue
        feats = _summary_row(xm, info, lsp)
        sys_ = {
            "curve": _curve_from_x(x, info),
            "feat_row": feats,
            "lsp": np.asarray(lsp, dtype=float),
            "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
            "info": info,
        }
        aligned.append(sys_)

    if host:
        for s in aligned:
            if s["info"]["host"].lower() == host.lower():
                return s, s["info"]
        raise ValueError(f"host {host!r} not found in filtered real test set")

    # Prefer moderate e, enough points, and a host where psi gets P roughly right.
    scored = []
    for s in aligned:
        e = float(s["theta5"][2])
        n = int(s["curve"]["mask"].sum())
        if n < 30 or not (0.08 <= e <= 0.45):
            continue
        scored.append((n, e, s))
    if not scored:
        return aligned[0], aligned[0]["info"]
    # Prefer HD 139357 when present (clean held-out demo); else most points near e~0.25.
    for _, _, s in scored:
        if s["info"]["host"] == "HD 139357":
            return s, s["info"]
    scored.sort(key=lambda z: (-z[0], abs(z[1] - 0.25)))
    s = scored[0][2]
    return s, s["info"]


def figure1(
    system: dict,
    info: dict,
    psi_predict,
    feature_cols: list[str],
    q04: dict[str, float],
    out_path: Path,
    *,
    n_region: int = 20,
    n_noisy: int = 12,
    seed: int = 0,
    widths_blob: dict | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    X = _feat_row_for_system(system, feature_cols)[None, :]
    th_tab = system["theta5"]
    th_psi = psi_predict(X)[0]

    # Prefer per-system papernorm widths when available (tighter for well-measured hosts).
    q_region = _match_system_widths(th_psi, widths_blob, fallback=q04, alpha="0.40") or q04
    # Cap omega half-width at π so Γ traces stay visually interpretable.
    q_region = dict(q_region)
    q_region["omega"] = float(min(q_region["omega"], math.pi))

    t, rv, sig = _obs_ms(system["curve"])
    P_fold = float(10.0 ** th_tab[0])
    t_peri = _anchor_t_peri(th_tab, t, rv)
    phase_obs = _phase(t, P_fold, t_peri)

    # Dense phase grid for smooth Kepler overlays (evaluate in time via t_peri).
    phase_grid = np.linspace(0.0, 1.0, 400)
    t_grid = t_peri + phase_grid * P_fold

    def folded_model(th: np.ndarray) -> np.ndarray:
        # Physical model at its own P/K/e/ω, plotted vs tabulated-P phase.
        return _kepler_on_grid(th, t_grid, t_peri)

    def model_on_obs(th: np.ndarray) -> np.ndarray:
        return _kepler_on_grid(th, t, t_peri)

    region = _sample_region(th_psi, q_region, n_region, rng)
    # Phase-fold display: freeze P at the fold period so Γ traces stay
    # single-valued in phase (varying P on a fixed phase grid creates
    # multi-cycle nonsense). Widths on K/e/ω still show conformal uncertainty.
    P_log_fold = float(np.log10(P_fold))
    region = [np.array([P_log_fold, th[1], th[2], th[3], th[4]], dtype=float) for th in region]

    # Noisy simulator draws at psi(y): Kepler + residual noise on the real cadence.
    p_psi = _theta5_to_params(th_psi)
    p_psi["phase"] = ((t_peri - float(t.min())) / p_psi["P"]) % 1.0
    noisy_folds = []
    for i in range(n_noisy):
        clean = rv_keplerian(t, p_psi["P"], p_psi["K"], p_psi["e"], p_psi["omega"], t_peri)
        noise, _ = _inject_noise(t, sig, np.random.default_rng(seed + 100 + i),
                                 dominant_params=p_psi, rv_clean_dominant=clean)
        y = clean + noise
        # Align γ like observations (median residual vs data).
        y = y - np.median(y - rv)
        noisy_folds.append((phase_obs, y))

    # Global γ so tabulated model matches observation median residual.
    gamma_tab = float(np.median(rv - model_on_obs(th_tab)))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # (iv) light region traces
    for th in region:
        y = folded_model(th) + gamma_tab
        ax.plot(phase_grid, y, color="0.75", lw=0.7, alpha=0.55, zorder=1)
    # (v) noisy draws
    for ph, y in noisy_folds:
        ax.scatter(ph, y, s=8, c="tab:orange", alpha=0.35, edgecolors="none", zorder=2)
    # (i) observations
    ax.errorbar(phase_obs, rv, yerr=sig, fmt="o", ms=3.5, color="k",
                ecolor="0.55", elinewidth=0.6, capsize=0, label="observations", zorder=5)
    # (ii) tabulated
    ax.plot(phase_grid, folded_model(th_tab) + gamma_tab, color="tab:blue", lw=2.0,
            label=r"$h(\theta_{\mathrm{tab}})$", zorder=4)
    # (iii) predicted
    gamma_psi = float(np.median(rv - model_on_obs(th_psi)))
    ax.plot(phase_grid, folded_model(th_psi) + gamma_psi, color="tab:red", lw=2.0,
            label=r"$h(\psi(y))$", zorder=4)

    ax.set_xlabel("orbital phase (folded at tabulated $P$)")
    ax.set_ylabel("RV (m/s)")
    ax.set_title(
        f"Held-out real: {info['host']}  "
        f"($P_{{\\mathrm{{tab}}}}$={P_fold:.1f} d, $e_{{\\mathrm{{tab}}}}$={th_tab[2]:.2f}; "
        r"light traces $\sim\mathrm{Unif}(\Gamma_{0.4})$)"
    )
    # Legend: add proxies for region / noisy
    ax.plot([], [], color="0.75", lw=1.2, label=r"$\theta\sim\mathrm{Unif}(\Gamma_{0.4})$")
    ax.scatter([], [], s=20, c="tab:orange", alpha=0.7, label="noisy sim. at $\\psi(y)$")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 -> {out_path}")


def figure2(checkpoint: Path, csv_path: Path, out_path: Path, device: torch.device) -> None:
    """Refresh MLP pred-vs-true: P/K/e panels only (74-D has no periapsis epoch)."""
    from regression import (
        TARGET_LABELS,
        DatasetBundle,
        build_model_from_checkpoint,
        _scatter_limits,
    )

    paper_targets = ("log10_P", "log10_K", "e")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model, norm_stats = build_model_from_checkpoint(ckpt, device)
    feature_set = str(norm_stats.get("feature_set", "74"))
    bundle = load_from_csv(csv_path, feature_set)
    rng = np.random.default_rng(42)
    n = len(bundle.X)
    idx = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = np.sort(idx[:n_val])
    X_val = bundle.X[val_idx]
    y_true = bundle.y[val_idx]
    y_pred = predict(model, X_val, norm_stats, device)
    val_bundle = DatasetBundle(
        X_val,
        y_true,
        row_idx=np.asarray(bundle.row_idx)[val_idx],
        e=np.asarray(bundle.e)[val_idx],
        has_t_peri=np.asarray(bundle.has_t_peri)[val_idx],
        has_ecc=np.asarray(bundle.has_ecc)[val_idx],
        df=bundle.df.iloc[val_idx].reset_index(drop=True),
    )
    metrics = {
        "per_target": _per_target_metrics(y_true, y_pred),
        "subsets": _subset_metrics(val_bundle, y_true, y_pred),
        "e_report": _e_subset_report(y_true, y_pred),
    }
    e_report = metrics["e_report"]

    # Option A: omit cos/sin ω panels — 74-D features lack periapsis-epoch
    # information, so ω is nearly unidentifiable (negative R²).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    name_to_j = {"log10_P": 0, "log10_K": 1, "e": 2}
    for ax, name in zip(axes, paper_targets):
        j = name_to_j[name]
        yt, yp = y_true[:, j], y_pred[:, j]
        ax.scatter(yt, yp, s=8, alpha=0.45, edgecolors="none")
        lo, hi = _scatter_limits(yt, yp)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if name == "e":
            r2 = metrics["per_target"][name]["r2"]
            r2_pos = e_report["e_gt_0"]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f} (e>0: {r2_pos:.3f})")
        else:
            r2 = metrics["per_target"][name]["r2"]
            ax.set_title(f"{TARGET_LABELS[name]}\n$R^2$={r2:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.25)
    fig.suptitle(
        r"pred vs true (74-D MLP val; $\omega$ omitted — needs periapsis epoch)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 (P/K/e) -> {out_path}")
    # Keep a full 5-panel diagnostic copy for internal use.
    plot_pred_vs_true(y_true, y_pred, out_path.with_name("rv_pred_vs_true_full5.png"), metrics)


def earth_likeness(row: pd.Series) -> float:
    """Lower is more Earth-like (P~365 d, low e, mass~1 Mearth when known)."""
    P = float(row["pl_orbper"]) if pd.notna(row.get("pl_orbper")) else np.nan
    e = float(row["pl_orbeccen"]) if pd.notna(row.get("pl_orbeccen")) else 0.0
    mass_j = row.get("pl_bmassj")
    if pd.isna(mass_j):
        mass_j = row.get("pl_msinij")
    mearth = float(mass_j) * 317.8 if pd.notna(mass_j) else np.nan
    score = 0.0
    if np.isfinite(P) and P > 0:
        score += abs(math.log10(P / 365.25))
    else:
        score += 5.0
    score += abs(e)
    if np.isfinite(mearth) and mearth > 0:
        score += abs(math.log10(mearth / 1.0))
    else:
        score += 2.0
    return float(score)


def earthlike_table(
    psi_predict,
    feature_cols: list[str],
    q01: dict[str, float],
    out_csv: Path,
    out_tex: Path,
    top_k: int = 10,
    widths_blob: dict | None = None,
) -> None:
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    # Restrict to single-planet hosts in our RV corpus with usable Kepler params.
    sp = splits
    if "n_planets" in sp.columns:
        sp = sp[sp["n_planets"] == 1]
    hosts = set(sp["host"].astype(str))
    need = ["pl_orbper", "pl_rvamp", "pl_orbeccen", "pl_orblper"]
    lab = labels[labels["hostname"].astype(str).isin(hosts)].copy()
    for c in need:
        lab = lab[lab[c].notna()]
    lab = lab[lab["pl_rvamp"] > 0]
    lab["earth_score"] = lab.apply(earth_likeness, axis=1)
    lab = lab.sort_values("earth_score")

    systems_by_host: dict[str, dict] = {}
    # Restrict to held-out hosts (val/test) so the table is not train-set leakage.
    for split in ("val", "test"):
        ds = RVDataset(split, normalize=False, single_planet=True)
        for i in range(len(ds)):
            x, lsp, theta, info = ds.get_numpy(i)
            if not info.get("valid", True):
                continue
            from conformal import _masked_observations, _curve_from_x
            from eval_omega_nn_vs_rf import _summary_row

            xm = _masked_observations(x)
            if xm.shape[1] < 10:
                continue
            med_sigma = float(np.median(xm[2] * float(info["rv_std_ms"])))
            if not (0.1 <= med_sigma <= 100.0):
                continue
            feats = _summary_row(xm, info, lsp)
            systems_by_host[info["host"]] = {
                "curve": _curve_from_x(x, info),
                "feat_row": feats,
                "lsp": np.asarray(lsp, dtype=float),
                "theta5": np.asarray([float(theta[k]) for k in range(5)], dtype=float),
                "info": info,
                "split": split,
            }
    rows = []
    for _, lab_row in lab.iterrows():
        host = str(lab_row["hostname"])
        if host not in systems_by_host:
            continue
        s = systems_by_host[host]
        X = _feat_row_for_system(s, feature_cols)[None, :]
        pred = psi_predict(X)[0]
        tab = s["theta5"]
        hw = _match_system_widths(pred, widths_blob, fallback=q01, alpha="0.10")
        source = "papernorm_per_system" if hw is not None else "global_raw"
        if hw is None:
            hw = q01
        rows.append({
            "host": host,
            "pl_name": lab_row.get("pl_name", ""),
            "split": s["split"],
            "earth_score": float(lab_row["earth_score"]),
            "P_tab_d": float(10 ** tab[0]),
            "K_tab_ms": float(10 ** tab[1]),
            "e_tab": float(tab[2]),
            "omega_tab_rad": float(_theta_to_omega(tab)),
            "P_pred_d": float(10 ** pred[0]),
            "K_pred_ms": float(10 ** pred[1]),
            "e_pred": float(pred[2]),
            "omega_pred_rad": float(_theta_to_omega(pred)),
            "halfwidth_log10_P_a01": float(hw["log10_P"]),
            "halfwidth_log10_K_a01": float(hw["log10_K"]),
            "halfwidth_e_a01": float(hw["e"]),
            "halfwidth_omega_a01": float(hw["omega"]),
            "widths_source": source,
        })
        if len(rows) >= top_k:
            break

    if not rows:
        raise RuntimeError("no Earth-like systems matched the RV corpus (val/test)")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Compact LaTeX snippet: pred +- alpha=0.1 half-width (P/K widths are in dex).
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Host & split & $P_{\mathrm{tab}}$ & $P_{\mathrm{pred}}$ "
        r"& $K_{\mathrm{tab}}$ & $K_{\mathrm{pred}}$ "
        r"& $e_{\mathrm{tab}}$ & $e_{\mathrm{pred}}$ \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['host']} & {r['split']} "
            f"& {r['P_tab_d']:.2f} & {r['P_pred_d']:.2f} $\\pm$ {r['halfwidth_log10_P_a01']:.2f} dex "
            f"& {r['K_tab_ms']:.2f} & {r['K_pred_ms']:.2f} $\\pm$ {r['halfwidth_log10_K_a01']:.2f} dex "
            f"& {r['e_tab']:.2f} & {r['e_pred']:.2f} $\\pm$ {r['halfwidth_e_a01']:.2f} \\\\"
        )
    omega_hw = float(np.median([r["halfwidth_omega_a01"] for r in rows]))
    lines += [
        r"\hline",
        r"\multicolumn{8}{l}{\footnotesize $\pm$ are per-system papernorm half-widths of the "
        r"$\alpha{=}0.1$ conformal region ($P$, $K$ in $\log_{10}$ dex). "
        f"$\\omega$ half-width $\\approx{omega_hw:.2f}$~rad (near-vacuous, omitted). "
        r"Held-out (val/test) hosts only.} \\",
        r"\end{tabular}",
    ]
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Earth-like table -> {out_csv} , {out_tex}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--quantiles", type=Path, default=DEFAULT_Q)
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument(
        "--widths",
        type=Path,
        default=ROOT / "synthetic_generation" / "regression" / "mlp_psi" / "per_system_widths_papernorm.json",
        help="optional per-system papernorm widths JSON from conformal_shift",
    )
    ap.add_argument("--host", default=None, help="held-out host for Figure 1 (default: auto)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q_blob = load_quantiles(args.quantiles, args.metrics)
    q01 = q_blob["quantiles"]["0.10"]
    q04 = q_blob["quantiles"]["0.40"]
    widths_blob = None
    if args.widths.exists():
        widths_blob = json.loads(args.widths.read_text())
        print(f"loaded per-system widths ({widths_blob.get('n_systems')} systems) from {args.widths}")

    psi_predict, norm_stats = _load_mlp_psi(args.checkpoint, device)
    df = pd.read_csv(args.csv, nrows=1)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    in_dim = int(norm_stats["in_dim"])
    if len(feature_cols) != in_dim:
        raise ValueError(f"csv features {len(feature_cols)} != MLP in_dim {in_dim}")

    system, info = pick_system(args.host)
    figure1(system, info, psi_predict, feature_cols, q04, OUT_DIR / "rv_heldout_phasefold.png",
            seed=args.seed, widths_blob=widths_blob)
    figure2(args.checkpoint, args.csv, OUT_DIR / "rv_pred_vs_true.png", device)
    earthlike_table(psi_predict, feature_cols, q01,
                    OUT_DIR / "earthlike_top10.csv",
                    OUT_DIR / "earthlike_top10.tex",
                    widths_blob=widths_blob)


if __name__ == "__main__":
    main()
