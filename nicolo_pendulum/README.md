# nicolo_pendulum — upstream snapshot

Nicolò's perturbed-pendulum work, vendored here so it is reachable from one
place alongside the RV code. **This is a snapshot, not a fork to edit.**

- Source: https://github.com/nicoloRHUL/Exoplanets_2026
- Upstream commit: `7db4c622e937cd66891f054e68d884ee0708773b`
  ("RF replaced by KNN, changes in the plots")
- Snapshot taken: 2026-07-27
- Excluded from the copy: `__pycache__/`

This is the toy experiment that currently occupies §4 of the Overleaf draft —
the exoplanet experiment the Abstract and Intro promise is not yet written up
there (see `handover.md`, workstream 2).

## Contents

| Path | What it is |
|------|------------|
| `pendulum_simulation.py` | Main script: integrates the pendulum, builds synthetic + "realistic" trajectory sets, fits length/ratio predictors (KNN, CNN), writes `Data/`. |
| `pendulum_simulation copy.py` | Paper-figure variant. Differs deliberately: axis label `y(t)` rather than `\theta(t)`, plot titles commented out, trajectory labels without ids, and `extract_embedding_features(n_components=40)` rather than `20`. |
| `Data/` | Trajectory CSVs and the current plots. |
| `Old Plots/` | Superseded figures, including the random-forest predictions from before the KNN switch. |

## ⚠️ The committed `Data/` does not match either script

Checked 2026-07-27. `extract_embedding_features` returns
`2 × n_components` Fourier features (real + imaginary parts) plus exactly 8
time-domain features, so the embedded dataset should have:

| script | `n_components` | expected feature columns |
|---|---|---|
| `pendulum_simulation.py` | 20 | 48 |
| `pendulum_simulation copy.py` | 40 | 88 |

`Data/synthetic_trajectories_embedded.csv` has **28** — i.e. it was generated
with `n_components = 10`, which **neither committed script uses**. The committed
data and plots therefore predate both scripts and are **not reproducible from
this snapshot**.

This matters for the paper: §4's figures currently come from this directory, so
any method description quoting 20 or 40 Fourier components would be wrong for
the figures actually shown. Before those figures go in, either regenerate
`Data/` from a named script or get the real provenance from Nicolò — and record
which of the two scripts is authoritative.

## Re-syncing

To refresh against upstream, re-clone and copy over the top rather than
merging — there is no shared history with this repo:

```bash
git clone --depth 1 https://github.com/nicoloRHUL/Exoplanets_2026.git /tmp/nicolo
rsync -a --exclude='.git' --exclude='__pycache__' /tmp/nicolo/ nicolo_pendulum/
```

Then update the commit hash and date above.
