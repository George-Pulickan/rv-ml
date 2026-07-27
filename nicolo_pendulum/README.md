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

## Re-syncing

To refresh against upstream, re-clone and copy over the top rather than
merging — there is no shared history with this repo:

```bash
git clone --depth 1 https://github.com/nicoloRHUL/Exoplanets_2026.git /tmp/nicolo
rsync -a --exclude='.git' --exclude='__pycache__' /tmp/nicolo/ nicolo_pendulum/
```

Then update the commit hash and date above.
