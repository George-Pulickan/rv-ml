# laptop-sync — branch-only payload

**This file exists only on the `laptop-sync` branch. Never merge this branch
into `main`.** It carries the gitignored binaries `main` deliberately omits, so
a fresh machine can pick the project up without re-running training.

Created 2026-07-27 off `main` @ `95519b9`, replacing the stale local
`laptop-sync` (preserved as `laptop-sync-local-20260719`, never pushed).

## Getting set up on the laptop

```bash
git clone https://github.com/George-Pulickan/rv-ml.git
cd rv-ml
git checkout laptop-sync      # code + payload
```

Work on `main`; use this branch only to obtain the binaries.

## What is here

| Path | Size | Notes |
|------|------|-------|
| `checkpoints/` | ~100 MB | Trained model checkpoints, including the pinned ψ the 07-26 numbers were regenerated against. |
| `data/` | ~46 MB | Everything except the pretrain caches — `rv_raw/`, `synthetic/`, `gp_residual_cache/`, `labels.csv`, `residuals.npz`, validation sets. |
| `models/` | ~1 MB | Already on `main`; listed for completeness. |

## What is NOT here, and how to get it

| Missing | Why | How to restore |
|---------|-----|----------------|
| `data/pretrain_cache{,_v2,_v3}.pt` | 5.8 GB — far past GitHub's limits | Regenerate from the code; they are caches, not inputs. |
| `synthetic_generation/datasets/synthetic_lsp_regression_10000.csv` | ~130 MB, regenerable | Regenerate via the synthetic generation scripts. |
| `handover.md` | `main` is public and it discusses an anonymised submission under review — `.gitignore` forbids committing it | Private gist. Same URL the 07-27 sync used. |
| `Exoplanets.txt`, paper PDFs | Same reason — anonymised submission under review | Not synced by design. Fetch from Overleaf. |
| `.claude/settings.local.json` | Machine-local config | Recreate as needed. |

## Refreshing this branch later

Rebuild it from `main` rather than merging into it, so it never drifts:

```bash
git checkout main && git pull
git branch -m laptop-sync laptop-sync-old-$(date +%Y%m%d)
git checkout -b laptop-sync
git add -f checkpoints/ 
find data -type f -not -name 'pretrain_cache*.pt' \
  -not -name 'synthetic_lsp_regression_10000.csv' -print0 | xargs -0 git add -f
git reset HEAD -- '*.pdf' '*__pycache__*' '*.pyc'
```

Then re-run the sensitive-file check before pushing:

```bash
git diff --cached --name-only | grep -iE \
  "handover|meeting_|Exoplanets\.txt|\.pdf$|settings\.local|reply_draft"
```

It must print nothing.
