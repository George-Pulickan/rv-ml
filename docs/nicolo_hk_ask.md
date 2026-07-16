# Message for Nicolò (h/k + reporting)

Hi Nicolò,

Quick ask while we regenerate Gate C / period-tolerance numbers.

**1. h/k targets.** We’d like to switch the shape heads from `(e, cos ω, sin ω)` to `(k, h) = (e cos ω, e sin ω)`, then decode `e = √(h²+k²)` and `ω = atan2(h, k)` for plots/metrics. Motivation: removes the ω-undefined-at-e≈0 degeneracy, kills the MSE hedging stripe at ω=0, and matches the usual exoplanet parameterization. OK to implement behind `--targets hk` (default stays current until you confirm)?

**2. Stratified headlines.** Plan to quote ω MAE / R² by e bands (0.1–0.3, 0.3–0.5, >0.5) and SNR tertiles, not only the full-sample aggregate. Fine for the paper story?

**3. Prior H.** We’ll leave the discrete 30-bin eccentricity prior alone for now (no KDE/Beta remesh) unless you and George want that changed.

Thanks —
Daksh
