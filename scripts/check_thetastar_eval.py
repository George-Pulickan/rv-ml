"""Unit check for the evaluate()/evaluate_union() changes.

Exercises the two things the pipeline run cannot cheaply re-test:
  1. targets= scores against the supplied vector, not s["theta5"]
  2. delta_s= produces the eq (32) band mass, and it behaves monotonically

No simulator, no GP, no corpus: runs in seconds. Not a substitute for the
end-to-end run, but it catches the shape/indexing/arithmetic bugs that would
otherwise surface four hours into one.
"""
import numpy as np

import conformal_shift as cs

COORDS = cs.COORDS
rng = np.random.default_rng(0)

n_cal, n_test = 200, 50
d5 = 5


def _theta5(vals):
    v = np.zeros(d5)
    v[:len(vals)] = vals
    return v


# Calibration scores: one array per coordinate.
cal_scores = {c: np.abs(rng.normal(0, 1, n_cal)) for c in COORDS}
sup = {c: (-10.0, 10.0) for c in COORDS}

# Test systems. theta5 is the default target; theta_hats are the predictions.
systems = [{"theta5": _theta5(rng.normal(0, 1, len(COORDS)))} for _ in range(n_test)]
theta_hats = [_theta5(rng.normal(0, 1, len(COORDS))) for _ in range(n_test)]

# --- 1. targets= actually redirects the evaluation target --------------------
base = cs.evaluate(cal_scores, systems, theta_hats, sup)

# Targets identical to the predictions => every coordinate covered, always.
perfect = [np.array(th, dtype=float) for th in theta_hats]
exact = cs.evaluate(cal_scores, systems, theta_hats, sup, targets=perfect)
for a in exact:
    assert exact[a]["joint_coverage"] == 1.0, (a, exact[a]["joint_coverage"])

# Targets pushed far outside the support => nothing covered.
far = [_theta5([1e6] * len(COORDS)) for _ in range(n_test)]
none_ = cs.evaluate(cal_scores, systems, theta_hats, sup, targets=far)
for a in none_:
    assert none_[a]["joint_coverage"] == 0.0, (a, none_[a]["joint_coverage"])

# And the default path is unchanged by the edit.
again = cs.evaluate(cal_scores, systems, theta_hats, sup, targets=None)
for a in base:
    assert base[a]["joint_coverage"] == again[a]["joint_coverage"]
print("targets= redirects the target, and default path unchanged  OK")

# --- 2. delta_s= produces the eq (32) band mass ------------------------------
no_gap = cs.evaluate(cal_scores, systems, theta_hats, sup)
assert "gap" not in no_gap[next(iter(no_gap))], "gap reported without delta_s"

zero = cs.evaluate(cal_scores, systems, theta_hats, sup,
                   delta_s={c: 0.0 for c in COORDS})
for a in zero:
    for c in COORDS:
        assert zero[a]["gap"][c] >= 0.0
        # A zero-width band can only pick up ties at the quantile itself.
        assert zero[a]["gap"][c] < 0.05, (a, c, zero[a]["gap"][c])

wide = cs.evaluate(cal_scores, systems, theta_hats, sup,
                   delta_s={c: 100.0 for c in COORDS})
for a in wide:
    for c in COORDS:
        # A band wider than the score range captures everything below q,
        # i.e. approximately the nominal level.
        assert wide[a]["gap"][c] > 0.5, (a, c, wide[a]["gap"][c])

mid = cs.evaluate(cal_scores, systems, theta_hats, sup,
                  delta_s={c: 0.3 for c in COORDS})
for a in mid:
    for c in COORDS:
        lo, hi = zero[a]["gap"][c], wide[a]["gap"][c]
        assert lo <= mid[a]["gap"][c] <= hi, (a, c, mid[a]["gap"][c], lo, hi)
print("delta_s= gap is monotone in the band width and absent when unset  OK")

# --- 3. weighting is respected ----------------------------------------------
w = rng.random(n_cal) + 0.1
gw = cs.evaluate(cal_scores, systems, theta_hats, sup,
                 w_cal=w, w_test=np.ones(n_test),
                 delta_s={c: 0.5 for c in COORDS})
for a in gw:
    for c in COORDS:
        assert 0.0 <= gw[a]["gap"][c] <= 1.0
print("gap stays a probability under non-uniform calibration weights  OK")

print("\nALL CHECKS PASSED")
