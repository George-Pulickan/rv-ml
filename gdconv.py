import numpy as np
from conformal_shift import NoiseProxy, estimate_constants
proxy = NoiseProxy()
for steps in (200, 1000, 3000):
    out = estimate_constants(proxy, 8, 41, gd_steps=steps, gd_lr=0.02)
    lm = out["lambda_min_H"]
    print("gd_steps=%5d  PD_frac=%.2f  lam_min med=%9.3g  min=%9.3g" % (
        steps, out["frac_positive_definite"], lm["median"], lm["min"]))
