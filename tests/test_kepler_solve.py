"""Unit tests for the differentiable Kepler solver's convergence behaviour.

The solver had no test at all, which is how a float64-era convergence tolerance
survived in a float32 code path for as long as it did: Newton's update stalls at
the dtype's rounding floor (~2.5e-7 in float32), never reaches a hard-coded
1e-10, and the loop therefore ran its full `maxiter` on every call -- 43 of 50
iterations moving E by less than one ulp while still costing a sin/cos pass
forward and 43 layers of autograd graph backward.

These tests pin the three properties that matter: float64 callers are unchanged,
float32 terminates early, and the answer still solves Kepler's equation to the
precision the dtype allows.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.kepler_torch import solve_kepler  # noqa: E402


def _residual(E: torch.Tensor, e: torch.Tensor, M: torch.Tensor) -> float:
    """max |E - e sin E - M| with M wrapped the way the solver wraps it."""
    Mw = torch.remainder(M + torch.pi, 2 * torch.pi) - torch.pi
    return float((E - e * torch.sin(E) - Mw).abs().max())


def _case(dtype: torch.dtype, ecc: float = 0.3, n: int = 512):
    g = torch.Generator().manual_seed(0)
    M = (torch.rand(4, n, generator=g, dtype=dtype) * 12.0) - 6.0
    e = torch.full((4, n), ecc, dtype=dtype)
    return M, e


class TestSolveKeplerTolerance(unittest.TestCase):
    def test_float64_matches_the_historical_tolerance_bitwise(self):
        """float64 behaviour must not move: the dtype-aware floor is capped at
        the 1e-10 the solver always used, so these are the same iteration."""
        M, e = _case(torch.float64)
        self.assertTrue(torch.equal(solve_kepler(M, e), solve_kepler(M, e, tol=1e-10)))

    def test_float32_solves_kepler_to_dtype_precision(self):
        """Terminating early must not cost accuracy: the residual has to match
        what 50 forced iterations achieve, not merely be 'small'."""
        M, e = _case(torch.float32)
        fast = solve_kepler(M, e)
        forced = solve_kepler(M, e, tol=1e-10)          # unreachable -> all 50
        self.assertLessEqual(_residual(fast, e, M), _residual(forced, e, M) * 1.01)
        # and the two answers agree to within float32 rounding of E itself
        floor = torch.finfo(torch.float32).eps * float(forced.abs().max())
        self.assertLessEqual(float((fast - forced).abs().max()), 4.0 * floor)

    def test_float32_terminates_early(self):
        """The point of the fix. Counted by how many times sin() is called."""
        M, e = _case(torch.float32)
        calls = {"n": 0}
        real_sin = torch.sin

        def counting_sin(x):
            calls["n"] += 1
            return real_sin(x)

        torch.sin = counting_sin
        try:
            solve_kepler(M, e)
            fast = calls["n"]
            calls["n"] = 0
            solve_kepler(M, e, tol=1e-10)
            forced = calls["n"]
        finally:
            torch.sin = real_sin
        # one sin() for the Danby guess, then one per Newton iteration
        self.assertLess(fast, 15, f"float32 took {fast - 1} Newton iterations")
        self.assertGreaterEqual(forced - fast, 30,
                                "forced tol=1e-10 should run far longer")

    def test_high_eccentricity_still_converges(self):
        """e -> 0.99 is where Newton from the Danby guess is slowest; the early
        exit must not truncate those before they have converged."""
        for ecc in (0.9, 0.95, 0.99):
            M, e = _case(torch.float32, ecc=ecc)
            r_fast = _residual(solve_kepler(M, e), e, M)
            r_forced = _residual(solve_kepler(M, e, tol=1e-10), e, M)
            self.assertLessEqual(r_fast, r_forced * 1.01, f"e={ecc}")

    def test_gradients_flow(self):
        """The solver sits inside the psi' descent, so it must stay
        differentiable with the early exit in place."""
        M, e = _case(torch.float32)
        scale = torch.ones(1, requires_grad=True)
        solve_kepler(M * scale, e).sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad).all())


if __name__ == "__main__":
    unittest.main()
