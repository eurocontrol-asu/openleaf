"""Tests for safe arithmetic helpers in physics/_safe.py.

These helpers exist to guard the aero functions against division by zero
near corner cases (very low dynamic pressure, zero surface area), which
otherwise propagate as NaN through the trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest


# Module-under-test is created in the same commit as this test.
safe = pytest.importorskip("openleaf.performance.physics._safe")


class TestSafeDivide:
    """Properties of the safe_divide helper."""

    def test_divides_normally_for_non_zero_denominator(self) -> None:
        """For ordinary inputs safe_divide is exact float division."""
        assert safe.safe_divide(6.0, 2.0) == 3.0

    def test_falls_back_to_default_at_zero(self) -> None:
        """At denominator exactly 0 the helper returns the requested default."""
        assert safe.safe_divide(1.0, 0.0, default=0.0) == 0.0

    def test_falls_back_within_epsilon(self) -> None:
        """At |denominator| < eps the helper still returns the default."""
        eps = 1e-10
        assert safe.safe_divide(1.0, eps / 2.0, default=0.0, eps=eps) == 0.0

    def test_handles_numpy_array_inputs(self) -> None:
        """Works element-wise for NumPy arrays without introducing NaNs."""
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([2.0, 0.0, 4.0])
        out = safe.safe_divide(num, den, default=0.0)
        # Element 0: 1/2 = 0.5
        # Element 1: zero denominator -> default
        # Element 2: 3/4 = 0.75
        assert np.allclose(out, [0.5, 0.0, 0.75])
        assert not np.any(np.isnan(out))

    def test_default_default_is_zero(self) -> None:
        """Default of ``default`` is 0.0 so callers can omit it for the common case."""
        assert safe.safe_divide(1.0, 0.0) == 0.0


class TestAerodynamicCoefficientCornerCases:
    """``aerodynamic_coefficient`` must not return NaN at zero dynamic pressure.

    This was previously a silent NaN source at very low CAS — easy to miss in
    point-value tests, hard to debug when it leaks through to a trajectory.
    """

    def test_zero_dynamic_pressure_returns_zero_not_nan(self) -> None:
        from openleaf.performance.physics.aero import aerodynamic_coefficient

        result = aerodynamic_coefficient(force=1000.0, q=0.0, surface_area=120.0)
        assert not np.isnan(result)
        assert result == 0.0

    def test_zero_surface_area_returns_zero_not_nan(self) -> None:
        from openleaf.performance.physics.aero import aerodynamic_coefficient

        result = aerodynamic_coefficient(force=1000.0, q=500.0, surface_area=0.0)
        assert not np.isnan(result)
        assert result == 0.0
