"""TotalEnergyModel must accept CasADi symbolic inputs (M4 regression)."""

from __future__ import annotations

import numpy as np
import pytest


ca = pytest.importorskip("casadi")

from openleaf.performance.physics.tem import TotalEnergyModel


class TestTotalEnergyModelSymbolic:
    def test_symbolic_thrust_resolves_without_crash(self):
        """np.isnan rejects MX; symbolic inputs count as 'provided'."""
        thrust_sym = ca.MX.sym("thrust", 3)
        model = TotalEnergyModel(
            drag=np.array([40e3, 41e3, 42e3]),
            mass=np.array([60e3, 60e3, 60e3]),
            true_airspeed=np.array([230.0, 231.0, 232.0]),
            altitude=np.array([10_000.0, 10_100.0, 10_200.0]),
            thrust=thrust_sym,
        )
        resolved = model.thrust
        assert isinstance(resolved, ca.MX)

    def test_numeric_path_unchanged(self):
        model = TotalEnergyModel(
            drag=np.array([40e3, 41e3]),
            mass=np.array([60e3, 60e3]),
            true_airspeed=np.array([230.0, 231.0]),
            altitude=np.array([10_000.0, 10_100.0]),
            thrust=np.array([45e3, np.nan]),
        )
        resolved = np.asarray(model.thrust)
        # Row 0: provided thrust; row 1: NaN falls back to default (drag)
        assert resolved[0] == pytest.approx(45e3)
        assert resolved[1] == pytest.approx(41e3)
