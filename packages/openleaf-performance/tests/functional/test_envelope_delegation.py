from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from openleaf.performance.core.types import EnvelopeMode


class TestDelegationPreservesApi:
    """After refactoring, AircraftTrajectoryData must still expose
    envelope_violations, check_envelope(), envelope_mode, and limit
    properties via delegation to EnvelopeChecker."""

    @pytest.fixture
    def traj_with_model(self):
        """Create a real AircraftTrajectoryData with mock internals."""
        from openleaf.performance.core.trajectory import AircraftTrajectoryData
        from openleaf.performance.core.trajectory import TrajectoryData

        # Mock TrajectoryData with required properties
        traj_data = MagicMock(spec=TrajectoryData)
        traj_data.__len__ = MagicMock(return_value=1)
        traj_data.__getitem__ = MagicMock(
            side_effect=lambda k: np.array([10_000.0]) if k == "altitude" else None
        )
        traj_data.mach_number = np.array([0.78])
        traj_data.calibrated_airspeed = np.array([230.0])  # m/s

        # Mock model
        model = MagicMock()
        model.oem = 40_000.0
        model.mtom = 80_000.0
        model.model = "TEST-A320"
        model.max_cas.return_value = np.asarray(350.0)
        model.min_cas.return_value = np.asarray(130.0)
        model.max_mach.return_value = np.asarray(0.82)
        model.max_altitude.return_value = np.asarray(41_000.0)

        # Build real AircraftTrajectoryData
        perf = AircraftTrajectoryData(traj_data, model, envelope_mode=EnvelopeMode.IGNORE)

        # Mock aircraft_mass (cached_property) by setting on instance
        perf.__dict__["aircraft_mass"] = np.array([60_000.0])

        return perf

    def test_delegation_preserves_api(self, traj_with_model):
        """All envelope-related attributes still accessible on trajectory."""
        perf = traj_with_model

        # Limit properties delegate correctly
        assert perf.max_cas is not None
        assert perf.min_cas is not None
        assert perf.max_mach is not None
        assert perf.max_altitude is not None

        # Envelope mode accessible
        assert perf.envelope_mode == EnvelopeMode.IGNORE

        # envelope_violations returns a dict with expected keys
        violations = perf.envelope_violations
        expected_keys = {
            "overspeed_cas",
            "overspeed_mach",
            "underspeed",
            "ceiling",
            "mass_overweight",
            "mass_underweight",
        }
        assert set(violations.keys()) == expected_keys

        # check_envelope returns same structure
        result = perf.check_envelope()
        assert set(result.keys()) == expected_keys
