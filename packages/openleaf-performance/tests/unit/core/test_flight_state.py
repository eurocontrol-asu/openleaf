from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.perfos import FlightPhase
from openleaf.performance.core.trajectory import AircraftTrajectoryData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeTrajectory:
    """Minimal trajectory dict-like for testing."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):
        return self._data[key]

    @property
    def delta_isa(self):
        return self._data.get("delta_isa", 0.0)


class _FakeModel:
    """Minimal model stub with configurable delegation returns."""

    def __init__(
        self,
        *,
        aero_config=None,
        high_lift_val=None,
        landing_gear_val=None,
    ) -> None:
        self._aero_config = aero_config
        self._high_lift_val = high_lift_val
        self._landing_gear_val = landing_gear_val

    def aerodynamic_configuration(self, data):
        return self._aero_config

    def high_lift(self, data):
        return self._high_lift_val

    def landing_gear(self, data):
        return self._landing_gear_val


@pytest.fixture
def _make_atd():
    """Factory fixture returning an AircraftTrajectoryData with given vs."""

    def _factory(
        vertical_speed=0.0,
        aircraft_mass=70_000.0,
        delta_isa=0.0,
        *,
        model_kwargs=None,
    ):
        traj = _FakeTrajectory(
            {
                "vertical_speed": np.asarray(vertical_speed),
                "aircraft_mass": np.asarray(aircraft_mass),
                "delta_isa": np.asarray(delta_isa),
            }
        )
        model = _FakeModel(**(model_kwargs or {}))
        atd = object.__new__(AircraftTrajectoryData)
        atd._traj = traj
        atd._model = model
        return atd

    return _factory


# ---------------------------------------------------------------------------
# Unit tests — flight_phase
# ---------------------------------------------------------------------------


class TestFlightPhase:
    def test_flight_phase_climb(self, _make_atd):
        atd = _make_atd(vertical_speed=5.0)
        assert np.all(np.equal(atd.flight_phase, FlightPhase.CLIMB))

    def test_flight_phase_cruise(self, _make_atd):
        atd = _make_atd(vertical_speed=0.0)
        assert np.all(np.equal(atd.flight_phase, FlightPhase.CRUISE))

    def test_flight_phase_descent(self, _make_atd):
        atd = _make_atd(vertical_speed=-5.0)
        assert np.all(np.equal(atd.flight_phase, FlightPhase.DESCENT))


# ---------------------------------------------------------------------------
# Unit tests — engine_setting
# ---------------------------------------------------------------------------


class TestEngineSetting:
    def test_engine_setting_climb(self, _make_atd):
        atd = _make_atd(vertical_speed=5.0)
        assert np.all(np.equal(atd.engine_setting, EngineSetting.MCMB))

    def test_engine_setting_cruise(self, _make_atd):
        atd = _make_atd(vertical_speed=0.0)
        assert np.all(np.equal(atd.engine_setting, EngineSetting.MCRZ))

    def test_engine_setting_descent(self, _make_atd):
        atd = _make_atd(vertical_speed=-5.0)
        assert np.all(np.equal(atd.engine_setting, EngineSetting.LIDL))


# ---------------------------------------------------------------------------
# Unit tests — model delegation
# ---------------------------------------------------------------------------


class TestModelDelegation:
    def test_aero_config_delegates_to_model(self, _make_atd, mocker):
        atd = _make_atd(model_kwargs={"aero_config": "CR"})
        spy = mocker.patch.object(atd._model, "aerodynamic_configuration", return_value="CR")
        result = atd.aerodynamic_configuration
        spy.assert_called_once_with(atd)
        assert result == "CR"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_boundary_vertical_speed_positive(self, _make_atd):
        """vs = +0.5 m/s exactly → CRUISE (not above threshold)."""
        atd = _make_atd(vertical_speed=0.5)
        assert np.all(np.equal(atd.flight_phase, FlightPhase.CRUISE))

    def test_boundary_vertical_speed_negative(self, _make_atd):
        """vs = -0.5 m/s exactly → CRUISE (not below threshold)."""
        atd = _make_atd(vertical_speed=-0.5)
        assert np.all(np.equal(atd.flight_phase, FlightPhase.CRUISE))

    def test_model_without_aero_config(self, _make_atd):
        """PS-like model returns None for aerodynamic_configuration."""
        atd = _make_atd(model_kwargs={"aero_config": None})
        assert atd.aerodynamic_configuration is None

    def test_model_without_landing_gear(self, _make_atd):
        """PS-like model returns None for landing_gear."""
        atd = _make_atd(model_kwargs={"landing_gear_val": None})
        assert atd.landing_gear is None
