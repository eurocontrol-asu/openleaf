"""Tests for TotalEnergyModel._f_from_speed_schedule."""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.physics import esf
from openleaf.performance.physics.tem import DEFAULT_ENERGY_SHARE_FACTOR
from openleaf.performance.physics.tem import TotalEnergyModel


@pytest.fixture
def base_kwargs():
    """Minimal kwargs for TotalEnergyModel (required fields only)."""
    altitude = np.array([35000.0 * 0.3048])  # 35000 ft in meters
    return {
        "drag": np.array([30000.0]),
        "mass": np.array([70000.0]),
        "true_airspeed": np.array([250.0]),
        "altitude": altitude,
    }


def _make_model(
    base_kwargs: dict,
    *,
    speed_schedule: esf.SpeedSchedule | None = None,
    mach: float | None = None,
) -> TotalEnergyModel:
    """Build a TotalEnergyModel with given schedule and mach."""
    kw = {**base_kwargs, "speed_schedule": speed_schedule}
    if mach is not None:
        kw["mach"] = np.array([mach])
    return TotalEnergyModel(**kw)


# --- Unit tests ---


def test_f_constant_mach(base_kwargs):
    """CONSTANT_MACH schedule returns f_constant_mach(mach, altitude)."""
    mach = 0.78
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.CONSTANT_MACH, mach=mach)
    result = model._f_from_speed_schedule()
    expected = esf.f_constant_mach(np.array([mach]), base_kwargs["altitude"])
    np.testing.assert_array_almost_equal(result, expected)


def test_f_constant_cas(base_kwargs):
    """CONSTANT_CAS schedule returns f_constant_cas(mach, altitude)."""
    mach = 0.78
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.CONSTANT_CAS, mach=mach)
    result = model._f_from_speed_schedule()
    expected = esf.f_constant_cas(np.array([mach]), base_kwargs["altitude"])
    np.testing.assert_array_almost_equal(result, expected)


def test_f_accel_climb(base_kwargs):
    """ACCEL_CLIMB schedule returns F_ACCEL_CLIMB constant."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.ACCEL_CLIMB)
    result = model._f_from_speed_schedule()
    assert result == esf.F_ACCEL_CLIMB


def test_f_no_schedule(base_kwargs):
    """No speed schedule returns DEFAULT_ENERGY_SHARE_FACTOR."""
    model = _make_model(base_kwargs, speed_schedule=None)
    result = model._f_from_speed_schedule()
    assert result == DEFAULT_ENERGY_SHARE_FACTOR


# --- Edge cases ---


def test_f_constant_mach_none_mach(base_kwargs):
    """CONSTANT_MACH with mach=None returns DEFAULT_ENERGY_SHARE_FACTOR."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.CONSTANT_MACH, mach=None)
    result = model._f_from_speed_schedule()
    assert result == DEFAULT_ENERGY_SHARE_FACTOR


# --- Remaining constant schedules (completeness for AC3) ---


def test_f_decel_climb(base_kwargs):
    """DECEL_CLIMB schedule returns F_DECEL_CLIMB constant."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.DECEL_CLIMB)
    result = model._f_from_speed_schedule()
    assert result == esf.F_DECEL_CLIMB


def test_f_accel_descent(base_kwargs):
    """ACCEL_DESCENT schedule returns F_ACCEL_DESCENT constant."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.ACCEL_DESCENT)
    result = model._f_from_speed_schedule()
    assert result == esf.F_ACCEL_DESCENT


def test_f_decel_descent(base_kwargs):
    """DECEL_DESCENT schedule returns F_DECEL_DESCENT constant."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.DECEL_DESCENT)
    result = model._f_from_speed_schedule()
    assert result == esf.F_DECEL_DESCENT


def test_f_constant_cas_none_mach(base_kwargs):
    """CONSTANT_CAS with mach=None returns DEFAULT_ENERGY_SHARE_FACTOR."""
    model = _make_model(base_kwargs, speed_schedule=esf.SpeedSchedule.CONSTANT_CAS, mach=None)
    result = model._f_from_speed_schedule()
    assert result == DEFAULT_ENERGY_SHARE_FACTOR
