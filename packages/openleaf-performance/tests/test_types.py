"""Tests for aetherx.core.types module."""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import EngineSetting
from openleaf.performance.core.types import EngineType
from openleaf.performance.core.types import EnvelopeViolationError
from openleaf.performance.core.types import FlightPhase
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.models.common.config import determine_flight_phase
from openleaf.performance.models.common.config import select_aerodynamic_configuration
from openleaf.performance.models.common.config import select_by_config
from openleaf.performance.numpy.determine_type import is_iterable


class TestEnvelopeViolationError:
    """Tests for EnvelopeViolationError."""

    def test_stores_violations(self):
        """Violations dict is accessible on the exception."""
        violations = {"overspeed": np.array([True, False])}
        err = EnvelopeViolationError("test", violations)
        assert err.violations == violations
        assert str(err) == "test"


class TestLandingGearPosition:
    """Tests for LandingGearPosition enum."""

    def test_from_string_up(self):
        assert LandingGearPosition.from_string("UP") == LandingGearPosition.UP

    def test_from_string_dn(self):
        """DN is a special alias for DOWN."""
        assert LandingGearPosition.from_string("DN") == LandingGearPosition.DOWN

    def test_from_string_down(self):
        assert LandingGearPosition.from_string("DOWN") == LandingGearPosition.DOWN


class TestEngineType:
    """Tests for EngineType enum."""

    def test_from_string(self):
        assert EngineType.from_string("JET") == EngineType.JET
        assert EngineType.from_string("TURBOPROP") == EngineType.TURBOPROP
        assert EngineType.from_string("PISTON") == EngineType.PISTON


class TestEngineSetting:
    """Tests for EngineSetting enum."""

    def test_from_string(self):
        assert EngineSetting.from_string("MCRZ") == EngineSetting.MCRZ
        assert EngineSetting.from_string("MTKF") == EngineSetting.MTKF
        assert EngineSetting.from_string("LIDL") == EngineSetting.LIDL


class TestAerodynamicConfig:
    """Tests for AerodynamicConfig enum."""

    def test_from_string(self):
        assert AerodynamicConfig.from_string("CR") == AerodynamicConfig.CR
        assert AerodynamicConfig.from_string("TO") == AerodynamicConfig.TO
        assert AerodynamicConfig.from_string("LD") == AerodynamicConfig.LD

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown aerodynamic configuration"):
            AerodynamicConfig.from_string("INVALID")


class TestDetermineFlightPhase:
    """Tests for determine_flight_phase()."""

    def test_climb(self):
        result = determine_flight_phase(5.0)
        assert int(result) == FlightPhase.CLIMB

    def test_descent(self):
        result = determine_flight_phase(-5.0)
        assert int(result) == FlightPhase.DESCENT

    def test_cruise(self):
        result = determine_flight_phase(0.1)
        assert int(result) == FlightPhase.CRUISE

    def test_vectorized(self):
        vs = np.array([5.0, 0.0, -5.0])
        result = determine_flight_phase(vs)
        assert int(result[0]) == FlightPhase.CLIMB
        assert int(result[1]) == FlightPhase.CRUISE
        assert int(result[2]) == FlightPhase.DESCENT


class TestSelectAerodynamicConfiguration:
    """Tests for select_aerodynamic_configuration()."""

    def test_climb_takeoff(self):
        """Low altitude climb -> TO config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([200.0]),
            flight_phase=np.array([FlightPhase.CLIMB]),
            airspeed_kt=np.array([150.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.TO

    def test_climb_initial_climb(self):
        """Mid altitude climb -> IC config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([1000.0]),
            flight_phase=np.array([FlightPhase.CLIMB]),
            airspeed_kt=np.array([200.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.IC

    def test_climb_cruise(self):
        """High altitude climb -> CR config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([5000.0]),
            flight_phase=np.array([FlightPhase.CLIMB]),
            airspeed_kt=np.array([250.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.CR

    def test_cruise_always_cr(self):
        """Cruise flight phase -> always CR config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([35000.0]),
            flight_phase=np.array([FlightPhase.CRUISE]),
            airspeed_kt=np.array([250.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.CR

    def test_descent_high_altitude(self):
        """Descent above h_max_ap -> CR config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([10000.0]),
            flight_phase=np.array([FlightPhase.DESCENT]),
            airspeed_kt=np.array([250.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.CR

    def test_descent_approach(self):
        """Descent in approach zone, slow -> AP config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([5000.0]),
            flight_phase=np.array([FlightPhase.DESCENT]),
            airspeed_kt=np.array([125.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.AP

    def test_descent_landing(self):
        """Descent in landing zone, very slow -> LD config."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([1000.0]),
            flight_phase=np.array([FlightPhase.DESCENT]),
            airspeed_kt=np.array([115.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.LD

    def test_descent_fast_stays_cr(self):
        """Fast descent at low altitude stays CR."""
        result = select_aerodynamic_configuration(
            altitude_ft=np.array([5000.0]),
            flight_phase=np.array([FlightPhase.DESCENT]),
            airspeed_kt=np.array([200.0]),
            vmin_cr_kt=np.array([130.0]),
            vmin_ap_kt=np.array([120.0]),
        )
        assert int(result[0]) == AerodynamicConfig.CR


class TestSelectByConfig:
    """Tests for select_by_config()."""

    def test_scalar(self):
        """Scalar config selects correct value."""
        result = select_by_config(
            AerodynamicConfig.CR,
            {
                AerodynamicConfig.CR: 1.0,
                AerodynamicConfig.TO: 2.0,
            },
            default=0.0,
        )
        assert float(result) == 1.0

    def test_vectorized(self):
        """Vectorized config selects per-element."""
        configs = np.array([AerodynamicConfig.CR, AerodynamicConfig.TO, AerodynamicConfig.LD])
        result = select_by_config(
            configs,
            {
                AerodynamicConfig.CR: 10.0,
                AerodynamicConfig.TO: 20.0,
                AerodynamicConfig.LD: 50.0,
            },
            default=0.0,
        )
        np.testing.assert_array_equal(result, [10.0, 20.0, 50.0])


class TestIsIterable:
    """Tests for is_iterable()."""

    def test_list_is_iterable(self):
        assert is_iterable([1, 2, 3]) is True

    def test_string_is_iterable(self):
        assert is_iterable("abc") is True

    def test_int_not_iterable(self):
        assert is_iterable(42) is False

    def test_none_not_iterable(self):
        assert is_iterable(None) is False

    def test_array_is_iterable(self):
        assert is_iterable(np.array([1, 2])) is True
