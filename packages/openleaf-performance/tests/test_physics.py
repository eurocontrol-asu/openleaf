"""
Tests for physics module (ISA and aerodynamics).

Tests ISA calculations including temperature, pressure, and altitude
conversions, plus airspeed conversions and aerodynamic calculations.
Includes property-based tests using Hypothesis.
"""

import casadi as ca
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st

import openleaf.performance.physics.isa as isa
import numpy as np
from openleaf.performance.physics.aero import AircraftAirspeed
from openleaf.performance.physics.aero import crossover_altitude
from openleaf.performance.physics.aero import dynamic_pressure
from openleaf.performance.physics.isa import air_pressure
from openleaf.performance.physics.isa import air_temperature
from openleaf.performance.physics.isa import delta
from openleaf.performance.physics.isa import pressure_altitude
from openleaf.performance.physics.isa import theta


# =============================================================================
# ISA Tests
# =============================================================================


class TestTheta:
    """Tests for theta (temperature ratio) function."""

    def test_sea_level(self):
        """Temperature ratio at sea level is 1."""
        assert np.isclose(theta(0), 1.0)

    def test_tropopause(self):
        """Temperature ratio at tropopause."""
        assert np.isclose(theta(isa.H_TROPO), isa.THETA_TROPO, rtol=1e-6)

    def test_array(self):
        """Test with numpy array input."""
        h = np.array([0, 5000, 10000])
        t = theta(h)
        assert t.shape == h.shape
        assert t[0] == 1.0
        assert np.all(np.diff(t) < 0)  # Decreasing with altitude

    def test_symbolic(self):
        """Test CasADi symbolic compatibility."""
        h_sym = ca.MX.sym("h")
        t_sym = theta(h_sym)
        assert isinstance(t_sym, ca.MX)


class TestDelta:
    """Tests for delta (pressure ratio) function."""

    def test_sea_level(self):
        """Pressure ratio at sea level is 1."""
        assert np.isclose(delta(0), 1.0)

    def test_decreases_with_altitude(self):
        """Pressure decreases with altitude."""
        h = np.array([0, 5000, 10000, 15000])
        d = delta(h)
        assert np.all(np.diff(d) < 0)

    def test_tropopause_continuity(self):
        """Pressure is continuous at tropopause boundary."""
        eps = 1.0  # 1 meter
        h_below = isa.H_TROPO - eps
        h_above = isa.H_TROPO + eps
        d_below = delta(h_below)
        d_above = delta(h_above)
        assert np.isclose(d_below, d_above, rtol=1e-3)

    def test_symbolic(self):
        """Test CasADi symbolic compatibility."""
        h_sym = ca.MX.sym("h")
        d_sym = delta(h_sym)
        assert isinstance(d_sym, ca.MX)


class TestAirTemperature:
    """Tests for air_temperature function (absolute values)."""

    def test_sea_level(self):
        """Temperature at sea level is TAU_SL."""
        assert np.isclose(air_temperature(0), isa.TAU_SL)

    def test_tropopause(self):
        """Temperature at tropopause is TAU_TROPO."""
        assert np.isclose(air_temperature(isa.H_TROPO), isa.TAU_TROPO, rtol=1e-6)

    def test_array(self):
        """Test with numpy array input."""
        h = np.array([0, 5000, 10000])
        T = air_temperature(h)
        assert T.shape == h.shape
        assert np.isclose(T[0], isa.TAU_SL)
        assert np.all(np.diff(T) < 0)  # Decreasing with altitude

    def test_symbolic(self):
        """Test CasADi symbolic compatibility."""
        h_sym = ca.MX.sym("h")
        T_sym = air_temperature(h_sym)
        assert isinstance(T_sym, ca.MX)


class TestAirPressure:
    """Tests for air_pressure function (absolute values)."""

    def test_sea_level(self):
        """Pressure at sea level is P_SL."""
        assert np.isclose(air_pressure(0), isa.P_SL)

    def test_decreases_with_altitude(self):
        """Pressure decreases with altitude."""
        h = np.array([0, 5000, 10000, 15000])
        p = air_pressure(h)
        assert np.all(np.diff(p) < 0)

    def test_tropopause_continuity(self):
        """Pressure is continuous at tropopause boundary."""
        eps = 1.0  # 1 meter
        h_below = isa.H_TROPO - eps
        h_above = isa.H_TROPO + eps
        p_below = air_pressure(h_below)
        p_above = air_pressure(h_above)
        assert np.isclose(p_below, p_above, rtol=1e-3)

    def test_symbolic(self):
        """Test CasADi symbolic compatibility."""
        h_sym = ca.MX.sym("h")
        p_sym = air_pressure(h_sym)
        assert isinstance(p_sym, ca.MX)


class TestPressureAltitude:
    """Tests for pressure_altitude function."""

    def test_roundtrip(self):
        """Test h → P → h roundtrip."""
        h_orig = np.array([0.0, 5000.0, 10000.0, 15000.0])
        p = air_pressure(h_orig)
        h_back = pressure_altitude(p)
        np.testing.assert_allclose(h_back, h_orig, rtol=1e-5)

    def test_sea_level(self):
        """Pressure P_SL gives altitude 0."""
        assert np.isclose(pressure_altitude(isa.P_SL), 0.0)

    def test_symbolic(self):
        """Test CasADi symbolic compatibility."""
        p_sym = ca.MX.sym("p")
        h_sym = pressure_altitude(p_sym)
        assert isinstance(h_sym, ca.MX)


class TestAtmosphericState:
    """Tests for AtmosphericState class."""

    def test_sea_level(self):
        """Test atmospheric state at sea level."""
        atm = isa.AtmosphericState(0.0)
        assert np.isclose(atm.theta, 1.0)
        assert np.isclose(atm.delta, 1.0)
        assert np.isclose(atm.sigma, 1.0)
        assert np.isclose(atm.temperature, isa.TAU_SL)
        assert np.isclose(atm.pressure, isa.P_SL)
        assert np.isclose(atm.density, isa.RHO_SL)
        assert np.isclose(atm.speed_of_sound, isa.A_SL)

    def test_consistency_with_functions(self):
        """Test that class gives same results as standalone functions."""
        h = 8000.0  # Troposphere
        atm = isa.AtmosphericState(h)
        assert np.isclose(atm.theta, theta(h))
        assert np.isclose(atm.delta, delta(h))
        assert np.isclose(atm.temperature, air_temperature(h))
        assert np.isclose(atm.pressure, air_pressure(h))

    def test_stratosphere(self):
        """Test atmospheric state in stratosphere."""
        h = 15000.0  # Above tropopause
        atm = isa.AtmosphericState(h)
        assert np.isclose(atm.theta, isa.THETA_TROPO)
        assert atm.delta < isa.DELTA_TROPO
        assert atm.temperature < isa.TAU_TROPO + 1  # Should be at or near TAU_TROPO

    def test_array_input(self):
        """Test with numpy array altitudes."""
        h = np.array([0.0, 5000.0, 10000.0, 15000.0])
        atm = isa.AtmosphericState(h)
        assert atm.theta.shape == h.shape
        assert atm.delta.shape == h.shape
        assert np.all(np.diff(atm.delta) < 0)  # Pressure decreases

    def test_caching(self):
        """Test that properties are cached (computed once)."""
        atm = isa.AtmosphericState(5000.0)
        # Access theta twice - should be same object due to caching
        theta1 = atm.theta
        theta2 = atm.theta
        assert theta1 is theta2

    def test_sigma_consistency(self):
        """Test that sigma = delta / theta."""
        h = np.array([0.0, 5000.0, 10000.0])
        atm = isa.AtmosphericState(h)
        np.testing.assert_allclose(atm.sigma, atm.delta / atm.theta)


# =============================================================================
# Aerodynamics Tests
# =============================================================================


class TestDynamicPressure:
    """Tests for dynamic pressure calculation."""

    def test_sea_level(self):
        """Test dynamic pressure at sea level."""
        v = 100.0  # m/s
        rho = isa.RHO_SL
        q = dynamic_pressure(v, rho)
        expected = 0.5 * rho * v**2
        assert np.isclose(q, expected, rtol=1e-6)

    def test_array(self):
        """Test dynamic pressure with arrays."""
        v = np.array([100.0, 200.0])
        rho = isa.RHO_SL
        q = dynamic_pressure(v, rho)
        expected = 0.5 * rho * v**2
        np.testing.assert_allclose(q, expected, rtol=1e-6)


class TestAircraftAirspeed:
    """Tests for AircraftAirspeed class."""

    @pytest.fixture
    def sea_level_state(self):
        """Sea level ISA conditions."""
        return {"T": isa.TAU_SL, "p": isa.P_SL}

    @pytest.fixture
    def cruise_state(self):
        """FL350 ISA conditions."""
        theta = 0.7519
        delta = 0.2234
        return {"T": theta * isa.TAU_SL, "p": delta * isa.P_SL}

    def test_tas_to_mach_roundtrip(self, sea_level_state):
        """Test TAS → Mach → TAS roundtrip."""
        tas = 250.0
        airspeed = AircraftAirspeed(**sea_level_state, v=tas)
        mach = airspeed.mach_number

        airspeed2 = AircraftAirspeed(**sea_level_state, mach=mach)
        tas_back = airspeed2.true_airspeed

        assert np.isclose(tas, tas_back, rtol=1e-6)

    def test_cas_to_tas_roundtrip(self, cruise_state):
        """Test CAS → TAS → CAS roundtrip."""
        cas = 250.0
        airspeed = AircraftAirspeed(**cruise_state, cas=cas)
        tas = airspeed.true_airspeed

        airspeed2 = AircraftAirspeed(**cruise_state, v=tas)
        cas_back = airspeed2.calibrated_airspeed

        assert np.isclose(cas, cas_back, rtol=1e-5)

    def test_mach_increases_with_altitude(self):
        """Test that same TAS gives higher Mach at altitude (colder air)."""
        tas = 250.0

        # Sea level
        mach_sl = AircraftAirspeed(T=isa.TAU_SL, p=isa.P_SL, v=tas).mach_number

        # FL350 (colder)
        theta = 0.7519
        mach_fl350 = AircraftAirspeed(
            T=theta * isa.TAU_SL, p=0.2234 * isa.P_SL, v=tas
        ).mach_number

        assert mach_fl350 > mach_sl

    def test_exactly_one_input_required(self):
        """Test that exactly one airspeed input is required."""
        with pytest.raises(ValueError):
            AircraftAirspeed(T=isa.TAU_SL, p=isa.P_SL)  # No input

        with pytest.raises(ValueError):
            AircraftAirspeed(T=isa.TAU_SL, p=isa.P_SL, v=100, mach=0.3)  # Two inputs


class TestAircraftAirspeedSymbolic:
    """Tests for CasADi symbolic compatibility."""

    def test_symbolic_mach(self):
        """Test symbolic Mach computation."""
        T = ca.MX.sym("T")
        p = ca.MX.sym("p")
        v = ca.MX.sym("v")

        airspeed = AircraftAirspeed(T, p, v=v)
        mach = airspeed.mach_number

        assert isinstance(mach, ca.MX)

    def test_symbolic_cas(self):
        """Test symbolic CAS computation from TAS."""
        T = ca.MX.sym("T")
        p = ca.MX.sym("p")
        v = ca.MX.sym("v")

        airspeed = AircraftAirspeed(T, p, v=v)
        cas = airspeed.calibrated_airspeed

        assert isinstance(cas, ca.MX)


class TestCrossoverAltitude:
    """Tests for crossover altitude calculation."""

    def test_typical_values(self):
        """Test crossover altitude for typical cruise conditions."""
        cas = 150.0  # m/s (~290 kt)
        mach = 0.78

        h = crossover_altitude(cas, mach)

        # Should be around FL300-FL350 for typical values
        assert 8000 < h < 12000  # meters


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================


class TestISAPropertyBased:
    """Property-based tests for ISA functions using Hypothesis."""

    @given(st.floats(min_value=0, max_value=20000, allow_nan=False))
    def test_theta_bounded(self, h: float) -> None:
        """Temperature ratio is always positive and <= 1."""
        t = theta(h)
        assert 0 < t <= 1.0

    @given(st.floats(min_value=0, max_value=20000, allow_nan=False))
    def test_delta_bounded(self, h: float) -> None:
        """Pressure ratio is always positive and <= 1."""
        d = delta(h)
        assert 0 < d <= 1.0

    @given(st.floats(min_value=0, max_value=20000, allow_nan=False))
    def test_temperature_positive(self, h: float) -> None:
        """Temperature is always positive (Kelvin)."""
        T = air_temperature(h)
        assert T > 0

    @given(st.floats(min_value=0, max_value=20000, allow_nan=False))
    def test_pressure_positive(self, h: float) -> None:
        """Pressure is always positive."""
        p = air_pressure(h)
        assert p > 0

    @given(st.floats(min_value=0, max_value=15000, allow_nan=False))
    def test_pressure_altitude_roundtrip(self, h: float) -> None:
        """Pressure altitude roundtrip: h -> P -> h."""
        p = air_pressure(h)
        h_back = pressure_altitude(p)
        assert np.isclose(h_back, h, rtol=1e-5)

    @given(st.floats(min_value=0, max_value=20000, allow_nan=False))
    def test_theta_delta_consistency(self, h: float) -> None:
        """Temperature and pressure ratios are consistent with absolute values."""
        t = theta(h)
        d = delta(h)
        T = air_temperature(h)
        p = air_pressure(h)

        assert np.isclose(T, t * isa.TAU_SL, rtol=1e-10)
        assert np.isclose(p, d * isa.P_SL, rtol=1e-10)

    @given(
        st.floats(min_value=0, max_value=20000, allow_nan=False),
        st.floats(min_value=0, max_value=20000, allow_nan=False),
    )
    def test_theta_monotonic(self, h1: float, h2: float) -> None:
        """Temperature ratio decreases (or stays constant) with altitude."""
        t1, t2 = theta(h1), theta(h2)
        if h1 < h2:
            assert t1 >= t2
        elif h1 > h2:
            assert t1 <= t2

    @given(
        st.floats(min_value=0, max_value=20000, allow_nan=False),
        st.floats(min_value=0, max_value=20000, allow_nan=False),
    )
    def test_delta_monotonic(self, h1: float, h2: float) -> None:
        """Pressure ratio strictly decreases with altitude."""
        assume(abs(h1 - h2) > 1.0)  # Need some separation
        d1, d2 = delta(h1), delta(h2)
        if h1 < h2:
            assert d1 > d2
        elif h1 > h2:
            assert d1 < d2


class TestAeroPropertyBased:
    """Property-based tests for aerodynamic functions."""

    @given(
        st.floats(min_value=200, max_value=350, allow_nan=False),  # Temperature K
        st.floats(min_value=10000, max_value=101325, allow_nan=False),  # Pressure Pa
        st.floats(min_value=50, max_value=300, allow_nan=False),  # TAS m/s
    )
    def test_dynamic_pressure_positive(self, T: float, p: float, v: float) -> None:
        """Dynamic pressure is always positive for positive airspeed."""
        rho = p / (isa.R_AIR * T)
        q = dynamic_pressure(rho, v)
        assert q > 0

    @given(
        st.floats(min_value=220, max_value=290, allow_nan=False),  # Temperature K
        st.floats(min_value=20000, max_value=101325, allow_nan=False),  # Pressure Pa
        st.floats(min_value=100, max_value=280, allow_nan=False),  # TAS m/s
    )
    def test_mach_positive(self, T: float, p: float, v: float) -> None:
        """Mach number is positive for positive airspeed."""
        airspeed = AircraftAirspeed(T, p, v=v)
        assert airspeed.mach_number > 0

    @given(
        st.floats(min_value=220, max_value=290, allow_nan=False),  # Temperature K
        st.floats(min_value=20000, max_value=101325, allow_nan=False),  # Pressure Pa
        st.floats(min_value=100, max_value=280, allow_nan=False),  # TAS m/s
    )
    def test_cas_positive(self, T: float, p: float, v: float) -> None:
        """Calibrated airspeed is positive for positive TAS."""
        airspeed = AircraftAirspeed(T, p, v=v)
        assert airspeed.calibrated_airspeed > 0
