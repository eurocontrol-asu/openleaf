"""Tests for Total Energy Model (TEM) functions."""

import pytest

import numpy as np
from openleaf.performance.physics import esf
from openleaf.performance.physics import isa
from openleaf.performance.physics import tem


class TestTEMFunctions:
    """Test pure TEM computation functions."""

    @pytest.fixture
    def flight_state(self):
        """Standard flight state for testing."""
        altitude = 10000.0  # m
        return {
            "thrust": 50000.0,  # N
            "drag": 30000.0,  # N
            "mass": 70000.0,  # kg
            "true_airspeed": 250.0,  # m/s
            "f": 1.0,
            "altitude": altitude,
            "air_temperature": isa.air_temperature(altitude),  # ISA
        }

    def test_rocd_pressure_positive_climb(self, flight_state):
        """Test ROCD is positive when thrust > drag."""
        rocd = tem.rocd_pressure(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )
        assert rocd > 0, "ROCD should be positive for climb (T > D)"

    def test_rocd_pressure_zero_for_cruise(self, flight_state):
        """Test ROCD is zero when thrust equals drag."""
        rocd = tem.rocd_pressure(
            thrust=flight_state["drag"],  # T = D
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )
        assert np.isclose(rocd, 0.0), "ROCD should be zero for cruise (T = D)"

    def test_rocd_pressure_negative_descent(self, flight_state):
        """Test ROCD is negative when thrust < drag."""
        rocd = tem.rocd_pressure(
            thrust=flight_state["drag"] - 10000,  # T < D
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )
        assert rocd < 0, "ROCD should be negative for descent (T < D)"

    def test_thrust_from_rocd_inverse(self, flight_state):
        """Test thrust_from_rocd is inverse of rocd_pressure."""
        # First compute ROCD
        rocd = tem.rocd_pressure(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        # Then recover thrust
        thrust_recovered = tem.thrust_from_rocd(
            rocd=rocd,
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        assert np.isclose(thrust_recovered, flight_state["thrust"], rtol=1e-10)

    def test_f_from_state_inverse(self, flight_state):
        """Test f_from_state recovers original f."""
        # Compute ROCD with known f
        rocd = tem.rocd_pressure(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        # Recover f
        f_recovered = tem.f_from_state(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            rocd=rocd,
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        assert np.isclose(f_recovered, flight_state["f"], rtol=1e-10)

    def test_residual_zero_when_consistent(self, flight_state):
        """Test residual is zero for consistent state."""
        rocd = tem.rocd_pressure(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        residual = tem.residual(
            thrust=flight_state["thrust"],
            drag=flight_state["drag"],
            mass=flight_state["mass"],
            true_airspeed=flight_state["true_airspeed"],
            f=flight_state["f"],
            rocd=rocd,
            air_temperature=flight_state["air_temperature"],
            altitude=flight_state["altitude"],
        )

        assert np.isclose(residual, 0.0, atol=1e-10)


class TestTemperatureCorrection:
    """Test temperature correction for non-ISA conditions."""

    def test_temp_correction_factor_isa(self):
        """Test correction factor is 1.0 for ISA conditions."""
        altitude = 10000.0
        T_isa = isa.air_temperature(altitude)

        factor = tem.temp_correction_factor(T_isa, altitude)
        assert np.isclose(factor, 1.0)

    def test_temp_correction_factor_hot(self):
        """Test correction factor < 1 for hot conditions."""
        altitude = 10000.0
        T_isa = isa.air_temperature(altitude)
        T_hot = T_isa + 15  # 15K above ISA

        factor = tem.temp_correction_factor(T_hot, altitude)
        assert factor < 1.0, "Correction factor should be < 1 for hot day"

    def test_temp_correction_factor_cold(self):
        """Test correction factor > 1 for cold conditions."""
        altitude = 10000.0
        T_isa = isa.air_temperature(altitude)
        T_cold = T_isa - 15  # 15K below ISA

        factor = tem.temp_correction_factor(T_cold, altitude)
        assert factor > 1.0, "Correction factor should be > 1 for cold day"

    def test_rocd_pressure_decreases_hot_day(self):
        """Test pressure ROCD is smaller on hot day for same thrust."""
        altitude = 10000.0
        T_isa = isa.air_temperature(altitude)
        T_hot = T_isa + 15

        thrust, drag, mass, tas, f = 50000, 30000, 70000, 250, 1.0

        rocd_isa = tem.rocd_pressure(thrust, drag, mass, tas, f, T_isa, altitude)
        rocd_hot = tem.rocd_pressure(thrust, drag, mass, tas, f, T_hot, altitude)

        assert rocd_hot < rocd_isa, "Pressure ROCD should be smaller on hot day"

    def test_rocd_geometric_conversion(self):
        """Test geometric ROCD conversion."""
        altitude = 10000.0
        T_isa = isa.air_temperature(altitude)
        T_hot = T_isa + 15

        rocd_pressure = 10.0  # m/s

        rocd_geo_isa = tem.rocd_geometric(rocd_pressure, T_isa, altitude)
        rocd_geo_hot = tem.rocd_geometric(rocd_pressure, T_hot, altitude)

        # Under ISA, geometric = pressure
        assert np.isclose(rocd_geo_isa, rocd_pressure)

        # On hot day, geometric > pressure (atmosphere expanded)
        assert rocd_geo_hot > rocd_pressure


class TestFSpeedSchedule:
    """Test f (energy share factor) computation for different speed schedules."""

    def test_constant_mach_above_tropopause(self):
        """Test f = 1.0 for constant Mach above tropopause."""
        altitude = 12000.0  # Above tropopause (11000m)
        mach = 0.78

        f = esf.f_constant_mach(mach, altitude)
        assert np.isclose(f, 1.0), "f should be 1.0 above tropopause for constant Mach"

    def test_constant_mach_below_tropopause(self):
        """Test f > 1 for constant Mach below tropopause."""
        altitude = 8000.0  # Below tropopause
        mach = 0.78

        f = esf.f_constant_mach(mach, altitude)
        assert f > 1.0, "f should be > 1.0 below tropopause for constant Mach"

    def test_constant_cas_below_tropopause(self):
        """Test f < 1 for constant CAS below tropopause."""
        altitude = 8000.0
        mach = 0.6

        f = esf.f_constant_cas(mach, altitude)
        assert f < 1.0, "f should be < 1.0 for constant CAS"

    def test_fixed_f_constants(self):
        """Test fixed f constants from BADA manual."""
        assert esf.F_ACCEL_CLIMB == 0.3
        assert esf.F_DECEL_CLIMB == 1.7
        assert esf.F_ACCEL_DESCENT == 1.7
        assert esf.F_DECEL_DESCENT == 0.3


class TestVectorizedTEM:
    """Test TEM functions work with arrays."""

    def test_rocd_pressure_vectorized(self):
        """Test rocd_pressure works with arrays."""
        n = 10
        thrust = np.linspace(40000, 60000, n)
        drag = np.full(n, 30000)
        mass = np.full(n, 70000)
        tas = np.full(n, 250)
        f = np.full(n, 1.0)
        altitude = np.full(n, 10000)
        T = isa.air_temperature(altitude)

        rocd = tem.rocd_pressure(thrust, drag, mass, tas, f, T, altitude)

        assert rocd.shape == (n,)
        assert np.all(rocd > 0)  # All climbing
        assert rocd[-1] > rocd[0]  # Higher thrust = higher ROCD

    def test_thrust_from_rocd_vectorized(self):
        """Test thrust_from_rocd works with arrays."""
        n = 10
        rocd = np.linspace(5, 15, n)
        drag = np.full(n, 30000)
        mass = np.full(n, 70000)
        tas = np.full(n, 250)
        f = np.full(n, 1.0)
        altitude = np.full(n, 10000)
        T = isa.air_temperature(altitude)

        thrust = tem.thrust_from_rocd(rocd, drag, mass, tas, f, T, altitude)

        assert thrust.shape == (n,)
        assert np.all(thrust > drag)  # Climbing requires T > D
        assert thrust[-1] > thrust[0]  # Higher ROCD = higher thrust


class TestTotalEnergyModel:
    """Test TotalEnergyModel class for TEM state resolution."""

    @pytest.fixture
    def base_state(self):
        """Base flight state without TEM variables."""
        altitude = np.array([10000.0])
        return {
            "drag": np.array([30000.0]),
            "mass": np.array([70000.0]),
            "true_airspeed": np.array([250.0]),
            "altitude": altitude,
            "air_temperature": isa.air_temperature(altitude),
        }

    def test_defaults_when_nothing_provided(self, base_state):
        """Test defaults: thrust=drag, vertical_speed=0, f=1.0."""
        model = tem.TotalEnergyModel(**base_state)

        np.testing.assert_array_almost_equal(model.thrust, base_state["drag"])
        np.testing.assert_array_almost_equal(model.vertical_speed, [0.0])
        np.testing.assert_array_almost_equal(model.f, [1.0])

    def test_thrust_and_f_compute_vertical_speed(self, base_state):
        """Test Branch 1: thrust + f → vertical_speed."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            f=np.array([1.0]),
        )

        # Should compute positive vertical_speed (climbing)
        assert model.vertical_speed > 0, "Should climb when thrust > drag"
        # Thrust and f should be as provided
        np.testing.assert_array_equal(model.thrust, [50000.0])
        np.testing.assert_array_equal(model.f, [1.0])

    def test_vertical_speed_and_f_compute_thrust(self, base_state):
        """Test Branch 2: vertical_speed + f → thrust."""
        model = tem.TotalEnergyModel(
            **base_state,
            vertical_speed=np.array([10.0]),
            f=np.array([1.0]),
        )

        # Should compute thrust > drag (climbing)
        assert model.thrust > base_state["drag"], "Climbing requires thrust > drag"
        # Vertical speed and f should be as provided
        np.testing.assert_array_equal(model.vertical_speed, [10.0])
        np.testing.assert_array_equal(model.f, [1.0])

    def test_thrust_and_vertical_speed_compute_f(self, base_state):
        """Test Branch 3: thrust + vertical_speed → f."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            vertical_speed=np.array([10.0]),
        )

        # Should compute f
        assert model.f > 0, "f should be positive"
        # Thrust and vertical_speed should be as provided
        np.testing.assert_array_equal(model.thrust, [50000.0])
        np.testing.assert_array_equal(model.vertical_speed, [10.0])

    def test_per_row_branching(self, base_state):
        """Test per-row branching with NaN values."""
        # 3 rows with different data available
        n = 3
        drag = np.full(n, 30000.0)
        mass = np.full(n, 70000.0)
        tas = np.full(n, 250.0)
        altitude = np.full(n, 10000.0)
        T = isa.air_temperature(altitude)

        # Row 0: thrust provided (compute vs)
        # Row 1: vertical_speed provided (compute thrust)
        # Row 2: neither (use defaults)
        thrust = np.array([50000.0, np.nan, np.nan])
        vertical_speed = np.array([np.nan, 10.0, np.nan])

        model = tem.TotalEnergyModel(
            drag=drag,
            mass=mass,
            true_airspeed=tas,
            altitude=altitude,
            air_temperature=T,
            thrust=thrust,
            vertical_speed=vertical_speed,
        )

        # Row 0: thrust=50000, vs computed
        assert model.thrust[0] == 50000.0
        assert model.vertical_speed[0] > 0  # Climbing

        # Row 1: vs=10, thrust computed
        assert model.vertical_speed[1] == 10.0
        assert model.thrust[1] > drag[1]  # Climbing requires T > D

        # Row 2: defaults
        assert model.thrust[2] == drag[2]  # T = D
        assert model.vertical_speed[2] == 0.0  # Level flight

    def test_all_three_provided_uses_inputs(self, base_state):
        """Test that provided values take precedence."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            vertical_speed=np.array([10.0]),
            f=np.array([0.8]),
        )

        # All should be as provided
        np.testing.assert_array_equal(model.thrust, [50000.0])
        np.testing.assert_array_equal(model.vertical_speed, [10.0])
        np.testing.assert_array_equal(model.f, [0.8])

    def test_isa_temperature_default(self):
        """Test that air_temperature defaults to ISA."""
        altitude = np.array([10000.0])
        model = tem.TotalEnergyModel(
            drag=np.array([30000.0]),
            mass=np.array([70000.0]),
            true_airspeed=np.array([250.0]),
            altitude=altitude,
            # No air_temperature provided
        )

        # Should still work with ISA temperature
        assert model.vertical_speed is not None

    def test_vertical_speed_geometric(self, base_state):
        """Test geometric vertical speed conversion."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            f=np.array([1.0]),
        )

        # Under ISA conditions, geometric ≈ pressure
        vs_pressure = model.vertical_speed
        vs_geometric = model.vertical_speed_geometric

        # At ISA, they should be approximately equal
        np.testing.assert_array_almost_equal(vs_pressure, vs_geometric, decimal=5)

    def test_speed_schedule_constant_mach(self, base_state):
        """Test f computed from CONSTANT_MACH speed schedule."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            mach=np.array([0.78]),
            speed_schedule=esf.SpeedSchedule.CONSTANT_MACH,
        )

        # f should be computed from speed schedule, not default 1.0
        f_expected = esf.f_constant_mach(np.array([0.78]), base_state["altitude"])
        np.testing.assert_array_almost_equal(model.f, f_expected)

    def test_speed_schedule_constant_cas(self, base_state):
        """Test f computed from CONSTANT_CAS speed schedule."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            mach=np.array([0.6]),
            speed_schedule=esf.SpeedSchedule.CONSTANT_CAS,
        )

        # f should be < 1.0 for constant CAS
        f_expected = esf.f_constant_cas(np.array([0.6]), base_state["altitude"])
        np.testing.assert_array_almost_equal(model.f, f_expected)
        assert model.f < 1.0

    def test_speed_schedule_accel_climb(self, base_state):
        """Test f computed from ACCEL_CLIMB speed schedule."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            speed_schedule=esf.SpeedSchedule.ACCEL_CLIMB,
        )

        # f should be 0.3 for accel climb
        np.testing.assert_array_almost_equal(model.f, [esf.F_ACCEL_CLIMB])

    def test_speed_schedule_decel_descent(self, base_state):
        """Test f computed from DECEL_DESCENT speed schedule."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            speed_schedule=esf.SpeedSchedule.DECEL_DESCENT,
        )

        # f should be 0.3 for decel descent
        np.testing.assert_array_almost_equal(model.f, [esf.F_DECEL_DESCENT])

    def test_explicit_f_overrides_speed_schedule(self, base_state):
        """Test that explicit f takes priority over speed_schedule."""
        model = tem.TotalEnergyModel(
            **base_state,
            thrust=np.array([50000.0]),
            f=np.array([0.85]),
            mach=np.array([0.78]),
            speed_schedule=esf.SpeedSchedule.CONSTANT_MACH,
        )

        # Explicit f should override speed schedule
        np.testing.assert_array_equal(model.f, [0.85])

    def test_speed_schedule_affects_thrust_calc(self, base_state):
        """Test that speed_schedule f is used in thrust computation."""
        # Without speed_schedule (f=1.0)
        model_default = tem.TotalEnergyModel(
            **base_state,
            vertical_speed=np.array([10.0]),
        )

        # With CONSTANT_MACH speed_schedule
        model_mach = tem.TotalEnergyModel(
            **base_state,
            vertical_speed=np.array([10.0]),
            mach=np.array([0.78]),
            speed_schedule=esf.SpeedSchedule.CONSTANT_MACH,
        )

        # Thrust should be different due to different f
        assert not np.allclose(model_mach.thrust, model_default.thrust)
