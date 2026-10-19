"""
Tests for TrajectoryData, AircraftTrajectoryData, and validation.
"""

import numpy as np
import pytest

from openleaf.performance.core.trajectory import TrajectoryData
from openleaf.performance.core.validation import StateSchema
from openleaf.performance.core.validation import get_required_fields
from openleaf.performance.core.validation import requires
from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.models.ps import PSModel
from openleaf.performance.physics import isa


# --- Fixtures ---


@pytest.fixture
def ps_model() -> PSModel:
    """Create a PS model for testing."""
    pytest.importorskip("pycontrails")
    return PSModel("A320")


@pytest.fixture
def basic_trajectory_data() -> dict:
    """Basic trajectory data with all required columns."""
    return {
        "altitude": np.array([10000.0, 10500.0, 11000.0]),
        "true_airspeed": np.array([230.0, 232.0, 234.0]),
        "vertical_speed": np.array([5.0, 5.0, 5.0]),
        "aircraft_mass": np.array([65000.0, 64900.0, 64800.0]),
        "acceleration": np.array([0.1, 0.1, 0.1]),
    }


@pytest.fixture
def level_flight_data() -> dict:
    """Level flight data (vs=0, a=0)."""
    return {
        "altitude": np.array([10000.0, 10000.0]),
        "true_airspeed": np.array([230.0, 230.0]),
        "vertical_speed": np.array([0.0, 0.0]),
        "aircraft_mass": np.array([65000.0, 65000.0]),
        "acceleration": np.array([0.0, 0.0]),
    }


# --- TrajectoryData Tests ---


class TestTrajectoryDataValidation:
    """Tests for TrajectoryData input validation."""

    def test_missing_required_column_raises_error(self):
        """Test that missing required columns raise ValueError."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            # Missing: vertical_speed, aircraft_mass
            "acceleration": np.array([0.1]),
        }

        with pytest.raises(ValueError, match="Missing required columns"):
            TrajectoryData(data)

    def test_missing_energy_column_raises_error(self):
        """Test that missing both f and acceleration raises ValueError."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([5.0]),
            "aircraft_mass": np.array([65000.0]),
            # Missing: f or acceleration
        }

        with pytest.raises(ValueError, match="Must provide 'f' or 'acceleration'"):
            TrajectoryData(data)

    def test_valid_data_with_acceleration(self, basic_trajectory_data):
        """Test valid data with acceleration column."""
        traj = TrajectoryData(basic_trajectory_data)
        assert len(traj) == 3

    def test_valid_data_with_f(self):
        """Test valid data with f column instead of acceleration."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([5.0]),
            "aircraft_mass": np.array([65000.0]),
            "f": np.array([1.0]),
        }
        traj = TrajectoryData(data)
        assert len(traj) == 1


class TestTrajectoryDataAccess:
    """Tests for TrajectoryData data access methods."""

    def test_getitem_data_column(self, basic_trajectory_data):
        """Test __getitem__ for data columns."""
        traj = TrajectoryData(basic_trajectory_data)

        np.testing.assert_array_equal(traj["altitude"], basic_trajectory_data["altitude"])
        np.testing.assert_array_equal(
            traj["true_airspeed"], basic_trajectory_data["true_airspeed"]
        )

    def test_getitem_computed_property(self, basic_trajectory_data):
        """Test __getitem__ for computed properties."""
        traj = TrajectoryData(basic_trajectory_data)

        # Access computed property via __getitem__
        mach = traj["mach_number"]
        assert mach is not None
        assert len(mach) == 3

    def test_getitem_missing_key_raises_error(self, basic_trajectory_data):
        """Test __getitem__ raises KeyError for missing keys."""
        traj = TrajectoryData(basic_trajectory_data)

        with pytest.raises(KeyError, match="nonexistent"):
            _ = traj["nonexistent"]

    def test_contains(self, basic_trajectory_data):
        """Test __contains__ for data and computed properties."""
        traj = TrajectoryData(basic_trajectory_data)

        # Data columns
        assert "altitude" in traj
        assert "true_airspeed" in traj

        # Computed properties
        assert "mach_number" in traj
        assert "air_temperature" in traj

        # Non-existent
        assert "nonexistent" not in traj

    def test_get_with_default(self, basic_trajectory_data):
        """Test get() with default value."""
        traj = TrajectoryData(basic_trajectory_data)

        # Existing key
        alt = traj.get("altitude")
        np.testing.assert_array_equal(alt, basic_trajectory_data["altitude"])

        # Non-existing key with default
        val = traj.get("nonexistent", 42)
        assert val == 42

    def test_keys_and_iter(self, basic_trajectory_data):
        """Test keys() and __iter__."""
        traj = TrajectoryData(basic_trajectory_data)

        keys = list(traj.keys())
        assert "altitude" in keys
        assert "true_airspeed" in keys

        iter_keys = list(traj)
        assert iter_keys == keys

    def test_len(self, basic_trajectory_data):
        """Test __len__."""
        traj = TrajectoryData(basic_trajectory_data)
        assert len(traj) == 3


class TestTrajectoryDataAtmospheric:
    """Tests for TrajectoryData atmospheric properties."""

    def test_air_temperature_isa(self, basic_trajectory_data):
        """Test air_temperature computed from ISA when not provided."""
        traj = TrajectoryData(basic_trajectory_data)

        T = traj.air_temperature
        T_expected = isa.air_temperature(basic_trajectory_data["altitude"])

        np.testing.assert_array_almost_equal(T, T_expected)

    def test_air_temperature_provided(self, basic_trajectory_data):
        """Test air_temperature uses provided value."""
        basic_trajectory_data["air_temperature"] = np.array([250.0, 248.0, 246.0])
        traj = TrajectoryData(basic_trajectory_data)

        np.testing.assert_array_equal(traj.air_temperature, [250.0, 248.0, 246.0])

    def test_air_pressure(self, basic_trajectory_data):
        """Test air_pressure computed from ISA."""
        traj = TrajectoryData(basic_trajectory_data)

        p = traj.air_pressure
        p_expected = isa.air_pressure(basic_trajectory_data["altitude"])

        np.testing.assert_array_almost_equal(p, p_expected)

    def test_air_density(self, basic_trajectory_data):
        """Test air_density computed from T and p."""
        traj = TrajectoryData(basic_trajectory_data)

        rho = traj.air_density
        rho_expected = traj.air_pressure / (isa.R * traj.air_temperature)

        np.testing.assert_array_almost_equal(rho, rho_expected)

    def test_delta_isa_zero_when_isa(self, basic_trajectory_data):
        """Test delta_isa is zero when no temperature provided (ISA)."""
        traj = TrajectoryData(basic_trajectory_data)

        delta = traj.delta_isa
        np.testing.assert_array_almost_equal(delta, np.zeros(3))

    def test_delta_isa_nonzero(self, basic_trajectory_data):
        """Test delta_isa when temperature deviates from ISA."""
        T_isa = isa.air_temperature(basic_trajectory_data["altitude"])
        basic_trajectory_data["air_temperature"] = T_isa + 10.0  # ISA+10

        traj = TrajectoryData(basic_trajectory_data)

        delta = traj.delta_isa
        np.testing.assert_array_almost_equal(delta, np.full(3, 10.0))


class TestTrajectoryDataAirspeed:
    """Tests for TrajectoryData airspeed conversions."""

    def test_mach_number(self, basic_trajectory_data):
        """Test mach_number computed from TAS."""
        traj = TrajectoryData(basic_trajectory_data)

        M = traj.mach_number
        # Mach = TAS / speed_of_sound
        a = np.sqrt(isa.GAMMA * isa.R * traj.air_temperature)
        M_expected = basic_trajectory_data["true_airspeed"] / a

        np.testing.assert_array_almost_equal(M, M_expected)

    def test_calibrated_airspeed(self, basic_trajectory_data):
        """Test calibrated_airspeed computed from TAS."""
        traj = TrajectoryData(basic_trajectory_data)

        CAS = traj.calibrated_airspeed
        # CAS should be less than TAS at altitude
        assert np.all(basic_trajectory_data["true_airspeed"] > CAS)
        assert np.all(CAS > 0)


class TestTrajectoryDataVerticalSpeed:
    """Tests for TrajectoryData vertical speed conversions."""

    def test_vertical_speed_geometric_isa(self, basic_trajectory_data):
        """Test vertical_speed_geometric under ISA conditions."""
        traj = TrajectoryData(basic_trajectory_data)

        vs_geo = traj.vertical_speed_geometric
        # Under ISA, correction factor = 1.0
        np.testing.assert_array_almost_equal(vs_geo, basic_trajectory_data["vertical_speed"])

    def test_vertical_speed_geometric_non_isa(self, basic_trajectory_data):
        """Test vertical_speed_geometric under non-ISA conditions."""
        T_isa = isa.air_temperature(basic_trajectory_data["altitude"])
        basic_trajectory_data["air_temperature"] = T_isa + 20.0  # ISA+20 (hot)

        traj = TrajectoryData(basic_trajectory_data)

        vs_geo = traj.vertical_speed_geometric
        # Hot conditions: vs_geo > vs_pressure
        assert np.all(vs_geo > basic_trajectory_data["vertical_speed"])


class TestTrajectoryDataEnergyFactor:
    """Tests for TrajectoryData energy share factor f and acceleration."""

    def test_f_from_data(self):
        """Test f taken directly from data."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([5.0]),
            "aircraft_mass": np.array([65000.0]),
            "f": np.array([0.9]),
        }
        traj = TrajectoryData(data)

        np.testing.assert_array_equal(traj.f, [0.9])

    def test_f_derived_from_acceleration(self, basic_trajectory_data):
        """Test f derived from acceleration when not provided."""
        traj = TrajectoryData(basic_trajectory_data)

        f = traj.f
        # f = vs_geo / (vs_geo + (V/g) * a)
        vs_geo = traj.vertical_speed_geometric
        V = basic_trajectory_data["true_airspeed"]
        a = basic_trajectory_data["acceleration"]
        f_expected = vs_geo / (vs_geo + (V / isa.G) * a)

        np.testing.assert_array_almost_equal(f, f_expected)

    def test_f_level_flight(self, level_flight_data):
        """Test f = 1.0 for level flight (vs=0, a=0)."""
        traj = TrajectoryData(level_flight_data)

        f = traj.f
        np.testing.assert_array_almost_equal(f, [1.0, 1.0])

    def test_acceleration_from_data(self, basic_trajectory_data):
        """Test acceleration taken directly from data."""
        traj = TrajectoryData(basic_trajectory_data)

        np.testing.assert_array_equal(traj.acceleration, basic_trajectory_data["acceleration"])

    def test_acceleration_derived_from_f(self):
        """Test acceleration derived from f when not provided."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([5.0]),
            "aircraft_mass": np.array([65000.0]),
            "f": np.array([0.9]),  # f < 1 means accelerating
        }
        traj = TrajectoryData(data)

        a = traj.acceleration
        # For f < 1 in climb, should be accelerating (a > 0)
        assert np.all(a > 0)

    def test_acceleration_constant_speed(self):
        """Test acceleration = 0 when f = 1 (constant speed)."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([5.0]),
            "aircraft_mass": np.array([65000.0]),
            "f": np.array([1.0]),
        }
        traj = TrajectoryData(data)

        a = traj.acceleration
        np.testing.assert_array_almost_equal(a, [0.0])


class TestTrajectoryDataCaching:
    """Tests for TrajectoryData property caching."""

    def test_computed_properties_cached(self, basic_trajectory_data):
        """Test that computed properties are cached."""
        traj = TrajectoryData(basic_trajectory_data)

        # Access mach_number twice
        mach1 = traj.mach_number
        mach2 = traj.mach_number

        # Should be the same object (cached)
        assert mach1 is mach2


# --- AircraftTrajectoryData Tests ---


class TestAircraftTrajectoryData:
    """Tests for AircraftTrajectoryData with aircraft model."""

    def test_with_aircraft_model_instance(self, basic_trajectory_data, ps_model):
        """Test with_aircraft with model instance."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        assert perf.model is ps_model
        assert perf.trajectory is traj

    def test_with_aircraft_string_requires_icao(self, basic_trajectory_data):
        """Test with_aircraft with string requires icao."""
        traj = TrajectoryData(basic_trajectory_data)

        with pytest.raises(ValueError, match="icao required"):
            traj.with_aircraft("PS")

    def test_delegation_to_trajectory(self, basic_trajectory_data, ps_model):
        """Test that AircraftTrajectoryData delegates to TrajectoryData."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        # Data columns via delegation
        np.testing.assert_array_equal(perf["altitude"], basic_trajectory_data["altitude"])

        # Computed properties via delegation
        np.testing.assert_array_equal(perf.mach_number, traj.mach_number)
        np.testing.assert_array_equal(perf.air_temperature, traj.air_temperature)

    def test_aircraft_mass(self, basic_trajectory_data, ps_model):
        """Test aircraft_mass from trajectory data."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        np.testing.assert_array_equal(perf.aircraft_mass, basic_trajectory_data["aircraft_mass"])

    def test_lift_coef(self, basic_trajectory_data, ps_model):
        """Test lift_coef computed by model."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        CL = perf.lift_coef
        assert np.all(CL > 0)
        assert np.all(CL < 2.0)

    def test_drag_coef(self, basic_trajectory_data, ps_model):
        """Test drag_coef computed by model."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        CD = perf.drag_coef
        assert np.all(CD > 0)
        assert np.all(CD < 0.1)

    def test_drag(self, basic_trajectory_data, ps_model):
        """Test drag computed by model."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        D = perf.drag
        assert np.all(D > 0)

    def test_thrust_level_flight(self, level_flight_data, ps_model):
        """Test thrust equals drag for level flight."""
        traj = TrajectoryData(level_flight_data)
        perf = traj.with_aircraft(ps_model)

        T = perf.thrust
        D = perf.drag

        np.testing.assert_array_almost_equal(T, D)

    def test_thrust_climbing(self, basic_trajectory_data, ps_model):
        """Test thrust > drag for climbing flight."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        T = perf.thrust
        D = perf.drag

        assert np.all(T > D)

    def test_thrust_descending(self, ps_model):
        """Test thrust < drag for descending flight."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([-5.0]),  # Descending
            "aircraft_mass": np.array([65000.0]),
            "f": np.array([1.0]),
        }
        traj = TrajectoryData(data)
        perf = traj.with_aircraft(ps_model)

        T = perf.thrust
        D = perf.drag

        assert np.all(T < D)

    @pytest.mark.parametrize(
        ("vertical_speed", "acceleration", "expected_excess"),
        [
            (0.0, 0.5, 32500.0),
            (0.0, -0.5, -32500.0),
            (5.0, 0.5, 46357.2),
            (5.0, 0.0, 13857.2),
        ],
    )
    def test_thrust_includes_acceleration(
        self, ps_model, vertical_speed, acceleration, expected_excess
    ):
        """T - D = m g dh/dt / V + m a, including level acceleration (issue #47)."""
        data = {
            "altitude": np.array([10000.0]),
            "true_airspeed": np.array([230.0]),
            "vertical_speed": np.array([vertical_speed]),
            "aircraft_mass": np.array([65000.0]),
            "acceleration": np.array([acceleration]),
        }
        perf = TrajectoryData(data).with_aircraft(ps_model)

        np.testing.assert_allclose(perf.thrust - perf.drag, expected_excess, atol=0.1)

    def test_thrust_from_acceleration_matches_f_path(self, ps_model):
        """Climbing and descending thrust is the same from acceleration or from f."""
        data = {
            "altitude": np.array([10000.0, 10000.0, 9000.0, 9000.0]),
            "true_airspeed": np.array([230.0, 230.0, 220.0, 220.0]),
            "vertical_speed": np.array([5.0, 5.0, -8.0, -8.0]),
            "aircraft_mass": np.array([65000.0, 65000.0, 60000.0, 60000.0]),
            "acceleration": np.array([0.5, -0.2, 0.3, -0.4]),
            "air_temperature": isa.air_temperature(np.array([10000.0, 10000.0, 9000.0, 9000.0]))
            + 15.0,
        }
        from_acceleration = TrajectoryData(data).with_aircraft(ps_model)
        f_data = {k: v for k, v in data.items() if k != "acceleration"}
        f_data["f"] = from_acceleration.f
        from_f = TrajectoryData(f_data).with_aircraft(ps_model)

        np.testing.assert_allclose(from_acceleration.thrust, from_f.thrust, rtol=1e-12)

    def test_fuel_flow(self, basic_trajectory_data, ps_model):
        """Test fuel_flow computed by model."""
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        ff = perf.fuel_flow
        assert np.all(ff > 0)

    def test_contains(self, basic_trajectory_data, ps_model):
        """Test __contains__ for input data and TrajectoryData properties.

        NOTE: __contains__ only returns True for input data and TrajectoryData
        computed properties (air_temperature, mach_number, etc.), NOT for
        AircraftTrajectoryData computed properties (drag, thrust, etc.).
        This is intentional to allow models to check for explicit user overrides.
        Use data["key"] or data.get("key") to access computed properties.
        """
        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        # From input data
        assert "altitude" in perf
        assert "true_airspeed" in perf

        # From TrajectoryData computed properties
        assert "mach_number" in perf
        assert "air_temperature" in perf

        # AircraftTrajectoryData computed properties are accessible but NOT in __contains__
        # This prevents recursion in model override checks
        assert perf["drag"] is not None  # Can access via __getitem__
        assert perf["thrust"] is not None
        assert perf["fuel_flow"] is not None
        assert perf.get("drag") is not None  # Can access via get()

        # These return False to allow model override patterns
        assert "drag" not in perf
        assert "thrust" not in perf


# --- Validation Decorator Tests ---


class TestValidationDecorator:
    """Tests for @requires decorator and StateSchema."""

    def test_state_schema_validates_required_fields(self):
        """Test StateSchema.validate raises error for missing fields."""
        schema = StateSchema(required=frozenset({"altitude", "mach_number"}))

        # Valid data
        valid_data = {"altitude": 10000.0, "mach_number": 0.78, "extra": 1.0}
        schema.validate(valid_data)  # Should not raise

        # Missing field
        invalid_data = {"altitude": 10000.0}
        with pytest.raises(MissingFieldError, match="mach_number"):
            schema.validate(invalid_data, method="test_method")

    def test_requires_decorator_validates_input(self):
        """Test @requires decorator validates data before calling method."""

        class TestModel:
            @requires("altitude", "mach_number")
            def compute(self, data):
                return data["altitude"] * data["mach_number"]

        model = TestModel()

        # Valid data
        result = model.compute({"altitude": 10000.0, "mach_number": 0.78})
        assert result == pytest.approx(7800.0)

        # Missing field
        with pytest.raises(MissingFieldError, match="altitude"):
            model.compute({"mach_number": 0.78})

    def test_requires_decorator_preserves_function_name(self):
        """Test @requires preserves function metadata."""

        class TestModel:
            @requires("x", "y")
            def my_function(self, data):
                """Docstring."""
                return data["x"] + data["y"]

        model = TestModel()
        assert model.my_function.__name__ == "my_function"
        assert model.my_function.__doc__ == "Docstring."

    def test_get_required_fields_returns_schema(self):
        """Test get_required_fields returns decorated function's required fields."""

        class TestModel:
            @requires("a", "b", "c")
            def method_with_requires(self, data):
                pass

            def method_without_requires(self, data):
                pass

        model = TestModel()

        fields = get_required_fields(model.method_with_requires)
        assert fields == frozenset({"a", "b", "c"})

        fields_none = get_required_fields(model.method_without_requires)
        assert fields_none == frozenset()

    def test_requires_with_trajectory_data(self, basic_trajectory_data, ps_model):
        """Test @requires works with TrajectoryData objects."""

        class TestModel:
            @requires("altitude", "true_airspeed")
            def compute(self, data):
                return data["altitude"] + data["true_airspeed"]

        model = TestModel()

        traj = TrajectoryData(basic_trajectory_data)
        perf = traj.with_aircraft(ps_model)

        result = model.compute(perf)
        expected = basic_trajectory_data["altitude"] + basic_trajectory_data["true_airspeed"]
        np.testing.assert_array_almost_equal(result, expected)
