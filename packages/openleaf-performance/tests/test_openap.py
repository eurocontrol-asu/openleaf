"""
Tests for the OpenAP aircraft performance model wrapper.

Tests cover aerodynamics, engine model, and CasADi symbolic compatibility.
"""

import casadi as ca
import numpy as np
import pytest

from openleaf.performance.core.perfos import EngineType
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


# Check if openap is available
pytest.importorskip("openap", reason="openap not installed")

from openleaf.performance.models.openap import OpenAPModel


# --- Fixtures ---


@pytest.fixture
def openap_model() -> OpenAPModel:
    """Create OpenAP model for A320."""
    return OpenAPModel("A320")


@pytest.fixture
def cruise_state() -> dict:
    """Typical FL350 cruise state."""
    h = 10668.0
    mach = 0.78
    theta = isa.theta(h)  # Temperature ratio [-]
    delta = isa.delta(h)  # Pressure ratio [-]
    T = isa.air_temperature(h)  # Absolute temperature [K]
    tas = mach * np.sqrt(isa.GAMMA * isa.R * T)
    return {
        "altitude": h,
        "mach_number": mach,
        "true_airspeed": tas,
        "air_temperature_ratio": theta,
        "air_pressure_ratio": delta,
        "aircraft_mass": 65000.0,
        "thrust": 30000.0,
        "vertical_speed": 0.0,
        "f": 1.0,
        "aerodynamic_configuration": AerodynamicConfig.CR,
    }


# --- Tests ---


class TestOpenAPModel:
    """Tests for OpenAP model creation."""

    def test_model_creation(self, openap_model: OpenAPModel):
        """Test model creation and properties."""
        assert openap_model.model == "A320"
        assert openap_model.engine_type == EngineType.JET
        assert openap_model.mtom > openap_model.oem > 0
        assert openap_model.engine.n_engines > 0

    def test_available_aircraft(self):
        """Test that available aircraft list is populated."""
        aircraft = OpenAPModel.available_aircraft()
        assert len(aircraft) > 0
        assert "a320" in aircraft


class TestOpenAPAerodynamics:
    """Tests for OpenAP aerodynamic model."""

    def test_drag_increases_with_mass(self, openap_model: OpenAPModel, cruise_state: dict):
        """Test induced drag behavior with mass."""
        cds = [
            openap_model.aerodynamics.drag_coef({**cruise_state, "aircraft_mass": m})
            for m in [55000, 65000, 75000]
        ]
        assert np.all(np.diff(cds) > 0)


class TestOpenAPEngine:
    """Tests for OpenAP engine model."""

    def test_thrust_envelope(self, openap_model: OpenAPModel, cruise_state: dict):
        """Test thrust min < max."""
        thrust_max = openap_model.engine.thrust_max(cruise_state)
        thrust_min = openap_model.engine.thrust_min(cruise_state)

        assert thrust_max > thrust_min >= 0

    def test_thrust_decreases_with_altitude(self, openap_model: OpenAPModel):
        """Test altitude effect on thrust."""
        thrusts = []
        for h in [0, 5000, 10000]:
            theta = isa.theta(h)  # Temperature ratio [-]
            delta = isa.delta(h)  # Pressure ratio [-]
            T = isa.air_temperature(h)  # Absolute temperature [K]
            tas = 0.6 * np.sqrt(isa.GAMMA * isa.R * T)
            state = {
                "altitude": h,
                "mach_number": 0.6,
                "true_airspeed": tas,
                "air_temperature_ratio": theta,
                "air_pressure_ratio": delta,
                "vertical_speed": 0.0,
            }
            thrusts.append(openap_model.engine.thrust_max(state))

        assert np.all(np.diff(thrusts) < 0)

    def test_fuel_flow_increases_with_mass(self, openap_model: OpenAPModel, cruise_state: dict):
        """Test fuel flow increases with mass (higher drag → higher thrust)."""
        altitude = cruise_state["altitude"]
        tas = cruise_state["true_airspeed"]
        delta = isa.delta(altitude)
        theta = isa.theta(altitude)
        rho = (delta / theta) * isa.RHO_SL
        q = 0.5 * rho * tas**2
        S = openap_model.aerodynamics.surface_area

        fuel_flows = []
        for m in [55000, 65000, 75000]:
            # Compute drag coefficient at this mass (in level flight: L = W, D = T)
            state_with_mass = {
                **cruise_state,
                "aircraft_mass": m,
            }
            cd = openap_model.aerodynamics.drag_coef(state_with_mass)
            # Thrust required for level flight equals drag
            thrust_required = cd * q * S
            # Fuel flow at that thrust
            ff = openap_model.engine.fuel_flow({"thrust": thrust_required})
            fuel_flows.append(ff)
        assert np.all(np.diff(fuel_flows) > 0)


class TestOpenAPIntegration:
    """Integration tests for OpenAP model."""

    def test_cruise_performance(self, openap_model: OpenAPModel):
        """Test complete cruise calculation chain."""
        h, mach, mass = 10668.0, 0.78, 65000.0
        theta = isa.theta(h)  # Temperature ratio
        delta = isa.delta(h)  # Pressure ratio
        T = isa.air_temperature(h)  # Absolute temperature [K]
        p = isa.air_pressure(h)  # Absolute pressure [Pa]
        tas = mach * np.sqrt(isa.GAMMA * isa.R * T)

        cl = openap_model.lift_coefficient(mass, p, mach)

        state = {
            "altitude": h,
            "mach_number": mach,
            "true_airspeed": tas,
            "air_temperature_ratio": theta,
            "air_pressure_ratio": delta,
            "aircraft_mass": mass,
            "vertical_speed": 0.0,
            "f": 1.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        cd = openap_model.aerodynamics.drag_coef(state)
        thrust = openap_model.compute_thrust_required(mass, cl, cd)

        state["thrust"] = thrust
        fuel_flow = openap_model.engine.fuel_flow(state)

        # Verify reasonable values
        assert 0.3 < cl < 0.7
        assert 0.01 < cd < 0.05
        assert 20000 < thrust < 60000
        assert 0.3 < fuel_flow < 2.0

        # L/D should be reasonable
        assert 10 < cl / cd < 25


class TestOpenAPSymbolic:
    """Tests for CasADi symbolic compatibility."""

    def test_symbolic_thrust(self, openap_model: OpenAPModel):
        """Test symbolic thrust computation.

        Note: vertical_speed must be numeric because OpenAP uses Python
        control flow (if/else) on this value, which is not CasADi-compatible.
        """
        mach_sym = ca.MX.sym("mach")
        tas_sym = ca.MX.sym("tas")
        alt_sym = ca.MX.sym("alt")

        state = {
            "mach_number": mach_sym,
            "true_airspeed": tas_sym,
            "altitude": alt_sym,
            "vertical_speed": 0.0,  # Must be numeric for OpenAP's if/else logic
        }
        thrust = openap_model.engine.thrust_max(state)
        assert isinstance(thrust, ca.MX)

    def test_symbolic_drag(self, openap_model: OpenAPModel):
        """Test symbolic drag computation with string config."""
        mass_sym = ca.MX.sym("mass")
        tas_sym = ca.MX.sym("tas")
        alt_sym = ca.MX.sym("alt")

        state = {
            "altitude": alt_sym,
            "true_airspeed": tas_sym,
            "aircraft_mass": mass_sym,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        cd = openap_model.aerodynamics.drag_coef(state)
        assert isinstance(cd, ca.MX)

    def test_symbolic_drag_with_symbolic_config(self, openap_model: OpenAPModel):
        """Test that drag computation is CasADi-compatible.

        The drag_coef method computes both clean and nonclean drag and uses
        np.where to select based on configuration. This enables:
        1. Symbolic inputs (mass, tas, altitude) with concrete config strings
        2. Future support for configuration-switching in trajectory optimization

        Both clean and nonclean configurations are evaluated internally,
        making the function differentiable even when the configuration
        affects which physics model is used.
        """
        mass_sym = ca.MX.sym("mass")
        tas_sym = ca.MX.sym("tas")
        alt_sym = ca.MX.sym("alt")

        # Test clean configuration with symbolic inputs
        state_clean = {
            "altitude": alt_sym,
            "true_airspeed": tas_sym,
            "aircraft_mass": mass_sym,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        cd_clean = openap_model.aerodynamics.drag_coef(state_clean)
        assert isinstance(cd_clean, ca.MX)

        # Test nonclean configuration with symbolic inputs
        state_nonclean = {
            "altitude": alt_sym,
            "true_airspeed": tas_sym,
            "aircraft_mass": mass_sym,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.AP,  # Non-clean config
            "flap_angle": 15.0,
            "landing_gear": True,
        }
        cd_nonclean = openap_model.aerodynamics.drag_coef(state_nonclean)
        assert isinstance(cd_nonclean, ca.MX)

        # Verify both can be evaluated numerically
        f_clean = ca.Function("cd_clean", [mass_sym, tas_sym, alt_sym], [cd_clean])
        f_nonclean = ca.Function("cd_nonclean", [mass_sym, tas_sym, alt_sym], [cd_nonclean])

        result_clean = float(f_clean(65000, 250, 10000))
        result_nonclean = float(f_nonclean(65000, 250, 10000))

        # Non-clean should have higher drag due to flaps and gear
        assert result_nonclean > result_clean
        assert result_clean > 0
        assert result_nonclean > 0

    def test_aerodynamic_config_enum(self, openap_model: OpenAPModel):
        """Test drag computation accepts AerodynamicConfig enum.

        AerodynamicConfig enum is the standard interface across all models.
        OpenAP uses it to select clean vs nonclean drag model.

        For non-clean configs, flap_angle and landing_gear must be provided
        explicitly since OpenAP does not define standard values per config.
        """
        from openleaf.performance.core.types import AerodynamicConfig

        state_base = {
            "altitude": 10000.0,
            "true_airspeed": 250.0,
            "aircraft_mass": 65000.0,
            "vertical_speed": 0.0,
        }

        # Cruise config → clean (no flaps/gear needed)
        state_cr = {**state_base, "aerodynamic_configuration": AerodynamicConfig.CR}
        cd_cruise = openap_model.aerodynamics.drag_coef(state_cr)

        # Non-clean configs require explicit flap_angle and landing_gear
        state_to = {
            **state_base,
            "aerodynamic_configuration": AerodynamicConfig.TO,
            "flap_angle": 15.0,
            "landing_gear": False,
        }
        cd_takeoff = openap_model.aerodynamics.drag_coef(state_to)

        state_ld = {
            **state_base,
            "aerodynamic_configuration": AerodynamicConfig.LD,
            "flap_angle": 35.0,
            "landing_gear": True,
        }
        cd_landing = openap_model.aerodynamics.drag_coef(state_ld)

        # Verify: clean < nonclean (with increasing flaps/gear)
        assert cd_cruise < cd_takeoff
        assert cd_takeoff < cd_landing


class TestOpenAPMultipleAircraft:
    """Test OpenAP with different aircraft types."""

    @pytest.mark.parametrize("aircraft", ["A320", "B738", "A333"])
    def test_different_aircraft(self, aircraft: str):
        """Test model creation for different aircraft."""
        model = OpenAPModel(aircraft)
        assert model.mtom > 0
        assert model.engine.n_engines > 0


# --- OpenAP Validation Tests (against reference openap library) ---

# Tolerance for comparison against reference openap implementation.
# Note: Small differences arise from ISA model variations between openleaf (standard ISA)
# and OpenAP (smoothed tropopause with slightly different density formula).
# At high altitudes (FL400+), these differences can reach ~0.3%.
OPENAP_TOLERANCE = 0.004  # 0.4% to account for ISA model differences


@pytest.fixture
def openap_refs():
    """Load reference OpenAP models.

    Note: We use the standard OpenAP models (not openap.casadi) because
    openleaf.performance.models.openap now uses standard OpenAP with openleaf.performance.numpy injection
    for CasADi compatibility, rather than the separate openap.casadi module.
    """
    from openap import Drag
    from openap import FuelFlow
    from openap import Thrust

    return {
        "drag": Drag("A320"),
        "thrust": Thrust("A320"),
        "fuel_flow": FuelFlow("A320"),
    }


class TestOpenAPModelValidation:
    """Validate OpenAP model against openap library.

    These tests ensure our wrapper produces IDENTICAL results to the
    original OpenAP library by junzi Sun for the same flight conditions.
    """

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt,mass",
        [
            (35000, 450, 65000),  # Typical cruise
            (35000, 450, 55000),  # Light
            (35000, 450, 73000),  # Heavy
            (30000, 400, 65000),  # Lower altitude
            (40000, 450, 65000),  # Higher altitude
            (20000, 350, 65000),  # FL200
            (10000, 300, 65000),  # FL100
        ],
    )
    def test_drag_matches_openap(self, openap_model, openap_refs, altitude_ft, tas_kt, mass):
        """Test drag matches openap library within tolerance."""
        altitude_m = altitude_ft * 0.3048
        theta = isa.theta(altitude_m)  # Temperature ratio [-]
        delta = isa.delta(altitude_m)  # Pressure ratio [-]

        state = {
            "aircraft_mass": mass,
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }

        # Reference OpenAP
        drag_ref_N = float(openap_refs["drag"].clean(mass=mass, tas=tas_kt, alt=altitude_ft, vs=0))

        # OpenLeaf - clean configuration specified in state dict
        drag_coef = openap_model.aerodynamics.drag_coef(state)
        q = 0.5 * (delta / theta) * isa.RHO_SL * (tas_kt * units.KT) ** 2
        drag_ae_N = drag_coef * q * openap_model.aerodynamics.surface_area

        assert drag_ae_N == pytest.approx(drag_ref_N, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt {mass}kg: "
            f"drag {drag_ae_N:.0f} N != openap {drag_ref_N:.0f} N"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt,mass,vs_fpm",
        [
            (35000, 450, 65000, 0),  # Level flight
            (35000, 450, 65000, 500),  # Slight climb
            (35000, 450, 65000, -500),  # Descent
            (20000, 350, 65000, 1500),  # Climb at FL200
            (10000, 300, 70000, 2000),  # Strong climb at FL100
        ],
    )
    def test_drag_with_vertical_speed_matches_openap(
        self, openap_model, openap_refs, altitude_ft, tas_kt, mass, vs_fpm
    ):
        """Test drag with vertical speed matches openap library."""
        altitude_m = altitude_ft * 0.3048
        vs_ms = vs_fpm * units.FPM
        theta = isa.theta(altitude_m)
        delta = isa.delta(altitude_m)

        state = {
            "aircraft_mass": mass,
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
            "vertical_speed": vs_ms,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }

        # Reference OpenAP
        drag_ref_N = float(
            openap_refs["drag"].clean(mass=mass, tas=tas_kt, alt=altitude_ft, vs=vs_fpm)
        )

        # OpenLeaf
        drag_coef = openap_model.aerodynamics.drag_coef(state)
        q = 0.5 * (delta / theta) * isa.RHO_SL * (tas_kt * units.KT) ** 2
        drag_ae_N = drag_coef * q * openap_model.aerodynamics.surface_area

        assert drag_ae_N == pytest.approx(drag_ref_N, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt VS{vs_fpm}fpm: "
            f"drag {drag_ae_N:.0f} N != openap {drag_ref_N:.0f} N"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt,mass",
        [
            (35000, 450, 65000),
            (35000, 450, 55000),
            (35000, 450, 73000),
            (30000, 400, 65000),
            (40000, 450, 65000),
        ],
    )
    def test_fuel_flow_matches_openap(self, openap_model, openap_refs, altitude_ft, tas_kt, mass):
        """Test fuel flow matches openap library within tolerance.

        This test uses the thrust-based fuel_flow method. We compute thrust
        from drag and level flight condition, then compare fuel flow.
        """
        altitude_m = altitude_ft * 0.3048
        tas_ms = tas_kt * units.KT

        # Compute drag coefficient
        state = {
            "aircraft_mass": mass,
            "altitude": altitude_m,
            "true_airspeed": tas_ms,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        cd = openap_model.aerodynamics.drag_coef(state)

        # Compute thrust for level flight (T = D = q * S * CD)
        delta = isa.delta(altitude_m)
        theta = isa.theta(altitude_m)
        rho = (delta / theta) * isa.RHO_SL
        q = 0.5 * rho * tas_ms**2
        S = openap_model.aerodynamics.surface_area
        thrust = cd * q * S

        # OpenLeaf fuel flow from thrust
        ff_ae = float(openap_model.engine.fuel_flow({"thrust": thrust}))

        # Reference OpenAP - use at_thrust for fair comparison
        ff_ref = float(openap_refs["fuel_flow"].at_thrust(thrust))

        assert ff_ae == pytest.approx(ff_ref, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt {mass}kg: "
            f"fuel {ff_ae:.4f} kg/s != openap {ff_ref:.4f} kg/s"
        )

    @pytest.mark.parametrize(
        "thrust",
        [10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000],
    )
    def test_fuel_flow_at_thrust_matches_openap(self, openap_model, openap_refs, thrust):
        """Test fuel_flow.at_thrust matches openap for various thrust levels."""
        # OpenLeaf
        ff_ae = float(openap_model.engine.fuel_flow({"thrust": thrust}))

        # Reference OpenAP
        ff_ref = float(openap_refs["fuel_flow"].at_thrust(thrust))

        assert ff_ae == pytest.approx(ff_ref, rel=OPENAP_TOLERANCE), (
            f"Thrust {thrust}N: fuel {ff_ae:.4f} kg/s != openap {ff_ref:.4f} kg/s"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt",
        [
            (0, 150),  # Takeoff
            (5000, 250),
            (10000, 300),
            (20000, 350),
            (30000, 400),
            (35000, 450),
            (40000, 450),
        ],
    )
    def test_takeoff_thrust_matches_openap(self, openap_model, openap_refs, altitude_ft, tas_kt):
        """Test takeoff thrust matches openap library."""
        from openleaf.performance.core.perfos import EngineSetting

        altitude_m = altitude_ft * 0.3048

        state = {
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
            "vertical_speed": 0.0,
            "engine_setting": EngineSetting.MTKF,
        }

        # Reference OpenAP
        thrust_ref = float(openap_refs["thrust"].takeoff(tas=tas_kt, alt=altitude_ft))

        # OpenLeaf with MTKF setting
        thrust_ae = float(openap_model.engine.thrust_max(state))

        assert thrust_ae == pytest.approx(thrust_ref, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt: "
            f"takeoff thrust {thrust_ae:.0f} N != openap {thrust_ref:.0f} N"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt,roc_fpm",
        [
            (10000, 300, 1500),
            (20000, 350, 1000),
            (30000, 400, 500),
            (35000, 450, 0),  # Cruise = climb with ROC=0
        ],
    )
    def test_climb_thrust_matches_openap(
        self, openap_model, openap_refs, altitude_ft, tas_kt, roc_fpm
    ):
        """Test climb thrust matches openap library."""
        from openleaf.performance.core.perfos import EngineSetting

        altitude_m = altitude_ft * 0.3048
        vs_ms = roc_fpm * units.FPM

        state = {
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
            "vertical_speed": vs_ms,
            "engine_setting": EngineSetting.MCMB,  # Max climb
        }

        # Reference OpenAP
        thrust_ref = float(openap_refs["thrust"].climb(tas=tas_kt, alt=altitude_ft, roc=roc_fpm))

        # OpenLeaf with MCMB setting
        thrust_ae = float(openap_model.engine.thrust_max(state))

        assert thrust_ae == pytest.approx(thrust_ref, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt ROC{roc_fpm}fpm: "
            f"climb thrust {thrust_ae:.0f} N != openap {thrust_ref:.0f} N"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt",
        [
            (35000, 450),
            (30000, 400),
            (40000, 450),
        ],
    )
    def test_cruise_thrust_matches_openap(self, openap_model, openap_refs, altitude_ft, tas_kt):
        """Test cruise thrust (climb with ROC=0) matches openap library."""
        altitude_m = altitude_ft * 0.3048

        state = {
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
            "vertical_speed": 0.0,
            # No engine_setting = default = climb with ROC=0
        }

        # Reference OpenAP - cruise = climb(roc=0)
        thrust_ref = float(openap_refs["thrust"].cruise(tas=tas_kt, alt=altitude_ft))

        # OpenLeaf default (no engine_setting)
        thrust_ae = float(openap_model.engine.thrust_max(state))

        assert thrust_ae == pytest.approx(thrust_ref, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt: "
            f"cruise thrust {thrust_ae:.0f} N != openap {thrust_ref:.0f} N"
        )

    @pytest.mark.parametrize(
        "altitude_ft,tas_kt",
        [
            (35000, 450),
            (30000, 400),
            (20000, 350),
            (10000, 300),
        ],
    )
    def test_idle_thrust_matches_openap(self, openap_model, openap_refs, altitude_ft, tas_kt):
        """Test idle/descent thrust matches openap library."""
        altitude_m = altitude_ft * 0.3048

        state = {
            "altitude": altitude_m,
            "true_airspeed": tas_kt * units.KT,
        }

        # Reference OpenAP
        thrust_ref = float(openap_refs["thrust"].descent_idle(tas=tas_kt, alt=altitude_ft))

        # OpenLeaf
        thrust_ae = float(openap_model.engine.thrust_min(state))

        assert thrust_ae == pytest.approx(thrust_ref, rel=OPENAP_TOLERANCE), (
            f"FL{altitude_ft / 100:.0f} TAS{tas_kt}kt: "
            f"idle thrust {thrust_ae:.0f} N != openap {thrust_ref:.0f} N"
        )


# --- Cross-Model Comparison Tests ---


class TestCrossModelComparison:
    """Compare PS and OpenAP models for consistency."""

    @pytest.fixture
    def ps_model(self):
        """Create PS model for A320 from pycontrails parameters."""
        pytest.importorskip("pycontrails")
        from openleaf.performance.models.ps import PSModel

        return PSModel.from_pycontrails("A320")

    def test_drag_trends_consistent(self, ps_model, openap_model):
        """Test that both models show similar drag trends with altitude."""
        pytest.importorskip("pycontrails")

        mass = 65000
        mach = 0.78
        altitudes = [5000, 10000, 12000]  # meters

        ps_drags = []
        openap_drags = []

        for h in altitudes:
            theta = isa.air_temperature(h)
            delta = isa.air_pressure(h)
            T = theta * isa.TAU_SL
            p = delta * isa.P_SL

            # PS drag
            c_lift = ps_model.lift_coefficient(mass, p, mach)
            c_drag = ps_model.aerodynamics.drag_coef_with_state(
                cl=c_lift, mach=mach, air_temperature=T, air_pressure=p
            )
            ps_drags.append(float(c_drag))

            # OpenAP drag (need TAS for OpenAP)
            a = np.sqrt(1.4 * 287.05 * T)
            tas = mach * a
            state = {
                "aircraft_mass": mass,
                "altitude": h,
                "true_airspeed": tas,
                "vertical_speed": 0.0,
                "air_temperature_ratio": theta,
                "air_pressure_ratio": delta,
                "aerodynamic_configuration": AerodynamicConfig.CR,
            }
            # Clean configuration specified in state dict
            openap_cd = openap_model.aerodynamics.drag_coef(state)
            openap_drags.append(float(openap_cd))

        # Both should show increasing drag with altitude (density decreases, CL increases)
        assert all(ps_drags[i] < ps_drags[i + 1] for i in range(len(ps_drags) - 1)), (
            "PS drag should increase with altitude"
        )

        assert all(openap_drags[i] < openap_drags[i + 1] for i in range(len(openap_drags) - 1)), (
            "OpenAP drag should increase with altitude"
        )

    def test_fuel_flow_in_reasonable_range(self, ps_model, openap_model):
        """Test that both models give fuel flow in reasonable range for A320."""
        pytest.importorskip("pycontrails")

        # Typical cruise: FL350, M0.78, 65t
        altitude_m = 10668
        mach = 0.78
        mass = 65000

        T = isa.air_temperature(altitude_m)  # Absolute temperature [K]
        p = isa.air_pressure(altitude_m)  # Absolute pressure [Pa]
        theta = isa.theta(altitude_m)  # Temperature ratio [-]
        delta = isa.delta(altitude_m)  # Pressure ratio [-]
        a = np.sqrt(isa.GAMMA * isa.R * T)
        tas = mach * a

        # PS fuel flow
        c_lift = ps_model.lift_coefficient(mass, p, mach)
        c_drag = ps_model.aerodynamics.drag_coef_with_state(
            cl=c_lift, mach=mach, air_temperature=T, air_pressure=p
        )
        thrust_req = ps_model.compute_thrust_required(mass, c_lift, c_drag)

        ps_state = {
            "mach_number": mach,
            "altitude": altitude_m,
            "air_temperature_ratio": theta,
            "air_pressure_ratio": delta,
            "air_temperature": T,
            "air_pressure": p,
            "thrust": thrust_req,
        }
        ff_ps = float(ps_model.engine.fuel_flow(ps_state))

        # OpenAP fuel flow - compute thrust from drag
        openap_state = {
            "aircraft_mass": mass,
            "altitude": altitude_m,
            "true_airspeed": tas,
            "vertical_speed": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        cd = openap_model.aerodynamics.drag_coef(openap_state)
        rho = (delta / theta) * isa.RHO_SL
        q = 0.5 * rho * tas**2
        S = openap_model.aerodynamics.surface_area
        thrust_openap = cd * q * S
        ff_openap = float(openap_model.engine.fuel_flow({"thrust": thrust_openap}))

        # A320 cruise fuel flow should be roughly 0.5-1.0 kg/s per engine
        # Total: 1.0-2.0 kg/s for twin engine
        # More typical: 0.6-0.8 kg/s
        assert 0.4 < ff_ps < 1.2, f"PS fuel flow {ff_ps} outside reasonable range"
        assert 0.4 < ff_openap < 1.2, f"OpenAP fuel flow {ff_openap} outside reasonable range"

        # Both should be in similar ballpark (within 30% of each other)
        ratio = ff_ps / ff_openap
        assert 0.7 < ratio < 1.4, (
            f"PS/OpenAP fuel ratio {ratio} outside expected range "
            f"(PS: {ff_ps:.3f}, OpenAP: {ff_openap:.3f})"
        )


class TestOpenAPCruiseTrajectory:
    """Tests for OpenAP fuel flow with cruise trajectory data."""

    def test_fuel_flow_with_cruise_trajectory(
        self, openap_model: OpenAPModel, a320_cruise_trajectory: dict
    ) -> None:
        """Test fuel_flow returns reasonable values for A320 cruise trajectory.

        Uses 10 cruise samples with realistic A320 data at FL350.
        Tests the full TrajectoryData → AircraftTrajectoryData → fuel_flow chain.
        Validates that fuel flow values are within expected physical bounds.
        """
        from openleaf.performance.core.trajectory import TrajectoryData

        # Create TrajectoryData from cruise samples
        traj = TrajectoryData(a320_cruise_trajectory)

        # Attach aircraft model and get performance data
        perf = traj.with_aircraft(openap_model)

        # Access fuel_flow - this triggers the entire computation chain:
        # altitude → atmosphere → drag → thrust (via TEM) → fuel_flow
        fuel_flow = np.asarray(perf.fuel_flow).flatten()

        # Validate output shape
        assert len(fuel_flow) == 10, "Expected 10 samples"

        # Validate fuel flow is positive
        assert np.all(fuel_flow > 0), "Fuel flow must be positive"

        # A320 twin-engine cruise fuel flow: ~0.5-1.2 kg/s total
        # (roughly 0.25-0.6 kg/s per engine)
        assert np.all(fuel_flow > 0.3), f"Fuel flow too low: min={np.min(fuel_flow):.3f} kg/s"
        assert np.all(fuel_flow < 1.5), f"Fuel flow too high: max={np.max(fuel_flow):.3f} kg/s"

        # Fuel flow should decrease as mass decreases (less drag)
        # Allow small variations due to numerical effects
        ff_diff = np.diff(fuel_flow)
        assert np.mean(ff_diff) < 0, "Fuel flow should generally decrease as mass decreases"


class TestOpenAPLimits:
    """Tests for OpenAP aircraft limits."""

    def test_max_altitude_exact(self, openap_model: OpenAPModel):
        """Verify max_altitude returns HMO in feet.

        OpenAP ceiling field is in meters (A320: 12500 m).
        The model converts to feet: 12500 * 3.28084 = 41010 ft.
        max_altitude() must return this in feet, not raw meters.
        """
        max_alt = openap_model.limits.max_altitude({})

        # A320 HMO: 12500 m = 41010 ft
        assert max_alt == pytest.approx(41010.5, rel=1e-3), (
            f"max_altitude should be ~41010 ft (12500m * 3.28084), got {max_alt}"
        )
        # Sanity: must be in feet range (30,000-50,000 ft), not meters (8,000-15,000)
        assert max_alt > 30_000, "HMO must be > 30000 ft (not in meters)"

    def test_max_cas_exact(self, openap_model: OpenAPModel):
        """Verify max_cas returns VMO in knots.

        OpenAP vmo field is already in knots for A320: 350 kt.
        """
        max_cas = openap_model.limits.max_cas({})
        assert max_cas == pytest.approx(350.0), f"VMO should be 350 kt, got {max_cas}"

    def test_max_mach_exact(self, openap_model: OpenAPModel):
        """Verify max_mach returns MMO dimensionless.

        OpenAP mmo field: A320 = 0.82.
        """
        max_mach = openap_model.limits.max_mach({})
        assert max_mach == pytest.approx(0.82), f"MMO should be 0.82, got {max_mach}"

    def test_limits_consistent_with_aircraft_data(self, openap_model: OpenAPModel):
        """Verify limits match the aircraft data stored in the model."""
        assert openap_model.limits.data.hmo == pytest.approx(openap_model.limits.max_altitude({}))
        assert openap_model.limits.data.vmo == pytest.approx(openap_model.limits.max_cas({}))
        assert openap_model.limits.data.mmo == pytest.approx(openap_model.limits.max_mach({}))

    def test_wtc_classification(self):
        """Verify Wake Turbulence Category classification thresholds.

        ICAO Doc 4444:
          H (Heavy):  MTOW > 136,000 kg
          M (Medium): 7,000 kg < MTOW <= 136,000 kg
          L (Light):  MTOW <= 7,000 kg
        """
        # A380 (MTOW ~575,000 kg) -> Heavy
        a380 = OpenAPModel("A388")
        assert a380.wtc == "H", f"A380 should be Heavy, got {a380.wtc}"

        # A320 (MTOW ~78,000 kg) -> Medium
        a320 = OpenAPModel("A320")
        assert a320.wtc == "M", f"A320 should be Medium, got {a320.wtc}"
