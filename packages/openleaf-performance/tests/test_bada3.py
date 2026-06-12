"""
Tests for BADA3 aircraft performance models.

Tests cover parsing, engine models (jet, turboprop, piston), aerodynamics,
and CasADi symbolic compatibility.
"""

import pytest


# Skip module if BADA3 not configured (proprietary data required)
try:
    import openleaf.performance

    openleaf.performance.get_bada3_path()
except Exception:
    pytest.skip("BADA3 data not configured", allow_module_level=True)

from pathlib import Path

import casadi as ca
import numpy as np
from ptd_parser import parse_ptd_file

from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.perfos import EngineType
from openleaf.performance.models.bada3 import BADA3AeroModel
from openleaf.performance.models.bada3 import BADA3Data
from openleaf.performance.models.bada3 import BADA3JetEngineModel
from openleaf.performance.models.bada3 import BADA3Model
from openleaf.performance.models.bada3 import BADA3PistonEngineModel
from openleaf.performance.models.bada3 import BADA3TurbopropEngineModel
from openleaf.performance.models.bada3 import parse_bada3_file
from openleaf.performance.models.bada_types import AerodynamicConfig
from openleaf.performance.physics import isa
from openleaf.performance.physics.units import FT
from openleaf.performance.physics.units import KT


class TestBADA3Parsing:
    """Tests for BADA3 file parsing."""

    def test_parse_jet_aircraft(self, bada3_a320_path):
        """Test parsing A320 jet aircraft data."""
        data = parse_bada3_file(str(bada3_a320_path))

        assert isinstance(data, BADA3Data)
        assert data.aircraft.engine_type == EngineType.JET
        assert data.aircraft.model == "A320__"
        assert data.aerodynamics.surface_area > 0
        assert data.engine.ct_1 > 0

    def test_parse_turboprop_aircraft(self, bada3_at72_path):
        """Test parsing ATR72 turboprop aircraft data."""
        data = parse_bada3_file(str(bada3_at72_path))

        assert data.aircraft.engine_type == EngineType.TURBOPROP
        assert data.aircraft.model == "AT72__"

    def test_parse_piston_aircraft(self, bada3_sr20_path):
        """Test parsing SR20 piston aircraft data."""
        data = parse_bada3_file(str(bada3_sr20_path))

        assert data.aircraft.engine_type == EngineType.PISTON
        assert data.aircraft.model == "SR20__"

    def test_mass_section_values_exact(self, bada3_a320_path):
        """Verify A320 mass section values against known OPF values.

        OPF mass section (in tonnes):
          reference=64t  minimum=39t  maximum=77t  max_payload=21.5t  mass_grad=0.4325 ft/kg

        The 5th value is the mass gradient Gw [ft/kg], NOT a mass.
        mmax must equal mtom (BADA3 Section 3.5: hmax defined at MTOM).
        """
        data = parse_bada3_file(str(bada3_a320_path))
        lim = data.limits

        assert lim.mref == pytest.approx(64_000.0), "mref should be 64t in kg"
        assert lim.oem == pytest.approx(39_000.0), "oem should be 39t in kg"
        assert lim.mtom == pytest.approx(77_000.0), "mtom should be 77t in kg"
        assert lim.mpl == pytest.approx(21_500.0), "mpl should be 21.5t in kg"
        # mmax must equal mtom — hmax is defined at MTOM per BADA3 Section 3.5
        assert lim.mmax == pytest.approx(lim.mtom), "mmax must equal mtom"
        # gw comes from mass section 5th value (0.4325 ft/kg, no unit conversion)
        assert lim.gw == pytest.approx(0.4325), "gw should be 0.4325 ft/kg"

    def test_mass_section_gw_turboprop(self, bada3_at72_path):
        """Verify AT72 mass gradient Gw parsed correctly from mass section."""
        data = parse_bada3_file(str(bada3_at72_path))
        lim = data.limits

        assert lim.mmax == pytest.approx(lim.mtom), "mmax must equal mtom"
        assert lim.gw == pytest.approx(0.81147), "AT72 gw should be 0.81147 ft/kg"

    def test_max_altitude_mass_correction_exact(self, bada3_a320_path):
        """Verify max_altitude uses gw correctly in the correction formula.

        Formula: hmax_act = min(hmo, hmax + Gt*(dTISA-CTc4) + Gw*(mmax-mact))
        A320 values: hmax=33295, gw=0.4325, mmax=mtom=77000
        At 70t mass reduction of 7000 kg: correction = 0.4325 * 7000 = 3027.5 ft
        """
        from openleaf.performance.models.bada3.limits import BADA3LimitsModel

        data = parse_bada3_file(str(bada3_a320_path))
        lim = BADA3LimitsModel(data.limits)

        # At MTOM, ISA: no corrections, should give hmax exactly
        h_mtom = lim._max_altitude_impl(mass=77_000.0, delta_isa=0.0)
        assert h_mtom == pytest.approx(33_295.0), "At MTOM/ISA expect hmax=33295 ft"

        # At 70t lighter mass: +0.4325 * 7000 = +3027.5 ft
        h_70t = lim._max_altitude_impl(mass=70_000.0, delta_isa=0.0)
        expected = 33_295.0 + 0.4325 * 7_000.0
        assert h_70t == pytest.approx(expected), (
            f"Mass correction wrong: got {h_70t}, expected {expected}"
        )

        # Altitude must not exceed HMO (41000 ft)
        h_min_mass = lim._max_altitude_impl(mass=data.limits.oem, delta_isa=0.0)
        assert h_min_mass <= data.limits.hmo, "max_altitude must not exceed HMO"


class TestBADA3Model:
    """Tests for BADA3 model instantiation and properties."""

    def test_jet_model(self, bada3_model):
        """Test jet model instantiation and engine type selection."""
        assert isinstance(bada3_model, BADA3Model)
        assert isinstance(bada3_model.engine, BADA3JetEngineModel)
        assert isinstance(bada3_model.aerodynamics, BADA3AeroModel)
        assert bada3_model.mtom > bada3_model.oem > 0

    def test_turboprop_model(self, bada3_turboprop_model):
        """Test turboprop model instantiation."""
        assert isinstance(bada3_turboprop_model.engine, BADA3TurbopropEngineModel)

    def test_piston_model(self, bada3_piston_model):
        """Test piston model instantiation."""
        assert isinstance(bada3_piston_model.engine, BADA3PistonEngineModel)


class TestBADA3Engine:
    """Tests for BADA3 engine models."""

    @pytest.mark.parametrize(
        "model_fixture,state_fixture",
        [
            ("bada3_model", "typical_cruise_state"),
            ("bada3_turboprop_model", "turboprop_cruise_state"),
            ("bada3_piston_model", "piston_cruise_state"),
        ],
    )
    def test_thrust_envelope(self, model_fixture, state_fixture, request):
        """Test thrust min < max for all engine types."""
        model = request.getfixturevalue(model_fixture)
        state = request.getfixturevalue(state_fixture)

        thrust_max = model.engine.thrust_max(state)
        thrust_min = model.engine.thrust_min(state)

        # >= 0: idle thrust can be exactly zero per the OPF data — e.g. the
        # AT72 has CTdes,high = 0.0, and its PTD lists 0 N above Hp,des.
        assert thrust_max > thrust_min >= 0

    def test_thrust_decreases_with_altitude(self, bada3_model):
        """Test that thrust decreases with altitude (density effect)."""
        thrusts = [
            bada3_model.engine.thrust_max(
                {
                    "true_airspeed": 200.0,
                    "altitude": h,
                    "engine_setting": EngineSetting.MCMB,
                    "delta_isa": 0.0,
                }
            )
            for h in [0, 5000, 10000]
        ]
        assert thrusts[0] > thrusts[-1]

    def test_fuel_flow_increases_with_thrust(self, bada3_model):
        """Test fuel flow monotonicity with thrust."""
        state = {
            "true_airspeed": 230.0,
            "altitude": 10000.0,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
        }
        fuel_flows = [
            bada3_model.engine.fuel_flow({**state, "thrust": t}) for t in [20000, 40000, 60000]
        ]
        assert np.all(np.diff(fuel_flows) > 0)

    def test_piston_constant_fuel_flow(self, bada3_piston_model):
        """Test that piston fuel flow is constant (BADA3 model characteristic)."""
        state = {
            "true_airspeed": 70.0,
            "altitude": 2000.0,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
        }
        fuel_flows = [
            bada3_piston_model.engine.fuel_flow({**state, "thrust": t}) for t in [1000, 2000, 3000]
        ]
        assert np.allclose(fuel_flows, fuel_flows[0])


class TestBADA3Aerodynamics:
    """Tests for BADA3 aerodynamic model."""

    def test_drag_increases_with_lift(self, bada3_model):
        """Test drag polar behavior."""
        cds = [
            bada3_model.aerodynamics.drag_coef(
                {
                    "lift_coef": cl,
                    "mach_number": 0.78,
                    "aerodynamic_configuration": AerodynamicConfig.CR,
                }
            )
            for cl in [0.2, 0.4, 0.6, 0.8]
        ]
        assert np.all(np.diff(cds) > 0)

    def test_configuration_drag_ordering(self, bada3_model):
        """Test clean < takeoff < landing drag ordering."""
        state = {"lift_coef": 0.5, "mach_number": 0.5}

        cd_clean = bada3_model.aerodynamics.drag_coef(
            {**state, "aerodynamic_configuration": AerodynamicConfig.CR}
        )
        cd_to = bada3_model.aerodynamics.drag_coef(
            {**state, "aerodynamic_configuration": AerodynamicConfig.TO}
        )
        cd_ld = bada3_model.aerodynamics.drag_coef(
            {**state, "aerodynamic_configuration": AerodynamicConfig.LD}
        )

        assert cd_clean < cd_to < cd_ld


class TestBADA3Integration:
    """Integration tests for complete performance calculations."""

    def test_cruise_level_flight(self, bada3_model):
        """Test that required thrust is within envelope for level flight."""
        h, v, m = 10668.0, 230.0, 65000.0
        theta = isa.theta(h)  # temperature ratio (dimensionless)
        delta = isa.delta(h)  # pressure ratio (dimensionless)
        sigma = delta / theta  # density ratio
        mach = v / (isa.A_SL * np.sqrt(theta))  # speed of sound = A_SL * sqrt(theta)

        cl = bada3_model.lift_coef_from_inputs(v=v, m=m, sigma=sigma)
        cd = bada3_model.aerodynamics.drag_coef_from_inputs(cl=cl, mach=mach)
        drag = bada3_model.drag_from_inputs(v=v, sigma=sigma, cd=cd)

        state = {
            "true_airspeed": v,
            "altitude": h,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }
        thrust_max = bada3_model.engine.thrust_max(state)
        thrust_min = bada3_model.engine.thrust_min(state)

        assert thrust_min < drag < thrust_max


class TestBADA3Symbolic:
    """Tests for CasADi symbolic compatibility."""

    def test_symbolic_thrust_and_gradient(self, bada3_model):
        """Test symbolic thrust computation and gradient extraction."""
        v_sym, h_sym = ca.MX.sym("v"), ca.MX.sym("h")

        thrust = bada3_model.engine.thrust_max(
            {
                "true_airspeed": v_sym,
                "altitude": h_sym,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
            }
        )
        assert isinstance(thrust, ca.MX)

        # Verify gradient computation
        jac = ca.jacobian(thrust, ca.vertcat(v_sym, h_sym))
        f = ca.Function("thrust_jac", [v_sym, h_sym], [jac])
        grad = np.array(f(230.0, 10000.0)).flatten()

        assert np.all(np.isfinite(grad))
        assert grad[1] < 0  # Thrust decreases with altitude

    def test_symbolic_drag(self, bada3_model):
        """Test symbolic drag coefficient computation."""
        cl_sym, mach_sym = ca.MX.sym("cl"), ca.MX.sym("mach")

        cd = bada3_model.aerodynamics.drag_coef(
            {
                "lift_coef": cl_sym,
                "mach_number": mach_sym,
                "aerodynamic_configuration": AerodynamicConfig.CR,
            }
        )
        assert isinstance(cd, ca.MX)


# ==============================================================================
# BADA3 PTD VALIDATION TESTS
# ==============================================================================

# Test tolerances (relative)
THRUST_TOLERANCE = 0.001  # 0.1%
FUEL_TOLERANCE = 0.001  # 0.1%
DRAG_TOLERANCE = 0.001  # 0.1%

# Propeller models need higher tolerance due to propeller efficiency approximations
PROPELLER_DRAG_TOLERANCE = 0.01  # 1%


from conftest import get_atm_ratios


@pytest.fixture
def bada3_ptd(data_dir: Path):
    """Load BADA3 PTD file."""
    path = data_dir / "A320__.PTD"
    if not path.exists():
        pytest.skip(f"PTD file not found: {path}")
    return parse_ptd_file(path)


@pytest.fixture
def bada3_at72_ptd(data_dir: Path):
    """Load BADA3 AT72 turboprop PTD file."""
    path = data_dir / "AT72__.PTD"
    if not path.exists():
        pytest.skip(f"PTD file not found: {path}")
    return parse_ptd_file(path)


@pytest.fixture
def bada3_sr20_ptd(data_dir: Path):
    """Load BADA3 SR20 piston PTD file."""
    path = data_dir / "SR20__.PTD"
    if not path.exists():
        pytest.skip(f"PTD file not found: {path}")
    return parse_ptd_file(path)


class TestBADA3PerformanceTableValidation:
    """Validate BADA3 model outputs against official performance tables."""

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_thrust_matches_ptd(self, bada3_model, bada3_ptd, mass_category):
        """Test maximum thrust matches PTD table values within tolerance."""
        section = bada3_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            state = {
                "true_airspeed": tas,
                "altitude": altitude,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
            }
            thrust_model = bada3_model.engine.thrust_max(state)

            assert thrust_model == pytest.approx(row.thrust, rel=THRUST_TOLERANCE), (
                f"FL{row.fl} {mass_category}: thrust {thrust_model:.0f} N != "
                f"PTD {row.thrust:.0f} N"
            )

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_fuel_flow_matches_ptd(self, bada3_model, bada3_ptd, mass_category):
        """Test fuel flow matches PTD table values within tolerance."""
        section = bada3_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            state = {
                "true_airspeed": tas,
                "altitude": altitude,
                "thrust": row.thrust,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
            }
            fuel_model = bada3_model.engine.fuel_flow(state) * 60  # kg/s to kg/min

            assert fuel_model == pytest.approx(row.fuel_flow, rel=FUEL_TOLERANCE), (
                f"FL{row.fl} {mass_category}: fuel {fuel_model:.2f} kg/min != "
                f"PTD {row.fuel_flow:.2f} kg/min"
            )

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_drag_matches_ptd(self, bada3_model, bada3_ptd, mass_category):
        """Test drag matches PTD table values within tolerance."""
        section = bada3_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            _theta, _delta, sigma = get_atm_ratios(altitude)

            # Get aerodynamic configuration from PTD (TO, IC, CR, AP, LD)
            config = (
                AerodynamicConfig.from_string(row.config) if row.config else AerodynamicConfig.CR
            )

            cl = bada3_model.lift_coef_from_inputs(v=tas, m=row.mass, sigma=sigma)
            cd = bada3_model.aerodynamics.drag_coef(
                {
                    "lift_coef": cl,
                    "mach_number": row.mach,
                    "aerodynamic_configuration": config,
                }
            )
            drag_model = bada3_model.drag_from_inputs(v=tas, sigma=sigma, cd=cd)

            assert drag_model == pytest.approx(row.drag, rel=DRAG_TOLERANCE), (
                f"FL{row.fl} {mass_category} {config.value}: drag {drag_model:.0f} N != "
                f"PTD {row.drag:.0f} N"
            )


class TestBADA3TurbopropPTDValidation:
    """Validate BADA3 turboprop model against official performance tables."""

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_thrust_matches_ptd(self, bada3_turboprop_model, bada3_at72_ptd, mass_category):
        """Test maximum thrust matches PTD table values within tolerance."""
        section = bada3_at72_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        # Propeller models need higher tolerance due to propeller efficiency approximations
        propeller_thrust_tolerance = 0.03  # 3%

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            state = {
                "true_airspeed": tas,
                "altitude": altitude,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
            }
            thrust_model = bada3_turboprop_model.engine.thrust_max(state)

            assert thrust_model == pytest.approx(row.thrust, rel=propeller_thrust_tolerance), (
                f"FL{row.fl} {mass_category}: thrust {thrust_model:.0f} N != "
                f"PTD {row.thrust:.0f} N"
            )

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_drag_matches_ptd(self, bada3_turboprop_model, bada3_at72_ptd, mass_category):
        """Test drag matches PTD table values within tolerance."""
        section = bada3_at72_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            _theta, _delta, sigma = get_atm_ratios(altitude)

            # Get aerodynamic configuration from PTD (TO, IC, CR, AP, LD)
            config = (
                AerodynamicConfig.from_string(row.config) if row.config else AerodynamicConfig.CR
            )

            cl = bada3_turboprop_model.lift_coef_from_inputs(v=tas, m=row.mass, sigma=sigma)
            cd = bada3_turboprop_model.aerodynamics.drag_coef(
                {
                    "lift_coef": cl,
                    "mach_number": row.mach,
                    "aerodynamic_configuration": config,
                }
            )
            drag_model = bada3_turboprop_model.drag_from_inputs(v=tas, sigma=sigma, cd=cd)

            assert drag_model == pytest.approx(row.drag, rel=PROPELLER_DRAG_TOLERANCE), (
                f"FL{row.fl} {mass_category} {config.value}: drag {drag_model:.0f} N != "
                f"PTD {row.drag:.0f} N"
            )


class TestBADA3PistonPTDValidation:
    """Validate BADA3 piston model against official performance tables."""

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_thrust_matches_ptd(self, bada3_piston_model, bada3_sr20_ptd, mass_category):
        """Test maximum thrust matches PTD table values within tolerance."""
        section = bada3_sr20_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        # Propeller models need higher tolerance due to propeller efficiency approximations
        propeller_thrust_tolerance = 0.03  # 3%

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            state = {
                "true_airspeed": tas,
                "altitude": altitude,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
            }
            thrust_model = bada3_piston_model.engine.thrust_max(state)

            assert thrust_model == pytest.approx(row.thrust, rel=propeller_thrust_tolerance), (
                f"FL{row.fl} {mass_category}: thrust {thrust_model:.0f} N != "
                f"PTD {row.thrust:.0f} N"
            )

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_drag_matches_ptd(self, bada3_piston_model, bada3_sr20_ptd, mass_category):
        """Test drag matches PTD table values within tolerance."""
        section = bada3_sr20_ptd.get_section(mass_category, "climb")
        if section is None:
            pytest.skip(f"No {mass_category} mass section in PTD")

        for row in section.rows:
            altitude = row.fl * 100 * FT
            tas = row.tas * KT

            _theta, _delta, sigma = get_atm_ratios(altitude)

            # Get aerodynamic configuration from PTD (TO, IC, CR, AP, LD)
            config = (
                AerodynamicConfig.from_string(row.config) if row.config else AerodynamicConfig.CR
            )

            cl = bada3_piston_model.lift_coef_from_inputs(v=tas, m=row.mass, sigma=sigma)
            cd = bada3_piston_model.aerodynamics.drag_coef(
                {
                    "lift_coef": cl,
                    "mach_number": row.mach,
                    "aerodynamic_configuration": config,
                }
            )
            drag_model = bada3_piston_model.drag_from_inputs(v=tas, sigma=sigma, cd=cd)

            assert drag_model == pytest.approx(row.drag, rel=PROPELLER_DRAG_TOLERANCE), (
                f"FL{row.fl} {mass_category} {config.value}: drag {drag_model:.0f} N != "
                f"PTD {row.drag:.0f} N"
            )


class TestBADA3CruiseTrajectory:
    """Tests for BADA3 fuel flow with cruise trajectory data."""

    def test_fuel_flow_with_cruise_trajectory(
        self, bada3_model, a320_cruise_trajectory: dict
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
        perf = traj.with_aircraft(bada3_model)

        # Access fuel_flow - this triggers the entire computation chain:
        # altitude → atmosphere → drag → thrust (via TEM) → fuel_flow
        fuel_flow = np.asarray(perf.fuel_flow).flatten()

        # Validate output shape
        assert len(fuel_flow) == 10, "Expected 10 samples"

        # Validate fuel flow is positive
        assert np.all(fuel_flow > 0), "Fuel flow must be positive"

        # A320 twin-engine cruise fuel flow: ~0.5-1.2 kg/s total
        assert np.all(fuel_flow > 0.3), f"Fuel flow too low: min={np.min(fuel_flow):.3f} kg/s"
        assert np.all(fuel_flow < 1.5), f"Fuel flow too high: max={np.max(fuel_flow):.3f} kg/s"


# ==============================================================================
# BADA3 LIMITS MODEL TESTS
# ==============================================================================


class TestBADA3LimitsScalar:
    """Tests for BADA3 limits with scalar inputs."""

    def test_min_cas_sensible_values(self, bada3_model):
        """Test min_cas returns sensible stall speeds for different configs."""
        mass = 65000.0  # Typical cruise mass

        for config in [
            AerodynamicConfig.CR,
            AerodynamicConfig.TO,
            AerodynamicConfig.LD,
        ]:
            state = {
                "aerodynamic_configuration": config,
                "aircraft_mass": mass,
            }
            min_cas = bada3_model.limits.min_cas(state)

            # Stall speed should be positive and < 200 kt for transport aircraft
            assert min_cas > 0, f"min_cas for {config} must be positive"
            assert min_cas < 200, f"min_cas for {config} unreasonably high: {min_cas}"

    def test_min_cas_increases_with_mass(self, bada3_model):
        """Test stall speed increases with mass (sqrt relationship)."""
        config = AerodynamicConfig.CR
        masses = [50000.0, 65000.0, 78000.0]

        min_cas_values = [
            bada3_model.limits.min_cas({"aerodynamic_configuration": config, "aircraft_mass": m})
            for m in masses
        ]

        assert np.all(np.diff(min_cas_values) > 0), "Stall speed should increase with mass"

    def test_max_altitude_sensible_values(self, bada3_model):
        """Test max_altitude returns sensible ceiling."""
        state = {
            "aircraft_mass": 65000.0,
            "delta_isa": 0.0,
        }
        max_alt = bada3_model.limits.max_altitude(state)

        # Max altitude should be between FL200 and FL500 for jets
        assert max_alt > 20000, f"max_altitude too low: {max_alt} ft"
        assert max_alt < 50000, f"max_altitude too high: {max_alt} ft"

    def test_max_altitude_decreases_with_mass(self, bada3_model):
        """Test ceiling decreases with heavier aircraft."""
        state_base = {"delta_isa": 0.0}
        masses = [50000.0, 65000.0, 78000.0]

        max_alts = [
            bada3_model.limits.max_altitude({**state_base, "aircraft_mass": m}) for m in masses
        ]

        assert np.all(np.diff(max_alts) <= 0), "Ceiling should decrease with mass"

    def test_max_altitude_decreases_with_hot_temperature(self, bada3_model):
        """Test ceiling decreases in hot conditions (ISA+)."""
        state_base = {"aircraft_mass": 65000.0}
        delta_isas = [0.0, 10.0, 20.0]  # ISA, ISA+10, ISA+20

        max_alts = [
            bada3_model.limits.max_altitude({**state_base, "delta_isa": dt}) for dt in delta_isas
        ]

        # Hot day reduces ceiling (Gt is negative)
        assert np.all(np.diff(max_alts) <= 0), "Ceiling should decrease in hot conditions"


class TestBADA3LimitsArray:
    """Tests for BADA3 limits with NumPy array inputs."""

    def test_min_cas_with_array_mass(self, bada3_model):
        """Test min_cas handles array mass input."""
        masses = np.array([50000.0, 60000.0, 70000.0, 78000.0])
        state = {
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "aircraft_mass": masses,
        }

        min_cas = bada3_model.limits.min_cas(state)

        # Result should be array with same shape
        assert isinstance(min_cas, np.ndarray), "Expected array output"
        assert min_cas.shape == masses.shape, "Shape mismatch"
        assert np.all(min_cas > 0), "All stall speeds must be positive"
        assert np.all(np.diff(min_cas) > 0), "Should increase with mass"

    def test_max_altitude_with_array_mass(self, bada3_model):
        """Test max_altitude handles array mass input."""
        masses = np.array([50000.0, 60000.0, 70000.0, 78000.0])
        state = {
            "aircraft_mass": masses,
            "delta_isa": 0.0,
        }

        max_alt = bada3_model.limits.max_altitude(state)

        assert isinstance(max_alt, np.ndarray), "Expected array output"
        assert max_alt.shape == masses.shape, "Shape mismatch"
        assert np.all(max_alt > 20000), "All ceilings should be > FL200"

    def test_max_altitude_with_array_delta_isa(self, bada3_model):
        """Test max_altitude handles array delta_isa input."""
        delta_isas = np.array([-10.0, 0.0, 10.0, 20.0])
        state = {
            "aircraft_mass": 65000.0,
            "delta_isa": delta_isas,
        }

        max_alt = bada3_model.limits.max_altitude(state)

        assert isinstance(max_alt, np.ndarray), "Expected array output"
        assert max_alt.shape == delta_isas.shape, "Shape mismatch"

    def test_max_altitude_with_both_arrays(self, bada3_model):
        """Test max_altitude with both mass and delta_isa as arrays."""
        masses = np.array([50000.0, 60000.0, 70000.0])
        delta_isas = np.array([0.0, 10.0, 20.0])
        state = {
            "aircraft_mass": masses,
            "delta_isa": delta_isas,
        }

        max_alt = bada3_model.limits.max_altitude(state)

        assert isinstance(max_alt, np.ndarray), "Expected array output"
        assert max_alt.shape == masses.shape, "Shape mismatch"


class TestBADA3LimitsSymbolic:
    """Tests for BADA3 limits with CasADi symbolic inputs."""

    def test_min_cas_symbolic_mass(self, bada3_model):
        """Test min_cas with CasADi symbolic mass."""
        mass_sym = ca.MX.sym("mass")
        state = {
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "aircraft_mass": mass_sym,
        }

        min_cas = bada3_model.limits.min_cas(state)

        # Should return CasADi expression
        assert isinstance(min_cas, ca.MX), "Expected CasADi MX output"

        # Create function and evaluate
        f = ca.Function("min_cas", [mass_sym], [min_cas])
        result = float(f(65000.0))

        assert result > 0, "Evaluated stall speed must be positive"
        assert result < 200, "Evaluated stall speed unreasonably high"

    def test_min_cas_symbolic_gradient(self, bada3_model):
        """Test gradient of min_cas w.r.t. mass."""
        mass_sym = ca.MX.sym("mass")
        state = {
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "aircraft_mass": mass_sym,
        }

        min_cas = bada3_model.limits.min_cas(state)
        grad = ca.jacobian(min_cas, mass_sym)

        f = ca.Function("grad_min_cas", [mass_sym], [grad])
        grad_val = float(f(65000.0))

        # Gradient should be positive (stall speed increases with mass)
        assert grad_val > 0, "d(min_cas)/d(mass) should be positive"
        assert np.isfinite(grad_val), "Gradient should be finite"

    def test_max_altitude_symbolic(self, bada3_model):
        """Test max_altitude with CasADi symbolic inputs."""
        mass_sym = ca.MX.sym("mass")
        delta_isa_sym = ca.MX.sym("delta_isa")
        state = {
            "aircraft_mass": mass_sym,
            "delta_isa": delta_isa_sym,
        }

        max_alt = bada3_model.limits.max_altitude(state)

        assert isinstance(max_alt, ca.MX), "Expected CasADi MX output"

        # Evaluate at reference conditions
        f = ca.Function("max_alt", [mass_sym, delta_isa_sym], [max_alt])
        result = float(f(65000.0, 0.0))

        assert result > 20000, "Ceiling should be > FL200"
        assert result < 50000, "Ceiling should be < FL500"

    def test_max_altitude_symbolic_gradient(self, bada3_model):
        """Test gradients of max_altitude w.r.t. mass and delta_isa."""
        mass_sym = ca.MX.sym("mass")
        delta_isa_sym = ca.MX.sym("delta_isa")
        state = {
            "aircraft_mass": mass_sym,
            "delta_isa": delta_isa_sym,
        }

        max_alt = bada3_model.limits.max_altitude(state)
        jac = ca.jacobian(max_alt, ca.vertcat(mass_sym, delta_isa_sym))

        f = ca.Function("jac_max_alt", [mass_sym, delta_isa_sym], [jac])
        grad = np.array(f(65000.0, 10.0)).flatten()

        # d(max_alt)/d(mass) should be <= 0 (heavier = lower ceiling)
        assert grad[0] <= 0, "d(max_alt)/d(mass) should be non-positive"
        # d(max_alt)/d(delta_isa) should be <= 0 (hotter = lower ceiling)
        assert grad[1] <= 0, "d(max_alt)/d(delta_isa) should be non-positive"
        assert np.all(np.isfinite(grad)), "Gradients should be finite"


class TestBADA3DescentPTDValidation:
    """Validate descent (idle) thrust against the official PTD descent tables.

    This is the reference check for the Desc(low)/Desc(high) OPF column
    order: the PTD descent thrust column is computed by EUROCONTROL from
    those coefficients, including

    - the A320 step-UP in thrust crossing Hp,des (~FL298): CTdes,high
      (0.13603) applies above, CTdes,low (0.10847) below — at FL290 the
      table shows 7,512 N vs 8,803 N at FL310;
    - the AT72 zero-thrust rows at/above FL120 (CTdes,high = 0.0).

    Rows below FL30 are excluded: the reference tables switch to
    approach/landing configurations there, while this check pins the
    clean-configuration coefficients.
    """

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_a320_descent_thrust_matches_ptd(self, bada3_model, bada3_ptd, mass_category):
        section = bada3_ptd.get_section(mass_category, "descent")
        if section is None:
            pytest.skip(f"No {mass_category} mass descent section in PTD")

        checked = 0
        for row in section.rows:
            if row.fl < 30:
                continue
            state = {
                "true_airspeed": row.tas * KT,
                "altitude": row.fl * 100 * FT,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
                "aerodynamic_configuration": AerodynamicConfig.CR,
            }
            thrust_model = bada3_model.engine.thrust_min(state)
            assert thrust_model == pytest.approx(row.thrust, rel=THRUST_TOLERANCE, abs=1.0), (
                f"FL{row.fl} {mass_category}: descent thrust {thrust_model:.0f} N != "
                f"PTD {row.thrust:.0f} N"
            )
            checked += 1
        assert checked > 0, "No descent rows at/above FL30 found in PTD"

    @pytest.mark.parametrize("mass_category", ["low", "medium", "high"])
    def test_at72_descent_thrust_matches_ptd(
        self, bada3_turboprop_model, bada3_at72_ptd, mass_category
    ):
        section = bada3_at72_ptd.get_section(mass_category, "descent")
        if section is None:
            pytest.skip(f"No {mass_category} mass descent section in PTD")

        checked = zero_rows = 0
        for row in section.rows:
            if row.fl < 30:
                continue
            state = {
                "true_airspeed": row.tas * KT,
                "altitude": row.fl * 100 * FT,
                "engine_setting": EngineSetting.MCMB,
                "delta_isa": 0.0,
                "aerodynamic_configuration": AerodynamicConfig.CR,
            }
            thrust_model = bada3_turboprop_model.engine.thrust_min(state)
            assert thrust_model == pytest.approx(row.thrust, rel=THRUST_TOLERANCE, abs=1.0), (
                f"FL{row.fl} {mass_category}: descent thrust {thrust_model:.0f} N != "
                f"PTD {row.thrust:.0f} N"
            )
            checked += 1
            if row.thrust == 0:
                zero_rows += 1
        assert checked > 0, "No descent rows at/above FL30 found in PTD"
        # AT72 CTdes,high = 0.0: the reference table itself lists 0 N above
        # Hp,des — guards against any future "fix" that reintroduces a floor.
        assert zero_rows > 0, "Expected zero-thrust descent rows above Hp,des for AT72"
