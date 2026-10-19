"""Tests for the Poll-Schumann aircraft performance model.

Tests cover aerodynamics (including Reynolds effects), engine model,
and CasADi symbolic compatibility.
"""

import casadi as ca
import numpy as np
import pytest

import openleaf.performance.models.ps as ps
from openleaf.performance.core.perfos import EngineType
from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.models.ps import PSModel
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


# Tolerances for PS validation against pycontrails
PS_CD_TOLERANCE = 0.01  # 1% for drag coefficient (cruise conditions)
PS_FUEL_TOLERANCE = 0.005  # 0.5% for fuel flow

# --- Fixtures ---


@pytest.fixture
def ps_model() -> PSModel:
    """Create PS model for A320 from pycontrails database."""
    pytest.importorskip("pycontrails")
    return PSModel("A320")


@pytest.fixture
def cruise_state() -> dict:
    """Typical FL350 cruise state."""
    h = 10668.0
    return {
        "altitude": h,
        "mach_number": 0.78,
        "air_temperature": isa.air_temperature(h),  # absolute temperature [K]
        "air_pressure": isa.air_pressure(h),  # absolute pressure [Pa]
        "aircraft_mass": 65000.0,
        "thrust": 25000.0,
    }


# --- Tests ---


class TestPSModel:
    """Tests for PS model creation."""

    def test_model_creation(self, ps_model: PSModel):
        """Test model creation and properties."""
        assert ps_model.model == "A320"
        assert ps_model.engine_type == EngineType.JET
        assert ps_model.mtom > ps_model.oem > 0

    def test_from_data(self):
        """Test model creation from data object."""
        pytest.importorskip("pycontrails")
        # Create a B738 model from pycontrails
        model = PSModel("B738")
        assert model.model == "B738"
        assert model.aerodynamics.data.surface_area > 0


class TestPSAerodynamics:
    """Tests for PS aerodynamic model."""

    def test_drag_increases_with_lift(self, ps_model: PSModel):
        """Test drag polar behavior."""
        h = 10668.0
        T = isa.air_temperature(h)
        p = isa.air_pressure(h)
        cds = [
            ps_model.aerodynamics._drag_coef_clean(
                cl=cl, mach=0.78, air_temperature=T, air_pressure=p
            )
            for cl in [0.3, 0.5, 0.7]
        ]
        assert np.all(np.diff(cds) > 0)

    def test_reynolds_effect(self, ps_model: PSModel):
        """Test that atmospheric state affects drag (Reynolds number)."""
        cl, mach = 0.5, 0.78

        # High altitude (cold, low pressure) vs low altitude
        cd_high = ps_model.aerodynamics._drag_coef_clean(
            cl=cl, mach=mach, air_temperature=220.0, air_pressure=22600.0
        )
        cd_low = ps_model.aerodynamics._drag_coef_clean(
            cl=cl, mach=mach, air_temperature=288.0, air_pressure=101325.0
        )

        # Different Reynolds numbers should give different drag
        assert cd_high != cd_low


class TestPSEngine:
    """Tests for PS engine model."""

    def test_thrust_envelope(self, ps_model: PSModel, cruise_state: dict):
        """Test thrust min < max."""
        thrust_max = ps_model.engine.thrust_max(cruise_state)
        thrust_min = ps_model.engine.thrust_min(cruise_state)

        assert thrust_max > thrust_min >= 0

    def test_thrust_decreases_with_altitude(self, ps_model: PSModel):
        """Test altitude effect on thrust."""
        thrusts = []
        for h in [0, 5000, 10000]:
            state = {
                "altitude": h,
                "mach_number": 0.6,
                "air_temperature": isa.air_temperature(h),
                "air_pressure": isa.air_pressure(h),
            }
            thrusts.append(ps_model.engine.thrust_max(state))

        assert np.all(np.diff(thrusts) < 0)

    def test_fuel_flow_increases_with_thrust(self, ps_model: PSModel, cruise_state: dict):
        """Test fuel flow monotonicity."""
        fuel_flows = [
            ps_model.engine.fuel_flow({**cruise_state, "thrust": t}) for t in [15000, 25000, 35000]
        ]
        assert np.all(np.diff(fuel_flows) > 0)


class TestPSOverallPropulsionEfficiency:
    """Tests for the overall propulsion efficiency used by fuel_flow."""

    @pytest.fixture
    def state(self) -> dict:
        """FL350, M0.78, ISA cruise state without thrust."""
        h = 35000.0 * units.FT
        return {
            "altitude": h,
            "mach_number": 0.78,
            "air_temperature": isa.air_temperature(h),
            "air_pressure": isa.air_pressure(h),
        }

    @pytest.fixture
    def cruise_thrust(self, ps_model: PSModel, state: dict) -> float:
        """Thrust equal to drag at 65 t in the cruise state [N]."""
        cl = ps_model.lift_coefficient(65000.0, state["air_pressure"], state["mach_number"])
        cd = ps_model.aerodynamics.drag_coef_with_state(
            cl=cl,
            mach=state["mach_number"],
            air_temperature=state["air_temperature"],
            air_pressure=state["air_pressure"],
        )
        return ps_model.compute_thrust_required(65000.0, cl, cd)

    def test_reference_values(self, ps_model: PSModel, state: dict, cruise_thrust: float):
        """Test values at cruise, above thrust_max and near zero thrust."""
        thrust_max = ps_model.engine.thrust_max(state)
        cases = [
            (cruise_thrust, 0.30639),
            (1.2 * thrust_max, 0.29525),
            (2.0 * thrust_max, 0.29525),
            (50.0, 0.01),
            (0.0, 0.01),
        ]
        for thrust, expected in cases:
            eta = ps_model.engine.overall_propulsion_efficiency({**state, "thrust": thrust})
            assert eta == pytest.approx(expected, abs=5e-6), f"thrust={thrust:.0f} N"

    def test_engine_deterioration_scales_fuel_flow(self, state: dict, cruise_thrust: float):
        """Test that the deterioration factor is a fractional fuel flow increase."""
        point = {**state, "thrust": cruise_thrust}
        clean = PSModel("A320", engine_deterioration_factor=0.0)
        worn = PSModel("A320", engine_deterioration_factor=0.025)

        assert worn.engine.fuel_flow(point) == pytest.approx(
            1.025 * clean.engine.fuel_flow(point), rel=1e-12
        )
        assert PSModel("A320").engine.data.engine_deterioration_factor == 0.025

    def test_equals_raw_correlation_inside_envelope(
        self, ps_model: PSModel, state: dict, cruise_thrust: float
    ):
        """Test that no guard is active for an unclipped cruise point."""
        d = ps_model.engine.data
        mach = state["mach_number"]
        c_t = ps_model.engine.thrust_coefficient(cruise_thrust, mach, state["air_pressure"])
        c_t_eta_b = ps.ps_physics.thrust_coefficient_at_max_efficiency(mach, d.m_des, d.c_t_des)
        raw = ps.overall_propulsion_efficiency(
            mach,
            c_t,
            c_t_eta_b,
            d.m_des,
            d.eta_1,
            d.eta_2,
            engine_deterioration_factor=d.engine_deterioration_factor,
        )

        eta = ps_model.engine.overall_propulsion_efficiency({**state, "thrust": cruise_thrust})
        assert eta == pytest.approx(raw, rel=1e-12)

    @pytest.mark.parametrize(
        "reference,factor",
        [("cruise", 0.5), ("cruise", 1.0), ("thrust_max", 1.2), ("thrust_max", 2.0)],
    )
    def test_consistent_with_fuel_flow(
        self, ps_model: PSModel, state: dict, cruise_thrust: float, reference: str, factor: float
    ):
        """Test that fuel_flow equals F V / (eta Q) with F clipped to thrust_max.

        None of these points is clipped by the idle or maximum fuel flow.
        """
        thrust_max = ps_model.engine.thrust_max(state)
        thrust = factor * (cruise_thrust if reference == "cruise" else thrust_max)
        flight_state = {**state, "thrust": thrust}

        eta = ps_model.engine.overall_propulsion_efficiency(flight_state)
        tas = state["mach_number"] * np.sqrt(units.GAMMA * units.R_AIR * state["air_temperature"])
        expected = np.minimum(thrust, thrust_max) * tas / (eta * units.Q_FUEL)

        assert ps_model.engine.fuel_flow(flight_state) == pytest.approx(expected, rel=1e-10)

    def test_array_input(self, ps_model: PSModel, state: dict, cruise_thrust: float):
        """Test element-wise evaluation on an array of thrusts."""
        thrust = np.array([0.0, cruise_thrust, 3.0 * cruise_thrust])
        eta = np.asarray(
            ps_model.engine.overall_propulsion_efficiency({**state, "thrust": thrust})
        )

        assert eta.shape == thrust.shape
        for i, t in enumerate(thrust):
            scalar = ps_model.engine.overall_propulsion_efficiency({**state, "thrust": t})
            assert eta[i] == pytest.approx(scalar, rel=1e-12)

    def test_missing_field_raises(self, ps_model: PSModel, state: dict):
        """Test that thrust is a required field."""
        with pytest.raises(MissingFieldError):
            ps_model.engine.overall_propulsion_efficiency(state)

    def test_raw_function_reexported(self):
        """Test that the raw correlation is re-exported from models.ps."""
        assert ps.overall_propulsion_efficiency is ps.ps_physics.overall_propulsion_efficiency
        assert "overall_propulsion_efficiency" in ps.__all__


class TestPSIntegration:
    """Integration tests for PS model."""

    def test_cruise_performance(self, ps_model: PSModel):
        """Test complete cruise calculation chain."""
        h, mach, mass = 10668.0, 0.78, 65000.0
        air_temperature = isa.air_temperature(h)
        air_pressure = isa.air_pressure(h)

        cl = ps_model.lift_coefficient(mass, air_pressure, mach)
        cd = ps_model.aerodynamics._drag_coef_clean(
            cl=cl, mach=mach, air_temperature=air_temperature, air_pressure=air_pressure
        )
        thrust = ps_model.compute_thrust_required(mass, cl, cd)

        state = {
            "altitude": h,
            "mach_number": mach,
            "air_temperature": air_temperature,
            "air_pressure": air_pressure,
            "thrust": thrust,
        }
        fuel_flow = ps_model.engine.fuel_flow(state)

        # Verify reasonable values
        assert 0.3 < cl < 0.7
        assert cd > 0
        assert thrust > 0
        assert 0.1 < fuel_flow < 2.0

        # L/D should be reasonable
        assert 10 < cl / cd < 80


class TestPSSymbolic:
    """Tests for CasADi symbolic compatibility."""

    def test_symbolic_thrust_and_gradient(self, ps_model: PSModel):
        """Test symbolic thrust and gradient computation."""
        mach_sym = ca.MX.sym("mach")
        T_sym = ca.MX.sym("T")  # absolute temperature [K]
        p_sym = ca.MX.sym("p")  # absolute pressure [Pa]

        state = {
            "mach_number": mach_sym,
            "air_temperature": T_sym,
            "air_pressure": p_sym,
        }
        thrust = ps_model.engine.thrust_max(state)
        assert isinstance(thrust, ca.MX)

        jac = ca.jacobian(thrust, ca.vertcat(mach_sym, T_sym, p_sym))
        f = ca.Function("thrust_jac", [mach_sym, T_sym, p_sym], [jac])
        h = 10668.0
        grad = np.array(f(0.78, isa.air_temperature(h), isa.air_pressure(h))).flatten()

        assert np.all(np.isfinite(grad))

    def test_symbolic_drag(self, ps_model: PSModel):
        """Test symbolic drag computation."""
        cl_sym, mach_sym = ca.MX.sym("cl"), ca.MX.sym("mach")
        T_sym, p_sym = ca.MX.sym("T"), ca.MX.sym("p")

        cd = ps_model.aerodynamics._drag_coef_clean(
            cl=cl_sym, mach=mach_sym, air_temperature=T_sym, air_pressure=p_sym
        )
        assert isinstance(cd, ca.MX)


class TestPSHelperFunctions:
    """Tests for PS physics helper functions."""

    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        from openleaf.performance.models.ps.ps_physics import reynolds_number

        rn = reynolds_number(
            surface_area=122.6,
            mach_num=0.78,
            air_temperature=220.0,
            air_pressure=22600.0,
        )
        assert 1e6 < rn < 1e9

    def test_skin_friction(self):
        """Test skin friction coefficient."""
        from openleaf.performance.models.ps.ps_physics import skin_friction_coefficient

        cf = skin_friction_coefficient(rn=5e7)
        assert 0.001 < cf < 0.01

    def test_oswald_efficiency(self):
        """Test Oswald efficiency factor."""
        from openleaf.performance.models.ps.ps_physics import oswald_efficiency_factor

        e = oswald_efficiency_factor(
            c_drag_0=0.02,
            wing_aspect_ratio=9.5,
            cos_sweep=0.866,
            delta_2=0.0,
            winglets=True,
        )
        assert 0.6 < e < 1.1

    def test_wave_drag(self):
        """Test wave drag coefficient."""
        from openleaf.performance.models.ps.ps_physics import wave_drag_coefficient

        cdw = wave_drag_coefficient(
            mach_num=0.78,
            c_lift=0.5,
            cos_sweep=0.866,
            wing_constant=0.935,
            j_1=0.021,
            j_2=0.92,
            j_3=100.0,
            x_ref=1.0,
        )
        assert 0.0 <= cdw < 0.01


# --- PS Model Validation Fixtures ---


@pytest.fixture
def ps_model_validation():
    """Create PS model for A320 from pycontrails parameters."""
    pytest.importorskip("pycontrails")
    from openleaf.performance.models.ps import PSModel

    return PSModel.from_pycontrails("A320")


@pytest.fixture
def pycontrails_params():
    """Load pycontrails A320 parameters."""
    pytest.importorskip("pycontrails")
    from pycontrails.models.ps_model.ps_aircraft_params import load_aircraft_engine_params

    return load_aircraft_engine_params()["A320"]


# --- PS Model Validation Tests ---


class TestPSModelValidation:
    """Validate PS model against pycontrails implementation."""

    @pytest.mark.parametrize(
        "altitude_ft,mach,mass",
        [
            (350, 0.78, 65000),  # Typical cruise
            (350, 0.78, 55000),  # Light cruise
            (350, 0.78, 73000),  # Heavy cruise
            (300, 0.78, 65000),  # Lower altitude
            (350, 0.70, 65000),  # Lower Mach
        ],
    )
    def test_drag_coefficient_matches_pycontrails(
        self, ps_model_validation, pycontrails_params, altitude_ft, mach, mass
    ):
        """Test drag coefficient matches pycontrails calculation within tolerance."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        altitude_m = altitude_ft * 100 * 0.3048
        T = isa.air_temperature(altitude_m)  # Absolute temperature [K]
        p = isa.air_pressure(altitude_m)  # Absolute pressure [Pa]

        # OpenLeaf calculation
        c_lift_ae = ps_model_validation.lift_coefficient(mass, p, mach)
        c_drag_ae = ps_model_validation.aerodynamics.drag_coef_with_state(
            cl=c_lift_ae, mach=mach, air_temperature=T, air_pressure=p
        )

        # Pycontrails calculation using their functions directly
        params = pycontrails_params

        # Lift coefficient from pycontrails
        c_lift_pc = pc_ps.lift_coefficient(
            wing_surface_area=params.wing_surface_area,
            aircraft_mass=mass,
            air_pressure=p,
            mach_num=mach,
            climb_angle=0.0,
        )

        # Reynolds number and skin friction
        rn = pc_ps.reynolds_number(params.wing_surface_area, mach, T, p)
        c_f = pc_ps.skin_friction_coefficient(rn)
        c_drag_0 = pc_ps.zero_lift_drag_coefficient(c_f, params.psi_0)

        # Wave drag using pycontrails function
        c_drag_w = pc_ps.wave_drag_coefficient(mach, c_lift_pc, params)

        # Oswald efficiency using pycontrails function
        e_ls = pc_ps.oswald_efficiency_factor(c_drag_0, params)

        # Total drag coefficient using pycontrails function
        c_drag_pc = pc_ps.airframe_drag_coefficient(
            c_drag_0, c_drag_w, c_lift_pc, e_ls, params.wing_aspect_ratio
        )

        assert c_drag_ae == pytest.approx(c_drag_pc, rel=PS_CD_TOLERANCE), (
            f"FL{altitude_ft} M{mach} {mass}kg: CD {c_drag_ae:.6f} != pycontrails {c_drag_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "altitude_ft,mach,mass",
        [
            (350, 0.78, 65000),  # Typical cruise
            (350, 0.78, 55000),  # Light cruise
            (350, 0.78, 73000),  # Heavy cruise
            (300, 0.78, 65000),  # Lower altitude
            (350, 0.70, 65000),  # Lower Mach
            (200, 0.78, 65000),  # FL200
            (100, 0.70, 65000),  # FL100
        ],
    )
    def test_fuel_flow_matches_pycontrails(
        self, ps_model_validation, pycontrails_params, altitude_ft, mach, mass
    ):
        """Test fuel flow matches pycontrails calculation within tolerance."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        altitude_m = altitude_ft * 100 * 0.3048
        T = isa.air_temperature(altitude_m)  # Absolute temperature [K]
        p = isa.air_pressure(altitude_m)  # Absolute pressure [Pa]
        theta = isa.theta(altitude_m)  # Temperature ratio [-]
        delta = isa.delta(altitude_m)  # Pressure ratio [-]

        params = pycontrails_params

        # OpenLeaf calculation
        c_lift = ps_model_validation.lift_coefficient(mass, p, mach)
        c_drag = ps_model_validation.aerodynamics.drag_coef_with_state(
            cl=c_lift, mach=mach, air_temperature=T, air_pressure=p
        )
        thrust_req = ps_model_validation.compute_thrust_required(mass, c_lift, c_drag)
        _c_t = ps_model_validation.engine.thrust_coefficient(thrust_req, mach, p)

        state = {
            "mach_number": mach,
            "altitude": altitude_m,
            "air_temperature_ratio": theta,
            "air_pressure_ratio": delta,
            "air_temperature": T,
            "air_pressure": p,
            "thrust": thrust_req,
        }
        ff_openleaf = ps_model_validation.engine.fuel_flow(state)

        # Pycontrails calculation using their functions directly
        c_lift_pc = pc_ps.lift_coefficient(
            params.wing_surface_area, mass, p, mach, climb_angle=0.0
        )

        # Calculate drag coefficient for thrust
        rn = pc_ps.reynolds_number(params.wing_surface_area, mach, T, p)
        c_f = pc_ps.skin_friction_coefficient(rn)
        c_drag_0 = pc_ps.zero_lift_drag_coefficient(c_f, params.psi_0)
        c_drag_w = pc_ps.wave_drag_coefficient(mach, c_lift_pc, params)
        e_ls = pc_ps.oswald_efficiency_factor(c_drag_0, params)
        c_drag_pc = pc_ps.airframe_drag_coefficient(
            c_drag_0, c_drag_w, c_lift_pc, e_ls, params.wing_aspect_ratio
        )

        # Thrust force using pycontrails
        thrust_pc = pc_ps.thrust_force(
            aircraft_mass=mass,
            c_l=c_lift_pc,
            c_d=c_drag_pc,
            dv_dt=0.0,
            theta=0.0,  # level flight
        )

        # Thrust coefficient using pycontrails
        c_t_pc = pc_ps.engine_thrust_coefficient(thrust_pc, mach, p, params.wing_surface_area)

        # Thrust coefficient at max efficiency
        c_t_eta_b = pc_ps.thrust_coefficient_at_max_efficiency(mach, params.m_des, params.c_t_des)

        # Overall propulsion efficiency using pycontrails
        eta = pc_ps.overall_propulsion_efficiency(
            mach,
            c_t_pc,
            c_t_eta_b,
            params,
            engine_deterioration_factor=ps_model_validation.engine.data.engine_deterioration_factor,
        )

        # Fuel flow using pycontrails
        # Use same Q_FUEL as openleaf for consistency (43.13 MJ/kg)
        q_fuel = units.Q_FUEL
        ff_pycontrails = pc_ps.fuel_mass_flow_rate(
            p, T, mach, c_t_pc, eta, params.wing_surface_area, q_fuel
        )

        assert ff_openleaf == pytest.approx(ff_pycontrails, rel=PS_FUEL_TOLERANCE), (
            f"FL{altitude_ft} M{mach} {mass}kg: fuel {ff_openleaf:.4f} kg/s != "
            f"pycontrails {ff_pycontrails:.4f} kg/s"
        )

    def test_lift_coefficient_matches_pycontrails(self, ps_model_validation, pycontrails_params):
        """Test lift coefficient calculation matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        altitude_m = 10668  # FL350
        mach = 0.78
        mass = 65000

        delta = isa.air_pressure(altitude_m)
        p = delta * isa.P_SL

        # OpenLeaf
        c_lift_ae = ps_model_validation.lift_coefficient(mass, p, mach)

        # Pycontrails
        c_lift_pc = pc_ps.lift_coefficient(
            wing_surface_area=pycontrails_params.wing_surface_area,
            aircraft_mass=mass,
            air_pressure=p,
            mach_num=mach,
            climb_angle=0.0,
        )

        assert c_lift_ae == pytest.approx(c_lift_pc, rel=0.001)

    def test_reynolds_number_matches_pycontrails(self, ps_model_validation, pycontrails_params):
        """Test Reynolds number calculation matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import reynolds_number as ae_reynolds

        altitude_m = 10668
        mach = 0.78

        theta = isa.air_temperature(altitude_m)
        delta = isa.air_pressure(altitude_m)
        T = theta * isa.TAU_SL
        p = delta * isa.P_SL

        S = pycontrails_params.wing_surface_area

        # OpenLeaf
        rn_ae = ae_reynolds(S, mach, T, p)

        # Pycontrails
        rn_pc = pc_ps.reynolds_number(S, mach, T, p)

        assert rn_ae == pytest.approx(rn_pc, rel=0.001)

    def test_skin_friction_matches_pycontrails(self, pycontrails_params):
        """Test skin friction calculation matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import skin_friction_coefficient as ae_cf

        # Test at various Reynolds numbers
        for rn in [1e7, 5e7, 1e8]:
            cf_ae = ae_cf(rn)
            cf_pc = pc_ps.skin_friction_coefficient(rn)
            assert cf_ae == pytest.approx(cf_pc, rel=0.001)

    def test_oswald_efficiency_matches_pycontrails(self, pycontrails_params):
        """Test Oswald efficiency calculation matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import oswald_efficiency_factor as ae_oswald

        params = pycontrails_params

        # Test at various CD0 values
        for c_drag_0 in [0.015, 0.018, 0.020]:
            e_ae = ae_oswald(
                c_drag_0,
                params.wing_aspect_ratio,
                params.cos_sweep,
                params.delta_2,
                params.winglets,
            )
            e_pc = pc_ps.oswald_efficiency_factor(c_drag_0, params)
            assert e_ae == pytest.approx(e_pc, rel=0.001)


class TestPSPhysicsExhaustive:
    """Exhaustive function-by-function validation of PS physics against pycontrails.

    Note: Slight tolerance differences may exist due to minor constant variations
    between openleaf (R_AIR=287.05287) and pycontrails (R_d=287.05).
    """

    # Tolerance for Reynolds number (affected by R_AIR constant difference)
    RE_TOL = 0.001  # 0.1%

    @pytest.mark.parametrize(
        "mach,T,p",
        [
            (0.78, 220.0, 22600.0),  # FL350
            (0.70, 226.0, 30100.0),  # FL300
            (0.65, 243.0, 47200.0),  # FL200
            (0.50, 265.0, 74700.0),  # FL100
            (0.82, 218.0, 19800.0),  # FL400
        ],
    )
    def test_reynolds_number_exhaustive(self, pycontrails_params, mach, T, p):
        """Test Reynolds number matches pycontrails at various conditions."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import reynolds_number as ae_reynolds

        S = pycontrails_params.wing_surface_area

        rn_ae = ae_reynolds(S, mach, T, p)
        rn_pc = pc_ps.reynolds_number(S, mach, T, p)

        assert rn_ae == pytest.approx(rn_pc, rel=self.RE_TOL), (
            f"M{mach} T{T}K p{p}Pa: Re {rn_ae:.2e} != pycontrails {rn_pc:.2e}"
        )

    @pytest.mark.parametrize(
        "rn",
        [5e6, 1e7, 2e7, 5e7, 1e8, 2e8],
    )
    def test_skin_friction_coefficient_exhaustive(self, rn):
        """Test skin friction coefficient matches pycontrails at various Re."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import skin_friction_coefficient as ae_cf

        cf_ae = ae_cf(rn)
        cf_pc = pc_ps.skin_friction_coefficient(rn)

        assert cf_ae == pytest.approx(cf_pc, rel=1e-6), (
            f"Re={rn:.1e}: Cf {cf_ae:.6f} != pycontrails {cf_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "cf,psi_0",
        [
            (0.002, 0.0065),  # Low psi_0
            (0.003, 0.0075),  # Medium
            (0.002, 0.0085),  # High psi_0
            (0.0025, 0.0072),  # A320-like
        ],
    )
    def test_zero_lift_drag_coefficient_exhaustive(self, cf, psi_0):
        """Test zero-lift drag coefficient matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import zero_lift_drag_coefficient as ae_cd0

        cd0_ae = ae_cd0(cf, psi_0)
        cd0_pc = pc_ps.zero_lift_drag_coefficient(cf, psi_0)

        assert cd0_ae == pytest.approx(cd0_pc, rel=1e-6), (
            f"Cf={cf} psi_0={psi_0}: CD0 {cd0_ae:.6f} != pycontrails {cd0_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "mach,c_lift",
        [
            (0.78, 0.50),  # Typical cruise
            (0.78, 0.40),  # Light
            (0.78, 0.60),  # Heavy
            (0.70, 0.50),  # Lower Mach
            (0.82, 0.45),  # Higher Mach
            (0.85, 0.40),  # Near transonic
        ],
    )
    def test_wave_drag_coefficient_exhaustive(self, pycontrails_params, mach, c_lift):
        """Test wave drag coefficient matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import wave_drag_coefficient as ae_cdw

        params = pycontrails_params

        cdw_ae = ae_cdw(
            mach_num=mach,
            c_lift=c_lift,
            cos_sweep=params.cos_sweep,
            wing_constant=params.wing_constant,
            j_1=params.j_1,
            j_2=params.j_2,
            j_3=params.j_3,
            x_ref=params.x_ref,
        )
        cdw_pc = pc_ps.wave_drag_coefficient(mach, c_lift, params)

        assert cdw_ae == pytest.approx(cdw_pc, rel=1e-6), (
            f"M{mach} CL={c_lift}: CDw {cdw_ae:.6f} != pycontrails {cdw_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "cd0",
        [0.015, 0.018, 0.020, 0.022, 0.025],
    )
    def test_oswald_efficiency_exhaustive(self, pycontrails_params, cd0):
        """Test Oswald efficiency factor matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import oswald_efficiency_factor as ae_oswald

        params = pycontrails_params

        e_ae = ae_oswald(
            c_drag_0=cd0,
            wing_aspect_ratio=params.wing_aspect_ratio,
            cos_sweep=params.cos_sweep,
            delta_2=params.delta_2,
            winglets=params.winglets,
        )
        e_pc = pc_ps.oswald_efficiency_factor(cd0, params)

        assert e_ae == pytest.approx(e_pc, rel=1e-6), (
            f"CD0={cd0}: e={e_ae:.4f} != pycontrails {e_pc:.4f}"
        )

    @pytest.mark.parametrize(
        "cd0,cdw,cl,e,ar",
        [
            (0.018, 0.001, 0.50, 0.85, 9.5),  # Typical A320
            (0.018, 0.002, 0.55, 0.82, 9.5),  # Heavier load
            (0.020, 0.003, 0.60, 0.80, 9.5),  # High CL
            (0.016, 0.000, 0.40, 0.88, 10.5),  # Efficient cruise
        ],
    )
    def test_airframe_drag_coefficient_exhaustive(self, cd0, cdw, cl, e, ar):
        """Test airframe drag coefficient matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import airframe_drag_coefficient as ae_cd

        cd_ae = ae_cd(cd0, cdw, cl, e, ar)
        cd_pc = pc_ps.airframe_drag_coefficient(cd0, cdw, cl, e, ar)

        assert cd_ae == pytest.approx(cd_pc, rel=1e-6), (
            f"CD0={cd0} CDw={cdw} CL={cl} e={e}: CD={cd_ae:.6f} != {cd_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "mach,m_des,c_t_des",
        [
            (0.78, 0.8, 0.018),  # Typical cruise
            (0.70, 0.75, 0.020),
            (0.82, 0.82, 0.016),
            (0.65, 0.72, 0.022),
        ],
    )
    def test_thrust_coefficient_at_max_efficiency_exhaustive(self, mach, m_des, c_t_des):
        """Test thrust coefficient at max efficiency matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import (
            thrust_coefficient_at_max_efficiency as ae_ct_eta,
        )

        ct_ae = ae_ct_eta(mach, m_des, c_t_des)
        ct_pc = pc_ps.thrust_coefficient_at_max_efficiency(mach, m_des, c_t_des)

        assert ct_ae == pytest.approx(ct_pc, rel=1e-6), (
            f"M{mach}: c_t_eta_b={ct_ae:.6f} != pycontrails {ct_pc:.6f}"
        )

    @pytest.mark.parametrize(
        "mach,c_t,c_t_eta_b",
        [
            (0.78, 0.015, 0.018),  # Below optimal
            (0.78, 0.018, 0.018),  # At optimal
            (0.78, 0.022, 0.018),  # Above optimal
            (0.70, 0.020, 0.022),
            (0.82, 0.012, 0.015),
        ],
    )
    def test_overall_propulsion_efficiency_exhaustive(
        self, pycontrails_params, mach, c_t, c_t_eta_b
    ):
        """Test overall propulsion efficiency matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        from openleaf.performance.models.ps.ps_physics import (
            overall_propulsion_efficiency as ae_eta,
        )

        params = pycontrails_params

        factor = 0.025
        eta_ae = ae_eta(
            mach,
            c_t,
            c_t_eta_b,
            params.m_des,
            params.eta_1,
            params.eta_2,
            engine_deterioration_factor=factor,
        )
        eta_pc = pc_ps.overall_propulsion_efficiency(
            mach, c_t, c_t_eta_b, params, engine_deterioration_factor=factor
        )

        assert eta_ae == pytest.approx(eta_pc, rel=1e-6), (
            f"M{mach} c_t={c_t}: eta={eta_ae:.4f} != pycontrails {eta_pc:.4f}"
        )


class TestPSCruiseTrajectory:
    """Tests for PS fuel flow with cruise trajectory data."""

    def test_fuel_flow_with_cruise_trajectory(
        self, ps_model: PSModel, a320_cruise_trajectory: dict
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
        perf = traj.with_aircraft(ps_model)

        # Access fuel_flow - this triggers the entire computation chain:
        # altitude → atmosphere → drag → thrust (via TEM) → fuel_flow
        fuel_flow = np.asarray(perf.fuel_flow).flatten()

        # Validate output shape
        assert len(fuel_flow) == 10, "Expected 10 samples"

        # Validate fuel flow is positive
        assert np.all(fuel_flow > 0), "Fuel flow must be positive"

        # PS model returns per-aircraft fuel flow for A320 cruise
        # Values typically range from ~0.15-0.35 kg/s for the PS model
        # (lower than other models due to different efficiency assumptions)
        assert np.all(fuel_flow > 0.1), f"Fuel flow too low: min={np.min(fuel_flow):.3f} kg/s"
        assert np.all(fuel_flow < 1.0), f"Fuel flow too high: max={np.max(fuel_flow):.3f} kg/s"

        # Fuel flow should decrease as mass decreases (less drag)
        # Allow small variations due to numerical effects
        ff_diff = np.diff(fuel_flow)
        assert np.mean(ff_diff) < 0, "Fuel flow should generally decrease as mass decreases"


# ==============================================================================
# PS LIMITS MODEL TESTS
# ==============================================================================


class TestPSLimitsScalar:
    """Tests for PS limits with scalar inputs."""

    def test_max_mach_returns_mmo(self, ps_model: PSModel):
        """Test max_mach returns MMO when no altitude provided."""
        max_mach = ps_model.limits.max_mach({})

        # MMO typically 0.78-0.85 for transport jets
        assert max_mach > 0.7, f"max_mach too low: {max_mach}"
        assert max_mach < 0.95, f"max_mach too high: {max_mach}"

    def test_max_mach_with_altitude(self, ps_model: PSModel):
        """Test max_mach at various altitudes."""
        # At cruise altitude, should be close to MMO
        max_mach_high = ps_model.limits.max_mach({"altitude": 35000.0})
        assert max_mach_high > 0.7, "max_mach at FL350 too low"

        # At low altitude with ATM speed limit, should be reduced
        max_mach_low = ps_model.limits.max_mach(
            {
                "altitude": 5000.0,
                "atm_speed_limit": True,
            }
        )
        # 250 kt at low altitude corresponds to ~0.38M at sea level
        # but increases with altitude, so at FL050 could be ~0.4-0.5M
        assert max_mach_low > 0, "max_mach at low altitude should be positive"

    def test_max_mach_atm_limit_effect(self, ps_model: PSModel):
        """Test ATM 250kt limit below FL100."""
        altitude = 8000.0  # Below FL100

        max_mach_limited = ps_model.limits.max_mach(
            {
                "altitude": altitude,
                "atm_speed_limit": True,
            }
        )
        max_mach_unlimited = ps_model.limits.max_mach(
            {
                "altitude": altitude,
                "atm_speed_limit": False,
            }
        )

        # With ATM limit, max Mach should be lower or equal
        assert max_mach_limited <= max_mach_unlimited, (
            "ATM limit should reduce or maintain max Mach"
        )

    def test_hmo_exact(self, ps_model: PSModel):
        """Verify PS HMO in feet (A320: FL410 in the pycontrails database)."""
        assert ps_model.limits.hmo == pytest.approx(41000.0)

    def test_max_mach_exact(self, ps_model: PSModel):
        """Verify PS max_mach at high altitude returns MMO (A320: 0.82).

        At cruise altitude where impact pressure limit is not binding,
        MMO drives the limit.
        """
        max_mach = ps_model.limits.max_mach({"altitude": 35000.0})
        # At FL350, impact pressure limit is not active, so should return MMO+buffer
        assert max_mach == pytest.approx(0.82, abs=0.03), (
            f"PS A320 max_mach at FL350 should be ~MMO=0.82, got {max_mach}"
        )

    def test_vmo_is_infinite_cruise_model(self, ps_model: PSModel):
        """Verify PS VMO is infinite (cruise-only model, no CAS limit).

        PS is a cruise-focused model. VMO is not defined; the speed limit
        is expressed as a Mach limit (MMO) or impact pressure limit.
        """
        import math

        assert math.isinf(ps_model.limits.data.vmo), (
            "PS VMO should be infinite (cruise model has no CAS limit)"
        )

    def test_high_lift_returns_zero(self, ps_model: PSModel):
        """Test high_lift always returns 0 (cruise model)."""
        result = ps_model.high_lift({})
        assert result == 0, "PS model high_lift should always be 0"

    def test_landing_gear_returns_false(self, ps_model: PSModel):
        """Test landing_gear always returns False (cruise model)."""
        result = ps_model.landing_gear({})
        assert result is False, "PS model landing_gear should always be False"


class TestPSLimitsArray:
    """Tests for PS limits with NumPy array inputs."""

    def test_max_mach_with_array_altitude(self, ps_model: PSModel):
        """Test max_mach handles array altitude input."""
        altitudes = np.array([5000.0, 15000.0, 25000.0, 35000.0])
        state = {"altitude": altitudes}

        max_mach = ps_model.limits.max_mach(state)

        assert isinstance(max_mach, np.ndarray), "Expected array output"
        assert max_mach.shape == altitudes.shape, "Shape mismatch"
        assert np.all(max_mach > 0), "All max_mach must be positive"

    def test_max_mach_with_array_pressure(self, ps_model: PSModel):
        """Test max_mach handles array air_pressure input."""
        altitudes = np.array([5000.0, 15000.0, 25000.0, 35000.0])
        pressures = np.array([isa.air_pressure(h * units.FT) for h in altitudes])
        state = {
            "altitude": altitudes,
            "air_pressure": pressures,
        }

        max_mach = ps_model.limits.max_mach(state)

        assert isinstance(max_mach, np.ndarray), "Expected array output"
        assert max_mach.shape == altitudes.shape, "Shape mismatch"


class TestPSLimitsSymbolic:
    """Tests for PS limits with CasADi symbolic inputs."""

    def test_max_mach_symbolic_altitude(self, ps_model: PSModel):
        """Test max_mach with CasADi symbolic altitude."""
        alt_sym = ca.MX.sym("altitude")
        # Compute pressure from altitude symbolically
        pressure_sym = isa.air_pressure(alt_sym * units.FT)
        state = {
            "altitude": alt_sym,
            "air_pressure": pressure_sym,
            "atm_speed_limit": False,  # Use scalar to simplify
        }

        max_mach = ps_model.limits.max_mach(state)

        assert isinstance(max_mach, ca.MX), "Expected CasADi MX output"

        # Evaluate at cruise altitude
        f = ca.Function("max_mach", [alt_sym], [max_mach])
        result = float(f(35000.0))

        assert result > 0.7, "max_mach at FL350 too low"
        assert result < 0.95, "max_mach at FL350 too high"

    def test_max_mach_symbolic_gradient(self, ps_model: PSModel):
        """Test gradient of max_mach w.r.t. altitude."""
        alt_sym = ca.MX.sym("altitude")
        pressure_sym = isa.air_pressure(alt_sym * units.FT)
        state = {
            "altitude": alt_sym,
            "air_pressure": pressure_sym,
            "atm_speed_limit": False,
        }

        max_mach = ps_model.limits.max_mach(state)
        grad = ca.jacobian(max_mach, alt_sym)

        f = ca.Function("grad_max_mach", [alt_sym], [grad])
        grad_val = float(f(20000.0))

        # Gradient should be finite (relationship may vary with altitude)
        assert np.isfinite(grad_val), "Gradient should be finite"

    def test_max_altitude_symbolic_gradient(self, ps_model: PSModel):
        """max_altitude is a CasADi expression in mass and delta_isa."""
        m_sym = ca.MX.sym("m")
        disa_sym = ca.MX.sym("disa")
        max_alt = ps_model.limits.max_altitude(
            {"aircraft_mass": m_sym, "delta_isa": disa_sym, "mach_number": 0.78}
        )
        assert isinstance(max_alt, ca.MX)

        f = ca.Function("f", [m_sym, disa_sym], [max_alt, ca.gradient(max_alt, m_sym)])
        h, dh_dm = (float(v) for v in f(73500.0, 15.0))
        h_ref = float(
            ps_model.limits.max_altitude(
                {"aircraft_mass": 73500.0, "delta_isa": 15.0, "mach_number": 0.78}
            )
        )
        assert h == pytest.approx(h_ref, rel=1e-12)
        assert dh_dm < 0.0


class TestPSMaxMachAltitudeUnits:
    """Regression: max_mach must treat data["altitude"] as meters.

    The package-wide convention is altitude in meters; this method used to
    consume the value as feet, applying the 250 kt ATM limit up to 10,000 m
    (FL328) and evaluating pressure at the wrong altitude.
    """

    def test_altitude_interpreted_as_meters(self, ps_model: PSModel):
        from openleaf.performance.models.ps.ps_physics import max_mach_number_by_altitude
        from openleaf.performance.physics import isa
        from openleaf.performance.physics import units

        lim = ps_model.limits
        altitude_m = 3500.0  # 11,483 ft: above the 10,000 ft ATM threshold

        expected = max_mach_number_by_altitude(
            altitude_m * units.M_TO_FT,
            isa.air_pressure(altitude_m),
            lim.data.mmo,
            lim.data.p_i_max,
            lim.data.p_inf_co,
            atm_speed_limit=True,
            buffer=0.02,
        )
        result = lim.max_mach({"altitude": altitude_m})
        assert result == pytest.approx(expected)

    def test_atm_limit_releases_above_10000_ft(self, ps_model: PSModel):
        lim = ps_model.limits
        # 2,500 m (8,202 ft): ATM 250 kt cap applies -> reduced Mach limit.
        # 3,500 m (11,483 ft): above the cap -> impact-pressure/MMO governs.
        low = float(lim.max_mach({"altitude": 2500.0}))
        high = float(lim.max_mach({"altitude": 3500.0}))
        assert high > low


class TestPSMaxAltitude:
    """PS max_altitude: min(hmo, thrust-limited ceiling, lift-limited ceiling).

    The thrust-limited ceiling is checked against the residual climb rate
    computed independently from PSModel.thrust_max and drag, as in issue #44.
    """

    MACH = 0.78

    @staticmethod
    def _residual_climb_fpm(model, mass, delta_isa, mach, altitude_ft):
        """Residual climb rate at maximum thrust, (T_max - D) V / (m g) [ft/min]."""
        altitude_m = altitude_ft * units.FT
        p = isa.air_pressure(altitude_m)
        t = isa.air_temperature(altitude_m) + delta_isa
        cl = model.lift_coefficient(mass, p, mach)
        cd = model.aerodynamics.drag_coef_with_state(
            cl=cl, mach=mach, air_temperature=t, air_pressure=p
        )
        drag = model.compute_thrust_required(mass, cl, cd)
        thrust_max = model.thrust_max(
            {"mach_number": mach, "air_temperature": t, "air_pressure": p}
        )
        tas = mach * np.sqrt(units.GAMMA * units.R_AIR * t)
        return (thrust_max - drag) * tas / (mass * units.G) * units.MS_TO_FPM

    def _max_altitude(self, model, mass, delta_isa, mach=MACH):
        return model.max_altitude(
            {"aircraft_mass": mass, "delta_isa": delta_isa, "mach_number": mach}
        )

    @pytest.mark.parametrize(
        "mass,delta_isa",
        [
            (55000.0, 0.0),
            (65000.0, 0.0),
            (65000.0, 15.0),
            (73500.0, 0.0),
            (73500.0, 15.0),
        ],
    )
    def test_matches_residual_climb_criterion(self, ps_model: PSModel, mass, delta_isa):
        """Below hmo the result is where the residual climb is 300 ft/min."""
        grid_ft = np.arange(30000.0, 45000.0 + 1.0, 10.0)
        roc = self._residual_climb_fpm(ps_model, mass, delta_isa, self.MACH, grid_ft)
        grid_ceiling = grid_ft[roc >= 300.0].max()

        h_max = float(self._max_altitude(ps_model, mass, delta_isa))

        if grid_ceiling >= ps_model.hmo:
            assert h_max == ps_model.hmo
        else:
            assert grid_ceiling <= h_max < grid_ceiling + 10.0
            roc_at_ceiling = self._residual_climb_fpm(ps_model, mass, delta_isa, self.MACH, h_max)
            assert roc_at_ceiling == pytest.approx(300.0, abs=1e-3)

    def test_issue_44_figures(self, ps_model: PSModel):
        """73.5 t at ISA+15 is thrust-limited near 33,110 ft; 55 t at ISA is hmo."""
        assert float(self._max_altitude(ps_model, 73500.0, 15.0)) == pytest.approx(
            33110.0, abs=10.0
        )
        assert float(self._max_altitude(ps_model, 55000.0, 0.0)) == 41000.0

    def test_decreases_with_mass(self, ps_model: PSModel):
        masses = np.linspace(55000.0, 73500.0, 12)
        h_max = self._max_altitude(ps_model, masses, 0.0)
        assert h_max.shape == masses.shape
        assert np.all(np.diff(h_max) <= 0.0)
        below = h_max < ps_model.hmo
        assert below.sum() >= 2
        assert np.all(np.diff(h_max[below]) < 0.0)

    def test_decreases_with_delta_isa(self, ps_model: PSModel):
        delta_isa = np.linspace(0.0, 25.0, 6)
        h_max = self._max_altitude(ps_model, 70000.0, delta_isa)
        assert h_max.shape == delta_isa.shape
        assert np.all(np.diff(h_max) <= 0.0)
        below = h_max < ps_model.hmo
        assert below.sum() >= 2
        assert np.all(np.diff(h_max[below]) < 0.0)

    def test_clamped_at_hmo(self, ps_model: PSModel):
        """A light, cold aircraft is limited by hmo, never above it."""
        h_max = self._max_altitude(ps_model, np.array([45000.0, 50000.0]), -10.0)
        np.testing.assert_array_equal(h_max, ps_model.hmo)

    def test_defaults_to_design_mach(self, ps_model: PSModel):
        m_des = ps_model.engine.data.m_des
        without_mach = ps_model.max_altitude({"aircraft_mass": 70000.0, "delta_isa": 10.0})
        with_mach = self._max_altitude(ps_model, 70000.0, 10.0, mach=m_des)
        assert float(without_mach) == float(with_mach)

    @pytest.mark.parametrize("missing", ["aircraft_mass", "delta_isa"])
    def test_requires_mass_and_delta_isa(self, ps_model: PSModel, missing):
        from openleaf.performance.exceptions import MissingFieldError

        state = {"aircraft_mass": 65000.0, "delta_isa": 0.0}
        del state[missing]
        with pytest.raises(MissingFieldError, match=missing):
            ps_model.max_altitude(state)

    def test_lift_limited_at_low_mach(self, ps_model: PSModel):
        """At M0.6 the lift limit binds: lift reaches the maximum usable lift."""
        from openleaf.performance.models.ps.ps_physics import max_usable_lift_coefficient

        mach, mass = 0.6, 65000.0
        h_max = float(self._max_altitude(ps_model, mass, 0.0, mach=mach))
        assert h_max < ps_model.hmo
        assert self._residual_climb_fpm(ps_model, mass, 0.0, mach, h_max) > 300.0

        c_lift = ps_model.lift_coefficient(mass, isa.air_pressure(h_max * units.FT), mach)
        c_l_maxu = max_usable_lift_coefficient(
            mach, ps_model.engine.data.m_des, ps_model.aerodynamics.data.c_l_do
        )
        assert c_lift == pytest.approx(c_l_maxu, rel=1e-9)

    def test_search_floor_when_margin_never_met(self, ps_model: PSModel):
        """Hot, heavy and fast: no 300 ft/min margin above FL200."""
        from openleaf.performance.models.ps.limits import CEILING_FLOOR_FT

        h_max = float(self._max_altitude(ps_model, 73500.0, 35.0, mach=0.82))
        assert h_max == CEILING_FLOOR_FT
        roc_at_floor = self._residual_climb_fpm(ps_model, 73500.0, 35.0, 0.82, CEILING_FLOOR_FT)
        assert roc_at_floor < 300.0

    def test_low_mach_states_report_floor_not_violation(self, ps_model: PSModel):
        """Climb and ground points: the lift limit never drops below FL200.

        At M0.25 the clean-wing lift limit alone lies below sea level, and at
        Mach 0 it is -inf; neither may flag a ceiling violation.
        """
        import warnings

        from openleaf.performance.core import TrajectoryData
        from openleaf.performance.models.ps.limits import CEILING_FLOOR_FT

        altitude = np.array([0.0, 1500.0, 5000.0]) * units.FT
        mach = np.array([0.0, 0.25, 0.35])
        temperature = isa.air_temperature(altitude)
        with warnings.catch_warnings():
            # Mach 0 divides by zero in the PS lift coefficient, as for drag.
            warnings.simplefilter("ignore", RuntimeWarning)
            perf = TrajectoryData(
                {
                    "altitude": altitude,
                    "air_temperature": temperature,
                    "true_airspeed": mach * np.sqrt(units.GAMMA * units.R_AIR * temperature),
                    "vertical_speed": np.full(3, 10.0),
                    "acceleration": np.zeros(3),
                    "aircraft_mass": np.full(3, 70000.0),
                }
            ).with_aircraft(ps_model)
            max_altitude = np.asarray(perf.max_altitude)
            ceiling = perf.envelope_violations["ceiling"]

        np.testing.assert_array_equal(max_altitude, CEILING_FLOOR_FT)
        np.testing.assert_array_equal(ceiling, [False, False, False])

    def test_thrust_ceiling_zeroes_pycontrails_excess_thrust(self, pycontrails_params):
        """At the thrust-limited ceiling, pycontrails' excess thrust is zero.

        pycontrails' get_excess_thrust_available uses a 0.2 thrust buffer and
        the 300 ft/min climb rate, so compare with thrust_buffer=0.2.
        """
        from pycontrails.models.ps_model.ps_operational_limits import get_excess_thrust_available

        model = PSModel("A320", thrust_buffer=0.2)
        mass, delta_isa, mach = 73500.0, 25.0, 0.80
        h_thrust = float(model.limits._thrust_limited_ceiling(mass, delta_isa, mach))
        assert h_thrust < model.hmo

        altitude_m = h_thrust * units.FT
        excess = get_excess_thrust_available(
            mach_number=mach,
            air_temperature=isa.air_temperature(altitude_m) + delta_isa,
            air_pressure=isa.air_pressure(altitude_m),
            aircraft_mass=mass,
            theta=0.0,
            atyp_param=pycontrails_params,
        )
        # The excess thrust coefficient falls by about 2e-6 per foot here; the
        # tolerance allows for constants that differ from pycontrails.
        assert excess == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("mach", [0.45, 0.6, 0.7, 0.78, 0.82])
    def test_max_usable_lift_coefficient_matches_pycontrails(self, pycontrails_params, mach):
        from pycontrails.models.ps_model import ps_operational_limits as pc_limits

        from openleaf.performance.models.ps.ps_physics import max_usable_lift_coefficient

        params = pycontrails_params
        expected = pc_limits.max_usable_lift_coefficient(mach, params.m_des, params.c_l_do)
        result = max_usable_lift_coefficient(mach, params.m_des, params.c_l_do)
        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("rocd", [0.0, 300.0, 1500.0])
    def test_required_thrust_coefficient_matches_pycontrails(self, rocd):
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_operational_limits as pc_limits

        from openleaf.performance.models.ps.ps_physics import required_thrust_coefficient

        expected = pc_limits.required_thrust_coefficient(0.5, 0.03, 230.0, rocd=rocd)
        result = required_thrust_coefficient(0.5, 0.03, 230.0, rocd=rocd)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_envelope_ceiling_flag(self, ps_model: PSModel):
        """Issue #44: at FL400, M0.78, 73.5 t at ISA+15 violates the ceiling."""
        from openleaf.performance.core import TrajectoryData

        altitude = np.array([40000.0, 40000.0]) * units.FT
        delta_isa = np.array([0.0, 15.0])
        mass = np.array([55000.0, 73500.0])
        temperature = isa.air_temperature(altitude) + delta_isa
        perf = TrajectoryData(
            {
                "altitude": altitude,
                "air_temperature": temperature,
                "true_airspeed": self.MACH * np.sqrt(units.GAMMA * units.R_AIR * temperature),
                "vertical_speed": np.zeros(2),
                "acceleration": np.zeros(2),
                "aircraft_mass": mass,
            }
        ).with_aircraft(ps_model)

        max_altitude = np.asarray(perf.max_altitude)
        assert max_altitude.shape == (2,)
        assert max_altitude[0] == ps_model.hmo
        assert max_altitude[1] == pytest.approx(33110.0, abs=10.0)
        np.testing.assert_array_equal(perf.envelope_violations["ceiling"], [False, True])
