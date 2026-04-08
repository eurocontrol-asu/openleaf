"""Tests for the Poll-Schumann aircraft performance model.

Tests cover aerodynamics (including Reynolds effects), engine model,
and CasADi symbolic compatibility.
"""

import casadi as ca
import pytest

import numpy as np
from openleaf.performance.core.perfos import EngineType
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

    def test_fuel_flow_increases_with_thrust(
        self, ps_model: PSModel, cruise_state: dict
    ):
        """Test fuel flow monotonicity."""
        fuel_flows = [
            ps_model.engine.fuel_flow({**cruise_state, "thrust": t})
            for t in [15000, 25000, 35000]
        ]
        assert np.all(np.diff(fuel_flows) > 0)


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
    from pycontrails.models.ps_model.ps_aircraft_params import (
        load_aircraft_engine_params,
    )

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

        # Aetherx calculation
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
            f"FL{altitude_ft} M{mach} {mass}kg: CD {c_drag_ae:.6f} != "
            f"pycontrails {c_drag_pc:.6f}"
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

        # Aetherx calculation
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
        ff_aetherx = ps_model_validation.engine.fuel_flow(state)

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
        c_t_pc = pc_ps.engine_thrust_coefficient(
            thrust_pc, mach, p, params.wing_surface_area
        )

        # Thrust coefficient at max efficiency
        c_t_eta_b = pc_ps.thrust_coefficient_at_max_efficiency(
            mach, params.m_des, params.c_t_des
        )

        # Overall propulsion efficiency using pycontrails
        eta = pc_ps.overall_propulsion_efficiency(mach, c_t_pc, c_t_eta_b, params)

        # Fuel flow using pycontrails
        # Use same Q_FUEL as aetherx for consistency (43.13 MJ/kg)
        q_fuel = units.Q_FUEL
        ff_pycontrails = pc_ps.fuel_mass_flow_rate(
            p, T, mach, c_t_pc, eta, params.wing_surface_area, q_fuel
        )

        assert ff_aetherx == pytest.approx(ff_pycontrails, rel=PS_FUEL_TOLERANCE), (
            f"FL{altitude_ft} M{mach} {mass}kg: fuel {ff_aetherx:.4f} kg/s != "
            f"pycontrails {ff_pycontrails:.4f} kg/s"
        )

    def test_lift_coefficient_matches_pycontrails(
        self, ps_model_validation, pycontrails_params
    ):
        """Test lift coefficient calculation matches pycontrails."""
        pytest.importorskip("pycontrails")
        from pycontrails.models.ps_model import ps_model as pc_ps

        altitude_m = 10668  # FL350
        mach = 0.78
        mass = 65000

        delta = isa.air_pressure(altitude_m)
        p = delta * isa.P_SL

        # Aetherx
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

    def test_reynolds_number_matches_pycontrails(
        self, ps_model_validation, pycontrails_params
    ):
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

        # Aetherx
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
    between aetherx (R_AIR=287.05287) and pycontrails (R_d=287.05).
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
    def test_thrust_coefficient_at_max_efficiency_exhaustive(
        self, mach, m_des, c_t_des
    ):
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

        from openleaf.performance.models.ps.ps_physics import overall_propulsion_efficiency as ae_eta

        params = pycontrails_params

        # aetherx function signature: (mach, c_t, c_t_eta_b, m_des, eta_1, eta_2)
        eta_ae = ae_eta(mach, c_t, c_t_eta_b, params.m_des, params.eta_1, params.eta_2)
        eta_pc = pc_ps.overall_propulsion_efficiency(mach, c_t, c_t_eta_b, params)

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
        assert np.all(fuel_flow > 0.1), (
            f"Fuel flow too low: min={np.min(fuel_flow):.3f} kg/s"
        )
        assert np.all(fuel_flow < 1.0), (
            f"Fuel flow too high: max={np.max(fuel_flow):.3f} kg/s"
        )

        # Fuel flow should decrease as mass decreases (less drag)
        # Allow small variations due to numerical effects
        ff_diff = np.diff(fuel_flow)
        assert np.mean(ff_diff) < 0, (
            "Fuel flow should generally decrease as mass decreases"
        )


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

    def test_max_altitude_returns_hmo(self, ps_model: PSModel):
        """Test max_altitude returns HMO."""
        max_alt = ps_model.limits.max_altitude({})

        # HMO typically FL350-FL450 for jets
        assert max_alt > 30000, f"max_altitude too low: {max_alt} ft"
        assert max_alt < 50000, f"max_altitude too high: {max_alt} ft"

    def test_max_altitude_exact(self, ps_model: PSModel):
        """Verify PS max_altitude returns HMO in feet (A320: 41000 ft).

        PS stores HMO from pycontrails aircraft database.
        A320: ceiling = 41000 ft.
        """
        max_alt = ps_model.limits.max_altitude({})
        assert max_alt == pytest.approx(41000.0), (
            f"PS A320 HMO should be 41000 ft, got {max_alt}"
        )

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

    def test_max_altitude_symbolic(self, ps_model: PSModel):
        """Test max_altitude returns constant (HMO) for symbolic input."""
        # PS max_altitude just returns HMO regardless of input
        max_alt = ps_model.limits.max_altitude({})

        # Should be a numeric constant
        assert isinstance(max_alt, (int, float, np.floating)), (
            "PS max_altitude should return numeric HMO"
        )
        assert max_alt > 30000, "HMO should be > FL300"
