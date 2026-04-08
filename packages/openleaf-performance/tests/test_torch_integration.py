"""End-to-end PyTorch backend integration tests.

Verifies that PyTorch tensors flow correctly through:
1. numpy dispatch layer (expanded coverage)
2. physics layer (ISA, aero, TEM, ESF)
3. model layer (OpenAP, Poll-Schumann)
4. gradient flow (autograd through physics computations)
"""

from __future__ import annotations

import pytest

import numpy as np
from openleaf.performance.numpy.backends import Backend


# Skip entire module if PyTorch is not installed
pytestmark = pytest.mark.skipif(
    not Backend.TORCH.available, reason="PyTorch not installed"
)

torch = pytest.importorskip("torch")


# =============================================================================
# Helpers
# =============================================================================


def to_torch(x, dtype=torch.float64, requires_grad=False):
    """Convert a value to a torch tensor with float64 for numerical parity."""
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)


def assert_parity(torch_result, numpy_result, rtol=1e-6, atol=1e-7):
    """Assert that a torch result matches a numpy result."""
    t = (
        torch_result.detach().numpy()
        if isinstance(torch_result, torch.Tensor)
        else torch_result
    )
    np.testing.assert_allclose(t, numpy_result, rtol=rtol, atol=atol)


def assert_is_tensor(result):
    """Assert result is a torch.Tensor."""
    assert isinstance(result, torch.Tensor), (
        f"Expected torch.Tensor, got {type(result)}"
    )


# =============================================================================
# PART 1: Expanded numpy-layer tests
# =============================================================================


class TestCbrt:
    """Tests for cbrt with PyTorch tensors."""

    def test_positive_values(self):
        from openleaf.performance.numpy import cbrt

        x_np = np.array([1.0, 8.0, 27.0, 64.0])
        x_torch = to_torch([1.0, 8.0, 27.0, 64.0])

        result = cbrt(x_torch)
        assert_is_tensor(result)
        assert_parity(result, cbrt(x_np))

    def test_negative_values(self):
        from openleaf.performance.numpy import cbrt

        x_np = np.array([-1.0, -8.0, -27.0])
        x_torch = to_torch([-1.0, -8.0, -27.0])

        result = cbrt(x_torch)
        assert_is_tensor(result)
        assert_parity(result, cbrt(x_np))

    def test_mixed_sign(self):
        from openleaf.performance.numpy import cbrt

        x_np = np.array([-8.0, 0.0, 8.0])
        x_torch = to_torch([-8.0, 0.0, 8.0])

        result = cbrt(x_torch)
        assert_is_tensor(result)
        assert_parity(result, cbrt(x_np))


class TestHstackVstackDstack:
    """Tests for hstack/vstack/dstack with PyTorch tensors."""

    def test_hstack(self):
        from openleaf.performance.numpy import hstack

        a = to_torch([1.0, 2.0])
        b = to_torch([3.0, 4.0])

        result = hstack([a, b])
        assert_is_tensor(result)
        assert_parity(result, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_vstack(self):
        from openleaf.performance.numpy import vstack

        a = to_torch([1.0, 2.0])
        b = to_torch([3.0, 4.0])

        result = vstack([a, b])
        assert_is_tensor(result)
        assert_parity(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_dstack(self):
        from openleaf.performance.numpy import dstack

        a = to_torch([1.0, 2.0])
        b = to_torch([3.0, 4.0])

        result = dstack([a, b])
        assert_is_tensor(result)
        assert_parity(result, np.array([[[1.0, 3.0], [2.0, 4.0]]]))


class TestDiag:
    """Tests for diag with PyTorch tensors."""

    def test_create_diagonal_matrix(self):
        from openleaf.performance.numpy import diag

        v = to_torch([1.0, 2.0, 3.0])
        result = diag(v)
        assert_is_tensor(result)
        expected = np.diag([1.0, 2.0, 3.0])
        assert_parity(result, expected)

    def test_extract_diagonal(self):
        from openleaf.performance.numpy import diag

        m = to_torch([[1.0, 2.0], [3.0, 4.0]])
        result = diag(m)
        assert_is_tensor(result)
        assert_parity(result, np.array([1.0, 4.0]))


class TestTileRollLength:
    """Tests for tile, roll, length with PyTorch tensors."""

    def test_tile(self):
        from openleaf.performance.numpy import tile

        x = to_torch([1.0, 2.0])
        result = tile(x, (3,))
        assert_is_tensor(result)
        assert_parity(result, np.tile([1.0, 2.0], (3,)))

    def test_roll(self):
        from openleaf.performance.numpy import roll

        x = to_torch([1.0, 2.0, 3.0, 4.0])
        result = roll(x, 2)
        assert_is_tensor(result)
        assert_parity(result, np.roll([1.0, 2.0, 3.0, 4.0], 2))

    def test_length(self):
        from openleaf.performance.numpy import length

        x = to_torch([1.0, 2.0, 3.0])
        assert length(x) == 3

    def test_length_scalar(self):
        from openleaf.performance.numpy import length

        x = to_torch(5.0)
        assert length(x) == 1


class TestFloorCeilSign:
    """Tests for floor, ceil, sign with PyTorch tensors."""

    def test_floor(self):
        from openleaf.performance.numpy import floor

        x = to_torch([1.7, -0.3, 2.5])
        result = floor(x)
        assert_is_tensor(result)
        assert_parity(result, np.floor([1.7, -0.3, 2.5]))

    def test_ceil(self):
        from openleaf.performance.numpy import ceil

        x = to_torch([1.2, -0.7, 2.5])
        result = ceil(x)
        assert_is_tensor(result)
        assert_parity(result, np.ceil([1.2, -0.7, 2.5]))

    def test_sign(self):
        from openleaf.performance.numpy import sign

        x = to_torch([-3.0, 0.0, 5.0])
        result = sign(x)
        assert_is_tensor(result)
        assert_parity(result, np.sign([-3.0, 0.0, 5.0]))


class TestDegreeTrig:
    """Tests for degree-based trig functions with PyTorch tensors."""

    def test_sind(self):
        from openleaf.performance.numpy import sind

        x = to_torch([0.0, 30.0, 90.0])
        result = sind(x)
        assert_is_tensor(result)
        assert_parity(result, np.sin(np.radians([0.0, 30.0, 90.0])), atol=1e-6)

    def test_cosd(self):
        from openleaf.performance.numpy import cosd

        x = to_torch([0.0, 60.0, 90.0])
        result = cosd(x)
        assert_is_tensor(result)
        assert_parity(result, np.cos(np.radians([0.0, 60.0, 90.0])), atol=1e-6)

    def test_tand(self):
        from openleaf.performance.numpy import tand

        x = to_torch([0.0, 45.0])
        result = tand(x)
        assert_is_tensor(result)
        assert_parity(result, np.tan(np.radians([0.0, 45.0])), atol=1e-6)

    def test_arcsind(self):
        from openleaf.performance.numpy import arcsind

        x = to_torch([0.0, 0.5, 1.0])
        result = arcsind(x)
        assert_is_tensor(result)
        assert_parity(result, np.degrees(np.arcsin([0.0, 0.5, 1.0])), atol=1e-5)

    def test_arccosd(self):
        from openleaf.performance.numpy import arccosd

        x = to_torch([0.0, 0.5, 1.0])
        result = arccosd(x)
        assert_is_tensor(result)
        assert_parity(result, np.degrees(np.arccos([0.0, 0.5, 1.0])), atol=1e-5)

    def test_arctan2d(self):
        from openleaf.performance.numpy import arctan2d

        y = to_torch([1.0, 0.0])
        x = to_torch([0.0, 1.0])
        result = arctan2d(y, x)
        assert_is_tensor(result)
        assert_parity(
            result,
            np.degrees(np.arctan2([1.0, 0.0], [0.0, 1.0])),
            atol=1e-5,
        )

    def test_degrees_radians(self):
        from openleaf.performance.numpy import degrees
        from openleaf.performance.numpy import radians

        x_deg = to_torch([0.0, 90.0, 180.0, 360.0])
        x_rad = radians(x_deg)
        assert_is_tensor(x_rad)
        assert_parity(x_rad, np.radians([0.0, 90.0, 180.0, 360.0]))

        result = degrees(x_rad)
        assert_is_tensor(result)
        assert_parity(result, np.array([0.0, 90.0, 180.0, 360.0]), atol=1e-5)


class TestHyperbolicTrig:
    """Tests for hyperbolic trig with PyTorch tensors."""

    def test_sinh_cosh_tanh(self):
        from openleaf.performance.numpy import cosh
        from openleaf.performance.numpy import sinh
        from openleaf.performance.numpy import tanh

        x_np = np.array([0.0, 0.5, 1.0, 2.0])
        x_torch = to_torch([0.0, 0.5, 1.0, 2.0])

        assert_parity(sinh(x_torch), np.sinh(x_np))
        assert_parity(cosh(x_torch), np.cosh(x_np))
        assert_parity(tanh(x_torch), np.tanh(x_np))

    def test_arcsinh_arccosh_arctanh(self):
        from openleaf.performance.numpy import arccosh
        from openleaf.performance.numpy import arcsinh
        from openleaf.performance.numpy import arctanh

        x_sinh = to_torch([0.0, 0.5, 1.0])
        assert_parity(arcsinh(x_sinh), np.arcsinh([0.0, 0.5, 1.0]))

        x_cosh = to_torch([1.0, 1.5, 2.0])
        assert_parity(arccosh(x_cosh), np.arccosh([1.0, 1.5, 2.0]))

        x_tanh = to_torch([0.0, 0.5, 0.9])
        assert_parity(arctanh(x_tanh), np.arctanh([0.0, 0.5, 0.9]))


class TestPowerLog:
    """Tests for power and log variants with PyTorch tensors."""

    def test_power(self):
        from openleaf.performance.numpy import power

        x = to_torch([2.0, 3.0, 4.0])
        result = power(x, 2.0)
        assert_is_tensor(result)
        assert_parity(result, np.power([2.0, 3.0, 4.0], 2.0))

    def test_log2(self):
        from openleaf.performance.numpy import log2

        x = to_torch([1.0, 2.0, 4.0, 8.0])
        result = log2(x)
        assert_is_tensor(result)
        assert_parity(result, np.log2([1.0, 2.0, 4.0, 8.0]))

    def test_log10(self):
        from openleaf.performance.numpy import log10

        x = to_torch([1.0, 10.0, 100.0])
        result = log10(x)
        assert_is_tensor(result)
        assert_parity(result, np.log10([1.0, 10.0, 100.0]))


class TestProd:
    """Tests for prod with PyTorch tensors."""

    def test_prod_all(self):
        from openleaf.performance.numpy import prod

        x = to_torch([1.0, 2.0, 3.0, 4.0])
        result = prod(x)
        assert float(result) == pytest.approx(24.0)

    def test_prod_axis(self):
        from openleaf.performance.numpy import prod

        x = to_torch([[1.0, 2.0], [3.0, 4.0]])
        result = prod(x, axis=0)
        assert_is_tensor(result)
        assert_parity(result, np.array([3.0, 8.0]))


class TestArrayCreation:
    """Tests for array() with torch tensors in nested structures."""

    def test_list_of_tensors(self):
        from openleaf.performance.numpy import array

        a = to_torch([1.0, 2.0])
        b = to_torch([3.0, 4.0])
        result = array([a, b])
        assert_is_tensor(result)
        assert_parity(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_direct_tensor_passthrough(self):
        from openleaf.performance.numpy import array

        t = to_torch([1.0, 2.0, 3.0])
        result = array(t)
        assert result is t  # Should return same object

    def test_empty_full_like(self):
        from openleaf.performance.numpy import empty_like
        from openleaf.performance.numpy import full_like

        x = to_torch([1.0, 2.0, 3.0])
        result_empty = empty_like(x)
        assert_is_tensor(result_empty)
        assert result_empty.shape == (3,)

        result_full = full_like(x, 7.0)
        assert_is_tensor(result_full)
        assert_parity(result_full, np.array([7.0, 7.0, 7.0]))


class TestRavelReshape:
    """Tests for ravel and reshape with PyTorch tensors."""

    def test_ravel(self):
        from openleaf.performance.numpy import ravel

        x = to_torch([[1.0, 2.0], [3.0, 4.0]])
        result = ravel(x)
        assert_is_tensor(result)
        assert_parity(result, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_reshape_int(self):
        from openleaf.performance.numpy import reshape

        x = to_torch([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = reshape(x, (3, 2))
        assert_is_tensor(result)
        assert result.shape == (3, 2)


class TestMinMaxReductions:
    """Tests for min/max with axis on PyTorch tensors."""

    def test_max_with_axis(self):
        from openleaf.performance.numpy import max

        x = to_torch([[1.0, 5.0], [3.0, 2.0]])
        result = max(x, axis=0)
        assert_is_tensor(result)
        assert_parity(result, np.array([3.0, 5.0]))

    def test_min_with_axis(self):
        from openleaf.performance.numpy import min

        x = to_torch([[1.0, 5.0], [3.0, 2.0]])
        result = min(x, axis=1)
        assert_is_tensor(result)
        assert_parity(result, np.array([1.0, 2.0]))

    def test_max_global(self):
        from openleaf.performance.numpy import max

        x = to_torch([3.0, 1.0, 4.0, 1.0, 5.0])
        assert float(max(x)) == pytest.approx(5.0)


# =============================================================================
# PART 2: Physics-layer integration tests
# =============================================================================


class TestISAWithTorch:
    """Verify ISA atmosphere functions accept and return torch tensors."""

    def test_theta(self):
        from openleaf.performance.physics.isa import theta

        h_np = np.array([0.0, 5000.0, 10000.0, 12000.0])
        h_torch = to_torch([0.0, 5000.0, 10000.0, 12000.0])

        result = theta(h_torch)
        assert_is_tensor(result)
        assert_parity(result, theta(h_np))

    def test_delta(self):
        from openleaf.performance.physics.isa import delta

        h_np = np.array([0.0, 5000.0, 10000.0, 12000.0])
        h_torch = to_torch([0.0, 5000.0, 10000.0, 12000.0])

        result = delta(h_torch)
        assert_is_tensor(result)
        assert_parity(result, delta(h_np), rtol=1e-5)

    def test_air_temperature(self):
        from openleaf.performance.physics.isa import air_temperature

        h_torch = to_torch([0.0, 5000.0, 11000.0, 15000.0])
        result = air_temperature(h_torch)
        assert_is_tensor(result)

        # Sea level should be 288.15 K
        assert float(result[0]) == pytest.approx(288.15, rel=1e-6)
        # Above tropopause should be 216.65 K
        assert float(result[3]) == pytest.approx(216.65, rel=1e-6)

    def test_air_pressure(self):
        from openleaf.performance.physics.isa import air_pressure

        h_torch = to_torch([0.0, 11000.0])
        result = air_pressure(h_torch)
        assert_is_tensor(result)

        # Sea level should be 101325 Pa
        assert float(result[0]) == pytest.approx(101325.0, rel=1e-4)

    def test_pressure_altitude(self):
        from openleaf.performance.physics.isa import air_pressure
        from openleaf.performance.physics.isa import pressure_altitude

        h_original = to_torch([0.0, 5000.0, 11000.0, 15000.0])
        p = air_pressure(h_original)
        h_recovered = pressure_altitude(p)
        assert_is_tensor(h_recovered)
        assert_parity(h_recovered, h_original.numpy(), rtol=1e-4)

    def test_atmospheric_state(self):
        from openleaf.performance.physics.isa import AtmosphericState

        h = to_torch([0.0, 5000.0, 10000.0])
        atm = AtmosphericState(h)

        # All properties should return tensors
        assert_is_tensor(atm.theta)
        assert_is_tensor(atm.delta)
        assert_is_tensor(atm.sigma)
        assert_is_tensor(atm.temperature)
        assert_is_tensor(atm.pressure)
        assert_is_tensor(atm.density)
        assert_is_tensor(atm.speed_of_sound)

        # Check consistency
        assert_parity(
            atm.temperature,
            (atm.theta * 288.15).detach().numpy(),
        )


class TestAeroWithTorch:
    """Verify aerodynamics functions accept and return torch tensors."""

    def test_dynamic_pressure(self):
        from openleaf.performance.physics.aero import dynamic_pressure

        v = to_torch([100.0, 200.0, 250.0])
        rho = to_torch([1.225, 0.7364, 0.4135])

        result = dynamic_pressure(v, rho)
        assert_is_tensor(result)

        expected = (
            0.5
            * np.array([1.225, 0.7364, 0.4135])
            * np.array([100.0, 200.0, 250.0]) ** 2
        )
        assert_parity(result, expected)

    def test_aerodynamic_coefficient_and_force(self):
        from openleaf.performance.physics.aero import aerodynamic_coefficient
        from openleaf.performance.physics.aero import aerodynamic_force

        force = to_torch([50000.0])
        q = to_torch([10000.0])
        S = 122.6  # A320 wing area

        cl = aerodynamic_coefficient(force, q, S)
        assert_is_tensor(cl)

        force_back = aerodynamic_force(cl, q, S)
        assert_is_tensor(force_back)
        assert_parity(force_back, force.numpy(), rtol=1e-6)

    def test_aircraft_airspeed_from_mach(self):
        from openleaf.performance.physics.aero import AircraftAirspeed

        T = to_torch([250.0])
        p = to_torch([50000.0])
        mach = to_torch([0.78])

        state = AircraftAirspeed(T=T, p=p, mach=mach)
        tas = state.true_airspeed
        cas = state.calibrated_airspeed
        assert_is_tensor(tas)
        assert_is_tensor(cas)

        # Verify Mach round-trip
        assert_parity(state.mach_number, mach.numpy())

    def test_aircraft_airspeed_from_tas(self):
        from openleaf.performance.physics.aero import AircraftAirspeed

        T = to_torch([250.0])
        p = to_torch([50000.0])
        v = to_torch([240.0])

        state = AircraftAirspeed(T=T, p=p, v=v)
        assert_is_tensor(state.mach_number)
        assert_is_tensor(state.calibrated_airspeed)


class TestESFWithTorch:
    """Verify energy share factor functions with torch tensors."""

    def test_f_constant_mach(self):
        from openleaf.performance.physics.esf import f_constant_mach

        mach_np = np.array([0.78, 0.78, 0.82])
        alt_np = np.array([8000.0, 12000.0, 10000.0])

        mach_t = to_torch([0.78, 0.78, 0.82])
        alt_t = to_torch([8000.0, 12000.0, 10000.0])

        result = f_constant_mach(mach_t, alt_t)
        assert_is_tensor(result)
        assert_parity(result, f_constant_mach(mach_np, alt_np))

    def test_f_constant_cas(self):
        from openleaf.performance.physics.esf import f_constant_cas

        mach_np = np.array([0.6, 0.7, 0.65])
        alt_np = np.array([8000.0, 12000.0, 10000.0])

        mach_t = to_torch([0.6, 0.7, 0.65])
        alt_t = to_torch([8000.0, 12000.0, 10000.0])

        result = f_constant_cas(mach_t, alt_t)
        assert_is_tensor(result)
        assert_parity(result, f_constant_cas(mach_np, alt_np), rtol=1e-5)


class TestTEMWithTorch:
    """Verify Total Energy Model with torch tensors."""

    def test_rocd_pressure(self):
        from openleaf.performance.physics.tem import rocd_pressure

        thrust = to_torch([120000.0])
        drag = to_torch([60000.0])
        mass = to_torch([70000.0])
        tas = to_torch([230.0])
        f = to_torch([0.95])
        T_air = to_torch([250.0])
        alt = to_torch([10000.0])

        result = rocd_pressure(thrust, drag, mass, tas, f, T_air, alt)
        assert_is_tensor(result)

        # Sanity check: positive thrust excess → positive ROCD
        assert float(result) > 0

    def test_thrust_from_rocd(self):
        from openleaf.performance.physics.tem import rocd_pressure
        from openleaf.performance.physics.tem import thrust_from_rocd

        thrust_in = to_torch([120000.0])
        drag = to_torch([60000.0])
        mass = to_torch([70000.0])
        tas = to_torch([230.0])
        f = to_torch([0.95])
        T_air = to_torch([250.0])
        alt = to_torch([10000.0])

        # Compute ROCD from thrust, then recover thrust
        rocd = rocd_pressure(thrust_in, drag, mass, tas, f, T_air, alt)
        thrust_recovered = thrust_from_rocd(rocd, drag, mass, tas, f, T_air, alt)
        assert_is_tensor(thrust_recovered)
        assert_parity(thrust_recovered, thrust_in.numpy(), rtol=1e-4)

    def test_temp_correction_factor(self):
        from openleaf.performance.physics.tem import temp_correction_factor

        T_air = to_torch([250.0, 220.0])
        alt = to_torch([10000.0, 12000.0])

        result = temp_correction_factor(T_air, alt)
        assert_is_tensor(result)
        # ISA at 10000m ≈ 223.15K, so factor ≈ 223.15/250 < 1
        assert float(result[0]) < 1.0

    def test_rocd_geometric(self):
        from openleaf.performance.physics.tem import rocd_geometric

        rocd_p = to_torch([5.0])
        T_air = to_torch([250.0])
        alt = to_torch([10000.0])

        result = rocd_geometric(rocd_p, T_air, alt)
        assert_is_tensor(result)


class TestUnitsWithTorch:
    """Verify that unit constants work naturally with torch arithmetic."""

    def test_unit_conversions(self):
        from openleaf.performance.physics.units import FT
        from openleaf.performance.physics.units import KT
        from openleaf.performance.physics.units import NM

        alt_ft = to_torch([35000.0])
        alt_m = alt_ft * FT
        assert_is_tensor(alt_m)
        assert float(alt_m) == pytest.approx(35000.0 * 0.3048, rel=1e-6)

        speed_kt = to_torch([250.0])
        speed_ms = speed_kt * KT
        assert_is_tensor(speed_ms)

        dist_nm = to_torch([100.0])
        dist_m = dist_nm * NM
        assert_is_tensor(dist_m)


# =============================================================================
# PART 3: Model-layer smoke tests
# =============================================================================


class TestOpenAPWithTorch:
    """Verify OpenAP model works with PyTorch tensors via KeyedData dict API."""

    @pytest.fixture
    def openap_model(self):
        from openleaf.performance.models.openap import OpenAPModel

        return OpenAPModel("A320")

    @pytest.fixture
    def cruise_data(self):
        """Typical cruise state with torch tensors."""
        from openleaf.performance.core.types import AerodynamicConfig
        from openleaf.performance.physics import isa

        h = to_torch([10668.0])
        mach = to_torch([0.78])
        T = isa.air_temperature(h)
        tas = mach * torch.sqrt(
            torch.tensor(isa.GAMMA * isa.R, dtype=torch.float64) * T
        )
        return {
            "altitude": h,
            "mach_number": mach,
            "true_airspeed": tas,
            "air_temperature_ratio": isa.theta(h),
            "air_pressure_ratio": isa.delta(h),
            "aircraft_mass": to_torch([65000.0]),
            "thrust": to_torch([30000.0]),
            "vertical_speed": to_torch([0.0]),
            "f": to_torch([1.0]),
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }

    def test_drag_coef(self, openap_model, cruise_data):
        """OpenAP drag coefficient should accept torch tensors and return tensors."""
        cd = openap_model.drag_coef(cruise_data)
        assert_is_tensor(cd)
        # Drag coef should be in reasonable range
        assert 0.01 < float(cd) < 0.1

    def test_drag(self, openap_model, cruise_data):
        """OpenAP drag should accept torch tensors and return tensors."""
        from openleaf.performance.physics import isa

        cd = openap_model.drag_coef(cruise_data)
        sigma = cruise_data["air_pressure_ratio"] / cruise_data["air_temperature_ratio"]
        rho = sigma * isa.RHO_SL
        drag_data = {**cruise_data, "drag_coef": cd, "air_density": rho}
        drag = openap_model.drag(drag_data)
        assert_is_tensor(drag)
        assert float(drag) > 10000

    def test_thrust_max(self, openap_model, cruise_data):
        """OpenAP max thrust should accept torch tensors and return tensors."""
        result = openap_model.thrust_max(cruise_data)
        assert_is_tensor(result)
        assert float(result) > 0

    def test_fuel_flow(self, openap_model, cruise_data):
        """OpenAP fuel flow should accept torch tensors."""
        ff = openap_model.fuel_flow(cruise_data)
        assert_is_tensor(ff)
        assert float(ff) > 0


class TestPSWithTorch:
    """Verify Poll-Schumann model works with PyTorch tensors via KeyedData dict API."""

    @pytest.fixture
    def ps_model(self):
        from openleaf.performance.models.ps import PSModel

        return PSModel("A320")

    @pytest.fixture
    def cruise_data(self, ps_model):
        """Typical cruise state with torch tensors, including PS-specific fields."""
        from openleaf.performance.core.types import AerodynamicConfig
        from openleaf.performance.physics import isa
        from openleaf.performance.physics.aero import aerodynamic_coefficient
        from openleaf.performance.physics.aero import dynamic_pressure

        h = to_torch([10668.0])
        mach = to_torch([0.78])
        T = isa.air_temperature(h)
        p = isa.air_pressure(h)
        tas = mach * torch.sqrt(
            torch.tensor(isa.GAMMA * isa.R, dtype=torch.float64) * T
        )
        sigma = isa.delta(h) / isa.theta(h)
        rho = sigma * isa.RHO_SL
        mass = to_torch([70000.0])

        # PS model needs lift_coef
        q = dynamic_pressure(tas, rho)
        lift = mass * torch.tensor(isa.G, dtype=torch.float64)
        cl = aerodynamic_coefficient(lift, q, ps_model.aerodynamics.surface_area)

        return {
            "altitude": h,
            "mach_number": mach,
            "true_airspeed": tas,
            "air_temperature": T,
            "air_pressure": p,
            "air_temperature_ratio": isa.theta(h),
            "air_pressure_ratio": isa.delta(h),
            "air_density": rho,
            "aircraft_mass": mass,
            "lift_coef": cl,
            "thrust": to_torch([50000.0]),
            "vertical_speed": to_torch([0.0]),
            "f": to_torch([1.0]),
            "aerodynamic_configuration": AerodynamicConfig.CR,
        }

    def test_drag_coef(self, ps_model, cruise_data):
        """PS drag coefficient should accept torch tensors and return tensors."""
        cd = ps_model.drag_coef(cruise_data)
        assert_is_tensor(cd)
        assert 0.01 < float(cd) < 0.1

    def test_drag(self, ps_model, cruise_data):
        """PS drag should accept torch tensors."""
        cd = ps_model.drag_coef(cruise_data)
        drag_data = {**cruise_data, "drag_coef": cd}
        drag = ps_model.drag(drag_data)
        assert_is_tensor(drag)
        assert float(drag) > 10000

    def test_fuel_flow(self, ps_model, cruise_data):
        """PS fuel flow should accept torch tensors."""
        ff = ps_model.fuel_flow(cruise_data)
        assert_is_tensor(ff)
        assert float(ff) > 0


# =============================================================================
# PART 4: Gradient flow verification
# =============================================================================


class TestGradientFlow:
    """Verify that gradients flow through physics computations.

    This proves the 'differentiable' claim — the whole point of PyTorch support.
    """

    def test_isa_gradient(self):
        """Gradients should flow through ISA atmosphere computation."""
        from openleaf.performance.physics.isa import air_pressure
        from openleaf.performance.physics.isa import air_temperature

        h = torch.tensor([5000.0], dtype=torch.float64, requires_grad=True)

        T = air_temperature(h)
        T.backward()

        assert h.grad is not None
        assert torch.isfinite(h.grad).all()
        # Temperature decreases with altitude → gradient should be negative
        assert float(h.grad) < 0

        h2 = torch.tensor([5000.0], dtype=torch.float64, requires_grad=True)
        p = air_pressure(h2)
        p.backward()
        assert h2.grad is not None
        assert torch.isfinite(h2.grad).all()
        # Pressure decreases with altitude → gradient should be negative
        assert float(h2.grad) < 0

    def test_dynamic_pressure_gradient(self):
        """Gradients should flow through dynamic pressure."""
        from openleaf.performance.physics.aero import dynamic_pressure

        v = torch.tensor([230.0], dtype=torch.float64, requires_grad=True)
        rho = torch.tensor([0.4135], dtype=torch.float64, requires_grad=True)

        q = dynamic_pressure(v, rho)
        q.backward()

        # dq/dv = rho * v > 0
        assert v.grad is not None
        assert float(v.grad) > 0
        # dq/drho = 0.5 * v^2 > 0
        assert rho.grad is not None
        assert float(rho.grad) > 0

    def test_esf_gradient(self):
        """Gradients should flow through energy share factor."""
        from openleaf.performance.physics.esf import f_constant_mach

        mach = torch.tensor([0.78], dtype=torch.float64, requires_grad=True)
        alt = torch.tensor([8000.0], dtype=torch.float64)

        f = f_constant_mach(mach, alt)
        f.backward()

        assert mach.grad is not None
        assert torch.isfinite(mach.grad).all()

    def test_rocd_gradient_wrt_thrust(self):
        """Gradients should flow through TEM ROCD computation."""
        from openleaf.performance.physics.tem import rocd_pressure

        thrust = torch.tensor([120000.0], dtype=torch.float64, requires_grad=True)
        drag = torch.tensor([60000.0], dtype=torch.float64)
        mass = torch.tensor([70000.0], dtype=torch.float64)
        tas = torch.tensor([230.0], dtype=torch.float64)
        f = torch.tensor([0.95], dtype=torch.float64)
        T_air = torch.tensor([250.0], dtype=torch.float64)
        alt = torch.tensor([10000.0], dtype=torch.float64)

        rocd = rocd_pressure(thrust, drag, mass, tas, f, T_air, alt)
        rocd.backward()

        # More thrust → higher ROCD → positive gradient
        assert thrust.grad is not None
        assert float(thrust.grad) > 0

    def test_rocd_gradient_wrt_mass(self):
        """ROCD should decrease with increasing mass."""
        from openleaf.performance.physics.tem import rocd_pressure

        thrust = torch.tensor([120000.0], dtype=torch.float64)
        drag = torch.tensor([60000.0], dtype=torch.float64)
        mass = torch.tensor([70000.0], dtype=torch.float64, requires_grad=True)
        tas = torch.tensor([230.0], dtype=torch.float64)
        f = torch.tensor([0.95], dtype=torch.float64)
        T_air = torch.tensor([250.0], dtype=torch.float64)
        alt = torch.tensor([10000.0], dtype=torch.float64)

        rocd = rocd_pressure(thrust, drag, mass, tas, f, T_air, alt)
        rocd.backward()

        # More mass → lower ROCD → negative gradient
        assert mass.grad is not None
        assert float(mass.grad) < 0

    def test_chain_altitude_to_pressure_gradient(self):
        """Gradient chain: altitude → atmosphere → pressure → airspeed."""
        from openleaf.performance.physics.isa import AtmosphericState

        h = torch.tensor([10000.0], dtype=torch.float64, requires_grad=True)
        atm = AtmosphericState(h)

        # Speed of sound depends on altitude via temperature
        a = atm.speed_of_sound
        a.backward()

        assert h.grad is not None
        assert torch.isfinite(h.grad).all()

    def test_multi_variable_gradient(self):
        """Multiple variables should all receive gradients in a compound expression."""
        from openleaf.performance.physics.aero import dynamic_pressure
        from openleaf.performance.physics.aero import thrust_required

        v = torch.tensor([230.0], dtype=torch.float64, requires_grad=True)
        mass = torch.tensor([70000.0], dtype=torch.float64, requires_grad=True)
        rho = torch.tensor([0.4135], dtype=torch.float64)

        drag = dynamic_pressure(v, rho) * 0.03 * 122.6  # Simplified drag
        vs = torch.tensor([5.0], dtype=torch.float64)
        f = torch.tensor([0.95], dtype=torch.float64)
        T_req = thrust_required(drag, mass, v, vs, f)
        T_req.backward()

        assert v.grad is not None
        assert mass.grad is not None
        assert torch.isfinite(v.grad).all()
        assert torch.isfinite(mass.grad).all()
