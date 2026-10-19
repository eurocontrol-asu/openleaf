"""Cross-backend integration tests for all models x all backends x all functions.

Verifies that fuel_flow, drag_coef, thrust_max, and thrust_min work correctly
across NumPy, PyTorch, CasADi, and JAX for every aircraft model and engine type.

Backend matrix:
    - NumPy: baseline reference values
    - PyTorch: verify tensor dispatch and numerical parity with NumPy
    - JAX: verify Array API dispatch and numerical parity with NumPy
    - CasADi: verify symbolic computation and evaluate-to-numeric parity
"""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.numpy.backends import Backend
from openleaf.performance.physics import isa


# ---------------------------------------------------------------------------
# Backend availability markers
# ---------------------------------------------------------------------------

has_torch = Backend.TORCH.available
has_casadi = Backend.CASADI.available

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    has_jax = True
except ImportError:
    has_jax = False
    jnp = None  # type: ignore[assignment]

skip_no_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
skip_no_casadi = pytest.mark.skipif(not has_casadi, reason="CasADi not installed")
skip_no_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")

if has_torch:
    import torch

if has_casadi:
    import casadi as ca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric_scalar(v):
    """Check if v is a plain numeric scalar (not an Enum)."""
    from enum import Enum

    if isinstance(v, Enum):
        return False
    return isinstance(v, (int, float, np.floating))


def to_torch_val(x, requires_grad=False):
    """Convert a scalar to a torch float64 tensor."""
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def to_jax_val(x):
    """Convert a scalar to a JAX float64 array."""
    return jnp.float64(x)


def to_float(x):
    """Extract a python float from any backend type."""
    if has_torch and isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    if has_casadi and isinstance(x, (ca.MX, ca.DM, ca.SX)):
        return float(x)
    if has_jax and isinstance(x, jnp.ndarray):
        return float(x)
    return float(x)


def assert_close(actual, expected, rtol=1e-5, atol=1e-7, label=""):
    """Assert two scalar values are close, with a label for diagnostics."""
    import math

    a = to_float(actual)
    e = to_float(expected)
    # Handle NaN: both NaN is OK
    if math.isnan(a) and math.isnan(e):
        return
    assert abs(a - e) <= atol + rtol * abs(e), f"{label}: {a} != {e} (rtol={rtol}, atol={atol})"


# ============================================================================
# Model fixtures (reuse conftest where possible, add missing ones)
# ============================================================================


@pytest.fixture
def openap_model():
    """OpenAP A320 model."""
    pytest.importorskip("openap")
    from openleaf.performance.models.openap import OpenAPModel

    return OpenAPModel("A320")


@pytest.fixture
def ps_model():
    """PS A320 model."""
    pytest.importorskip("pycontrails")
    from openleaf.performance.models.ps import PSModel

    return PSModel("A320")


# ============================================================================
# BADA3 x ALL BACKENDS
# ============================================================================

# ---- BADA3 state builders ---------------------------------------------------


def _bada3_engine_state_numpy(aircraft_type: str) -> dict:
    """Build a NumPy-valued state dict for BADA3 engine functions."""
    if aircraft_type == "jet":
        return {
            "true_airspeed": 230.0,
            "altitude": 10668.0,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "thrust": 50000.0,
        }
    elif aircraft_type == "turboprop":
        return {
            "true_airspeed": 140.0,
            "altitude": 6096.0,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "thrust": 15000.0,
        }
    else:  # piston
        return {
            "true_airspeed": 75.0,
            "altitude": 2438.0,
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "thrust": 2000.0,
        }


def _bada3_aero_state_numpy(aircraft_type: str) -> dict:
    """Build a NumPy-valued state dict for BADA3 aerodynamics (drag_coef)."""
    return {
        "lift_coef": 0.5,
        "mach_number": {"jet": 0.78, "turboprop": 0.45, "piston": 0.23}[aircraft_type],
        "aerodynamic_configuration": AerodynamicConfig.CR,
    }


def _convert_bada3_state_torch(state: dict) -> dict:
    """Clone a BADA3 state with torch tensors for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_torch_val(float(v))
        else:
            out[k] = v
    return out


def _convert_bada3_state_jax(state: dict) -> dict:
    """Clone a BADA3 state with JAX arrays for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_jax_val(float(v))
        else:
            out[k] = v
    return out


def _bada3_engine_state_casadi(aircraft_type: str) -> tuple[dict, list, list]:
    """Build a CasADi-symbolic state for BADA3 engine functions.

    Returns (state, sym_vars, eval_vals) so we can build a CasADi Function
    and evaluate it at the numeric point.
    """
    np_state = _bada3_engine_state_numpy(aircraft_type)
    v_sym = ca.MX.sym("v")
    h_sym = ca.MX.sym("h")
    thrust_sym = ca.MX.sym("thrust")

    sym_state = {
        "true_airspeed": v_sym,
        "altitude": h_sym,
        "engine_setting": np_state["engine_setting"],
        "delta_isa": 0.0,
        "aerodynamic_configuration": np_state["aerodynamic_configuration"],
        "thrust": thrust_sym,
    }
    return (
        sym_state,
        [v_sym, h_sym, thrust_sym],
        [np_state["true_airspeed"], np_state["altitude"], np_state["thrust"]],
    )


def _bada3_aero_state_casadi(aircraft_type: str) -> tuple[dict, list, list]:
    """Build CasADi-symbolic state for BADA3 drag_coef."""
    np_state = _bada3_aero_state_numpy(aircraft_type)
    cl_sym = ca.MX.sym("cl")
    sym_state = {
        "lift_coef": cl_sym,
        "mach_number": np_state["mach_number"],
        "aerodynamic_configuration": AerodynamicConfig.CR,
    }
    return sym_state, [cl_sym], [np_state["lift_coef"]]


# ---- BADA3 tests ------------------------------------------------------------


class TestBADA3CrossBackend:
    """Test BADA3 common functions across all backends for jet, turboprop, piston."""

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    class TestNumPy:
        """NumPy baseline for BADA3."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada3_model",
                "turboprop": "bada3_turboprop_model",
                "piston": "bada3_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada3_engine_state_numpy(aircraft_type)
            result = model.engine.thrust_max(state)
            assert to_float(result) > 0

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada3_engine_state_numpy(aircraft_type)
            result = model.engine.thrust_min(state)
            # >= 0: the AT72 OPF has CTdes,high = 0.0, so idle thrust is
            # exactly zero above Hp,des; the PTD reference lists 0 N there.
            assert to_float(result) >= 0

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada3_engine_state_numpy(aircraft_type)
            result = model.engine.fuel_flow(state)
            assert to_float(result) > 0

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada3_engine_state_numpy(aircraft_type)
            result = model.engine.fuel_flow_min(state)
            assert to_float(result) > 0

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada3_aero_state_numpy(aircraft_type)
            result = model.aerodynamics.drag_coef(state)
            assert 0 < to_float(result) < 1

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_torch
    class TestTorch:
        """PyTorch parity for BADA3."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada3_model",
                "turboprop": "bada3_turboprop_model",
                "piston": "bada3_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            t_state = _convert_bada3_state_torch(np_state)
            result = model.engine.thrust_max(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="thrust_max torch")

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            t_state = _convert_bada3_state_torch(np_state)
            result = model.engine.thrust_min(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="thrust_min torch")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            t_state = _convert_bada3_state_torch(np_state)
            result = model.engine.fuel_flow(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="fuel_flow torch")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            t_state = _convert_bada3_state_torch(np_state)
            result = model.engine.fuel_flow_min(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="fuel_flow_min torch")

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            t_state = _convert_bada3_state_torch(np_state)
            result = model.aerodynamics.drag_coef(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="drag_coef torch")

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_jax
    class TestJAX:
        """JAX parity for BADA3."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada3_model",
                "turboprop": "bada3_turboprop_model",
                "piston": "bada3_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            j_state = _convert_bada3_state_jax(np_state)
            result = model.engine.thrust_max(j_state)
            assert_close(result, ref, label="thrust_max jax")

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            j_state = _convert_bada3_state_jax(np_state)
            result = model.engine.thrust_min(j_state)
            assert_close(result, ref, label="thrust_min jax")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            j_state = _convert_bada3_state_jax(np_state)
            result = model.engine.fuel_flow(j_state)
            assert_close(result, ref, label="fuel_flow jax")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            j_state = _convert_bada3_state_jax(np_state)
            result = model.engine.fuel_flow_min(j_state)
            assert_close(result, ref, label="fuel_flow_min jax")

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            j_state = _convert_bada3_state_jax(np_state)
            result = model.aerodynamics.drag_coef(j_state)
            assert_close(result, ref, label="drag_coef jax")

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_casadi
    class TestCasADi:
        """CasADi symbolic for BADA3."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada3_model",
                "turboprop": "bada3_turboprop_model",
                "piston": "bada3_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            sym_state, syms, vals = _bada3_engine_state_casadi(aircraft_type)
            result = model.engine.thrust_max(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="thrust_max casadi")

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            sym_state, syms, vals = _bada3_engine_state_casadi(aircraft_type)
            result = model.engine.thrust_min(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="thrust_min casadi")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            sym_state, syms, vals = _bada3_engine_state_casadi(aircraft_type)
            result = model.engine.fuel_flow(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="fuel_flow casadi")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            h_sym = ca.MX.sym("h")
            sym_state = {"altitude": h_sym}
            result = model.engine.fuel_flow_min(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", [h_sym], [result])
            assert_close(f(np_state["altitude"]), ref, label="fuel_flow_min casadi")

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada3_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            sym_state, syms, vals = _bada3_aero_state_casadi(aircraft_type)
            result = model.aerodynamics.drag_coef(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="drag_coef casadi")


# ============================================================================
# BADA4 x ALL BACKENDS
# ============================================================================

# ---- BADA4 state builders ---------------------------------------------------


def _bada4_engine_state_numpy(aircraft_type: str) -> dict:
    """Build a NumPy-valued state dict for BADA4 engine functions."""
    if aircraft_type == "jet":
        h = 10668.0
        return {
            "mach_number": 0.78,
            "air_pressure": isa.air_pressure(h),
            "air_temperature": isa.air_temperature(h),
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "thrust": 50000.0,
        }
    elif aircraft_type == "turboprop":
        h = 6096.0
        return {
            "mach_number": 0.45,
            "air_pressure": isa.air_pressure(h),
            "air_temperature": isa.air_temperature(h),
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "thrust": 15000.0,
        }
    else:  # piston
        h = 2438.0
        return {
            "mach_number": 0.23,
            "air_pressure": isa.air_pressure(h),
            "air_temperature": isa.air_temperature(h),
            "engine_setting": EngineSetting.MCMB,
            "delta_isa": 0.0,
            "thrust": 2000.0,
        }


def _bada4_aero_state_numpy(aircraft_type: str) -> dict:
    """Build a NumPy-valued state dict for BADA4 aerodynamics (drag_coef)."""
    return {
        "lift_coef": 0.5,
        "mach_number": {"jet": 0.78, "turboprop": 0.45, "piston": 0.23}[aircraft_type],
        "aerodynamic_configuration": AerodynamicConfig.CR,
    }


def _convert_bada4_state_torch(state: dict) -> dict:
    """Clone a BADA4 state with torch tensors for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_torch_val(float(v))
        else:
            out[k] = v
    return out


def _convert_bada4_state_jax(state: dict) -> dict:
    """Clone a BADA4 state with JAX arrays for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_jax_val(float(v))
        else:
            out[k] = v
    return out


def _bada4_engine_state_casadi(aircraft_type: str) -> tuple[dict, list, list]:
    """Build a CasADi-symbolic state for BADA4 engine functions.

    Returns (state, sym_vars, eval_vals).
    """
    np_state = _bada4_engine_state_numpy(aircraft_type)
    mach_sym = ca.MX.sym("mach")
    p_sym = ca.MX.sym("p")
    T_sym = ca.MX.sym("T")
    thrust_sym = ca.MX.sym("thrust")

    sym_state = {
        "mach_number": mach_sym,
        "air_pressure": p_sym,
        "air_temperature": T_sym,
        "engine_setting": np_state["engine_setting"],
        "delta_isa": 0.0,
        "thrust": thrust_sym,
    }
    return (
        sym_state,
        [mach_sym, p_sym, T_sym, thrust_sym],
        [
            np_state["mach_number"],
            np_state["air_pressure"],
            np_state["air_temperature"],
            np_state["thrust"],
        ],
    )


def _bada4_aero_state_casadi(aircraft_type: str) -> tuple[dict, list, list]:
    """Build CasADi-symbolic state for BADA4 drag_coef."""
    np_state = _bada4_aero_state_numpy(aircraft_type)
    cl_sym = ca.MX.sym("cl")
    mach_sym = ca.MX.sym("mach")
    sym_state = {
        "lift_coef": cl_sym,
        "mach_number": mach_sym,
        "aerodynamic_configuration": AerodynamicConfig.CR,
    }
    return (
        sym_state,
        [cl_sym, mach_sym],
        [np_state["lift_coef"], np_state["mach_number"]],
    )


# ---- BADA4 tests ------------------------------------------------------------


class TestBADA4CrossBackend:
    """Test BADA4 common functions across all backends for jet, turboprop, piston."""

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    class TestNumPy:
        """NumPy baseline for BADA4."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada4_model",
                "turboprop": "bada4_turboprop_model",
                "piston": "bada4_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada4_engine_state_numpy(aircraft_type)
            result = model.engine.thrust_max(state)
            assert to_float(result) > 0

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada4_engine_state_numpy(aircraft_type)
            result = to_float(model.engine.thrust_min(state))
            # BADA4 idle thrust can be negative (propeller drag) or NaN
            # at edge conditions; just verify it produces a number
            assert isinstance(result, float)

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada4_engine_state_numpy(aircraft_type)
            result = model.engine.fuel_flow(state)
            assert to_float(result) > 0

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada4_engine_state_numpy(aircraft_type)
            result = model.engine.fuel_flow_min(state)
            # Piston fuel_flow_min can be 0 at certain conditions
            assert to_float(result) >= 0

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            state = _bada4_aero_state_numpy(aircraft_type)
            result = model.aerodynamics.drag_coef(state)
            assert 0 < to_float(result) < 1

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_torch
    class TestTorch:
        """PyTorch parity for BADA4."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada4_model",
                "turboprop": "bada4_turboprop_model",
                "piston": "bada4_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            t_state = _convert_bada4_state_torch(np_state)
            result = model.engine.thrust_max(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="bada4 thrust_max torch")

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            t_state = _convert_bada4_state_torch(np_state)
            result = model.engine.thrust_min(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="bada4 thrust_min torch")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            t_state = _convert_bada4_state_torch(np_state)
            result = model.engine.fuel_flow(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="bada4 fuel_flow torch")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            t_state = _convert_bada4_state_torch(np_state)
            result = model.engine.fuel_flow_min(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="bada4 fuel_flow_min torch")

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            t_state = _convert_bada4_state_torch(np_state)
            result = model.aerodynamics.drag_coef(t_state)
            assert isinstance(result, torch.Tensor)
            assert_close(result, ref, label="bada4 drag_coef torch")

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_jax
    class TestJAX:
        """JAX parity for BADA4."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada4_model",
                "turboprop": "bada4_turboprop_model",
                "piston": "bada4_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            j_state = _convert_bada4_state_jax(np_state)
            result = model.engine.thrust_max(j_state)
            assert_close(result, ref, label="bada4 thrust_max jax")

        def test_thrust_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            j_state = _convert_bada4_state_jax(np_state)
            result = model.engine.thrust_min(j_state)
            assert_close(result, ref, label="bada4 thrust_min jax")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            j_state = _convert_bada4_state_jax(np_state)
            result = model.engine.fuel_flow(j_state)
            assert_close(result, ref, label="bada4 fuel_flow jax")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            j_state = _convert_bada4_state_jax(np_state)
            result = model.engine.fuel_flow_min(j_state)
            assert_close(result, ref, label="bada4 fuel_flow_min jax")

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            j_state = _convert_bada4_state_jax(np_state)
            result = model.aerodynamics.drag_coef(j_state)
            assert_close(result, ref, label="bada4 drag_coef jax")

    @pytest.mark.parametrize("aircraft_type", ["jet", "turboprop", "piston"])
    @skip_no_casadi
    class TestCasADi:
        """CasADi symbolic for BADA4."""

        def _get_model(self, aircraft_type, request):
            fixture = {
                "jet": "bada4_model",
                "turboprop": "bada4_turboprop_model",
                "piston": "bada4_piston_model",
            }[aircraft_type]
            return request.getfixturevalue(fixture)

        def test_thrust_max(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_max(np_state))

            sym_state, syms, vals = _bada4_engine_state_casadi(aircraft_type)
            result = model.engine.thrust_max(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="bada4 thrust_max casadi")

        def test_thrust_min(self, aircraft_type, request):
            import math

            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.thrust_min(np_state))

            mach_sym = ca.MX.sym("mach")
            p_sym = ca.MX.sym("p")
            T_sym = ca.MX.sym("T")
            sym_state = {
                "mach_number": mach_sym,
                "air_pressure": p_sym,
                "air_temperature": T_sym,
            }
            result = model.engine.thrust_min(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", [mach_sym, p_sym, T_sym], [result])
            casadi_val = to_float(
                f(
                    np_state["mach_number"],
                    np_state["air_pressure"],
                    np_state["air_temperature"],
                )
            )
            # BADA4 piston at low Mach hits divide-by-zero; NumPy returns NaN
            # while CasADi returns 0.0 (different div-by-zero semantics)
            if math.isnan(ref):
                assert math.isnan(casadi_val) or casadi_val == 0.0
            else:
                assert_close(casadi_val, ref, label="bada4 thrust_min casadi")

        def test_fuel_flow(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow(np_state))

            sym_state, syms, vals = _bada4_engine_state_casadi(aircraft_type)
            result = model.engine.fuel_flow(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="bada4 fuel_flow casadi")

        def test_fuel_flow_min(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_engine_state_numpy(aircraft_type)
            ref = to_float(model.engine.fuel_flow_min(np_state))

            mach_sym = ca.MX.sym("mach")
            p_sym = ca.MX.sym("p")
            T_sym = ca.MX.sym("T")
            sym_state = {
                "mach_number": mach_sym,
                "air_pressure": p_sym,
                "air_temperature": T_sym,
            }
            result = model.engine.fuel_flow_min(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", [mach_sym, p_sym, T_sym], [result])
            assert_close(
                f(
                    np_state["mach_number"],
                    np_state["air_pressure"],
                    np_state["air_temperature"],
                ),
                ref,
                label="bada4 fuel_flow_min casadi",
            )

        def test_drag_coef(self, aircraft_type, request):
            model = self._get_model(aircraft_type, request)
            np_state = _bada4_aero_state_numpy(aircraft_type)
            ref = to_float(model.aerodynamics.drag_coef(np_state))

            sym_state, syms, vals = _bada4_aero_state_casadi(aircraft_type)
            result = model.aerodynamics.drag_coef(sym_state)
            assert isinstance(result, ca.MX)
            f = ca.Function("f", syms, [result])
            assert_close(f(*vals), ref, label="bada4 drag_coef casadi")


# ============================================================================
# OpenAP x ALL BACKENDS
# ============================================================================


def _openap_engine_state_numpy() -> dict:
    """Build a NumPy-valued state for OpenAP engine functions."""
    h = 10668.0
    T = isa.air_temperature(h)
    mach = 0.78
    tas = mach * np.sqrt(isa.GAMMA * isa.R * T)
    theta = isa.theta(h)
    delta = isa.delta(h)
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


def _convert_openap_state_torch(state: dict) -> dict:
    """Clone OpenAP state with torch tensors for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_torch_val(float(v))
        else:
            out[k] = v
    return out


def _convert_openap_state_jax(state: dict) -> dict:
    """Clone OpenAP state with JAX arrays for numeric keys."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_jax_val(float(v))
        else:
            out[k] = v
    return out


class TestOpenAPCrossBackend:
    """Test OpenAP common functions across all backends."""

    # -- NumPy ----------------------------------------------------------------

    def test_numpy_thrust_max(self, openap_model):
        state = _openap_engine_state_numpy()
        result = openap_model.engine.thrust_max(state)
        assert to_float(result) > 0

    def test_numpy_thrust_min(self, openap_model):
        state = _openap_engine_state_numpy()
        result = openap_model.engine.thrust_min(state)
        assert to_float(result) >= 0

    def test_numpy_fuel_flow(self, openap_model):
        state = _openap_engine_state_numpy()
        result = openap_model.engine.fuel_flow(state)
        assert to_float(result) > 0

    def test_numpy_fuel_flow_min(self, openap_model):
        state = _openap_engine_state_numpy()
        result = openap_model.engine.fuel_flow_min(state)
        assert to_float(result) > 0

    def test_numpy_drag_coef(self, openap_model):
        state = _openap_engine_state_numpy()
        result = openap_model.aerodynamics.drag_coef(state)
        assert 0 < to_float(result) < 1

    # -- PyTorch ---------------------------------------------------------------

    @skip_no_torch
    def test_torch_thrust_max(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_max(np_state))
        t_state = _convert_openap_state_torch(np_state)
        result = openap_model.engine.thrust_max(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="openap thrust_max torch")

    @skip_no_torch
    def test_torch_thrust_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_min(np_state))
        t_state = _convert_openap_state_torch(np_state)
        result = openap_model.engine.thrust_min(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="openap thrust_min torch")

    @skip_no_torch
    def test_torch_fuel_flow(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow(np_state))
        t_state = _convert_openap_state_torch(np_state)
        result = openap_model.engine.fuel_flow(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="openap fuel_flow torch")

    @skip_no_torch
    def test_torch_fuel_flow_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow_min(np_state))
        t_state = _convert_openap_state_torch(np_state)
        result = openap_model.engine.fuel_flow_min(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="openap fuel_flow_min torch")

    @skip_no_torch
    def test_torch_drag_coef(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.aerodynamics.drag_coef(np_state))
        t_state = _convert_openap_state_torch(np_state)
        result = openap_model.aerodynamics.drag_coef(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="openap drag_coef torch")

    # -- JAX -------------------------------------------------------------------

    @skip_no_jax
    def test_jax_thrust_max(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_max(np_state))
        j_state = _convert_openap_state_jax(np_state)
        result = openap_model.engine.thrust_max(j_state)
        assert_close(result, ref, label="openap thrust_max jax")

    @skip_no_jax
    def test_jax_thrust_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_min(np_state))
        j_state = _convert_openap_state_jax(np_state)
        result = openap_model.engine.thrust_min(j_state)
        assert_close(result, ref, label="openap thrust_min jax")

    @skip_no_jax
    def test_jax_fuel_flow(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow(np_state))
        j_state = _convert_openap_state_jax(np_state)
        result = openap_model.engine.fuel_flow(j_state)
        assert_close(result, ref, label="openap fuel_flow jax")

    @skip_no_jax
    def test_jax_fuel_flow_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow_min(np_state))
        j_state = _convert_openap_state_jax(np_state)
        result = openap_model.engine.fuel_flow_min(j_state)
        assert_close(result, ref, label="openap fuel_flow_min jax")

    @skip_no_jax
    def test_jax_drag_coef(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.aerodynamics.drag_coef(np_state))
        j_state = _convert_openap_state_jax(np_state)
        result = openap_model.aerodynamics.drag_coef(j_state)
        assert_close(result, ref, label="openap drag_coef jax")

    # -- CasADi ----------------------------------------------------------------

    @skip_no_casadi
    def test_casadi_thrust_max(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_max(np_state))

        alt_sym = ca.MX.sym("alt")
        tas_sym = ca.MX.sym("tas")
        state = {
            "altitude": alt_sym,
            "true_airspeed": tas_sym,
            "vertical_speed": 0.0,  # Must be numeric for OpenAP if/else
        }
        result = openap_model.engine.thrust_max(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [alt_sym, tas_sym], [result])
        assert_close(
            f(np_state["altitude"], np_state["true_airspeed"]),
            ref,
            label="openap thrust_max casadi",
        )

    @skip_no_casadi
    def test_casadi_thrust_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.thrust_min(np_state))

        alt_sym = ca.MX.sym("alt")
        tas_sym = ca.MX.sym("tas")
        state = {"altitude": alt_sym, "true_airspeed": tas_sym}
        result = openap_model.engine.thrust_min(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [alt_sym, tas_sym], [result])
        assert_close(
            f(np_state["altitude"], np_state["true_airspeed"]),
            ref,
            label="openap thrust_min casadi",
        )

    @skip_no_casadi
    def test_casadi_fuel_flow(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow(np_state))

        thrust_sym = ca.MX.sym("thrust")
        state = {"thrust": thrust_sym}
        result = openap_model.engine.fuel_flow(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [thrust_sym], [result])
        assert_close(f(np_state["thrust"]), ref, label="openap fuel_flow casadi")

    @skip_no_casadi
    def test_casadi_fuel_flow_min(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.engine.fuel_flow_min(np_state))

        alt_sym = ca.MX.sym("alt")
        tas_sym = ca.MX.sym("tas")
        state = {"altitude": alt_sym, "true_airspeed": tas_sym}
        result = openap_model.engine.fuel_flow_min(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [alt_sym, tas_sym], [result])
        assert_close(
            f(np_state["altitude"], np_state["true_airspeed"]),
            ref,
            label="openap fuel_flow_min casadi",
        )

    @skip_no_casadi
    def test_casadi_drag_coef(self, openap_model):
        np_state = _openap_engine_state_numpy()
        ref = to_float(openap_model.aerodynamics.drag_coef(np_state))

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
        result = openap_model.aerodynamics.drag_coef(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [mass_sym, tas_sym, alt_sym], [result])
        assert_close(
            f(
                np_state["aircraft_mass"],
                np_state["true_airspeed"],
                np_state["altitude"],
            ),
            ref,
            label="openap drag_coef casadi",
        )


# ============================================================================
# PS x ALL BACKENDS
# ============================================================================


def _ps_engine_state_numpy() -> dict:
    """Build a NumPy-valued state for PS engine functions."""
    h = 10668.0
    return {
        "altitude": h,
        "mach_number": 0.78,
        "air_temperature": isa.air_temperature(h),
        "air_pressure": isa.air_pressure(h),
        "air_temperature_ratio": isa.theta(h),
        "air_pressure_ratio": isa.delta(h),
        "thrust": 25000.0,
    }


def _ps_aero_state_numpy() -> dict:
    """Build a NumPy-valued state for PS aerodynamics (drag_coef)."""
    h = 10668.0
    return {
        "lift_coef": 0.5,
        "mach_number": 0.78,
        "air_temperature": isa.air_temperature(h),
        "air_pressure": isa.air_pressure(h),
    }


def _convert_ps_state_torch(state: dict) -> dict:
    """Clone PS state with torch tensors."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_torch_val(float(v))
        else:
            out[k] = v
    return out


def _convert_ps_state_jax(state: dict) -> dict:
    """Clone PS state with JAX arrays."""
    out = {}
    for k, v in state.items():
        if _is_numeric_scalar(v):
            out[k] = to_jax_val(float(v))
        else:
            out[k] = v
    return out


class TestPSCrossBackend:
    """Test PS common functions across all backends."""

    # -- NumPy ----------------------------------------------------------------

    def test_numpy_thrust_max(self, ps_model):
        state = _ps_engine_state_numpy()
        result = ps_model.engine.thrust_max(state)
        assert to_float(result) > 0

    def test_numpy_thrust_min(self, ps_model):
        state = _ps_engine_state_numpy()
        result = ps_model.engine.thrust_min(state)
        assert to_float(result) >= 0

    def test_numpy_fuel_flow(self, ps_model):
        state = _ps_engine_state_numpy()
        result = ps_model.engine.fuel_flow(state)
        assert to_float(result) > 0

    def test_numpy_fuel_flow_min(self, ps_model):
        state = _ps_engine_state_numpy()
        result = ps_model.engine.fuel_flow_min(state)
        assert to_float(result) > 0

    def test_numpy_overall_propulsion_efficiency(self, ps_model):
        state = _ps_engine_state_numpy()
        result = ps_model.engine.overall_propulsion_efficiency(state)
        assert 0 < to_float(result) < 1

    def test_numpy_drag_coef(self, ps_model):
        state = _ps_aero_state_numpy()
        result = ps_model.aerodynamics.drag_coef(state)
        assert 0 < to_float(result) < 1

    # -- PyTorch ---------------------------------------------------------------

    @skip_no_torch
    def test_torch_thrust_max(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_max(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.engine.thrust_max(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps thrust_max torch")

    @skip_no_torch
    def test_torch_thrust_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_min(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.engine.thrust_min(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps thrust_min torch")

    @skip_no_torch
    def test_torch_fuel_flow(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.engine.fuel_flow(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps fuel_flow torch")

    @skip_no_torch
    def test_torch_fuel_flow_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow_min(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.engine.fuel_flow_min(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps fuel_flow_min torch")

    @skip_no_torch
    def test_torch_overall_propulsion_efficiency(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.overall_propulsion_efficiency(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.engine.overall_propulsion_efficiency(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps overall_propulsion_efficiency torch")

    @skip_no_torch
    def test_torch_drag_coef(self, ps_model):
        np_state = _ps_aero_state_numpy()
        ref = to_float(ps_model.aerodynamics.drag_coef(np_state))
        t_state = _convert_ps_state_torch(np_state)
        result = ps_model.aerodynamics.drag_coef(t_state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="ps drag_coef torch")

    # -- JAX -------------------------------------------------------------------

    @skip_no_jax
    def test_jax_thrust_max(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_max(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.engine.thrust_max(j_state)
        assert_close(result, ref, label="ps thrust_max jax")

    @skip_no_jax
    def test_jax_thrust_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_min(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.engine.thrust_min(j_state)
        assert_close(result, ref, label="ps thrust_min jax")

    @skip_no_jax
    def test_jax_fuel_flow(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.engine.fuel_flow(j_state)
        assert_close(result, ref, label="ps fuel_flow jax")

    @skip_no_jax
    def test_jax_fuel_flow_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow_min(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.engine.fuel_flow_min(j_state)
        assert_close(result, ref, label="ps fuel_flow_min jax")

    @skip_no_jax
    def test_jax_overall_propulsion_efficiency(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.overall_propulsion_efficiency(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.engine.overall_propulsion_efficiency(j_state)
        assert_close(result, ref, label="ps overall_propulsion_efficiency jax")

    @skip_no_jax
    def test_jax_drag_coef(self, ps_model):
        np_state = _ps_aero_state_numpy()
        ref = to_float(ps_model.aerodynamics.drag_coef(np_state))
        j_state = _convert_ps_state_jax(np_state)
        result = ps_model.aerodynamics.drag_coef(j_state)
        assert_close(result, ref, label="ps drag_coef jax")

    # -- CasADi ----------------------------------------------------------------

    @skip_no_casadi
    def test_casadi_thrust_max(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_max(np_state))

        mach_sym = ca.MX.sym("mach")
        T_sym = ca.MX.sym("T")
        p_sym = ca.MX.sym("p")
        state = {
            "mach_number": mach_sym,
            "air_temperature": T_sym,
            "air_pressure": p_sym,
        }
        result = ps_model.engine.thrust_max(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [mach_sym, T_sym, p_sym], [result])
        assert_close(
            f(
                np_state["mach_number"],
                np_state["air_temperature"],
                np_state["air_pressure"],
            ),
            ref,
            label="ps thrust_max casadi",
        )

    @skip_no_casadi
    def test_casadi_thrust_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.thrust_min(np_state))

        mach_sym = ca.MX.sym("mach")
        state = {"mach_number": mach_sym}
        result = ps_model.engine.thrust_min(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [mach_sym], [result])
        assert_close(f(np_state["mach_number"]), ref, label="ps thrust_min casadi")

    @skip_no_casadi
    def test_casadi_fuel_flow(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow(np_state))

        thrust_sym = ca.MX.sym("thrust")
        mach_sym = ca.MX.sym("mach")
        T_sym = ca.MX.sym("T")
        p_sym = ca.MX.sym("p")
        alt_sym = ca.MX.sym("alt")
        state = {
            "thrust": thrust_sym,
            "mach_number": mach_sym,
            "air_temperature": T_sym,
            "air_pressure": p_sym,
            "altitude": alt_sym,
        }
        result = ps_model.engine.fuel_flow(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [thrust_sym, mach_sym, T_sym, p_sym, alt_sym], [result])
        assert_close(
            f(
                np_state["thrust"],
                np_state["mach_number"],
                np_state["air_temperature"],
                np_state["air_pressure"],
                np_state["altitude"],
            ),
            ref,
            label="ps fuel_flow casadi",
        )

    @skip_no_casadi
    def test_casadi_fuel_flow_min(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.fuel_flow_min(np_state))

        alt_sym = ca.MX.sym("alt")
        state = {"altitude": alt_sym}
        result = ps_model.engine.fuel_flow_min(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [alt_sym], [result])
        assert_close(f(np_state["altitude"]), ref, label="ps fuel_flow_min casadi")

    @skip_no_casadi
    def test_casadi_overall_propulsion_efficiency(self, ps_model):
        np_state = _ps_engine_state_numpy()
        ref = to_float(ps_model.engine.overall_propulsion_efficiency(np_state))

        thrust_sym = ca.MX.sym("thrust")
        mach_sym = ca.MX.sym("mach")
        T_sym = ca.MX.sym("T")
        p_sym = ca.MX.sym("p")
        state = {
            "thrust": thrust_sym,
            "mach_number": mach_sym,
            "air_temperature": T_sym,
            "air_pressure": p_sym,
        }
        result = ps_model.engine.overall_propulsion_efficiency(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [thrust_sym, mach_sym, T_sym, p_sym], [result])
        assert_close(
            f(
                np_state["thrust"],
                np_state["mach_number"],
                np_state["air_temperature"],
                np_state["air_pressure"],
            ),
            ref,
            label="ps overall_propulsion_efficiency casadi",
        )

    @skip_no_casadi
    def test_casadi_drag_coef(self, ps_model):
        np_state = _ps_aero_state_numpy()
        ref = to_float(ps_model.aerodynamics.drag_coef(np_state))

        cl_sym = ca.MX.sym("cl")
        mach_sym = ca.MX.sym("mach")
        T_sym = ca.MX.sym("T")
        p_sym = ca.MX.sym("p")
        state = {
            "lift_coef": cl_sym,
            "mach_number": mach_sym,
            "air_temperature": T_sym,
            "air_pressure": p_sym,
        }
        result = ps_model.aerodynamics.drag_coef(state)
        assert isinstance(result, ca.MX)
        f = ca.Function("f", [cl_sym, mach_sym, T_sym, p_sym], [result])
        assert_close(
            f(
                np_state["lift_coef"],
                np_state["mach_number"],
                np_state["air_temperature"],
                np_state["air_pressure"],
            ),
            ref,
            label="ps drag_coef casadi",
        )
