"""Cross-backend tests for flight envelope limits across all models.

Verifies that min_cas, max_cas, max_mach, and max_altitude work correctly
across NumPy, PyTorch, JAX, and CasADi for BADA3, BADA4, PS, and OpenAP,
and across all relevant aerodynamic configurations.

Backend matrix:
    - NumPy: baseline reference values
    - PyTorch: tensor dispatch and numerical parity
    - JAX: Array API dispatch and numerical parity
    - CasADi: symbolic computation and evaluate-to-numeric parity
"""

from __future__ import annotations

import math

import pytest

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.numpy.backends import Backend
from openleaf.performance.physics import isa


# ---------------------------------------------------------------------------
# Backend availability
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


def to_float(x) -> float:
    if has_torch and isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    if has_casadi and isinstance(x, (ca.DM, ca.MX, ca.SX)):
        return float(x)
    if has_jax and isinstance(x, jax.Array):
        return float(x)
    return float(x)


def assert_close(actual, expected, rtol=1e-5, atol=1e-6, label=""):
    a, e = to_float(actual), to_float(expected)
    if math.isnan(a) and math.isnan(e):
        return
    assert abs(a - e) <= atol + rtol * abs(e), f"{label}: {a} != {e} (rtol={rtol}, atol={atol})"


def assert_positive(x, label=""):
    v = to_float(x)
    assert v > 0, f"{label}: expected positive, got {v}"


def assert_finite(x, label=""):
    v = to_float(x)
    assert math.isfinite(v), f"{label}: expected finite, got {v}"


# ---------------------------------------------------------------------------
# Shared flight conditions
# ---------------------------------------------------------------------------

# Cruise condition: FL350, Mach 0.78, 65 t, ISA
ALT_FT = 35_000.0  # feet
ALT_M = ALT_FT * 0.3048  # meters
MACH = 0.78
MASS_KG = 65_000.0
DELTA_ISA = 0.0
AIR_PRESSURE = float(isa.air_pressure(ALT_M))

ALL_CONFIGS = [
    AerodynamicConfig.CR,
    AerodynamicConfig.IC,
    AerodynamicConfig.TO,
    AerodynamicConfig.AP,
    AerodynamicConfig.LD,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ps_model():
    pytest.importorskip("pycontrails")
    from openleaf.performance.models.ps import PSModel

    return PSModel("B738")


@pytest.fixture
def openap_model():
    pytest.importorskip("openap")
    from openleaf.performance.models.openap import OpenAPModel

    return OpenAPModel("A320")


# ---------------------------------------------------------------------------
# BADA3 limits — min_cas and max_altitude across all backends and configs
# ---------------------------------------------------------------------------


class TestBADA3Limits:
    """BADA3 flight envelope limits across all backends."""

    # Reference values computed with NumPy, used to verify other backends.
    @pytest.fixture(autouse=True)
    def _ref(self, bada3_model):
        self.model = bada3_model

    def _min_cas_state(self, config, mass=MASS_KG):
        return {"aerodynamic_configuration": config, "aircraft_mass": mass}

    def _max_alt_state(self, mass=MASS_KG, delta_isa=DELTA_ISA):
        return {"aircraft_mass": mass, "delta_isa": delta_isa}

    # --- NumPy reference ---

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_numpy_min_cas(self, config):
        result = self.model.limits.min_cas(self._min_cas_state(config))
        # Stall speed should be positive (or 0 if config not defined for this aircraft)
        assert to_float(result) >= 0, f"min_cas[{config}] = {result}"
        if config == AerodynamicConfig.CR:
            assert_positive(result, "min_cas[CR]")

    def test_numpy_max_altitude(self):
        result = self.model.limits.max_altitude(self._max_alt_state())
        assert_positive(result, "max_altitude")
        assert to_float(result) <= self.model.limits.hmo + 1

    def test_numpy_max_altitude_mass_dependence(self):
        h_heavy = to_float(self.model.limits.max_altitude(self._max_alt_state(mass=70_000)))
        h_light = to_float(self.model.limits.max_altitude(self._max_alt_state(mass=50_000)))
        # Lighter aircraft can fly higher (or at same ceiling)
        assert h_light >= h_heavy - 1

    def test_numpy_max_altitude_isa_dependence(self):
        h_isa = to_float(self.model.limits.max_altitude(self._max_alt_state(delta_isa=0.0)))
        h_hot = to_float(self.model.limits.max_altitude(self._max_alt_state(delta_isa=15.0)))
        # Hotter day lowers ceiling
        assert h_hot <= h_isa + 1

    # --- Torch ---

    @skip_no_torch
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_torch_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._min_cas_state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
        }
        result = self.model.limits.min_cas(state)
        assert isinstance(result, torch.Tensor), "expected torch.Tensor"
        assert_close(result, ref, label=f"torch min_cas[{config}]")

    @skip_no_torch
    def test_torch_max_altitude(self):
        ref = to_float(self.model.limits.max_altitude(self._max_alt_state()))
        state = {
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
            "delta_isa": torch.tensor(DELTA_ISA, dtype=torch.float64),
        }
        result = self.model.limits.max_altitude(state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="torch max_altitude")

    # --- JAX ---

    @skip_no_jax
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_jax_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._min_cas_state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": jnp.float64(MASS_KG),
        }
        result = self.model.limits.min_cas(state)
        assert isinstance(result, jax.Array), "expected jax.Array"
        assert_close(result, ref, label=f"jax min_cas[{config}]")

    @skip_no_jax
    def test_jax_max_altitude(self):
        ref = to_float(self.model.limits.max_altitude(self._max_alt_state()))
        state = {
            "aircraft_mass": jnp.float64(MASS_KG),
            "delta_isa": jnp.float64(DELTA_ISA),
        }
        result = self.model.limits.max_altitude(state)
        assert isinstance(result, jax.Array)
        assert_close(result, ref, label="jax max_altitude")

    # --- CasADi ---

    @skip_no_casadi
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_casadi_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._min_cas_state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": ca.DM(MASS_KG),
        }
        result = self.model.limits.min_cas(state)
        assert_close(result, ref, label=f"casadi min_cas[{config}]")

    @skip_no_casadi
    def test_casadi_max_altitude(self):
        ref = to_float(self.model.limits.max_altitude(self._max_alt_state()))
        state = {
            "aircraft_mass": ca.DM(MASS_KG),
            "delta_isa": ca.DM(DELTA_ISA),
        }
        result = self.model.limits.max_altitude(state)
        assert_close(result, ref, label="casadi max_altitude")

    @skip_no_casadi
    def test_casadi_max_altitude_symbolic(self):
        """Verify gradient flows through max_altitude."""
        m_sym = ca.MX.sym("m")
        disa_sym = ca.MX.sym("disa")
        state = {"aircraft_mass": m_sym, "delta_isa": disa_sym}
        result = self.model.limits.max_altitude(state)
        grad = ca.jacobian(result, m_sym)
        F = ca.Function("F", [m_sym, disa_sym], [grad])
        g = float(F(MASS_KG, DELTA_ISA))
        # Gradient w.r.t. mass should be non-positive (heavier → lower ceiling)
        assert g <= 0 + 1e-9, f"d(max_alt)/dm should be ≤ 0, got {g}"


# ---------------------------------------------------------------------------
# BADA4 limits — all limit methods, all configs, all backends
# ---------------------------------------------------------------------------


class TestBADA4Limits:
    """BADA4 flight envelope limits across all backends and configurations."""

    @pytest.fixture(autouse=True)
    def _ref(self, bada4_model):
        self.model = bada4_model

    def _state(self, config=AerodynamicConfig.CR, mass=MASS_KG, mach=MACH):
        return {
            "aerodynamic_configuration": config,
            "aircraft_mass": mass,
            "mach_number": mach,
        }

    # --- NumPy reference ---

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_numpy_min_cas(self, config):
        result = to_float(self.model.limits.min_cas(self._state(config)))
        assert result > 0, f"min_cas[{config}] = {result}"

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_numpy_max_cas(self, config):
        result = to_float(self.model.limits.max_cas(self._state(config)))
        assert result > 0, f"max_cas[{config}] = {result}"

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_numpy_max_mach(self, config):
        result = to_float(self.model.limits.max_mach(self._state(config)))
        assert result > 0, f"max_mach[{config}] = {result}"

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_numpy_max_altitude(self, config):
        result = to_float(self.model.limits.max_altitude(self._state(config)))
        assert result > 0, f"max_altitude[{config}] = {result}"

    def test_numpy_hl_lowers_speed_limit(self):
        """Non-clean configs should have max_cas no greater than cruise.

        VFE may equal VMO for some aircraft (e.g. A320 BADA4), so we check ≤.
        """
        v_cr = to_float(self.model.limits.max_cas(self._state(AerodynamicConfig.CR)))
        v_ld = to_float(self.model.limits.max_cas(self._state(AerodynamicConfig.LD)))
        assert v_ld <= v_cr, f"LD max_cas ({v_ld}) should be ≤ CR max_cas ({v_cr})"

    def test_numpy_hl_lowers_altitude_limit(self):
        """High-lift configs should have lower max_altitude than cruise."""
        h_cr = to_float(self.model.limits.max_altitude(self._state(AerodynamicConfig.CR)))
        h_ap = to_float(self.model.limits.max_altitude(self._state(AerodynamicConfig.AP)))
        assert h_ap < h_cr, f"AP max_alt ({h_ap}) should be < CR max_alt ({h_cr})"

    def test_numpy_min_cas_mass_dependence(self):
        """Heavier aircraft has higher stall speed."""
        v_heavy = to_float(self.model.limits.min_cas(self._state(mass=70_000)))
        v_light = to_float(self.model.limits.min_cas(self._state(mass=50_000)))
        assert v_heavy > v_light

    # --- Torch ---

    @skip_no_torch
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_torch_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
            "mach_number": torch.tensor(MACH, dtype=torch.float64),
        }
        result = self.model.limits.min_cas(state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label=f"torch min_cas[{config}]")

    @skip_no_torch
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_torch_max_cas(self, config):
        # max_cas only uses aerodynamic_configuration (an enum), not tensor fields.
        # The test verifies the function accepts a state with torch tensors without
        # crashing, and that the numerical result is correct.
        ref = to_float(self.model.limits.max_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
            "mach_number": torch.tensor(MACH, dtype=torch.float64),
        }
        result = self.model.limits.max_cas(state)
        assert_finite(result, f"torch max_cas[{config}]")
        assert_close(result, ref, label=f"torch max_cas[{config}]")

    @skip_no_torch
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_torch_max_mach(self, config):
        # max_mach only uses aerodynamic_configuration (an enum), not tensor fields.
        ref = to_float(self.model.limits.max_mach(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
            "mach_number": torch.tensor(MACH, dtype=torch.float64),
        }
        result = self.model.limits.max_mach(state)
        assert_finite(result, f"torch max_mach[{config}]")
        assert_close(result, ref, label=f"torch max_mach[{config}]")

    @skip_no_torch
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_torch_max_altitude(self, config):
        # max_altitude only uses aerodynamic_configuration (an enum), not tensor fields.
        ref = to_float(self.model.limits.max_altitude(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
            "mach_number": torch.tensor(MACH, dtype=torch.float64),
        }
        result = self.model.limits.max_altitude(state)
        assert_finite(result, f"torch max_altitude[{config}]")
        assert_close(result, ref, label=f"torch max_altitude[{config}]")

    # --- JAX ---

    @skip_no_jax
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_jax_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": jnp.float64(MASS_KG),
            "mach_number": jnp.float64(MACH),
        }
        result = self.model.limits.min_cas(state)
        assert isinstance(result, jax.Array)
        assert_close(result, ref, label=f"jax min_cas[{config}]")

    @skip_no_jax
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_jax_max_cas(self, config):
        # max_cas only uses aerodynamic_configuration (an enum), not tensor fields.
        ref = to_float(self.model.limits.max_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": jnp.float64(MASS_KG),
            "mach_number": jnp.float64(MACH),
        }
        result = self.model.limits.max_cas(state)
        assert_finite(result, f"jax max_cas[{config}]")
        assert_close(result, ref, label=f"jax max_cas[{config}]")

    @skip_no_jax
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_jax_max_mach(self, config):
        # max_mach only uses aerodynamic_configuration (an enum), not tensor fields.
        ref = to_float(self.model.limits.max_mach(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": jnp.float64(MASS_KG),
            "mach_number": jnp.float64(MACH),
        }
        result = self.model.limits.max_mach(state)
        assert_finite(result, f"jax max_mach[{config}]")
        assert_close(result, ref, label=f"jax max_mach[{config}]")

    @skip_no_jax
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_jax_max_altitude(self, config):
        # max_altitude only uses aerodynamic_configuration (an enum), not tensor fields.
        ref = to_float(self.model.limits.max_altitude(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": jnp.float64(MASS_KG),
            "mach_number": jnp.float64(MACH),
        }
        result = self.model.limits.max_altitude(state)
        assert_finite(result, f"jax max_altitude[{config}]")
        assert_close(result, ref, label=f"jax max_altitude[{config}]")

    # --- CasADi ---

    @skip_no_casadi
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_casadi_min_cas(self, config):
        ref = to_float(self.model.limits.min_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": ca.DM(MASS_KG),
            "mach_number": ca.DM(MACH),
        }
        result = self.model.limits.min_cas(state)
        assert_close(result, ref, label=f"casadi min_cas[{config}]")

    @skip_no_casadi
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_casadi_max_cas(self, config):
        ref = to_float(self.model.limits.max_cas(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": ca.DM(MASS_KG),
            "mach_number": ca.DM(MACH),
        }
        result = self.model.limits.max_cas(state)
        assert_close(result, ref, label=f"casadi max_cas[{config}]")

    @skip_no_casadi
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_casadi_max_mach(self, config):
        ref = to_float(self.model.limits.max_mach(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": ca.DM(MASS_KG),
            "mach_number": ca.DM(MACH),
        }
        result = self.model.limits.max_mach(state)
        assert_close(result, ref, label=f"casadi max_mach[{config}]")

    @skip_no_casadi
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_casadi_max_altitude(self, config):
        ref = to_float(self.model.limits.max_altitude(self._state(config)))
        state = {
            "aerodynamic_configuration": config,
            "aircraft_mass": ca.DM(MASS_KG),
            "mach_number": ca.DM(MACH),
        }
        result = self.model.limits.max_altitude(state)
        assert_close(result, ref, label=f"casadi max_altitude[{config}]")

    @skip_no_casadi
    def test_casadi_min_cas_symbolic(self):
        """Verify gradient flows through min_cas w.r.t. mass."""
        m_sym = ca.MX.sym("m")
        state = {
            "aerodynamic_configuration": AerodynamicConfig.CR,
            "aircraft_mass": m_sym,
            "mach_number": ca.DM(MACH),
        }
        result = self.model.limits.min_cas(state)
        grad = ca.jacobian(result, m_sym)
        F = ca.Function("F", [m_sym], [grad])
        g = float(F(MASS_KG))
        # Heavier aircraft → higher stall speed → positive gradient
        assert g > 0, f"d(min_cas)/dm should be > 0, got {g}"


# ---------------------------------------------------------------------------
# PS limits — max_mach (altitude-dependent) and max_altitude
# ---------------------------------------------------------------------------


class TestPSLimits:
    """PS flight envelope limits across all backends."""

    @pytest.fixture(autouse=True)
    def _ref(self, ps_model):
        self.model = ps_model

    def _state(self, alt_ft=ALT_FT, air_pressure=AIR_PRESSURE):
        return {"altitude": alt_ft, "air_pressure": air_pressure}

    # --- NumPy reference ---

    def test_numpy_max_mach_cruise(self):
        result = to_float(self.model.limits.max_mach(self._state()))
        assert 0 < result <= self.model.limits.mmo + 0.1

    def test_numpy_max_mach_low_altitude(self):
        """Below FL100, ATM 250kt limit applies — max Mach should be lower."""
        alt_low = 5_000.0
        p_low = float(isa.air_pressure(alt_low * 0.3048))
        m_low = to_float(self.model.limits.max_mach({"altitude": alt_low, "air_pressure": p_low}))
        m_high = to_float(self.model.limits.max_mach(self._state()))
        assert m_low < m_high

    def test_numpy_max_altitude(self):
        result = to_float(self.model.limits.max_altitude({}))
        assert result == self.model.limits.hmo

    # --- Torch ---

    @skip_no_torch
    def test_torch_max_mach(self):
        ref = to_float(self.model.limits.max_mach(self._state()))
        state = {
            "altitude": torch.tensor(ALT_FT, dtype=torch.float64),
            "air_pressure": torch.tensor(AIR_PRESSURE, dtype=torch.float64),
        }
        result = self.model.limits.max_mach(state)
        assert isinstance(result, torch.Tensor)
        assert_close(result, ref, label="torch ps max_mach")

    @skip_no_torch
    def test_torch_max_mach_array(self):
        """Verify vectorized evaluation over altitude array."""
        alts = torch.tensor([5_000.0, 15_000.0, 35_000.0], dtype=torch.float64)
        pressures = torch.tensor(
            [float(isa.air_pressure(a * 0.3048)) for a in [5_000, 15_000, 35_000]],
            dtype=torch.float64,
        )
        result = self.model.limits.max_mach({"altitude": alts, "air_pressure": pressures})
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        # Low altitude should have lower max Mach than cruise
        assert float(result[0]) < float(result[2])

    # --- JAX ---

    @skip_no_jax
    def test_jax_max_mach(self):
        ref = to_float(self.model.limits.max_mach(self._state()))
        state = {
            "altitude": jnp.float64(ALT_FT),
            "air_pressure": jnp.float64(AIR_PRESSURE),
        }
        result = self.model.limits.max_mach(state)
        assert isinstance(result, jax.Array)
        assert_close(result, ref, label="jax ps max_mach")

    @skip_no_jax
    def test_jax_max_mach_array(self):
        alts = jnp.array([5_000.0, 15_000.0, 35_000.0])
        pressures = jnp.array(
            [float(isa.air_pressure(a * 0.3048)) for a in [5_000, 15_000, 35_000]]
        )
        result = self.model.limits.max_mach({"altitude": alts, "air_pressure": pressures})
        assert isinstance(result, jax.Array)
        assert result.shape == (3,)
        assert float(result[0]) < float(result[2])

    # --- CasADi ---

    @skip_no_casadi
    def test_casadi_max_mach(self):
        ref = to_float(self.model.limits.max_mach(self._state()))
        state = {
            "altitude": ca.DM(ALT_FT),
            "air_pressure": ca.DM(AIR_PRESSURE),
        }
        result = self.model.limits.max_mach(state)
        assert_close(result, ref, label="casadi ps max_mach")

    @skip_no_casadi
    def test_casadi_max_mach_symbolic(self):
        """Verify CasADi can differentiate max_mach w.r.t. altitude."""
        h_sym = ca.MX.sym("h")
        p_sym = ca.MX.sym("p")
        state = {"altitude": h_sym, "air_pressure": p_sym}
        result = self.model.limits.max_mach(state)
        grad = ca.jacobian(result, h_sym)
        F = ca.Function("F", [h_sym, p_sym], [grad])
        g = float(F(ALT_FT, AIR_PRESSURE))
        # At cruise altitude above crossover, max Mach is constant → gradient ≈ 0
        assert math.isfinite(g)


# ---------------------------------------------------------------------------
# OpenAP limits — constant limits, all backends
# ---------------------------------------------------------------------------


class TestOpenAPLimits:
    """OpenAP flight envelope limits across all backends."""

    @pytest.fixture(autouse=True)
    def _ref(self, openap_model):
        self.model = openap_model

    # OpenAP limits are constants — verify they return sensible values on all backends

    def test_numpy_vmo(self):
        assert self.model.limits.vmo > 0

    def test_numpy_mmo(self):
        assert self.model.limits.mmo > 0

    def test_numpy_hmo(self):
        assert self.model.limits.hmo > 0

    def test_numpy_max_cas(self):
        result = to_float(self.model.limits.max_cas({}))
        assert result == self.model.limits.vmo

    def test_numpy_max_mach(self):
        result = to_float(self.model.limits.max_mach({}))
        assert result == self.model.limits.mmo

    def test_numpy_max_altitude(self):
        result = to_float(self.model.limits.max_altitude({}))
        assert result == self.model.limits.hmo

    # OpenAP limits ignore inputs — verify they work when called with any backend values

    @skip_no_torch
    def test_torch_limits_accept_tensor_state(self):
        """OpenAP limits should work even when state contains tensors."""
        state = {
            "altitude": torch.tensor(ALT_FT, dtype=torch.float64),
            "aircraft_mass": torch.tensor(MASS_KG, dtype=torch.float64),
        }
        assert to_float(self.model.limits.max_cas(state)) == self.model.limits.vmo
        assert to_float(self.model.limits.max_mach(state)) == self.model.limits.mmo
        assert to_float(self.model.limits.max_altitude(state)) == self.model.limits.hmo

    @skip_no_jax
    def test_jax_limits_accept_array_state(self):
        state = {
            "altitude": jnp.float64(ALT_FT),
            "aircraft_mass": jnp.float64(MASS_KG),
        }
        assert to_float(self.model.limits.max_cas(state)) == self.model.limits.vmo
        assert to_float(self.model.limits.max_mach(state)) == self.model.limits.mmo
        assert to_float(self.model.limits.max_altitude(state)) == self.model.limits.hmo

    @skip_no_casadi
    def test_casadi_limits_accept_symbolic_state(self):
        state = {
            "altitude": ca.DM(ALT_FT),
            "aircraft_mass": ca.DM(MASS_KG),
        }
        assert to_float(self.model.limits.max_cas(state)) == self.model.limits.vmo
        assert to_float(self.model.limits.max_mach(state)) == self.model.limits.mmo
        assert to_float(self.model.limits.max_altitude(state)) == self.model.limits.hmo
