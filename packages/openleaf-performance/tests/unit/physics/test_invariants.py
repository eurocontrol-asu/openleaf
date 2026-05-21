"""Property-based invariants for the physics layer.

These tests assert structural properties that must hold for ANY valid input
in the documented operating envelope, rather than fixed numeric values at
specific points (which already exist in ``tests/test_physics.py``).

Properties encode the physics, not the implementation. If a function is
later rewritten (Phase 3's typing pass, Phase 4's refactor), these tests
catch behavioural drift even when point-value tests still pass.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from openleaf.performance.physics import aero
from openleaf.performance.physics import esf
from openleaf.performance.physics import isa


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Hypothesis settings: keep examples bounded so the suite stays fast.
FAST_SETTINGS = settings(max_examples=50, deadline=None)

# Tropospheric altitudes (where temperature lapse applies linearly).
tropospheric_altitudes = st.floats(min_value=0.0, max_value=isa.H_TROPO - 1.0)

# Full operating envelope for civil aviation.
flight_altitudes = st.floats(min_value=0.0, max_value=20_000.0)

# Pairs of altitudes guaranteed to be strictly increasing.
@st.composite
def altitude_pair(draw, max_alt: float = 20_000.0) -> tuple[float, float]:
    a = draw(st.floats(min_value=0.0, max_value=max_alt - 100.0))
    b = draw(st.floats(min_value=a + 1.0, max_value=max_alt))
    return a, b


# Cruise-regime Mach numbers.
mach_numbers = st.floats(min_value=0.1, max_value=0.95)

# Strictly positive speeds and densities.
positive_speed = st.floats(min_value=0.1, max_value=300.0)
positive_density = st.floats(min_value=0.05, max_value=1.5)

# Aerodynamic coefficients in a sane range.
coefficients = st.floats(min_value=-2.0, max_value=2.0)

# Wing surface areas in a sane range.
surface_areas = st.floats(min_value=1.0, max_value=500.0)


# ---------------------------------------------------------------------------
# ISA invariants
# ---------------------------------------------------------------------------


class TestISAInvariants:
    """Properties of the International Standard Atmosphere."""

    @FAST_SETTINGS
    @given(altitude_pair())
    def test_pressure_monotonically_decreasing(self, pair: tuple[float, float]) -> None:
        """Pressure must strictly decrease with altitude across the envelope."""
        lo, hi = pair
        assert isa.air_pressure(hi) < isa.air_pressure(lo)

    @FAST_SETTINGS
    @given(altitude_pair(max_alt=isa.H_TROPO))
    def test_temperature_strictly_decreasing_below_tropopause(
        self, pair: tuple[float, float]
    ) -> None:
        """Temperature strictly decreases up to the tropopause."""
        lo, hi = pair
        assert isa.air_temperature(hi) < isa.air_temperature(lo)

    @FAST_SETTINGS
    @given(altitude_pair(max_alt=20_000.0))
    def test_temperature_never_increases(self, pair: tuple[float, float]) -> None:
        """Above the tropopause temperature is constant; below it decreases.

        Either way, going up never gives a hotter atmosphere in the ISA
        standard model used here.
        """
        lo, hi = pair
        assert isa.air_temperature(hi) <= isa.air_temperature(lo) + 1e-9

    @FAST_SETTINGS
    @given(tropospheric_altitudes)
    def test_temperature_linear_in_troposphere(self, h: float) -> None:
        """In the troposphere, T(h) == TAU_SL - TAU_H * h."""
        expected = isa.TAU_SL - isa.TAU_H * h
        assert np.isclose(isa.air_temperature(h), expected, rtol=1e-9, atol=1e-9)

    @FAST_SETTINGS
    @given(flight_altitudes)
    def test_ratios_consistent_with_absolute(self, h: float) -> None:
        """theta(h) * T_SL == T(h) and delta(h) * P_SL == P(h)."""
        assert np.isclose(isa.theta(h) * isa.TAU_SL, isa.air_temperature(h), rtol=1e-9)
        assert np.isclose(isa.delta(h) * isa.P_SL, isa.air_pressure(h), rtol=1e-9)

    @FAST_SETTINGS
    @given(flight_altitudes)
    def test_pressure_altitude_inverts_air_pressure(self, h: float) -> None:
        """pressure_altitude is the inverse of air_pressure within the envelope."""
        recovered = isa.pressure_altitude(isa.air_pressure(h))
        # Tropopause has a piecewise-defined inverse; allow a few metres of slop.
        assert np.isclose(recovered, h, rtol=1e-6, atol=1.0)


# ---------------------------------------------------------------------------
# Aerodynamics invariants
# ---------------------------------------------------------------------------


class TestAeroInvariants:
    """Algebraic identities of the aerodynamic helpers."""

    @FAST_SETTINGS
    @given(positive_speed, positive_density)
    def test_dynamic_pressure_non_negative(
        self, v: float, rho: float
    ) -> None:
        """q = 0.5 * rho * v^2 is non-negative for non-negative inputs."""
        assert aero.dynamic_pressure(v, rho) >= 0.0

    @FAST_SETTINGS
    @given(coefficients, positive_speed, positive_density, surface_areas)
    def test_force_and_coefficient_round_trip(
        self,
        cd: float,
        v: float,
        rho: float,
        s: float,
    ) -> None:
        """aerodynamic_coefficient is the left-inverse of aerodynamic_force.

        For a valid (q, S), starting from a coefficient, computing the
        force, and recovering the coefficient must yield the original.
        """
        q = aero.dynamic_pressure(v, rho)
        force = aero.aerodynamic_force(cd, q, s)
        recovered = aero.aerodynamic_coefficient(force, q, s)
        assert np.isclose(recovered, cd, rtol=1e-9, atol=1e-12)

    @FAST_SETTINGS
    @given(coefficients, positive_speed, positive_density, surface_areas)
    def test_aerodynamic_force_matches_definition(
        self,
        cd: float,
        v: float,
        rho: float,
        s: float,
    ) -> None:
        """F == C * q * S, the textbook definition."""
        q = aero.dynamic_pressure(v, rho)
        assert np.isclose(aero.aerodynamic_force(cd, q, s), cd * q * s, rtol=1e-12)


# ---------------------------------------------------------------------------
# Energy share factor invariants
# ---------------------------------------------------------------------------


class TestESFInvariants:
    """Properties of the BADA energy-share-factor formulas.

    Important: ESF is NOT bounded above by 1. The documented domain is
    [0, +inf): constant-Mach climbs in the troposphere give f > 1
    (commonly ~1.05); the fixed deceleration constants give 1.7. The
    properties below only assert non-negativity and known structural
    facts (constant-Mach above the tropopause is exactly 1), which is
    what physical reality and the BADA manual actually guarantee.
    """

    @FAST_SETTINGS
    @given(mach_numbers, flight_altitudes)
    def test_constant_mach_is_non_negative(
        self, mach: float, alt: float
    ) -> None:
        """f_constant_mach >= 0 across the operating envelope."""
        assert float(esf.f_constant_mach(mach, alt)) >= 0.0

    @FAST_SETTINGS
    @given(mach_numbers, flight_altitudes)
    def test_constant_cas_is_non_negative(
        self, mach: float, alt: float
    ) -> None:
        """f_constant_cas >= 0 across the operating envelope."""
        assert float(esf.f_constant_cas(mach, alt)) >= 0.0

    @FAST_SETTINGS
    @given(mach_numbers, st.floats(min_value=isa.H_TROPO + 100.0, max_value=20_000.0))
    def test_constant_mach_above_tropopause_is_one(
        self, mach: float, alt: float
    ) -> None:
        """Above the tropopause at constant Mach, f = 1 exactly (BADA 3.1-6)."""
        assert np.isclose(float(esf.f_constant_mach(mach, alt)), 1.0, rtol=1e-12)

    @FAST_SETTINGS
    @given(
        mach_numbers,
        st.floats(min_value=0.0, max_value=isa.H_TROPO - 1.0),
    )
    def test_constant_mach_below_tropopause_at_least_one(
        self, mach: float, alt: float
    ) -> None:
        """Below the tropopause at constant Mach, f >= 1.

        TAS decreases as the air cools, so some climb thrust becomes
        kinetic→potential transfer; the result is f >= 1.
        """
        assert float(esf.f_constant_mach(mach, alt)) >= 1.0 - 1e-9

    def test_fixed_constants_are_non_negative(self) -> None:
        """The four documented fixed f values are all non-negative."""
        assert esf.F_ACCEL_CLIMB >= 0.0
        assert esf.F_DECEL_CLIMB >= 0.0
        assert esf.F_ACCEL_DESCENT >= 0.0
        assert esf.F_DECEL_DESCENT >= 0.0
