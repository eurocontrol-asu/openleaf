# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Poll-Schumann aircraft limitations model.

Provides the PSLimits class implementing altitude-dependent Mach limits
based on impact pressure constraints and ATM speed restrictions, and a
mass- and temperature-dependent maximum altitude from the PS thrust,
drag and maximum usable lift models.

Reference: Poll & Schumann (2021), The Aeronautical Journal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.core.validation import requires
from openleaf.performance.models.ps.ps_physics import CEILING_CLIMB_RATE_FPM
from openleaf.performance.models.ps.ps_physics import max_mach_number_by_altitude
from openleaf.performance.models.ps.ps_physics import max_usable_lift_coefficient
from openleaf.performance.models.ps.ps_physics import required_thrust_coefficient
from openleaf.performance.models.ps.schemas import PSLimitsData
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


if TYPE_CHECKING:
    from openleaf.performance.models.ps.models import PSAeroModel
    from openleaf.performance.models.ps.models import PSEngineModel
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = ["PSLimits"]

CEILING_FLOOR_FT = 20_000.0
"""Lowest maximum altitude the PS limits report [ft].

PS is a cruise model, so both ceilings are bounded below by FL200. Below
about FL200 the excess thrust coefficient at a cruise Mach number falls again
with decreasing altitude (higher temperature lowers the maximum thrust
coefficient), so a thrust-ceiling bracket reaching lower would admit a
second, non-physical root. The clean-wing lift limit falls below FL200 at
the low Mach numbers of climb and approach, where it would report spurious
ceiling violations for a regime that the underspeed limit covers.
"""

CEILING_BISECTION_STEPS = 12
"""Fixed number of bisection steps on the ceiling bracket.

Twelve halvings resolve the bracket ``[CEILING_FLOOR_FT, hmo]`` to
under 10 ft for any hmo below 60,000 ft, before the final secant step.
"""


class PSLimits(LimitsModel[PSLimitsData]):
    """Poll-Schuman aircraft operational limits.

    Provides Mach number limits based on altitude, impact pressure constraints,
    and ATM speed restrictions, and a maximum altitude that depends on mass and
    temperature through the PS engine and aerodynamic models. Compatible with
    the AircraftLimits interface.
    """

    def __init__(
        self,
        data: PSLimitsData,
        *,
        engine: PSEngineModel,
        aerodynamics: PSAeroModel,
        ceiling_climb_rate: float = CEILING_CLIMB_RATE_FPM,
    ):
        """Initialize PS limits.

        Args:
            data: PS limits data container.
            engine: PS engine model (maximum thrust coefficient, design Mach).
            aerodynamics: PS aerodynamic model (drag, wing area, design
                optimum lift coefficient).
            ceiling_climb_rate: Residual climb rate at maximum thrust that
                defines the thrust-limited ceiling [ft/min].
        """
        super().__init__(data)
        self._engine = engine
        self._aerodynamics = aerodynamics
        self._ceiling_climb_rate = ceiling_climb_rate

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum Mach number at given altitude.

        PS model provides altitude-dependent Mach limits based on impact
        pressure constraints and ATM speed restrictions.

        Args:
            data: Flight state dictionary with optional keys:
                - altitude: Altitude in meters (package-wide convention).
                  If not provided, returns MMO.
                - air_pressure: Ambient pressure [Pa]. If not provided,
                  computed from ISA.
                - atm_speed_limit: Whether to apply ATM 250kt limit below
                  FL100 (default True).

        Returns:
            Maximum allowable Mach number.
        """
        # PS model always has mmo defined
        mmo = self.data.mmo
        if mmo is None:
            return np.inf

        altitude_m = data.get("altitude")
        if altitude_m is None:
            return mmo

        # "altitude" is meters everywhere in the package; the PS physics
        # helper (and its 10,000 ft ATM-limit threshold) works in feet.
        altitude_ft = altitude_m * units.M_TO_FT

        air_pressure = data.get("air_pressure")
        if air_pressure is None:
            air_pressure = isa.air_pressure(altitude_m)

        atm_speed_limit = bool(data.get("atm_speed_limit", True))

        return max_mach_number_by_altitude(
            altitude_ft,
            air_pressure,
            mmo,
            self.data.p_i_max,
            self.data.p_inf_co,
            atm_speed_limit=atm_speed_limit,
            buffer=0.02,
        )

    @requires("aircraft_mass", "delta_isa")
    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum operating altitude considering mass and temperature.

        Returns ``min(hmo, h_thrust, h_lift)`` where

        - ``h_thrust`` is the thrust-limited ceiling: the pressure altitude at
          which the maximum available thrust coefficient equals the thrust
          coefficient required for level flight plus a residual climb of
          ``ceiling_climb_rate`` (300 ft/min by default, the criterion of
          pycontrails' ``required_thrust_coefficient``),
        - ``h_lift`` is the lift-limited ceiling: the pressure altitude at
          which the lift coefficient reaches the maximum usable lift
          coefficient.

        Both are evaluated at the Mach number in ``data`` when present, and at
        the design Mach number otherwise, and both are bounded below by
        ``CEILING_FLOOR_FT`` (FL200): PS is a cruise model, so states below
        FL200 (including low-Mach climb, approach and ground points) never
        report a ceiling violation. The result is differentiable with respect
        to mass, temperature deviation and Mach number between FL200 and hmo
        (see ``_thrust_limited_ceiling``).

        Args:
            data: Flight state with:
                - aircraft_mass: Aircraft mass [kg] (required)
                - delta_isa: Temperature deviation from ISA [K] (required)
                - mach_number: Mach number (optional, default: design Mach)

        Returns:
            Maximum altitude in feet (pressure altitude).

        Raises:
            MissingFieldError: If any required field is missing.
        """
        mach = data.get("mach_number")
        if mach is None:
            mach = self._engine.data.m_des

        return self._max_altitude_impl(
            mass=data["aircraft_mass"],
            delta_isa=data["delta_isa"],
            mach=mach,
        )

    def _max_altitude_impl(
        self,
        mass: ArrayScalarLike,
        delta_isa: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate maximum altitude from typed inputs.

        Args:
            mass: Aircraft mass [kg].
            delta_isa: Temperature deviation from ISA [K].
            mach: Mach number.

        Returns:
            Maximum altitude in feet.
        """
        h_thrust = self._thrust_limited_ceiling(mass, delta_isa, mach)
        h_lift = self._lift_limited_ceiling(mass, mach)
        return np.minimum(np.minimum(h_thrust, h_lift), self.data.hmo)

    def _excess_thrust_coefficient(
        self,
        altitude_ft: ArrayScalarLike,
        mass: ArrayScalarLike,
        delta_isa: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Maximum minus required thrust coefficient at a pressure altitude.

        Positive below the thrust-limited ceiling, negative above it.

        Args:
            altitude_ft: Pressure altitude [ft].
            mass: Aircraft mass [kg].
            delta_isa: Temperature deviation from ISA [K].
            mach: Mach number.

        Returns:
            Excess thrust coefficient.
        """
        altitude_m = altitude_ft * units.FT
        air_pressure = isa.air_pressure(altitude_m)
        air_temperature = isa.air_temperature(altitude_m) + delta_isa

        # Level flight, L = m g; mass on the left keeps torch/JAX tensor types
        c_lift = (mass * units.G) / (
            0.5 * units.GAMMA * air_pressure * mach**2 * self._aerodynamics.data.surface_area
        )
        c_drag = self._aerodynamics.drag_coef_with_state(
            cl=c_lift,
            mach=mach,
            air_temperature=air_temperature,
            air_pressure=air_pressure,
        )
        true_airspeed = mach * np.sqrt(units.GAMMA * units.R_AIR * air_temperature)

        c_t_required = required_thrust_coefficient(
            c_lift, c_drag, true_airspeed, rocd=self._ceiling_climb_rate
        )
        c_t_max = self._engine.max_thrust_coefficient(mach, air_temperature)
        return c_t_max - c_t_required

    def _thrust_limited_ceiling(
        self,
        mass: ArrayScalarLike,
        delta_isa: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Solve for the pressure altitude where the excess thrust is zero.

        Fixed-step bisection on ``[CEILING_FLOOR_FT, hmo]``, so the
        computation is vectorised and free of data-dependent control flow on
        every backend (CasADi turns each step into ``if_else`` nodes). The
        bisection result is piecewise constant in the inputs; one secant
        (Newton) step on the final bracket makes the root a smooth function
        of mass, temperature deviation and Mach number, so automatic
        differentiation returns the implicit-function derivative.

        If the margin holds at hmo, hmo is returned. If it fails at the
        search floor, the floor is returned: the thrust-limited ceiling then
        lies below the cruise range the PS model covers.

        Args:
            mass: Aircraft mass [kg].
            delta_isa: Temperature deviation from ISA [K].
            mach: Mach number.

        Returns:
            Thrust-limited ceiling in feet, clipped to the search bracket.
        """

        def excess(altitude_ft: ArrayScalarLike) -> ArrayScalarLike:
            return self._excess_thrust_coefficient(altitude_ft, mass, delta_isa, mach)

        excess_lo = excess(CEILING_FLOOR_FT)
        excess_hi = excess(self.data.hmo)
        # Bracket arrays take the backend, dtype and shape of the state, so
        # that np.where never promotes Python floats to a narrower dtype.
        lo = np.full_like(excess_lo, CEILING_FLOOR_FT)
        hi = np.full_like(excess_hi, self.data.hmo)

        for _ in range(CEILING_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            excess_mid = excess(mid)
            below = excess_mid >= 0.0
            lo = np.where(below, mid, lo)
            excess_lo = np.where(below, excess_mid, excess_lo)
            hi = np.where(below, hi, mid)
            excess_hi = np.where(below, excess_hi, excess_mid)

        # Secant step inside a bracket that straddles the root. The guarded
        # denominator keeps the unselected branch finite, so gradients
        # through np.where stay finite on JAX and PyTorch.
        straddles = np.logical_and(excess_lo >= 0.0, excess_hi < 0.0)
        slope = np.where(straddles, excess_lo - excess_hi, 1.0)
        root = lo + (hi - lo) * excess_lo / slope

        return np.where(straddles, root, np.where(excess_hi >= 0.0, hi, lo))

    def _lift_limited_ceiling(
        self,
        mass: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Pressure altitude at which lift reaches the maximum usable lift.

        Closed form: ``p_min = m g / (0.5 gamma M**2 S c_l_maxu(M))``.

        Args:
            mass: Aircraft mass [kg].
            mach: Mach number.

        Returns:
            Lift-limited ceiling in feet, bounded below by ``CEILING_FLOOR_FT``.
        """
        aero = self._aerodynamics.data
        c_l_maxu = max_usable_lift_coefficient(mach, self._engine.data.m_des, aero.c_l_do)
        p_min = (mass * units.G) / (0.5 * units.GAMMA * mach**2 * aero.surface_area * c_l_maxu)
        return np.maximum(isa.pressure_altitude(p_min) * units.M_TO_FT, CEILING_FLOOR_FT)
