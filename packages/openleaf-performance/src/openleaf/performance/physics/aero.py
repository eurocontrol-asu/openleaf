# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
Aerodynamic conversion and calculation functions.

This module provides functions for converting between different airspeed
definitions (TAS, CAS, Mach) and calculating aerodynamic properties.
"""

from __future__ import annotations


__all__ = [
    "AircraftAirspeed",
    "aerodynamic_coefficient",
    "aerodynamic_force",
    "crossover_altitude",
    "dynamic_pressure",
    "f_from_gradient",
    "thrust_required",
]

from functools import cached_property

import openleaf.performance.numpy as np
from openleaf.performance.physics import isa
from openleaf.performance.utils.types import ArrayScalarLike


# --- Fundamental Aerodynamic Equations ---


def dynamic_pressure(
    true_airspeed: ArrayScalarLike, air_density: ArrayScalarLike
) -> ArrayScalarLike:
    """Calculate dynamic pressure.

    Args:
        true_airspeed: True airspeed [m/s].
        air_density: Air density [kg/m³].

    Returns:
        Dynamic pressure [Pa].

    Note:
        Can also use sigma (density ratio) with: air_density = sigma * isa.RHO_SL
    """
    return 0.5 * air_density * true_airspeed**2


def aerodynamic_coefficient(
    force: ArrayScalarLike,
    q: ArrayScalarLike,
    surface_area: float,
) -> ArrayScalarLike:
    """Calculate aerodynamic coefficient.

    C = F / (q * S)

    Args:
        force: Aerodynamic force [N].
        q: Dynamic pressure [Pa].
        surface_area: Wing reference area [m²].

    Returns:
        Lift coefficient (dimensionless).
    """
    return force / (q * surface_area)


def aerodynamic_force(
    coefficient: ArrayScalarLike,
    q: ArrayScalarLike,
    surface_area: float,
) -> ArrayScalarLike:
    """Calculate aerodynamic force.

    F = C * q * S

    Args:
        C: Aerodynamic coefficient (dimensionless).
        q: Dynamic pressure [Pa].
        surface_area: Wing reference area [m²].

    Returns:
        Aerodynamic force [N].
    """
    return coefficient * q * surface_area


def f_from_gradient(true_airspeed: ArrayScalarLike, dv_dh: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate energy share factor f from speed gradient (generic formula).

    This is the fundamental f formula from the TEM derivation.
    For speed-schedule specific f (constant Mach, constant CAS), use
    the functions in openleaf.performance.physics.esf module instead.

    f = 1 / (1 + V/g · dV/dh)

    Args:
        true_airspeed: True airspeed [m/s].
        dv_dh: Speed change with altitude [1/s] (dV/dh).

    Returns:
        Energy share factor f (dimensionless).
        - f = 1: Constant speed (all energy to climb)
        - f < 1: Accelerating climb (dV/dh > 0)
        - f > 1: Decelerating climb (dV/dh < 0)

    See Also:
        openleaf.performance.physics.esf.f_constant_mach: f for constant Mach schedule
        openleaf.performance.physics.esf.f_constant_cas: f for constant CAS schedule
    """
    return 1.0 / (1.0 + true_airspeed / isa.G * dv_dh)


def thrust_required(
    drag: ArrayScalarLike,
    mass: ArrayScalarLike,
    true_airspeed: ArrayScalarLike,
    vertical_speed: ArrayScalarLike,
    f: ArrayScalarLike,
) -> ArrayScalarLike:
    """Calculate thrust required for a given flight condition.

    Simple TEM formula assuming GEOMETRIC altitude rate (dh_geo/dt).
    Does NOT include temperature correction for non-ISA conditions.

    For temperature-corrected computations using PRESSURE altitude rate
    (dHp/dt), use openleaf.performance.physics.tem.thrust_from_rocd instead.

    T = D + m · g · (dh/dt) / (V · f)

    Args:
        drag: Aerodynamic drag force [N].
        mass: Aircraft mass [kg].
        true_airspeed: True airspeed [m/s].
        vertical_speed: GEOMETRIC altitude rate [m/s] (positive = climbing).
        f: Energy share factor [-].

    Returns:
        Required thrust [N].

    See Also:
        openleaf.performance.physics.tem.thrust_from_rocd: Temperature-corrected version
            using pressure altitude rate.
    """
    return drag + mass * isa.G * vertical_speed / (true_airspeed * f)


# --- Speed Conversions ---


def crossover_altitude(cas: ArrayScalarLike, mach: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the altitude where CAS and Mach constraints intersect.

    At this altitude, flying at a given calibrated airspeed is equivalent
    to flying at a given Mach number.

    Args:
        cas: Calibrated airspeed in m/s.
        mach: Mach number (dimensionless).

    Returns:
        Pressure altitude in meters.
    """
    # Absolute pressure at the transition
    p_trans = isa.P_SL * (
        (
            np.power(
                1 + ((isa.GAMMA - 1.0) / 2.0) * ((cas / isa.A_SL) ** 2),
                np.power(isa.MU, -1),
            )
            - 1.0
        )
        / (np.power(1 + ((isa.GAMMA - 1.0) / 2.0) * (mach**2), np.power(isa.MU, -1)) - 1.0)
    )

    return isa.pressure_altitude(p_trans)


class AircraftAirspeed:
    """Unified CAS / TAS / Mach converter for a given atmospheric state.

    Converts between different airspeed representations (calibrated airspeed,
    true airspeed, Mach number) using the static air temperature and pressure.
    All conversions are lazily computed and cached.

    Example:
        >>> from openleaf.performance.physics.aero import AircraftAirspeed  # noqa: E501
        >>> state = AircraftAirspeed(T=288.15, p=101325, cas=150.0)
        >>> state.mach_number  # Convert CAS to Mach
        >>> state.true_airspeed  # Convert CAS to TAS
    """

    def __init__(
        self,
        T: ArrayScalarLike,
        p: ArrayScalarLike,
        v: ArrayScalarLike = None,
        cas: ArrayScalarLike = None,
        mach: ArrayScalarLike = None,
    ):
        """Initialize airspeed converter with atmospheric state.

        Args:
            T: Static air temperature [K].
            p: Static air pressure [Pa].
            v: True airspeed [m/s]. Provide exactly one of v, cas, or mach.
            cas: Calibrated airspeed [m/s]. Provide exactly one of v, cas, or mach.
            mach: Mach number [-]. Provide exactly one of v, cas, or mach.

        Raises:
            ValueError: If not exactly one of v, cas, mach is provided.
        """
        self.T = T  # static air temperature [K]
        self.p = p  # static air pressure [Pa]

        # Validate exactly one input provided
        inputs = [cas, v, mach]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError("Provide exactly one airspeed input")

        self._cas = cas
        self._tas = v
        self._mach = mach

        # Precompute commonly used values
        self._a = np.sqrt(isa.GAMMA * isa.R * T)  # speed of sound
        self._rho = p / (isa.R * T)

    @cached_property
    def mach_number(self) -> ArrayScalarLike:
        """Convert to Mach number."""
        if self._mach is not None:
            return self._mach
        if self._tas is not None:
            return self._tas / self._a
        mu = isa.MU
        qc_p0 = (1 + 0.5 * mu * self._cas**2 * isa.RHO_SL / isa.P_SL) ** (1 / mu) - 1
        # Mach from impact-pressure ratio: coefficient is 2/(gamma-1), NOT 2/mu.
        # mu = (gamma-1)/gamma; using 2/mu here overestimates Mach by sqrt(gamma).
        return np.sqrt((2.0 / (isa.GAMMA - 1.0)) * ((1 + qc_p0 * isa.P_SL / self.p) ** mu - 1))

    @cached_property
    def true_airspeed(self) -> ArrayScalarLike:
        """Convert to true airspeed."""
        return self.mach_number * self._a

    @cached_property
    def calibrated_airspeed(self) -> ArrayScalarLike:
        """Convert to calibrated airspeed."""
        mu = isa.MU
        # Impact-pressure ratio from Mach: coefficient is 0.5*(gamma-1), NOT 0.5*mu.
        # (The outer 2/mu * P_SL/RHO_SL term is correct: it carries a0^2 = gamma*P_SL/RHO_SL.)
        qc_p = (1 + 0.5 * (isa.GAMMA - 1.0) * self.mach_number**2) ** (1 / mu) - 1
        qc_p0 = qc_p * self.p / isa.P_SL
        return np.sqrt((2 / mu) * (isa.P_SL / isa.RHO_SL) * ((1 + qc_p0) ** mu - 1))
