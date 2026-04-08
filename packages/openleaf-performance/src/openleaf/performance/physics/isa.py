"""
International Standard Atmosphere (ISA) implementation.

Defines atmospheric constants and functions for calculating
air properties at different altitudes.

The module provides:
- AtmosphericState class: efficient computation of all properties at an altitude
- Absolute values: air_temperature() [K], air_pressure() [Pa]
- Dimensionless ratios: theta() [T/T_SL], delta() [P/P_SL]
"""

from __future__ import annotations


__all__ = [
    "A_SL",
    "DELTA_TROPO",
    "GAMMA",
    "H_TROPO",
    "MU",
    "P_SL",
    "P_TROPO",
    "RHO_SL",
    "TAU_H",
    "TAU_SL",
    "TAU_TROPO",
    "THETA_TROPO",
    "AtmosphericState",
    "G",
    "R",
    "air_pressure",
    "air_temperature",
    "delta",
    "pressure_altitude",
    "theta",
]

from functools import cached_property

import openleaf.performance.numpy as np
from openleaf.performance.physics.units import GAMMA
from openleaf.performance.physics.units import R_AIR
from openleaf.performance.physics.units import G
from openleaf.performance.utils.types import ArrayScalarLike


# Re-export for backward compatibility
R = R_AIR  # Specific gas constant for air [J/(kg·K)]
TAU_H = 0.0065  # Temperature gradient [K/m]
TAU_SL = 288.15  # Sea level temperature [K]
TAU_TROPO = 216.65  # Tropopause temperature [K]
H_TROPO = 11000.0  # Tropopause altitude [m]
P_TROPO = 22632.0  # Tropopause pressure [Pa]
P_SL = 101325.0  # Sea level pressure [Pa]
RHO_SL = 1.225  # Sea level air density [kg/m³]
A_SL = 340.29  # Sea level speed of sound [m/s]
MU = (GAMMA - 1) / GAMMA  # Derived constant

# Pre-computed ratios - computed to ensure continuity at tropopause
THETA_TROPO = TAU_TROPO / TAU_SL  # Tropopause temperature ratio
# Compute DELTA_TROPO using the tropospheric formula for exact continuity
_theta_at_tropo = 1.0 - (TAU_H / TAU_SL) * H_TROPO
DELTA_TROPO = _theta_at_tropo ** (G / (TAU_H * R_AIR))  # Tropopause pressure ratio


# --- Dimensionless Ratios ---


def theta(h: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate dimensionless temperature ratio (theta = T/T_SL).

    Args:
        h: Geometric altitude in meters.

    Returns:
        Dimensionless temperature ratio theta = T/T_SL.
        At sea level: theta = 1.0
        At tropopause: theta ≈ 0.7519
    """
    return np.where(
        h < H_TROPO,
        1.0 - (TAU_H / TAU_SL) * h,
        THETA_TROPO,
    )


def delta(h: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate dimensionless pressure ratio (delta = P/P_SL).

    Args:
        h: Geometric altitude in meters.

    Returns:
        Dimensionless pressure ratio delta = P/P_SL.
        At sea level: delta = 1.0
        At tropopause: delta ≈ 0.2234
    """
    t = theta(h)
    return np.where(
        h < H_TROPO,
        np.power(t, G / (TAU_H * R)),
        DELTA_TROPO * np.exp(-G / (R * TAU_TROPO) * (h - H_TROPO)),
    )


# --- Absolute Values ---


def air_temperature(h: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate air temperature in Kelvin.

    Args:
        h: Geometric altitude in meters.

    Returns:
        Air temperature [K].
        At sea level: T = 288.15 K
        At tropopause: T = 216.65 K
    """
    return theta(h) * TAU_SL


def air_pressure(h: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate air pressure in Pascals.

    Args:
        h: Geometric altitude in meters.

    Returns:
        Air pressure [Pa].
        At sea level: P = 101325 Pa
        At tropopause: P ≈ 22632 Pa
    """
    return delta(h) * P_SL


# --- Inverse Functions ---


def pressure_altitude(p: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate pressure altitude from absolute pressure.

    Args:
        p: Absolute air pressure in Pascals.

    Returns:
        Pressure altitude in meters.
    """
    d = p / P_SL  # Convert to pressure ratio
    return np.where(
        d > DELTA_TROPO,
        (TAU_SL / TAU_H) * (1.0 - np.power(d, R * TAU_H / G)),
        H_TROPO - (R * TAU_TROPO / G) * np.log(d / DELTA_TROPO),
    )


# --- Atmospheric State Class ---


class AtmosphericState:
    """Efficient ISA atmospheric state at a given altitude.

    Computes all atmospheric properties from altitude, avoiding redundant
    calculations. All properties are lazily evaluated and cached.

    Example:
        >>> atm = AtmosphericState(10000)  # 10 km altitude
        >>> atm.temperature  # 223.15 K
        >>> atm.pressure  # 26436 Pa
        >>> atm.density  # 0.412 kg/m³
        >>> atm.theta  # 0.7744 (temperature ratio)
        >>> atm.delta  # 0.2609 (pressure ratio)
        >>> atm.sigma  # 0.3369 (density ratio)
    """

    def __init__(self, altitude: ArrayScalarLike):
        """Initialize atmospheric state at given altitude.

        Args:
            altitude: Geometric altitude in meters.
        """
        self.altitude = altitude
        self._in_troposphere = altitude < H_TROPO

    @cached_property
    def theta(self) -> ArrayScalarLike:
        """Temperature ratio (T / T_SL)."""
        return np.where(
            self._in_troposphere,
            1.0 - (TAU_H / TAU_SL) * self.altitude,
            THETA_TROPO,
        )

    @cached_property
    def delta(self) -> ArrayScalarLike:
        """Pressure ratio (P / P_SL)."""
        return np.where(
            self._in_troposphere,
            np.power(self.theta, G / (TAU_H * R)),
            DELTA_TROPO * np.exp(-G / (R * TAU_TROPO) * (self.altitude - H_TROPO)),
        )

    @cached_property
    def sigma(self) -> ArrayScalarLike:
        """Density ratio (rho / rho_SL)."""
        return self.delta / self.theta

    @cached_property
    def temperature(self) -> ArrayScalarLike:
        """Air temperature [K]."""
        return self.theta * TAU_SL

    @cached_property
    def pressure(self) -> ArrayScalarLike:
        """Air pressure [Pa]."""
        return self.delta * P_SL

    @cached_property
    def density(self) -> ArrayScalarLike:
        """Air density [kg/m³]."""
        return self.sigma * RHO_SL

    @cached_property
    def speed_of_sound(self) -> ArrayScalarLike:
        """Speed of sound [m/s]."""
        return A_SL * np.sqrt(self.theta)
