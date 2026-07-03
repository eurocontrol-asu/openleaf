# SPDX-License-Identifier: LGPL-3.0-only
#
# Derived from OpenAP's aero module (https://github.com/junzis/openap)
# Copyright (c) Junzi Sun / TU Delft - GNU LGPL v3 (see LICENSES/LGPL-3.0.txt).
# Modified by EUROCONTROL (2026) for multi-backend (NumPy/CasADi) dispatch; this
# file remains available under the LGPL-3.0.

"""OpenAP-compatible aero module using openleaf.performance.numpy.

This module provides the same API as `openap.aero` but uses `openleaf.performance.numpy`
for all mathematical operations. This enables seamless CasADi symbolic
compatibility while returning numpy arrays for numerical inputs.

The module is designed to be injected into OpenAP classes via:
    thrust = openap.Thrust('A320')
    thrust.aero = openap_aero

References:
    - OpenAP: https://github.com/junzis/openap
    - ISA: ICAO Standard Atmosphere (Doc 7488)
"""

from __future__ import annotations

import openleaf.performance.numpy as np
from openleaf.performance.physics import isa
from openleaf.performance.utils.types import ArrayScalarLike


# =============================================================================
# Constants (same as openap.aero)
# =============================================================================

# Unit conversions
kts = 0.514444  # knot -> m/s
ft = 0.3048  # ft -> m
fpm = 0.00508  # ft/min -> m/s
inch = 0.0254  # inch -> m
sqft = 0.09290304  # 1 square foot
nm = 1852.0  # nautical mile -> m
lbs = 0.453592  # pound -> kg

# Physical constants
g0 = 9.80665  # m/s², sea level gravity
R = 287.05287  # m²/(s² K), gas constant
p0 = 101325.0  # Pa, sea level pressure ISA
rho0 = 1.225  # kg/m³, sea level density ISA
T0 = 288.15  # K, sea level temperature ISA
gamma = 1.40  # cp/cv for air
gamma1 = 0.2  # (gamma-1)/2
gamma2 = 3.5  # gamma/(gamma-1)
beta = -0.0065  # K/m, ISA temp gradient below tropopause
r_earth = 6371000.0  # m, average earth radius
a0 = 340.293988  # m/s, sea level speed of sound ISA


# =============================================================================
# Atmospheric functions
# =============================================================================


def pressure(h: float, dT: float = 0) -> ArrayScalarLike:
    """Compute air pressure at a given altitude.

    Args:
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Air pressure in Pa.

    Note:
        OpenAP's original implementation uses a smoothed tropopause
        transition. We use the standard ISA model from
        openleaf.performance.physics.isa.
    """
    # Use OpenLeaf ISA which supports CasADi
    return isa.air_pressure(h)


def density(h: float, dT: float = 0) -> ArrayScalarLike:
    """Compute air density at a given altitude.

    Args:
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Air density in kg/m³.
    """
    p = pressure(h, dT)
    T = temperature(h, dT)
    return p / (R * T)


def temperature(h: float, dT: float = 0) -> ArrayScalarLike:
    """Compute air temperature at a given altitude.

    Args:
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Air temperature in K.
    """
    return isa.air_temperature(h) + dT


def vsound(h: float, dT: float = 0) -> ArrayScalarLike:
    """Compute speed of sound at a given altitude.

    Args:
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Speed of sound in m/s.
    """
    T = temperature(h, dT)
    return np.sqrt(gamma * R * T)


# =============================================================================
# Airspeed conversions
# =============================================================================


def tas2mach(v_tas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert true airspeed to Mach number.

    Args:
        v_tas: True airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Mach number.
    """
    a = vsound(h, dT)
    return v_tas / a


def mach2tas(mach: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert Mach number to true airspeed.

    Args:
        mach: Mach number.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        True airspeed in m/s.
    """
    a = vsound(h, dT)
    return mach * a


def tas2cas(v_tas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert true airspeed to calibrated airspeed.

    Uses the compressible flow equations for airspeed conversion.

    Args:
        v_tas: True airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Calibrated airspeed in m/s.
    """
    p = pressure(h, dT)
    rho = density(h, dT)

    # Impact pressure at altitude
    qdyn = p * (np.power(1.0 + rho * v_tas * v_tas / (7.0 * p), 3.5) - 1.0)

    # CAS from impact pressure at sea level
    v_cas = np.sqrt(7.0 * p0 / rho0 * (np.power(qdyn / p0 + 1.0, 2.0 / 7.0) - 1.0))
    return v_cas


def cas2tas(v_cas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert calibrated airspeed to true airspeed.

    Uses the compressible flow equations for airspeed conversion.

    Args:
        v_cas: Calibrated airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        True airspeed in m/s.
    """
    p = pressure(h, dT)
    rho = density(h, dT)

    # Impact pressure from CAS at sea level
    qdyn = p0 * (np.power(1.0 + rho0 * v_cas * v_cas / (7.0 * p0), 3.5) - 1.0)

    # TAS from impact pressure at altitude
    v_tas = np.sqrt(7.0 * p / rho * (np.power(1.0 + qdyn / p, 2.0 / 7.0) - 1.0))
    return v_tas


def mach2cas(mach: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert Mach number to calibrated airspeed.

    Args:
        mach: Mach number.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Calibrated airspeed in m/s.
    """
    v_tas = mach2tas(mach, h, dT)
    return tas2cas(v_tas, h, dT)


def cas2mach(v_cas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert calibrated airspeed to Mach number.

    Args:
        v_cas: Calibrated airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Mach number.
    """
    v_tas = cas2tas(v_cas, h, dT)
    return tas2mach(v_tas, h, dT)


def eas2tas(v_eas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert equivalent airspeed to true airspeed.

    Args:
        v_eas: Equivalent airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        True airspeed in m/s.
    """
    rho = density(h, dT)
    return v_eas * np.sqrt(rho0 / rho)


def tas2eas(v_tas: float, h: float, dT: float = 0) -> ArrayScalarLike:
    """Convert true airspeed to equivalent airspeed.

    Args:
        v_tas: True airspeed in m/s.
        h: Altitude in meters.
        dT: Temperature deviation from ISA in K. Defaults to 0.

    Returns:
        Equivalent airspeed in m/s.
    """
    rho = density(h, dT)
    return v_tas * np.sqrt(rho / rho0)
