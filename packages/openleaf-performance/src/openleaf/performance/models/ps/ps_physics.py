# SPDX-License-Identifier: EUPL-1.2 AND Apache-2.0
# Copyright (c) 2026 EUROCONTROL
#
# Derived from pycontrails (https://github.com/contrailcirrus/pycontrails)
# Copyright (c) 2021-present Contrails.org and the Breakthrough Energy Foundation
# Licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt and
# LICENSES/NOTICE-pycontrails.txt). Modified by EUROCONTROL (2026): refactored into
# standalone functions and vectorized for multi-backend (NumPy/CasADi) dispatch.

"""Poll-Schumann (PS) physics and aerodynamics functions.

This module contains the physics-based functions specific to the Poll-Schumann
aircraft performance model for drag, propulsion efficiency, and thrust calculations.

Reference:
    Poll, D.I.A., and Schumann, U. (2021). "An estimation method for the fuel burn
    and other performance characteristics of civil transport aircraft in the cruise."
    The Aeronautical Journal, 125(1284), 257-295.
    https://doi.org/10.1017/aer.2020.62

Reference implementation:
    pycontrails (Apache-2.0), https://github.com/contrailcirrus/pycontrails
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.physics import units


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike


# =============================================================================
# Physical Constants (Poll & Schumann 2021)
# =============================================================================

# ATM speed limit threshold and impact pressure (corresponds to 250kt below FL100)
ATM_SPEED_LIMIT_ALTITUDE_FT = 10000.0
"""Altitude threshold for ATM 250kt speed limit [ft]."""

ATM_SPEED_LIMIT_IMPACT_PRESSURE_PA = 10510.0
"""Impact pressure corresponding to 250kt at sea level [Pa]."""

# Non-vortex lift-dependent drag factor coefficients (Eq. 7)
K1_BASE_COEFF = 0.8
"""Base coefficient for non-vortex lift-dependent drag factor."""

K1_SWEEP_COEFF = 0.53
"""Sweep correction coefficient for non-vortex drag."""

# Oswald efficiency factor coefficients (Eq. 6-8)
WINGLET_BONUS_FACTOR = 1.075
"""Efficiency bonus factor for aircraft with winglets."""

INDUCED_DRAG_BASE_FACTOR = 0.03
"""Base induced drag factor increment."""

# Wave drag coefficient (Eq. 9-11)
MCC_CL_CORRECTION_FACTOR = 0.10
"""Lift coefficient correction factor for critical Mach calculation."""


__all__ = [
    "airframe_drag_coefficient",
    "fluid_dynamic_viscosity",
    "low_speed_lift_dependent_drag_factor",
    "max_available_thrust_coefficient",
    "max_mach_number_by_altitude",
    "max_overall_propulsion_efficiency",
    "non_vortex_lift_dependent_drag_factor",
    "normalised_max_throttle_parameter",
    "oswald_efficiency_factor",
    "overall_propulsion_efficiency",
    "propulsion_efficiency_over_max",
    "reynolds_number",
    "skin_friction_coefficient",
    "thrust_coefficient_at_max_efficiency",
    "wave_drag_coefficient",
    "zero_lift_drag_coefficient",
]


# --- Operational Limits ---


def max_mach_number_by_altitude(
    altitude_ft: ArrayScalarLike,
    air_pressure: ArrayScalarLike,
    max_mach_num: float,
    p_i_max: float,
    p_inf_co: float,
    *,
    atm_speed_limit: bool = True,
    buffer: float = 0.02,
) -> ArrayScalarLike:
    """Calculate maximum Mach number considering altitude and pressure limits.

    Below 10,000 ft with ATM speed limit enabled, uses reduced impact pressure.
    Above crossover pressure altitude, returns aircraft max Mach.
    In between, computes Mach from impact pressure constraint.

    Args:
        altitude_ft: Altitude in feet.
        air_pressure: Ambient pressure [Pa].
        max_mach_num: Aircraft maximum operating Mach number.
        p_i_max: Maximum impact pressure [Pa].
        p_inf_co: Crossover pressure [Pa].
        atm_speed_limit: Whether to apply ATM 250kt limit below FL100.
        buffer: Safety buffer added to computed Mach (default 0.02).

    Returns:
        Maximum allowable Mach number.

    References:
        pycontrails ps_operational_limits.max_mach_number_by_altitude
    """
    # ATM speed limit of 250kt below FL100 corresponds to ~10510 Pa impact pressure
    if atm_speed_limit:
        p_i_max_eff = np.where(
            altitude_ft < ATM_SPEED_LIMIT_ALTITUDE_FT,
            ATM_SPEED_LIMIT_IMPACT_PRESSURE_PA,
            p_i_max,
        )
    else:
        p_i_max_eff = p_i_max

    # Below crossover pressure, compute Mach from impact pressure limit
    # Above crossover, use max_mach_num
    mach_from_pressure = (
        2.0**0.5 * ((1.0 + (2.0 / units.GAMMA) * (p_i_max_eff / air_pressure)) ** 0.5 - 1.0) ** 0.5
    )

    return (
        np.where(
            air_pressure > p_inf_co,
            mach_from_pressure,
            max_mach_num,
        )
        + buffer
    )


# --- Skin Friction and Viscosity ---


def fluid_dynamic_viscosity(air_temperature: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate fluid dynamic viscosity using Sutherland's Law.

    Args:
        air_temperature: Air temperature [K].

    Returns:
        Dynamic viscosity [Pa*s].

    References:
        Sutherland's Law for air viscosity.
    """
    return 1.458e-6 * np.power(air_temperature, 1.5) / (110.4 + air_temperature)


def reynolds_number(
    surface_area: float,
    mach_num: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    air_pressure: ArrayScalarLike,
) -> ArrayScalarLike:
    """Calculate Reynolds number.

    Uses wing reference area square root as characteristic length.

    Args:
        surface_area: Wing reference area [m**2].
        mach_num: Mach number.
        air_temperature: Air temperature [K].
        air_pressure: Air pressure [Pa].

    Returns:
        Reynolds number.

    References:
        Poll & Schumann (2021), Eq. 3.
    """
    mu = fluid_dynamic_viscosity(air_temperature)
    L_ref = np.sqrt(surface_area)
    inv_speed_of_sound_factor = np.sqrt(units.GAMMA / (units.R_AIR * air_temperature))
    return L_ref * mach_num * (air_pressure / mu) * inv_speed_of_sound_factor


def skin_friction_coefficient(rn: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate aircraft skin friction coefficient.

    Args:
        rn: Reynolds number.

    Returns:
        Skin friction coefficient.

    References:
        Poll & Schumann (2021), Eq. 4.
    """
    return 0.0269 / np.power(rn, 0.14)


# --- Drag Components ---


def zero_lift_drag_coefficient(c_f: ArrayScalarLike, psi_0: float) -> ArrayScalarLike:
    """Calculate zero-lift drag coefficient.

    Args:
        c_f: Skin friction coefficient.
        psi_0: Parasite drag area factor.

    Returns:
        Zero-lift drag coefficient.

    References:
        Poll & Schumann (2021), Eq. 5.
    """
    return c_f * psi_0


def non_vortex_lift_dependent_drag_factor(
    c_drag_0: ArrayScalarLike, cos_sweep: float
) -> ArrayScalarLike:
    """Calculate non-vortex lift-dependent drag factor.

    Args:
        c_drag_0: Zero-lift drag coefficient.
        cos_sweep: Cosine of wing sweep angle.

    Returns:
        Non-vortex lift-dependent drag factor.

    References:
        Poll & Schumann (2021), Eq. 7.
    """
    return K1_BASE_COEFF * (1.0 - K1_SWEEP_COEFF * cos_sweep) * c_drag_0


def oswald_efficiency_factor(
    c_drag_0: ArrayScalarLike,
    wing_aspect_ratio: float,
    cos_sweep: float,
    delta_2: float,
    winglets: bool,
) -> ArrayScalarLike:
    """Calculate Oswald efficiency factor.

    Args:
        c_drag_0: Zero-lift drag coefficient.
        wing_aspect_ratio: Wing aspect ratio.
        cos_sweep: Cosine of wing sweep angle.
        delta_2: Induced drag factor increment.
        winglets: Whether aircraft has winglets.

    Returns:
        Oswald efficiency factor.

    References:
        Poll & Schumann (2021), Eq. 6-8.
    """
    numer = WINGLET_BONUS_FACTOR if winglets else 1.0
    k_1 = non_vortex_lift_dependent_drag_factor(c_drag_0, cos_sweep)
    denom = 1.0 + INDUCED_DRAG_BASE_FACTOR + delta_2 + (k_1 * np.pi * wing_aspect_ratio)
    return numer / denom


def wave_drag_coefficient(
    mach_num: ArrayScalarLike,
    c_lift: ArrayScalarLike,
    cos_sweep: float,
    wing_constant: float,
    j_1: float,
    j_2: float,
    j_3: float,
    x_ref: float,
) -> ArrayScalarLike:
    """Calculate wave drag coefficient.

    Args:
        mach_num: Mach number.
        c_lift: Lift coefficient.
        cos_sweep: Cosine of wing sweep angle.
        wing_constant: Wing constant for wave drag.
        j_1: Wave drag coefficient 1.
        j_2: Wave drag coefficient 2.
        j_3: Wave drag coefficient 3.
        x_ref: Reference x for wave drag.

    Returns:
        Wave drag coefficient.

    References:
        Poll & Schumann (2021), Eq. 9-11.
    """
    m_cc = wing_constant - MCC_CL_CORRECTION_FACTOR * (c_lift / (cos_sweep**2))
    x = mach_num * cos_sweep / m_cc

    c_d_w_base = np.where(
        x < j_2,
        0.0,
        (cos_sweep**3) * j_1 * np.power(x - j_2, 2),
    )

    c_d_w = np.where(
        x < x_ref,
        c_d_w_base,
        c_d_w_base + j_3 * np.power(x - x_ref, 4),
    )

    return c_d_w


def low_speed_lift_dependent_drag_factor(
    e_ls: ArrayScalarLike, wing_aspect_ratio: float
) -> ArrayScalarLike:
    """Calculate low-speed lift-dependent drag factor.

    Args:
        e_ls: Oswald efficiency factor.
        wing_aspect_ratio: Wing aspect ratio.

    Returns:
        Low-speed lift-dependent drag factor.

    References:
        Poll & Schumann (2021), induced drag term.
    """
    return 1.0 / (np.pi * wing_aspect_ratio * e_ls)


def airframe_drag_coefficient(
    c_drag_0: ArrayScalarLike,
    c_drag_w: ArrayScalarLike,
    c_lift: ArrayScalarLike,
    e_ls: ArrayScalarLike,
    wing_aspect_ratio: float,
) -> ArrayScalarLike:
    """Calculate total airframe drag coefficient.

    Args:
        c_drag_0: Zero-lift drag coefficient.
        c_drag_w: Wave drag coefficient.
        c_lift: Lift coefficient.
        e_ls: Oswald efficiency factor.
        wing_aspect_ratio: Wing aspect ratio.

    Returns:
        Total drag coefficient.

    References:
        Poll & Schumann (2021), Eq. 2.
    """
    k = low_speed_lift_dependent_drag_factor(e_ls, wing_aspect_ratio)
    return c_drag_0 + (k * c_lift**2) + c_drag_w


# --- Propulsion Efficiency ---


def thrust_coefficient_at_max_efficiency(
    mach_num: ArrayScalarLike,
    m_des: float,
    c_t_des: float,
) -> ArrayScalarLike:
    """Calculate thrust coefficient at maximum overall propulsion efficiency.

    Args:
        mach_num: Mach number.
        m_des: Design Mach number.
        c_t_des: Design thrust coefficient.

    Returns:
        Thrust coefficient at maximum efficiency.

    References:
        Poll & Schumann (2021), Eq. 17.
    """
    m_over_m_des = mach_num / m_des
    h_2 = ((1.0 + 0.55 * mach_num) / (1.0 + 0.55 * m_des)) / (m_over_m_des**2)
    return h_2 * c_t_des


def max_overall_propulsion_efficiency(
    mach_num: ArrayScalarLike, mach_num_des: float, eta_1: float, eta_2: float
) -> ArrayScalarLike:
    """Calculate maximum overall propulsion efficiency.

    Args:
        mach_num: Mach number.
        mach_num_des: Design Mach number.
        eta_1: Propulsion efficiency coefficient 1.
        eta_2: Propulsion efficiency coefficient 2.

    Returns:
        Maximum propulsion efficiency.

    References:
        Poll & Schumann (2021), Eq. 15.
    """
    return eta_1 * np.power(mach_num, eta_2)


def propulsion_efficiency_over_max(
    mach_num: ArrayScalarLike,
    c_t: ArrayScalarLike,
    c_t_eta_b: ArrayScalarLike,
) -> ArrayScalarLike:
    """Calculate ratio of propulsion efficiency to maximum propulsion efficiency.

    Args:
        mach_num: Mach number.
        c_t: Thrust coefficient.
        c_t_eta_b: Thrust coefficient at max efficiency.

    Returns:
        Efficiency ratio.

    References:
        Poll & Schumann (2021), Eq. 16.
    """
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    sigma = np.where(mach_num < 0.4, 1.3 * (0.4 - mach_num), 0.0)

    eta_over_eta_b_low = (
        10.0 * (1.0 + 0.8 * (sigma - 0.43) - 0.6027 * sigma * 0.43) * c_t_over_c_t_eta_b
        + 33.3333
        * (-1.0 - 0.97 * (sigma - 0.43) + 0.8281 * sigma * 0.43)
        * (c_t_over_c_t_eta_b**2)
        + 37.037 * (1.0 + (sigma - 0.43) - 0.9163 * sigma * 0.43) * (c_t_over_c_t_eta_b**3)
    )

    eta_over_eta_b_hi = (
        (1.0 + (sigma - 0.43) - sigma * 0.43)
        + (4.0 * sigma * 0.43 - 2.0 * (sigma - 0.43)) * c_t_over_c_t_eta_b
        + ((sigma - 0.43) - 6 * sigma * 0.43) * (c_t_over_c_t_eta_b**2)
        + 4.0 * sigma * 0.43 * (c_t_over_c_t_eta_b**3)
        - sigma * 0.43 * (c_t_over_c_t_eta_b**4)
    )

    return np.where(c_t_over_c_t_eta_b < 0.3, eta_over_eta_b_low, eta_over_eta_b_hi)


def overall_propulsion_efficiency(
    mach_num: ArrayScalarLike,
    c_t: ArrayScalarLike,
    c_t_eta_b: ArrayScalarLike,
    m_des: float,
    eta_1: float,
    eta_2: float,
) -> ArrayScalarLike:
    """Calculate overall propulsion efficiency.

    Args:
        mach_num: Mach number.
        c_t: Thrust coefficient.
        c_t_eta_b: Thrust coefficient at max efficiency.
        m_des: Design Mach number.
        eta_1: Propulsion efficiency coefficient 1.
        eta_2: Propulsion efficiency coefficient 2.

    Returns:
        Overall propulsion efficiency.

    References:
        Poll & Schumann (2021), Eq. 14-16.
    """
    eta_over_eta_b = propulsion_efficiency_over_max(mach_num, c_t, c_t_eta_b)
    eta_b = max_overall_propulsion_efficiency(mach_num, m_des, eta_1, eta_2)
    return eta_over_eta_b * eta_b


# --- Thrust Limits ---


def normalised_max_throttle_parameter(
    air_temperature: ArrayScalarLike,
    mach_number: ArrayScalarLike,
    tet_mcc: float,
    tr_ec: float,
    m_ec: float,
    buffer: float = 0.20,
) -> ArrayScalarLike:
    """Calculate normalised maximum throttle parameter.

    Args:
        air_temperature: Air temperature [K].
        mach_number: Mach number.
        tet_mcc: Maximum combustor temperature [K].
        tr_ec: Engine temperature ratio at max efficiency.
        m_ec: Mach number at engine corner point.
        buffer: Safety buffer.

    Returns:
        Normalised maximum throttle parameter.

    References:
        Poll & Schumann (2021), Eq. 18-19.
    """
    return (
        (tet_mcc / air_temperature)
        / (tr_ec * (1.0 - 0.53 * (mach_number - m_ec) ** 2) * (1.0 + 0.2 * mach_number**2))
        * (1.0 + buffer)
    )


def max_available_thrust_coefficient(
    air_temperature: ArrayScalarLike,
    mach_number: ArrayScalarLike,
    c_t_eta_b: ArrayScalarLike,
    tet_mcc: float,
    tr_ec: float,
    m_ec: float,
    buffer: float = 0.20,
) -> ArrayScalarLike:
    """Calculate maximum available thrust coefficient.

    Args:
        air_temperature: Air temperature [K].
        mach_number: Mach number.
        c_t_eta_b: Thrust coefficient at max efficiency.
        tet_mcc: Maximum combustor temperature [K].
        tr_ec: Engine temperature ratio at max efficiency.
        m_ec: Mach number at engine corner point.
        buffer: Safety buffer.

    Returns:
        Maximum available thrust coefficient.

    References:
        Poll & Schumann (2021), Eq. 18-20.
    """
    tr_max = normalised_max_throttle_parameter(
        air_temperature,
        mach_number,
        tet_mcc,
        tr_ec,
        m_ec,
        buffer,
    )
    c_t_max_over_c_t_eta_b = 1.0 + 2.5 * (tr_max - 1.0)
    return c_t_max_over_c_t_eta_b * c_t_eta_b
