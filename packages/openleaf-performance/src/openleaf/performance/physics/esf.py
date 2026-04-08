"""
Energy share factor (f) calculations for different speed schedules.

The energy share factor determines how excess thrust power is distributed
between potential energy (altitude change) and kinetic energy (speed change).

From BADA User Manual (Section 3.1):
    dHp/dt = [(T-D)·V_TAS / (m·g)] · f

Where f depends on the speed schedule:
- f = 1.0: All power to altitude (constant Mach above tropopause)
- f > 1.0: Kinetic → potential transfer (constant Mach below tropopause)
- f < 1.0: Some power to acceleration (constant CAS)

Reference: EUROCONTROL BADA User Manual, Section 3.1
"""

from __future__ import annotations


__all__ = [
    # Enum
    "SpeedSchedule",
    # Fixed constants
    "F_ACCEL_CLIMB",
    "F_ACCEL_DESCENT",
    "F_DECEL_CLIMB",
    "F_DECEL_DESCENT",
    # Functions
    "f_constant_mach",
    "f_constant_cas",
]

from enum import Enum

import openleaf.performance.numpy as np
from openleaf.performance.physics import isa
from openleaf.performance.utils.types import ArrayScalarLike


class SpeedSchedule(Enum):
    """Speed schedule for energy share factor computation.

    Determines how excess thrust is distributed between climb and acceleration.

    Reference: BADA User Manual Section 3.1
    """

    CONSTANT_MACH = "constant_mach"
    """Constant Mach number - f depends on altitude (1.0 above tropopause)."""

    CONSTANT_CAS = "constant_cas"
    """Constant calibrated airspeed - f < 1 (some power to acceleration)."""

    ACCEL_CLIMB = "accel_climb"
    """Accelerating during climb - fixed f = 0.3."""

    DECEL_CLIMB = "decel_climb"
    """Decelerating during climb - fixed f = 1.7."""

    ACCEL_DESCENT = "accel_descent"
    """Accelerating during descent - fixed f = 1.7."""

    DECEL_DESCENT = "decel_descent"
    """Decelerating during descent - fixed f = 0.3."""


# -----------------------------------------------------------------------------
# Fixed f values from BADA User Manual Section 3.1
# Used when neither constant Mach nor constant CAS is maintained
# -----------------------------------------------------------------------------

F_ACCEL_CLIMB = 0.3
"""f for acceleration during climb (majority of power to speed change)."""

F_DECEL_CLIMB = 1.7
"""f for deceleration during climb (kinetic energy transfers to potential)."""

F_ACCEL_DESCENT = 1.7
"""f for acceleration during descent (potential energy transfers to kinetic)."""

F_DECEL_DESCENT = 0.3
"""f for deceleration during descent (majority of power to speed change)."""


# -----------------------------------------------------------------------------
# Speed-schedule based f computations
# -----------------------------------------------------------------------------


def f_constant_mach(mach: ArrayScalarLike, altitude: ArrayScalarLike) -> ArrayScalarLike:
    """Energy share factor for constant Mach number speed schedule.

    Above tropopause: f = 1.0 (temperature constant, no acceleration needed)
    Below tropopause: f > 1.0 (TAS decreases with altitude at constant Mach)

    Formula (BADA 3.1-6):
        f = [1 + (gamma·R·k_T / 2g) · M^2]^(-1)

    Where k_T = -0.0065 K/m is the ISA temperature lapse rate.

    Args:
        mach: Mach number (dimensionless).
        altitude: Geometric altitude [m].

    Returns:
        Energy share factor f (dimensionless).
        Typically 1.0 to 1.1 for transport aircraft.
    """
    f_above = 1.0
    f_below = 1.0 / (1.0 - 0.5 * isa.GAMMA * isa.R * isa.TAU_H * mach * mach / isa.G)
    return np.where(altitude > isa.H_TROPO, f_above, f_below)


def f_constant_cas(mach: ArrayScalarLike, altitude: ArrayScalarLike) -> ArrayScalarLike:
    """Energy share factor for constant calibrated airspeed schedule.

    Below tropopause: f < 1.0 (TAS increases with altitude at constant CAS)
    Above tropopause: f < 1.0 (but different formula)

    Formula (BADA 3.1-7 below tropopause, 3.1-8 above):
        Complex compressibility-corrected formula.

    Args:
        mach: Mach number (dimensionless).
        altitude: Geometric altitude [m].

    Returns:
        Energy share factor f (dimensionless).
        Typically 0.8 to 0.95 for transport aircraft.
    """
    # Compressibility factor A = 1 + (gamma-1)/2 · M^2
    A = 1.0 + 0.5 * (isa.GAMMA - 1.0) * mach * mach

    # B term from isentropic relations
    B = np.power(A, -1.0 / (isa.GAMMA - 1.0)) * (np.power(A, isa.GAMMA / (isa.GAMMA - 1.0)) - 1.0)

    # Above tropopause (3.1-8): no temperature lapse term
    f_above = 1.0 / (1.0 + B)

    # Below tropopause (3.1-7): includes temperature lapse contribution
    f_below = 1.0 / (1.0 - 0.5 * isa.GAMMA * isa.TAU_H * isa.R * mach * mach / isa.G + B)

    return np.where(altitude > isa.H_TROPO, f_above, f_below)
