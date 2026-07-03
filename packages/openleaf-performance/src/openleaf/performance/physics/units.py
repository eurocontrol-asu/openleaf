# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Unit conversion constants and functions.

Provides a centralized location for all unit conversions and physical constants
used in the library.
"""

from __future__ import annotations


__all__ = [
    # Numerical stability constants
    "EPSILON_CLAMP",
    "EPSILON_TOLERANCE",
    # Vertical speed conversions
    "FPM",
    # Length conversions
    "FT",
    "GAMMA",
    # Power conversions
    "HP",
    "KG_TO_LB",
    # Speed conversions
    "KT",
    # Mass conversions
    "LB",
    "MS_TO_FPM",
    "MS_TO_KT",
    "M_TO_FT",
    "M_TO_NM",
    "NM",
    "Q_FUEL",
    "R_AIR",
    "W_TO_HP",
    # Physical constants
    "G",
]

# ------------------------------------------------------------------------------
# Numerical stability constants
# ------------------------------------------------------------------------------
EPSILON_CLAMP = 1e-10
"""Small value for clamping denominators to avoid division by zero."""

EPSILON_TOLERANCE = 1e-9
"""Tolerance for floating-point comparisons and near-zero checks."""

# ------------------------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------------------------
G = 9.80665  # Standard gravitational acceleration [m/s²]
GAMMA = 1.4  # Heat capacity ratio for air (cp/cv)
R_AIR = 287.05287  # Specific gas constant for dry air [J/(kg·K)]
Q_FUEL = 43.13e6  # Fuel lower heating value [J/kg]

# ------------------------------------------------------------------------------
# Length conversions
# ------------------------------------------------------------------------------
FT = 0.3048  # Feet to meters
M_TO_FT = 1.0 / FT  # Meters to feet
NM = 1852.0  # Nautical miles to meters
M_TO_NM = 1.0 / NM  # Meters to nautical miles

# ------------------------------------------------------------------------------
# Speed conversions
# ------------------------------------------------------------------------------
KT = 0.514444  # Knots to m/s
MS_TO_KT = 1.0 / KT  # m/s to knots

# ------------------------------------------------------------------------------
# Vertical speed conversions
# ------------------------------------------------------------------------------
FPM = FT / 60.0  # Feet per minute to m/s
MS_TO_FPM = 1.0 / FPM  # m/s to feet per minute

# ------------------------------------------------------------------------------
# Mass conversions
# ------------------------------------------------------------------------------
LB = 0.453592  # Pounds to kilograms
KG_TO_LB = 1.0 / LB  # Kilograms to pounds

# ------------------------------------------------------------------------------
# Power conversions
# ------------------------------------------------------------------------------
HP = 745.699872  # Horsepower to watts
W_TO_HP = 1.0 / HP  # Watts to horsepower
