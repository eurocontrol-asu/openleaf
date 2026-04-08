"""Common utilities shared across aircraft performance models.

This module provides shared utilities and helpers used by multiple models
(BADA3, BADA4, OpenAP, PS). It does NOT provide common base classes for
engine models because:

1. BADA3 uses altitude (h) and airspeed (v) parameterization
2. BADA4 uses normalized atmospheric parameters (mach, delta, theta)

These are fundamentally different approaches per their respective manuals,
and forcing common inheritance would violate physics-faithful principles.

Each model retains its own implementation faithful to its reference manual.
"""

from __future__ import annotations

from openleaf.performance.models.common.config import AERO_CONFIG_THRESHOLDS
from openleaf.performance.models.common.config import AeroConfigThresholds
from openleaf.performance.models.common.config import ConfigurationSelector
from openleaf.performance.models.common.config import determine_flight_phase
from openleaf.performance.models.common.config import select_aerodynamic_configuration
from openleaf.performance.models.common.config import select_by_config


__all__ = [
    "AERO_CONFIG_THRESHOLDS",
    "AeroConfigThresholds",
    "ConfigurationSelector",
    "determine_flight_phase",
    "select_aerodynamic_configuration",
    "select_by_config",
]
