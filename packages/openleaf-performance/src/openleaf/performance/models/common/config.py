# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Configuration selection utilities shared across models.

This module provides the ConfigurationSelector protocol and common helpers
for determining aerodynamic configuration from flight state.

Reference: BADA3 Section 3.5, BADA4 Appendix B
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

from pydantic import BaseModel
from pydantic import ConfigDict

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import FlightPhase
from openleaf.performance.core.types import AerodynamicConfig


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = [
    "AERO_CONFIG_THRESHOLDS",
    "AeroConfigThresholds",
    "ConfigurationSelector",
    "determine_flight_phase",
    "select_aerodynamic_configuration",
    "select_by_config",
]


class AeroConfigThresholds(BaseModel):
    """Altitude thresholds for aerodynamic configuration selection.

    These thresholds define altitude bands (in feet AGL) for each configuration.
    Used by both BADA3 and BADA4 models with identical default values.

    Reference: BADA3 Section 5.6, BADA4 Appendix B / ARPM Table 2
    """

    model_config = ConfigDict(frozen=True)

    h_max_to: float = 400.0
    """Maximum altitude for take-off configuration [ft]."""

    h_max_ic: float = 2000.0
    """Maximum altitude for initial climb configuration [ft]."""

    h_max_ap: float = 8000.0
    """Maximum altitude for approach configuration [ft]."""

    h_max_ld: float = 3000.0
    """Maximum altitude for landing configuration [ft]."""

    speed_margin: float = 10.0
    """Speed margin above Vmin for configuration transitions [kt]."""

    c_vmin: float = 1.3
    """Minimum speed coefficient: Vmin = c_vmin * Vstall (Section 5.7)."""


# Default thresholds per BADA3/BADA4 manuals
AERO_CONFIG_THRESHOLDS = AeroConfigThresholds()


def select_aerodynamic_configuration(
    *,
    altitude_ft: ArrayScalarLike,
    flight_phase: ArrayScalarLike,
    airspeed_kt: ArrayScalarLike,
    vmin_cr_kt: ArrayScalarLike,
    vmin_ap_kt: ArrayScalarLike,
    thresholds: AeroConfigThresholds = AERO_CONFIG_THRESHOLDS,
) -> ArrayScalarLike:
    """Select aerodynamic configuration from flight state.

    This is the shared configuration selection logic used by BADA3 and BADA4.
    The selection is based on flight phase, altitude, and speed relative to
    minimum speeds.

    Configuration selection rules:
    - TO: Climb, altitude <= h_max_to
    - IC: Climb, h_max_to < altitude <= h_max_ic
    - CR: Climb above h_max_ic, OR descent above h_max_ap,
          OR descent with speed >= vmin_cr + margin
    - AP: Descent, h_max_ld < altitude <= h_max_ap, speed < vmin_cr + margin
    - LD: Descent, altitude <= h_max_ld, speed < vmin_ap + margin

    Args:
        altitude_ft: Altitude AGL in feet.
        flight_phase: FlightPhase enum (CLIMB, CRUISE, DESCENT).
        airspeed_kt: Calibrated airspeed in knots.
        vmin_cr_kt: Minimum speed for cruise configuration [kt].
        vmin_ap_kt: Minimum speed for approach configuration [kt].
        thresholds: Altitude thresholds for configuration bands.

    Returns:
        AerodynamicConfig enum value (vectorized).

    Note:
        The caller is responsible for computing vmin_cr and vmin_ap from
        model-specific stall speed formulas. BADA3 uses sqrt(m/mref) scaling,
        BADA4 uses CL_max buffet model.
    """
    # Speed margins
    v_cr_margin = vmin_cr_kt + thresholds.speed_margin
    v_ap_margin = vmin_ap_kt + thresholds.speed_margin

    # Boolean conditions for flight phase
    is_descent = np.equal(flight_phase, FlightPhase.DESCENT)
    is_cruise = np.equal(flight_phase, FlightPhase.CRUISE)

    # Altitude conditions
    in_to_zone = np.less_equal(altitude_ft, thresholds.h_max_to)
    in_ic_zone = np.logical_and(
        np.greater(altitude_ft, thresholds.h_max_to),
        np.less_equal(altitude_ft, thresholds.h_max_ic),
    )
    above_ap_zone = np.greater(altitude_ft, thresholds.h_max_ap)
    in_ld_zone = np.less_equal(altitude_ft, thresholds.h_max_ld)

    # Speed conditions for descent
    fast_for_cr = np.greater_equal(airspeed_kt, v_cr_margin)
    slow_for_ld = np.less(airspeed_kt, v_ap_margin)

    # Climb configuration selection (based on altitude only)
    climb_config = np.where(
        in_to_zone,
        AerodynamicConfig.TO,
        np.where(in_ic_zone, AerodynamicConfig.IC, AerodynamicConfig.CR),
    )

    # Descent configuration selection (based on altitude and speed)
    # CR: above h_max_ap OR fast enough (speed >= vmin_cr + margin)
    # AP: below h_max_ap AND slow AND above h_max_ld
    # LD: below h_max_ld AND slow (speed < vmin_ap + margin)
    descent_config = np.where(
        np.logical_or(above_ap_zone, fast_for_cr),
        AerodynamicConfig.CR,
        np.where(
            np.logical_and(in_ld_zone, slow_for_ld),
            AerodynamicConfig.LD,
            AerodynamicConfig.AP,
        ),
    )

    # Select based on flight phase
    return np.where(
        is_descent,
        descent_config,
        np.where(is_cruise, AerodynamicConfig.CR, climb_config),
    )


class ConfigurationSelector(Protocol):
    """Protocol for configuration selection from flight state.

    All models that implement aerodynamic configuration selection should
    follow this interface. The implementation details vary by model
    (BADA3 uses altitude thresholds, BADA4 uses ARPM tables, etc.).
    """

    def aerodynamic_configuration(self, data: KeyedData) -> ArrayScalarLike:
        """Determine aerodynamic configuration from flight state.

        Args:
            data: Flight state dictionary with model-specific keys.

        Returns:
            AerodynamicConfig enum value (can be vectorized).
        """
        ...

    def high_lift(self, data: KeyedData) -> ArrayScalarLike | None:
        """Get high-lift device state from flight state.

        Args:
            data: Flight state dictionary.

        Returns:
            High-lift indicator (model-specific type) or None if not applicable.
        """
        ...

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state.

        Args:
            data: Flight state dictionary.

        Returns:
            Boolean indicating gear extended (True) or retracted (False).
        """
        ...


def determine_flight_phase(
    vertical_speed: ArrayScalarLike,
    threshold: float = 0.5,
) -> ArrayScalarLike:
    """Determine flight phase from vertical speed.

    This is a common helper used when flight_phase is not explicitly provided.
    The determination is based on rate of climb/descent with a small threshold
    to avoid oscillation near level flight.

    Args:
        vertical_speed: Vertical speed in m/s (positive = climb).
        threshold: Threshold for detecting climb/descent [m/s].
            Vertical speeds within +/- threshold are considered cruise.
            Default 0.5 m/s (~100 ft/min).

    Returns:
        FlightPhase enum value (CLIMB, CRUISE, or DESCENT).

    Example:
        >>> phase = determine_flight_phase(5.0)  # 5 m/s climb
        >>> phase == FlightPhase.CLIMB
        True
    """
    return np.where(
        np.greater(vertical_speed, threshold),
        FlightPhase.CLIMB,
        np.where(
            np.less(vertical_speed, -threshold),
            FlightPhase.DESCENT,
            FlightPhase.CRUISE,
        ),
    )


def select_by_config(
    config: ArrayScalarLike,
    mapping: dict,
    default: ArrayScalarLike = None,
) -> ArrayScalarLike:
    """Select values based on aerodynamic configuration using vectorized np.where.

    This is a utility function to reduce code duplication when selecting
    values based on AerodynamicConfig enum. It handles vectorized config
    values for CasADi compatibility.

    Args:
        config: AerodynamicConfig enum value (scalar or array).
        mapping: Dict mapping AerodynamicConfig to values.
        default: Default value if config not in mapping (last fallback).

    Returns:
        Selected value(s) based on config.

    Example:
        >>> hlid = select_by_config(
        ...     config,
        ...     {
        ...         AerodynamicConfig.CR: "0",
        ...         AerodynamicConfig.IC: "1",
        ...         AerodynamicConfig.TO: "2",
        ...         AerodynamicConfig.AP: "1",
        ...         AerodynamicConfig.LD: "3",
        ...     },
        ... )
    """
    result = default
    for config_key, value in mapping.items():
        result = np.where(np.equal(config, config_key), value, result)
    return result
