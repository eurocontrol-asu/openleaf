"""
Core type definitions for AetherX aircraft performance models.

This module provides unified type definitions used across all performance models
(BADA3, BADA4, OpenAP, PS). These types ensure consistency and interoperability
between different model implementations.

Reference: BADA3 Manual Section 3.5, BADA4 Manual Section 6
"""

from __future__ import annotations

from enum import Enum
from enum import IntEnum

from openleaf.performance.exceptions import EnvelopeError


# Import FlightPhase from perfos to maintain single source of truth
# Note: circular import avoided via TYPE_CHECKING

__all__ = [
    "AerodynamicConfig",
    "EngineSetting",
    "EngineType",
    "EnvelopeMode",
    "EnvelopeViolationError",
    "FlightPhase",
    "LandingGearPosition",
]


class EnvelopeMode(Enum):
    """How to handle flight envelope violations.

    Used by AircraftTrajectoryData to control behavior when trajectory
    points exceed aircraft limits (overspeed, underspeed, ceiling, mass).

    Values:
        IGNORE: Do nothing, just compute values (default). Violations can
            be queried via the envelope_violations property.
        WARN: Log a warning message for each violation type detected,
            but continue computing values.
        RAISE: Raise an EnvelopeViolationError when any violation is detected.
            This happens on first access to envelope_violations property.

    Example:
        >>> perf = AircraftTrajectoryData(traj, model, envelope_mode=EnvelopeMode.WARN)
        >>> _ = perf.envelope_violations  # Logs warnings if violations exist
    """

    IGNORE = "ignore"
    WARN = "warn"
    RAISE = "raise"


class EnvelopeViolationError(EnvelopeError):
    """Raised when flight envelope violations are detected in a trajectory.

    This exception is raised when EnvelopeMode.RAISE is set and the
    trajectory contains points that exceed aircraft limits. Unlike
    EnvelopeError which handles single-point violations, this handles
    batch/trajectory violations with arrays of violation flags.

    Attributes:
        violations: Dictionary mapping violation type to boolean array.

    Example:
        >>> try:
        ...     _ = perf.envelope_violations
        ... except EnvelopeViolationError as e:
        ...     print(f"Violations: {e.violations.keys()}")
    """

    def __init__(self, message: str, violations: dict):
        """Initialize with message and violations dict.

        Args:
            message: Human-readable description of violations.
            violations: Dictionary mapping violation type to boolean array.
        """
        super().__init__(message)
        self.violations = violations


class LandingGearPosition(IntEnum):
    """Landing gear position (universal across all models).

    Uses IntEnum for array API compatibility - values can be used directly
    in np.where and other array operations.

    Values:
        UP: Landing gear retracted (cruise, climb, approach).
        DOWN: Landing gear extended (landing, go-around).
    """

    UP = 0
    DOWN = 1

    @classmethod
    def from_string(cls, name: str) -> LandingGearPosition:
        """Create from string name (for XML/OPF parsing compatibility).

        Args:
            name: Position name ("UP" or "DN").

        Returns:
            Corresponding LandingGearPosition enum value.
        """
        if name == "DN":
            return cls.DOWN
        return cls[name]


class EngineType(IntEnum):
    """Enumeration of aircraft engine types.

    Uses IntEnum for array API compatibility.
    """

    TURBOPROP = 1
    JET = 2
    PISTON = 3

    @classmethod
    def from_string(cls, name: str) -> EngineType:
        """Create from string name."""
        return cls[name]


class EngineSetting(IntEnum):
    """Enumeration of engine throttle settings.

    Uses IntEnum for array API compatibility - values can be used directly
    in np.where and other array operations.
    """

    MTKF = 1  # Maximum Takeoff
    MCRZ = 2  # Maximum Continuous Cruise
    MCMB = 3  # Maximum Continuous Climb
    LIDL = 4  # Idle

    @classmethod
    def from_string(cls, name: str) -> EngineSetting:
        """Create from string name."""
        return cls[name]


class FlightPhase(IntEnum):
    """Enumeration of flight phases.

    Uses IntEnum for array API compatibility - values can be used directly
    in np.where and other array operations.
    """

    CLIMB = 1
    CRUISE = 2
    DESCENT = 3


class AerodynamicConfig(IntEnum):
    """Unified aerodynamic configuration across all models.

    These configurations represent discrete flight phases with different
    aerodynamic characteristics (flap/gear settings).

    Uses IntEnum for array API compatibility - values can be used directly
    in np.where and other array operations.

    BADA3 Manual Section 3.5 defines altitude thresholds:
        - TO: Take-off (up to 400 ft AGL in climb)
        - IC: Initial Climb (400-2000 ft AGL in climb)
        - CR: Cruise (above 2000 ft climb, above 8000 ft descent, or fast descent)
        - AP: Approach (below 8000 ft descent when V < Vmin_CR + 10 kt)
        - LD: Landing (below 3000 ft descent when V < Vmin_AP + 10 kt)

    BADA4 Manual Section 6 uses similar logic with aircraft-specific thresholds.
    """

    TO = 1  # Take-off
    IC = 2  # Initial Climb
    CR = 3  # Cruise (clean)
    AP = 4  # Approach
    LD = 5  # Landing

    @classmethod
    def from_string(cls, name: str) -> AerodynamicConfig:
        """Create from string name (for XML/OPF parsing compatibility).

        Args:
            name: Configuration name ("TO", "IC", "CR", "AP", "LD").

        Returns:
            Corresponding AerodynamicConfig enum value.

        Raises:
            ValueError: If name is not a valid configuration.
        """
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"Unknown aerodynamic configuration: {name}") from None
