# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Flight envelope checking logic.

Extracted from AircraftTrajectoryData to encapsulate all envelope
violation detection and handling in a single class.
"""

from __future__ import annotations


__all__ = ["EnvelopeChecker"]

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import openleaf.performance.numpy as np
from openleaf.performance.core.types import EnvelopeMode
from openleaf.performance.core.types import EnvelopeViolationError
from openleaf.performance.logging import get_logger
from openleaf.performance.utils.types import ArrayScalarLike


if TYPE_CHECKING:
    from openleaf.performance.core.perfos import AircraftModel

logger = get_logger(__name__)


@runtime_checkable
class FlightState(Protocol):
    """Protocol for objects providing flight state data to EnvelopeChecker."""

    @property
    def cas_kt(self) -> ArrayScalarLike: ...

    @property
    def mach(self) -> ArrayScalarLike: ...

    @property
    def altitude_ft(self) -> ArrayScalarLike: ...

    @property
    def mass(self) -> ArrayScalarLike: ...

    def __len__(self) -> int: ...


class EnvelopeChecker:
    """Encapsulates flight envelope violation detection and handling.

    Args:
        model: Aircraft performance model providing limits and mass data.
        flight_state: Object exposing cas_kt, mach, altitude_ft, mass.
        envelope_mode: How to handle detected violations.
    """

    def __init__(
        self,
        model: AircraftModel,
        flight_state: Any,
        envelope_mode: EnvelopeMode = EnvelopeMode.IGNORE,
    ) -> None:
        self._model = model
        self._flight_state = flight_state
        self._envelope_mode = envelope_mode

    @property
    def envelope_mode(self) -> EnvelopeMode:
        """Current envelope violation handling mode."""
        return self._envelope_mode

    @cached_property
    def max_cas(self) -> ArrayScalarLike:
        """Maximum calibrated airspeed [kt]."""
        return self._model.max_cas(self._flight_state)

    @cached_property
    def min_cas(self) -> ArrayScalarLike:
        """Minimum calibrated airspeed (stall speed) [kt]."""
        result = self._model.min_cas(self._flight_state)
        return result

    @cached_property
    def max_mach(self) -> ArrayScalarLike:
        """Maximum Mach number [-]."""
        return self._model.max_mach(self._flight_state)

    @cached_property
    def max_altitude(self) -> ArrayScalarLike:
        """Maximum operating altitude [ft]."""
        return self._model.max_altitude(self._flight_state)

    @cached_property
    def envelope_violations(self) -> dict[str, ArrayScalarLike]:
        """Detect flight envelope violations (pure computation, no side effects).

        Returns:
            Dictionary with boolean violation flags.
        """
        fs = self._flight_state
        oem = self._model.oem
        mtom = self._model.mtom

        min_cas = self.min_cas
        underspeed = np.asarray(False) if min_cas is None else fs.cas_kt < min_cas

        # mmo is optional (e.g. piston aircraft); treat "no MMO" as no violation
        max_mach = self.max_mach
        overspeed_mach = np.asarray(False) if max_mach is None else fs.mach > max_mach

        return {
            "overspeed_cas": fs.cas_kt > self.max_cas,
            "overspeed_mach": overspeed_mach,
            "underspeed": underspeed,
            "ceiling": fs.altitude_ft > self.max_altitude,
            "mass_overweight": fs.mass > mtom,
            "mass_underweight": fs.mass < oem,
        }

    def check_envelope(self) -> dict[str, ArrayScalarLike]:
        """Check envelope violations and handle according to envelope_mode.

        Returns:
            Dictionary with violation flags.

        Raises:
            EnvelopeViolationError: If envelope_mode is RAISE and violations exist.
        """
        violations = self.envelope_violations
        self._handle_envelope_violations(violations)
        return violations

    def _handle_envelope_violations(self, violations: dict[str, ArrayScalarLike]) -> None:
        """Handle violations based on envelope_mode setting."""
        if self._envelope_mode == EnvelopeMode.IGNORE:
            return

        active_violations = {
            name: int(np.asarray(np.sum(mask)).item())
            for name, mask in violations.items()
            if np.any(mask)
        }

        if not active_violations:
            return

        total_points = len(self._flight_state)
        message = "Flight envelope violations detected"

        if self._envelope_mode == EnvelopeMode.WARN:
            logger.warning(
                message,
                violations=active_violations,
                total_points=total_points,
                aircraft=self._model.model,
            )
        elif self._envelope_mode == EnvelopeMode.RAISE:
            raise EnvelopeViolationError(f"{message}: {active_violations}", violations)
