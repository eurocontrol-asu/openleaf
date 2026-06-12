# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Mixins for AircraftTrajectoryData.

PerformanceMixin provides aerodynamic and propulsion computed properties
that delegate to an aircraft model or implement physics equations.
"""

from __future__ import annotations


__all__ = ["FlightStateMixin", "PerformanceMixin"]

from functools import cached_property
from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.physics import tem


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike


class FlightStateMixin:
    """Flight state properties derived from trajectory and aircraft model.

    Expects the container class to provide:
        - ``_model``: an ``AircraftModel`` instance
        - ``_traj``: a dict-like trajectory data store
    """

    @cached_property
    def aircraft_mass(self) -> ArrayScalarLike:
        """Aircraft mass [kg]."""
        return self._traj["aircraft_mass"]

    @cached_property
    def delta_isa(self) -> ArrayScalarLike:
        """Temperature deviation from ISA [K]."""
        return self._traj.delta_isa

    @cached_property
    def flight_phase(self) -> ArrayScalarLike:
        """Flight phase (CLIMB, CRUISE, DESCENT) from vertical speed.

        Derived from vertical_speed using standard thresholds:
        - vs > 0.5 m/s \u2192 CLIMB
        - vs < -0.5 m/s \u2192 DESCENT
        - otherwise \u2192 CRUISE
        """
        from openleaf.performance.models.common.config import determine_flight_phase

        return determine_flight_phase(self._traj["vertical_speed"])

    @cached_property
    def aerodynamic_configuration(self) -> ArrayScalarLike:
        """Aerodynamic configuration (TO, IC, CR, AP, LD) from model.

        Configuration depends on flight phase, altitude, and speed.
        Returns None for models without configuration (e.g., PS cruise-only).
        """
        return self._model.aerodynamic_configuration(self)

    @cached_property
    def high_lift(self) -> ArrayScalarLike:
        """High-lift device state (hlid) from model.

        Model-specific return type:
        - BADA4: HLid string (\"0\", \"1\", etc.)
        - OpenAP: Flap angle in degrees
        - BADA3/PS: None
        """
        return self._model.high_lift(self)

    @cached_property
    def landing_gear(self) -> ArrayScalarLike:
        """Landing gear state from model.

        Returns True if gear extended, False if retracted.
        May return None for models that don't track gear state.
        """
        return self._model.landing_gear(self)

    @cached_property
    def engine_setting(self) -> ArrayScalarLike:
        """Engine setting derived from flight phase.

        Maps flight phase to engine setting:
        - CLIMB \u2192 MCMB (Maximum Continuous Climb)
        - CRUISE \u2192 MCRZ (Maximum Continuous Cruise)
        - DESCENT \u2192 LIDL (Idle)
        """
        from openleaf.performance.core.perfos import EngineSetting
        from openleaf.performance.core.perfos import FlightPhase

        return np.where(
            np.equal(self.flight_phase, FlightPhase.CLIMB),
            EngineSetting.MCMB,
            np.where(
                np.equal(self.flight_phase, FlightPhase.DESCENT),
                EngineSetting.LIDL,
                EngineSetting.MCRZ,  # CRUISE default
            ),
        )


class PerformanceMixin:
    """Performance properties that delegate to an aircraft model.

    Expects the container class to provide:
        - ``_model``: an ``AircraftModel`` instance
        - ``_traj``: a dict-like trajectory data store
        - ``drag``, ``aircraft_mass``, ``f``, ``air_temperature``
          (for the TEM thrust calculation)
    """

    # ---- Aerodynamic coefficients ----

    @cached_property
    def lift_coef(self) -> ArrayScalarLike:
        """Lift coefficient [-]."""
        return self._model.lift_coef(self)

    @cached_property
    def drag_coef(self) -> ArrayScalarLike:
        """Drag coefficient [-].

        Requires aerodynamic_configuration for BADA3/BADA4/OpenAP.
        PS model uses cruise-only clean configuration.
        """
        return self._model.drag_coef(self)

    @cached_property
    def drag(self) -> ArrayScalarLike:
        """Aerodynamic drag [N]."""
        return self._model.drag(self)

    # ---- Thrust from TEM ----

    @cached_property
    def thrust(self) -> ArrayScalarLike:
        """Required thrust [N] from Total Energy Model.

        Uses the TEM equation to compute thrust given:
        - drag (from aerodynamic model)
        - vertical_speed (pressure altitude rate)
        - energy share factor f

        Formula (BADA Manual 3.1-4):
            T = D + (m \u00b7 g \u00b7 dh_geo/dt) / (V \u00b7 f)

        where dh_geo/dt is derived from pressure altitude rate using
        temperature correction.
        """
        return tem.thrust_from_rocd(
            rocd=self._traj["vertical_speed"],
            drag=self.drag,
            mass=self.aircraft_mass,
            true_airspeed=self._traj["true_airspeed"],
            f=self.f,
            air_temperature=self.air_temperature,
            altitude=self._traj["altitude"],
        )

    @cached_property
    def thrust_max(self) -> ArrayScalarLike:
        """Maximum available thrust [N]."""
        return self._model.thrust_max(self)

    @cached_property
    def thrust_min(self) -> ArrayScalarLike:
        """Minimum available thrust (idle) [N]."""
        return self._model.thrust_min(self)

    @cached_property
    def fuel_flow(self) -> ArrayScalarLike:
        """Fuel flow rate [kg/s]."""
        return self._model.fuel_flow(self)

    @cached_property
    def fuel_flow_min(self) -> ArrayScalarLike:
        """Minimum fuel flow rate (idle) [kg/s]."""
        return self._model.fuel_flow_min(self)
