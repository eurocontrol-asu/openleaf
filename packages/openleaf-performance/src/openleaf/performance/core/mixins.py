"""Mixins for AircraftTrajectoryData.

PerformanceMixin provides aerodynamic and propulsion computed properties
that delegate to an aircraft model or implement physics equations.
"""

from __future__ import annotations


__all__ = ["PerformanceMixin"]

from functools import cached_property
from typing import TYPE_CHECKING

from openleaf.performance.physics import tem


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike


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
