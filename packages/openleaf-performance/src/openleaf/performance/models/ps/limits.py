"""Poll-Schumann aircraft limitations model.

Provides the PSLimits class implementing altitude-dependent Mach limits
based on impact pressure constraints and ATM speed restrictions.

Reference: Poll & Schumann (2021), The Aeronautical Journal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.models.ps.ps_physics import max_mach_number_by_altitude
from openleaf.performance.models.ps.schemas import PSLimitsData
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = ["PSLimits"]


class PSLimits(LimitsModel[PSLimitsData]):
    """Poll-Schuman aircraft operational limits.

    Provides Mach number limits based on altitude, impact pressure constraints,
    and ATM speed restrictions. Compatible with the AircraftLimits interface.
    """

    def __init__(self, data: PSLimitsData):
        """Initialize PS limits.

        Args:
            data: PS limits data container.
        """
        super().__init__(data)

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum Mach number at given altitude.

        PS model provides altitude-dependent Mach limits based on impact
        pressure constraints and ATM speed restrictions.

        Args:
            data: Flight state dictionary with optional keys:
                - altitude: Altitude in meters (package-wide convention).
                  If not provided, returns MMO.
                - air_pressure: Ambient pressure [Pa]. If not provided,
                  computed from ISA.
                - atm_speed_limit: Whether to apply ATM 250kt limit below
                  FL100 (default True).

        Returns:
            Maximum allowable Mach number.
        """
        # PS model always has mmo defined
        mmo = self.data.mmo
        if mmo is None:
            return np.inf

        altitude_m = data.get("altitude")
        if altitude_m is None:
            return mmo

        # "altitude" is meters everywhere in the package; the PS physics
        # helper (and its 10,000 ft ATM-limit threshold) works in feet.
        altitude_ft = altitude_m * units.M_TO_FT

        air_pressure = data.get("air_pressure")
        if air_pressure is None:
            air_pressure = isa.air_pressure(altitude_m)

        atm_speed_limit = bool(data.get("atm_speed_limit", True))

        return max_mach_number_by_altitude(
            altitude_ft,
            air_pressure,
            mmo,
            self.data.p_i_max,
            self.data.p_inf_co,
            atm_speed_limit=atm_speed_limit,
            buffer=0.02,
        )

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum operating altitude.

        Args:
            data: KeyedData (unused for PS - returns constant HMO).

        Returns:
            Maximum altitude in feet.
        """
        return self.data.hmo
