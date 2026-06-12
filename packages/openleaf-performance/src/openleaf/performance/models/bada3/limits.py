"""BADA3 aircraft limitations model.

Provides the BADA3LimitsModel class implementing aircraft operational limits
including mass/temperature-dependent maximum altitude per BADA3 formula 3.5-1.

Reference: EUROCONTROL BADA3 User Manual Section 3.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.validation import requires
from openleaf.performance.models.bada3.schemas import BADA3LimitsData
from openleaf.performance.models.common import select_by_config
from openleaf.performance.physics.units import EPSILON_CLAMP


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = ["BADA3LimitsModel"]


class BADA3LimitsModel(LimitsModel[BADA3LimitsData]):
    """BADA3 aircraft limitations with mass/temperature dependent altitude.

    Implements BADA3 flight envelope limits including:
    - VMO: Maximum operating calibrated airspeed [kt]
    - MMO: Maximum operating Mach number
    - HMO: Maximum operating altitude [ft]
    - Mass/temperature dependent maximum altitude per BADA3 formula 3.5-1

    The actual maximum altitude is calculated as:
        hmax_act = MIN(hMO, hmax + Gt*(ΔTISA - CTc4) + Gw*(mmax - mact))

    where:
        - hmax: maximum altitude at MTOW under ISA [ft]
        - Gt: temperature gradient [ft/°C]
        - Gw: mass gradient [ft/kg]
        - CTc4: thrust temperature coefficient (from engine data)
        - mmax: maximum mass for altitude calculation [kg]
    """

    def __init__(self, data: BADA3LimitsData):
        """Initialize BADA3 limits.

        Args:
            data: BADA3 limits data container.
        """
        super().__init__(data)

    @requires("aerodynamic_configuration", "aircraft_mass")
    def min_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum calibrated airspeed (stall speed) by configuration.

        BADA3 provides fixed stall speeds by configuration from the OPF file.
        These are reference stall speeds at mref mass. For other masses,
        the stall speed scales as sqrt(m/mref).

        Args:
            data: Flight state with:
                - aerodynamic_configuration: AerodynamicConfig enum (required)
                - aircraft_mass: Aircraft mass [kg] (required)

        Returns:
            Minimum CAS in knots.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        return self._min_cas_impl(
            config=data["aerodynamic_configuration"],
            mass=data["aircraft_mass"],
        )

    def _min_cas_impl(
        self,
        config: ArrayScalarLike,
        mass: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum CAS from typed inputs.

        Args:
            config: Aerodynamic configuration enum (scalar or array).
            mass: Aircraft mass in kg (scalar or array).

        Returns:
            Minimum CAS in knots.

        Note:
            Fully vectorized - both config and mass can be arrays or CasADi symbolics.
        """
        # Select stall speed based on configuration using common utility
        vstall_ref = select_by_config(
            config,
            {
                AerodynamicConfig.CR: self.data.vstall.get(AerodynamicConfig.CR, 0.0),
                AerodynamicConfig.IC: self.data.vstall.get(AerodynamicConfig.IC, 0.0),
                AerodynamicConfig.TO: self.data.vstall.get(AerodynamicConfig.TO, 0.0),
                AerodynamicConfig.AP: self.data.vstall.get(AerodynamicConfig.AP, 0.0),
                AerodynamicConfig.LD: self.data.vstall.get(AerodynamicConfig.LD, 0.0),
            },
            default=0.0,
        )

        # Handle undefined stall speeds (vstall_ref <= 0)
        # Replace with small positive value to avoid division issues
        vstall_safe = np.where(vstall_ref > 0.0, vstall_ref, EPSILON_CLAMP)

        # Scale for mass: V_stall(m) = V_stall_ref * sqrt(m / mref)
        # mass on the left ensures torch/JAX tensor type propagation
        result = np.sqrt(mass / self.data.mref) * vstall_safe

        # Return 0 where vstall was undefined
        return np.where(vstall_ref > 0.0, result, 0.0)

    @requires("aircraft_mass", "delta_isa")
    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum operating altitude considering mass and temperature.

        Implements BADA3 formula 3.5-1:
            hmax_act = MIN(hMO, hmax + Gt*(ΔTISA - CTc4) + Gw*(mmax - mact))

        With constraints from manual:
            - if (ΔTISA - CTc4) < 0, then (ΔTISA - CTc4) = 0
            - Gt <= 0 (temperature gradient is negative)
            - Gw >= 0 (mass gradient is positive)

        Args:
            data: Flight state with:
                - aircraft_mass: Aircraft mass [kg] (required)
                - delta_isa: Temperature deviation from ISA [K] (required)

        Returns:
            Maximum altitude in feet (pressure altitude).

        Raises:
            MissingFieldError: If any required field is missing.
        """
        return self._max_altitude_impl(
            mass=data["aircraft_mass"],
            delta_isa=data["delta_isa"],
        )

    def _max_altitude_impl(
        self,
        mass: ArrayScalarLike,
        delta_isa: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate maximum altitude from typed inputs.

        Args:
            mass: Aircraft mass in kg.
            delta_isa: Temperature deviation from ISA in K.

        Returns:
            Maximum altitude in feet.
        """
        if self.data.hmax <= 0.0:
            return self.data.hmo

        # BADA3 formula 3.5-1
        # Per manual: if (ΔTISA - CTc4) < 0, then set to 0
        delta_isa_eff = np.maximum(delta_isa - self.data.ct_4, 0.0)
        h_temp_correction = self.data.gt * delta_isa_eff
        h_mass_correction = self.data.gw * (self.data.mmax - mass)
        h_max_actual = self.data.hmax + h_temp_correction + h_mass_correction

        return np.minimum(self.data.hmo, h_max_actual)

    def stall_speed(self, conf: AerodynamicConfig = AerodynamicConfig.CR) -> float:
        """Get stall speed for a given configuration.

        Args:
            conf: Aerodynamic configuration.

        Returns:
            Stall speed in knots CAS, or 0.0 if not available.
        """
        return self.data.vstall.get(conf, 0.0)

    def buffet_onset_cl(self, mach: ArrayScalarLike) -> ArrayScalarLike:
        """Calculate buffet onset lift coefficient.

        BADA3 low-speed buffeting limit (jet aircraft): the buffet-onset lift
        coefficient decreases linearly with Mach,

            CL_bo(M) = Clbo - k * M

        where ``Clbo`` is the buffet onset lift coefficient at M = 0 and ``k``
        is the buffeting gradient [1/M] from the OPF aerodynamics section.

        Args:
            mach: Mach number.

        Returns:
            Buffet onset lift coefficient.
        """
        return self.data.clbo - self.data.k * mach
