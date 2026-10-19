# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
BADA4 aircraft limitations model.

Provides the BADA4Limits class implementing the Aircraft Limitations Model (ALM)
per BADA4 User Manual Section 6.

Reference: EUROCONTROL BADA Family 4 User Manual Section 6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
import openleaf.performance.physics.isa as isa
from openleaf.performance.core.perfos import LimitsData
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.models.bada4.schemas import CVMIN_DEFAULT
from openleaf.performance.models.bada4.schemas import BADA4BuffetLimitsData
from openleaf.performance.models.bada4.schemas import BADA4GeometricLimitsData
from openleaf.performance.models.bada4.schemas import BADA4KinematicLimitsData
from openleaf.performance.models.bada4.schemas import BADA4LimitsData
from openleaf.performance.models.common.config import select_by_config
from openleaf.performance.physics.units import MS_TO_KT


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData
else:
    # Runtime imports for module-level functions
    from openleaf.performance.utils.types import ArrayScalarLike

__all__ = ["BADA4Limits"]

# Threshold for detecting infinity (CasADi-compatible alternative to np.isinf)
# Used because np.isinf doesn't work with CasADi symbolic types
_INF_THRESHOLD = 1e10

# Default CL_max fallback when configuration CL_max is undefined (infinite)
_CL_MAX_FALLBACK = 2.0


def _is_clean(config: ArrayScalarLike) -> ArrayScalarLike:
    """Check if configuration is clean (CR only). Fully vectorized.

    Only CR uses the clean configuration (HLid=0, no VFE restriction).
    IC, TO, AP, and LD all have HL deployed per the AeroConfSchedule.
    """
    return np.equal(config, AerodynamicConfig.CR)


def _has_hl_deployed(config: ArrayScalarLike) -> ArrayScalarLike:
    """Check if high-lift devices are deployed. Fully vectorized."""
    return np.logical_not(_is_clean(config))


def _has_lg_extended(config: ArrayScalarLike) -> ArrayScalarLike:
    """Check if landing gear is extended (LD config). Fully vectorized."""
    return np.equal(config, AerodynamicConfig.LD)


class BADA4Limits(LimitsModel[BADA4LimitsData]):
    """Combined BADA4 Aircraft Limitations Model (ALM) - Section 6.

    Extends the base AircraftLimits to provide BADA4-specific
    configuration-dependent limits including VFE, VLE, MFA, and buffet CL_max.

    All methods are fully vectorized and compatible with NumPy arrays
    and CasADi symbolic types when using aerodynamic_configuration.
    """

    def __init__(
        self,
        data: BADA4LimitsData,
        oem: float = 0.0,
        mtom: float = 0.0,
        mpl: float = 0.0,
        surface_area: float = 0.0,
        cvmin: float = CVMIN_DEFAULT,
    ):
        """Initialize BADA4 limits from data.

        Args:
            data: BADA4 limits data container.
            oem: Operating empty mass [kg].
            mtom: Maximum takeoff mass [kg].
            mpl: Maximum payload [kg].
            surface_area: Wing reference area [m²] for stall speed calculation.
            cvmin: Minimum speed coefficient (V_min = CVmin * V_stall).
        """
        # Create a base AircraftLimitsData for the parent class
        base_data = LimitsData(
            oem=oem,
            mtom=mtom,
            mpl=mpl,
            vmo=data.kinematic.vmo,
            mmo=data.kinematic.mmo,
            hmo=data.geometric.hmo,
        )
        super().__init__(base_data)
        self._bada4_data = data
        self._surface_area = surface_area
        self._cvmin = cvmin

    @property
    def data(self) -> BADA4LimitsData:
        """The underlying BADA4 limits data container."""
        return self._bada4_data

    # Convenience accessors for sub-data
    @property
    def geometric(self) -> BADA4GeometricLimitsData:
        """Geometric limitations data (GLM)."""
        return self._bada4_data.geometric

    @property
    def kinematic(self) -> BADA4KinematicLimitsData:
        """Kinematic limitations data (KLM)."""
        return self._bada4_data.kinematic

    @property
    def buffet(self) -> BADA4BuffetLimitsData:
        """Buffet limitations data (BLM)."""
        return self._bada4_data.buffet

    def max_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum calibrated airspeed (Eq. 6.2-1).

        BADA4 manual formula 6.2-1:
            V_CAS,max = VMO              when clean (CR/IC), LG up
            V_CAS,max = VFE              when HL deployed, LG up
            V_CAS,max = VLE              when clean, LG down
            V_CAS,max = min(VFE, VLE)    when HL deployed, LG down

        Args:
            data: Flight state dictionary with:
                - aerodynamic_configuration: AerodynamicConfig (preferred)
                - lg_extended: bool/array - landing gear state (if no config)

        Returns:
            Maximum CAS in knots. Fully vectorized.
        """
        kin = self._bada4_data.kinematic

        # Determine configuration state (fully vectorized)
        config = data.get("aerodynamic_configuration")
        if config is not None:
            is_clean = _is_clean(config)
            lg_extended = _has_lg_extended(config)
        else:
            # No config provided: assume clean, use lg_extended directly
            config = AerodynamicConfig.CR
            is_clean = True
            lg_extended = data.get("lg_extended", False)

        # Build per-config VFE using AeroConfSchedule mapping; fall back to VMO
        vfe_ic = kin.vfe_by_config.get(AerodynamicConfig.IC, kin.vmo)
        vfe_to = kin.vfe_by_config.get(AerodynamicConfig.TO, kin.vmo)
        vfe_ap = kin.vfe_by_config.get(AerodynamicConfig.AP, kin.vmo)
        vfe_ld = kin.vfe_by_config.get(AerodynamicConfig.LD, kin.vmo)

        # Vectorized VFE selection per config (non-clean only)
        vfe_nonclean = select_by_config(
            config,
            {
                AerodynamicConfig.IC: vfe_ic,
                AerodynamicConfig.TO: vfe_to,
                AerodynamicConfig.AP: vfe_ap,
                AerodynamicConfig.LD: vfe_ld,
            },
            default=kin.vmo,
        )

        # Vectorized selection per Eq. 6.2-1
        v_clean = np.where(lg_extended, kin.vle, kin.vmo)
        v_hl_deployed = np.where(lg_extended, np.minimum(vfe_nonclean, kin.vle), vfe_nonclean)
        return np.where(is_clean, v_clean, v_hl_deployed)

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum Mach number (Eq. 6.2-2).

        BADA4 provides gear-dependent Mach limits:
        - MMO: Clean configuration, gear up
        - MLE: Maximum Mach with landing gear extended

        Args:
            data: Flight state dictionary with:
                - aerodynamic_configuration: AerodynamicConfig (preferred)
                - lg_extended: bool/array - landing gear state (if no config)

        Returns:
            Maximum Mach number, or inf if not applicable. Fully vectorized.
        """
        kin = self._bada4_data.kinematic
        if kin.mmo is None:
            return np.inf

        # Determine lg_extended (fully vectorized)
        if "aerodynamic_configuration" in data:
            lg_extended = _has_lg_extended(data["aerodynamic_configuration"])
        else:
            lg_extended = data.get("lg_extended", False)

        return np.where(lg_extended, kin.mle, kin.mmo)

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum geopotential pressure altitude (Eq. 6.1-1).

        BADA4 provides configuration-dependent altitude limits:
        - HMO: Maximum operating altitude for clean configuration
        - MFA: Maximum high-lift devices use altitude

        Args:
            data: Flight state dictionary with:
                - aerodynamic_configuration: AerodynamicConfig (preferred)
                - hl_deployed: bool/array - high-lift state (if no config,
                  default: False)

        Returns:
            Maximum altitude in feet. Fully vectorized.
        """
        geo = self._bada4_data.geometric

        # Determine hl_deployed (fully vectorized)
        if "aerodynamic_configuration" in data:
            hl_deployed = _has_hl_deployed(data["aerodynamic_configuration"])
        else:
            hl_deployed = data.get("hl_deployed", False)

        return np.where(hl_deployed, geo.mfa, geo.hmo)

    def min_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum calibrated airspeed from stall speed (Section 6.3).

        BADA4 manual Section 6.3.3 defines the minimum speed as derived from
        the maximum lift coefficient CL_max:
            V_stall = sqrt(2 * m * g / (rho_SL * S * CL_max))
            V_min = CVmin * V_stall

        CL_max depends on aerodynamic configuration per formula 6.3-4:
            CL_max = CL_max,HL,LG (constant per HL/LG position)

        For clean configuration, CL_max is Mach-dependent per formula 6.3-2.

        Args:
            data: Flight state dictionary with:
                - aircraft_mass: Aircraft mass [kg] (default: MTOM)
                - aerodynamic_configuration: AerodynamicConfig (preferred)
                - mach_number: Mach number for clean CL_max (default: 0.3)
                - lg_extended: bool/array - landing gear state (if no config)

        Returns:
            Minimum CAS in knots. Fully vectorized.
        """
        mass = data.get("aircraft_mass", self._limits_data.mtom)
        mach = data.get("mach_number", 0.3)

        # Determine CL_max (fully vectorized)
        if "aerodynamic_configuration" in data:
            config = data["aerodynamic_configuration"]
            cl_max = self.cl_max_by_config(config, mach)
        else:
            # No config: use clean CL_max, with gear-down adjustment
            lg_extended = data.get("lg_extended", False)
            cl_max_clean = self.cl_max_clean(mach)
            # For gear-down without config, use a default non-clean CL_max
            cl_max_gear_dn = self._bada4_data.buffet.cl_max_config.get(
                AerodynamicConfig.LD, _CL_MAX_FALLBACK
            )
            cl_max = np.where(lg_extended, cl_max_gear_dn, cl_max_clean)

        # Handle undefined CL_max (use threshold comparison for CasADi compatibility)
        cl_max = np.where(cl_max > _INF_THRESHOLD, _CL_MAX_FALLBACK, cl_max)

        # Stall speed formula: V_stall = sqrt(2 * m * g / (rho_SL * S * CL_max))
        v_stall_ms = np.sqrt(2.0 * mass * isa.G / (isa.RHO_SL * self._surface_area * cl_max))

        # Convert to knots and apply CVmin coefficient
        return self._cvmin * v_stall_ms * MS_TO_KT

    # Buffet limit methods
    def _cl_poly(self, mach: ArrayScalarLike) -> ArrayScalarLike:
        """Evaluate CL_max polynomial (Eq. 6.3-1)."""
        bf = self._bada4_data.buffet.bf
        return bf[0] + bf[1] * mach + bf[2] * mach**2 + bf[3] * mach**3 + bf[4] * mach**4

    def _cl_der(self, mach: ArrayScalarLike) -> ArrayScalarLike:
        """Evaluate derivative of CL_max polynomial (Eq. 6.3-3)."""
        bf = self._bada4_data.buffet.bf
        return bf[1] + 2 * bf[2] * mach + 3 * bf[3] * mach**2 + 4 * bf[4] * mach**3

    def cl_max_clean(self, mach: ArrayScalarLike) -> ArrayScalarLike:
        """Get maximum lift coefficient in clean configuration (Eq. 6.3-2).

        Uses the Mach-dependent buffet limit model polynomial.

        Args:
            mach: Mach number (scalar or array).

        Returns:
            Maximum lift coefficient in clean configuration.
        """
        if not self._bada4_data.buffet.has_clean_model:
            if isinstance(mach, np.ndarray):
                return np.full_like(mach, np.inf, dtype=float)
            return np.inf

        return self._cl_poly(mach)

    def cl_max_nonclean_config(
        self,
        hl_position: str,
        lg_position: LandingGearPosition,
    ) -> float:
        """Get maximum lift coefficient for non-clean configuration (Eq. 6.3-4).

        Args:
            hl_position: High-lift devices position identifier.
            lg_position: Landing gear position (LandingGearPosition enum).

        Returns:
            Maximum lift coefficient, or inf if not defined.
        """
        return self._bada4_data.buffet.cl_max_nonclean.get((hl_position, lg_position), np.inf)

    def cl_max_by_config(
        self,
        config: ArrayScalarLike,
        mach: ArrayScalarLike = 0.3,
    ) -> ArrayScalarLike:
        """Get maximum lift coefficient for a given aerodynamic configuration.

        For the clean configuration (CR only), uses the Mach-dependent buffet model.
        For non-clean configurations (IC, TO, AP, LD), uses the configuration-mapped
        CL_max values from the AeroConfSchedule.

        Args:
            config: Aerodynamic configuration (scalar or array).
            mach: Mach number (used for clean configuration only).

        Returns:
            Maximum lift coefficient, or inf if not defined.

        Note:
            Fully vectorized - config can be scalar or array/CasADi symbolic.
        """
        # Get CL_max for clean configuration (Mach-dependent)
        cl_max_clean = self.cl_max_clean(mach)

        # Get CL_max for each non-clean configuration from lookup table
        cl_max_ic = self._bada4_data.buffet.cl_max_config.get(AerodynamicConfig.IC, np.inf)
        cl_max_to = self._bada4_data.buffet.cl_max_config.get(AerodynamicConfig.TO, np.inf)
        cl_max_ap = self._bada4_data.buffet.cl_max_config.get(AerodynamicConfig.AP, np.inf)
        cl_max_ld = self._bada4_data.buffet.cl_max_config.get(AerodynamicConfig.LD, np.inf)

        # Only CR is truly clean; IC, TO, AP, LD all have HL deployed
        is_clean = _is_clean(config)

        # Build non-clean result using nested np.where
        cl_max_nonclean = np.where(
            np.equal(config, AerodynamicConfig.IC),
            cl_max_ic,
            np.where(
                np.equal(config, AerodynamicConfig.TO),
                cl_max_to,
                np.where(
                    np.equal(config, AerodynamicConfig.AP),
                    cl_max_ap,
                    np.where(
                        np.equal(config, AerodynamicConfig.LD),
                        cl_max_ld,
                        np.inf,  # Default for unknown config
                    ),
                ),
            ),
        )

        return np.where(is_clean, cl_max_clean, cl_max_nonclean)
