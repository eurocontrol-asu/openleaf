"""
BADA4 data structures and schemas.

Provides typed Pydantic data classes for BADA4 aircraft performance data.
These classes represent the parsed data from XML files.

Reference: EUROCONTROL BADA Family 4 User Manual
"""

from __future__ import annotations

import numpy.typing as npt
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

import openleaf.performance.numpy as np
import openleaf.performance.physics.isa as isa
from openleaf.performance.core.perfos import AerodynamicsData
from openleaf.performance.core.perfos import AircraftData
from openleaf.performance.core.perfos import EngineData
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.physics.units import FT


__all__ = [
    # ARPM configuration
    "BADA4ARPMConfig",
    # Aerodynamics and aircraft data
    "BADA4AerodynamicsData",
    "BADA4AircraftData",
    "BADA4BuffetLimitsData",
    "BADA4Data",
    # Engine data structures
    "BADA4EngineData",
    # Limits data structures
    "BADA4GeometricLimitsData",
    "BADA4JetEngineData",
    "BADA4JetOrTurbopropEngineData",
    "BADA4KinematicLimitsData",
    "BADA4LimitsData",
    "BADA4PistonEngineData",
    "BADA4TurbopropEngineData",
    "BADA4TurbopropOrPistonEngineData",
    # Constants
    "CVMIN_DEFAULT",
    "CVMIN_TO_DEFAULT",
    "HMAX_AP_DEFAULT",
    "HMAX_IC_DEFAULT",
    "HMAX_LD_DEFAULT",
    "HMAX_TO_DEFAULT",
]

# =============================================================================
# ARPM DEFAULT CONSTANTS (Section 8 / Appendix B)
# =============================================================================

# Default altitude thresholds [ft AGL]
HMAX_TO_DEFAULT = 400.0
HMAX_IC_DEFAULT = 2000.0
HMAX_AP_DEFAULT = 8000.0
HMAX_LD_DEFAULT = 3000.0

# Default minimum speed coefficients
CVMIN_TO_DEFAULT = 1.13
CVMIN_DEFAULT = 1.23


# =============================================================================
# LIMITATIONS DATA STRUCTURES (Section 6)
# =============================================================================


class BADA4GeometricLimitsData(BaseModel):
    """Geometric Limitations Model (GLM) data - Section 6.1.

    Pure data container for geometric limits.

    Attributes:
        hmo: Maximum operating altitude [ft].
        mfa: Maximum high-lift devices use altitude [ft], defaults to hmo.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hmo: float
    mfa: float | None = None

    @model_validator(mode="after")
    def _set_mfa_default(self) -> BADA4GeometricLimitsData:
        if self.mfa is None:
            object.__setattr__(self, "mfa", self.hmo)
        return self


class BADA4KinematicLimitsData(BaseModel):
    """Kinematic Limitations Model (KLM) data - Section 6.2.

    Pure data container for kinematic limits.

    Attributes:
        vmo: Maximum operating speed [kt].
        mmo: Maximum operating Mach number, None if not applicable.
        vle: Maximum speed with landing gear extended [kt], defaults to vmo.
        mle: Maximum Mach with landing gear extended, defaults to mmo.
        vfe: Dict mapping HLid string to placard speed [kt] (raw lookup table).
        vfe_by_config: Dict mapping AerodynamicConfig to placard VFE [kt].
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    vmo: float
    mmo: float | None = None
    vle: float | None = None
    mle: float | None = None
    vfe: dict[str, float] = Field(default_factory=dict)
    vfe_by_config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_defaults(self) -> BADA4KinematicLimitsData:
        if self.vle is None:
            object.__setattr__(self, "vle", self.vmo)
        if self.mle is None and self.mmo is not None:
            object.__setattr__(self, "mle", self.mmo)
        return self


class BADA4BuffetLimitsData(BaseModel):
    """Buffet Limitations Model (BLM) data - Section 6.3.

    Pure data container for buffet limits.

    Attributes:
        bf: Polynomial coefficients [bf1..bf5] for clean CL_max.
        mach_min: Lower boundary of polynomial domain.
        mach_max: Upper boundary of polynomial domain.
        cl_mach0: Maximum lift coefficient at M=0.
        cl_max_nonclean: Dict mapping (HL, LG) to constant CL_max values.
            Keys are (hlid_str, lg_pos) where lg_pos is LandingGearPosition enum.
        cl_max_config: Dict mapping AerodynamicConfig to CL_max values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    bf: npt.NDArray | None = None
    mach_min: float | None = None
    mach_max: float | None = None
    cl_mach0: float | None = None
    cl_max_nonclean: dict[tuple[str, LandingGearPosition], float] = Field(
        default_factory=dict
    )
    cl_max_config: dict[AerodynamicConfig, float] = Field(default_factory=dict)

    @property
    def has_clean_model(self) -> bool:
        """Check if clean CL_max model is available."""
        return self.bf is not None and len(self.bf) >= 5


class BADA4LimitsData(BaseModel):
    """Combined BADA4 Aircraft Limitations data - Section 6.

    Pure data container composing geometric, kinematic, and buffet limits.

    Attributes:
        geometric: Geometric limitations data (GLM).
        kinematic: Kinematic limitations data (KLM).
        buffet: Buffet limitations data (BLM).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    geometric: BADA4GeometricLimitsData
    kinematic: BADA4KinematicLimitsData
    buffet: BADA4BuffetLimitsData

    @property
    def vmo(self) -> float:
        """Maximum operating speed [kt]."""
        return self.kinematic.vmo

    @property
    def mmo(self) -> float | None:
        """Maximum operating Mach number."""
        return self.kinematic.mmo

    @property
    def hmo(self) -> float:
        """Maximum operating altitude [ft]."""
        return self.geometric.hmo


# =============================================================================
# ARPM CONFIGURATION DATA (Section 8 / Appendix B)
# =============================================================================


class BADA4ARPMConfig(BaseModel):
    """Airline Procedure Model (ARPM) configuration - Section 8 / Appendix B.

    Defines altitude thresholds for configuration changes, minimum
    speed coefficients, and high-lift/landing-gear mapping per configuration.

    Attributes:
        hmax_to: Max altitude threshold for take-off [ft AGL].
        hmax_ic: Max altitude threshold for initial climb [ft AGL].
        hmax_ap: Max altitude threshold for approach [ft AGL].
        hmax_ld: Max altitude threshold for landing [ft AGL].
        cvmin_to: Minimum speed coefficient for take-off configuration [-].
        cvmin: Minimum speed coefficient for other configurations [-].
        hlid_map: Mapping from AerodynamicConfig to HLid (from AeroConfSchedule).
        lg_map: Mapping from AerodynamicConfig to LandingGearPosition (UP/DOWN).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hmax_to: float = HMAX_TO_DEFAULT
    hmax_ic: float = HMAX_IC_DEFAULT
    hmax_ap: float = HMAX_AP_DEFAULT
    hmax_ld: float = HMAX_LD_DEFAULT
    cvmin_to: float = CVMIN_TO_DEFAULT
    cvmin: float = CVMIN_DEFAULT
    hlid_map: dict[AerodynamicConfig, str] = Field(default_factory=dict)
    lg_map: dict[AerodynamicConfig, LandingGearPosition] = Field(default_factory=dict)

    def vmin_config(
        self,
        vs: npt.ArrayLike,
        config: AerodynamicConfig,
    ) -> npt.ArrayLike:
        """Calculate minimum operational speed for configuration (Eq. B.5-1).

        Args:
            vs: Stall speed in CAS [kt].
            config: Aerodynamic configuration.

        Returns:
            Minimum operational speed in CAS [kt].
        """
        return np.where(
            np.equal(config, AerodynamicConfig.TO),
            self.cvmin_to * vs,
            self.cvmin * vs,
        )


# =============================================================================
# ENGINE DATA STRUCTURES
# =============================================================================


class BADA4EngineData(EngineData):
    """BADA4 base engine data common to all engine types.

    Attributes:
        engines: Number of engines.
        lhv: Lower heating value of fuel (J/kg).
        wref: Reference weight for engine thrust [N] (= MREF * g).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    engines: int
    lhv: float
    wref: float


class BADA4JetOrTurbopropEngineData(BaseModel):
    """Common data for jet and turboprop engines.

    Attributes:
        a: Mach-dependent thrust coefficients.
        f: Altitude-dependent thrust coefficients.
        fi: Integrated thrust coefficients.
        ti: Integrated thrust coefficients.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    a: npt.NDArray
    f: npt.NDArray
    fi: npt.NDArray
    ti: npt.NDArray


class BADA4TurbopropOrPistonEngineData(BaseModel):
    """Common data for turboprop and piston engines.

    Attributes:
        prop_dia: Propeller diameter (m).
        max_eff: Maximum propeller efficiency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prop_dia: float
    max_eff: float


class BADA4JetEngineData(
    BADA4EngineData,
    BADA4JetOrTurbopropEngineData,
):
    """BADA4 jet engine data.

    Contains throttle-dependent thrust and fuel flow coefficients
    for different engine settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    b: dict[EngineSetting, npt.NDArray]
    c: dict[EngineSetting, npt.NDArray]
    kink: dict[EngineSetting, float]


class BADA4TurbopropEngineData(
    BADA4EngineData,
    BADA4JetOrTurbopropEngineData,
    BADA4TurbopropOrPistonEngineData,
):
    """BADA4 turboprop engine data.

    Contains power-dependent thrust and fuel flow coefficients
    for different engine settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    p: dict[EngineSetting, npt.NDArray]
    max_power: dict[EngineSetting, float]


class BADA4PistonEngineData(
    BADA4EngineData,
    BADA4TurbopropOrPistonEngineData,
):
    """BADA4 piston engine data (PEM) - Section 3.3.3.

    Attributes:
        hd_turbo: Turbocharger density altitude limit [ft]. Zero for atmospheric.
        cpsfc: Power-specific fuel consumption coefficient [-].
        p: Maximum one-engine power at MSL in standard atmosphere [hp].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hd_turbo: float
    cpsfc: float
    p: float

    @property
    def sigma_turbo(self) -> float:
        """Density ratio at turbocharger limit altitude."""
        if self.hd_turbo <= 0:
            return 1.0
        h_m = self.hd_turbo * FT
        return isa.air_pressure(h_m) / (isa.air_temperature(h_m) * isa.RHO_SL * isa.R)

    @property
    def delta_turbo(self) -> float:
        """Pressure ratio at turbocharger limit altitude."""
        if self.hd_turbo <= 0:
            return 1.0
        return isa.air_pressure(self.hd_turbo * FT) / isa.P_SL

    @property
    def theta_turbo(self) -> float:
        """Temperature ratio at turbocharger limit altitude."""
        if self.hd_turbo <= 0:
            return 1.0
        return isa.air_temperature(self.hd_turbo * FT) / isa.TAU_SL


# =============================================================================
# AERODYNAMICS AND AIRCRAFT DATA
# =============================================================================


class BADA4AerodynamicsData(AerodynamicsData):
    """BADA4 aerodynamic data.

    Attributes:
        surface_area: Wing reference area in m².
        mach_max: Maximum Mach number.
        scalar: Scaling factor for drag coefficients.
        d_clean: Drag coefficients for clean configuration (HLid=0, LG=UP).
        d_nonclean: Drag coefficients by (hlid, lg) tuple.
            Keys are (hlid_str, lg_pos) where hlid_str is "0","1","2",... and
            lg_pos is LandingGearPosition.UP or LandingGearPosition.DOWN.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    mach_max: float
    scalar: float

    d_clean: npt.NDArray
    d_nonclean: dict[tuple[str, LandingGearPosition], npt.NDArray]


class BADA4AircraftData(AircraftData):
    """BADA4 aircraft data.

    Extends the base aircraft data with BADA4-specific parameters.

    Attributes:
        vmo: Maximum operating calibrated airspeed [kt].
        mmo: Maximum operating Mach number.
        hmo: Maximum operating altitude [ft].
        designator: ICAO aircraft designator code.
        oem: Operating empty mass [kg].
        mtom: Maximum takeoff mass [kg].
        mpl: Maximum payload [kg].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vmo: float
    mmo: float | None
    hmo: float
    designator: str
    oem: float
    mtom: float
    mpl: float


class BADA4Data(BaseModel):
    """Container for complete BADA4 aircraft data.

    Attributes:
        aircraft: Aircraft specifications and performance limits.
        aero: Aerodynamic coefficients and wing area.
        engine: Engine thrust and fuel flow coefficients.
        limits: Aircraft limitations (ALM), optional. Type is BADA4Limits from limits.py.
        arpm: Airline procedure model configuration, optional.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    aircraft: BADA4AircraftData
    aero: BADA4AerodynamicsData
    engine: BADA4EngineData
    # Use object type to avoid circular import with BADA4Limits from limits.py
    # The actual runtime type is BADA4Limits
    limits: object | None = None
    arpm: BADA4ARPMConfig | None = None
