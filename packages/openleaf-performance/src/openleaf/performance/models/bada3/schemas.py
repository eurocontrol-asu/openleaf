"""
BADA3 data structures and schemas.

Provides typed Pydantic data classes for BADA3 aircraft performance data.
These classes represent the parsed data from OPF files.

Reference: EUROCONTROL Base of Aircraft Data (BADA) Family 3 User Manual
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from openleaf.performance.core.perfos import AerodynamicsData
from openleaf.performance.core.perfos import AircraftData
from openleaf.performance.core.perfos import EngineData
from openleaf.performance.core.perfos import LimitsData
from openleaf.performance.core.types import AerodynamicConfig


__all__ = [
    "BADA3AerodynamicsData",
    "BADA3AircraftData",
    "BADA3Data",
    "BADA3EngineData",
    "BADA3LimitsData",
]


class BADA3EngineData(EngineData):
    """BADA3 engine thrust and fuel flow coefficients.

    Attributes:
        cf_1: Nominal fuel flow coefficient 1 [kg/(min·kN)]
        cf_2: Nominal fuel flow coefficient 2 [kt]
        cf_3: Idle fuel flow coefficient 1 [kg/min]
        cf_4: Idle fuel flow coefficient 2 [ft]
        cf_crz: Cruise fuel flow correction factor [-]
        ct_1: Max thrust coefficient 1 [N]
        ct_2: Max thrust coefficient 2 [ft]
        ct_3: Max thrust coefficient 3 [1/ft²]
        ct_4: Max thrust temperature coefficient [K]
        ct_5: Max thrust temperature coefficient [1/K]
        ct_des_high: Descent thrust coefficient (high altitude) [-]
        ct_des_low: Descent thrust coefficient (low altitude) [-]
        h_des: Descent thrust transition altitude [ft]
        ct_des_app: Approach thrust coefficient [-]
        ct_des_ld: Landing thrust coefficient [-]
        ct_crz: Cruise thrust coefficient [-]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Fuel flow coefficients (Section 3.9)
    cf_1: float  # Nominal coefficient 1
    cf_2: float  # Nominal coefficient 2
    cf_3: float  # Idle coefficient 1
    cf_4: float  # Idle coefficient 2
    cf_crz: float  # Cruise correction

    # Max thrust coefficients (Section 3.7)
    ct_1: float
    ct_2: float
    ct_3: float
    ct_4: float
    ct_5: float

    # Idle thrust coefficients (Section 3.8)
    ct_des_high: float
    ct_des_low: float
    h_des: float
    ct_des_app: float
    ct_des_ld: float

    # Cruise thrust coefficient
    ct_crz: float = 0.95


class BADA3LimitsData(LimitsData):
    """BADA3 flight envelope and buffet limits.

    Per Section 3.5 of BADA3 User Manual.

    Attributes:
        vmo: Maximum operating CAS [kt]
        mmo: Maximum operating Mach [-]
        hmo: Maximum operating altitude [ft]
        oem: Operating empty mass [kg]
        mtom: Maximum takeoff mass [kg]
        mpl: Maximum payload [kg]
        hmax: Maximum altitude at MTOW, ISA [ft]
        gt: Temperature gradient for max altitude [ft/K]
        gw: Mass gradient for max altitude [ft/kg]
        mref: Reference mass [kg]
        mmax: Maximum mass at which hmax is defined [kg]; equals mtom per BADA3 Section 3.5
        clbo: Lift coefficient for buffet onset [-]
        k: Mach-dependent buffet factor [-]
        ct_4: Thrust temperature coefficient [K] (for formula 3.5-1)
        vstall: Stall speeds by configuration [kt]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mass data (Section 3.1)
    mref: float
    mmax: float

    # Buffet onset (Section 3.6)
    clbo: float
    k: float

    # Max altitude coefficients (Section 3.5, formula 3.5-1)
    hmax: float = 0.0
    gt: float = 0.0
    gw: float = 0.0
    ct_4: float = 0.0

    # Stall speeds by configuration
    vstall: dict[AerodynamicConfig, float] = Field(default_factory=dict)


class BADA3AerodynamicsData(AerodynamicsData):
    """BADA3 aerodynamic coefficients.

    Per Section 3.4 of BADA3 User Manual.

    Attributes:
        surface_area: Wing reference area [m²]
        cd_0: Zero-lift drag coefficient by configuration [-]
        cd_2: Induced drag coefficient by configuration [-]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cd_0: dict[AerodynamicConfig, float]
    cd_2: dict[AerodynamicConfig, float]


class BADA3AircraftData(AircraftData):
    """BADA3 aircraft identification data.

    Attributes:
        model: Aircraft model designator
        wtc: Wake turbulence category (H/M/L)
        engine_type: Engine type (JET/TURBOPROP/PISTON)
        engines: Number of engines
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    engines: int


class BADA3Data(BaseModel):
    """Complete BADA3 aircraft performance data.

    Container for all parsed OPF data including aircraft specifications,
    aerodynamic coefficients, engine parameters, and operational limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aircraft: BADA3AircraftData
    aerodynamics: BADA3AerodynamicsData
    engine: BADA3EngineData
    limits: BADA3LimitsData
