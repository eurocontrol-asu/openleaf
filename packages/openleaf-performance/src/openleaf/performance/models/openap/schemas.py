"""OpenAP data structures and schemas.

Provides typed Pydantic data classes for OpenAP aircraft performance data.
These classes represent the data loaded from OpenAP's aircraft database.

Reference: OpenAP https://github.com/junzis/openap
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict

from openleaf.performance.core.perfos import AerodynamicsData
from openleaf.performance.core.perfos import AircraftData
from openleaf.performance.core.perfos import EngineData
from openleaf.performance.core.perfos import LimitsData


__all__ = [
    "OpenAPAerodynamicsData",
    "OpenAPAircraftData",
    "OpenAPData",
    "OpenAPEngineData",
    "OpenAPLimitsData",
]


class OpenAPEngineData(EngineData):
    """OpenAP engine data.

    Attributes:
        n_engines: Number of engines.
        engine_name: Default engine model name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_engines: int
    engine_name: str


class OpenAPLimitsData(LimitsData):
    """OpenAP aircraft operational limits.

    Extends base LimitsData with OpenAP-specific fields.

    Attributes (inherited from LimitsData):
        vmo: Maximum operating speed [kt].
        mmo: Maximum operating Mach number [-].
        hmo: Maximum operating altitude (service ceiling) [ft].
        mtom: Maximum takeoff mass [kg].
        oem: Operating empty mass [kg].
        mpl: Maximum payload [kg].

    Attributes (OpenAP-specific):
        mlm: Maximum landing mass [kg].
        mfc: Maximum fuel capacity [litres] (volume, per the OpenAP database).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mlm: float
    mfc: float


class OpenAPAerodynamicsData(AerodynamicsData):
    """OpenAP aerodynamic data.

    Attributes:
        surface_area: Wing reference area [m²].
        wing_span: Wing span [m].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wing_span: float


class OpenAPAircraftData(AircraftData):
    """OpenAP aircraft identification data.

    Attributes:
        model: Aircraft model designator (ICAO code).
        wtc: Wake turbulence category (H/M/L).
        engine_type: Engine type (JET/TURBOPROP/PISTON).
        n_engines: Number of engines.
        engine_name: Default engine model name.
        vmo: Maximum operating speed [kt].
        mmo: Maximum operating Mach number [-].
        hmo: Maximum operating altitude [ft].
        oem: Operating empty mass [kg].
        mtom: Maximum takeoff mass [kg].
        mpl: Maximum payload [kg].
        mlm: Maximum landing mass [kg].
        mfc: Maximum fuel capacity [litres] (volume, per the OpenAP database).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_engines: int
    engine_name: str
    vmo: float
    mmo: float
    hmo: float
    oem: float
    mtom: float
    mpl: float
    mlm: float
    mfc: float


class OpenAPData(BaseModel):
    """Complete OpenAP aircraft data container.

    Container for all OpenAP data including aircraft specifications,
    aerodynamic data, engine parameters, and operational limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aircraft: OpenAPAircraftData
    engine: OpenAPEngineData
    aerodynamics: OpenAPAerodynamicsData
    limits: OpenAPLimitsData
