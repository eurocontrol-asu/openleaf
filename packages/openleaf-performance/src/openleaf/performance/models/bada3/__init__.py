"""BADA3 aircraft performance model implementation."""

from __future__ import annotations

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.models.bada3.limits import BADA3LimitsModel
from openleaf.performance.models.bada3.models import BADA3AeroModel
from openleaf.performance.models.bada3.models import BADA3EngineModel
from openleaf.performance.models.bada3.models import BADA3JetEngineModel
from openleaf.performance.models.bada3.models import BADA3Model
from openleaf.performance.models.bada3.models import BADA3PistonEngineModel
from openleaf.performance.models.bada3.models import BADA3PropellerModel
from openleaf.performance.models.bada3.models import BADA3TurbineModel
from openleaf.performance.models.bada3.models import BADA3TurbopropEngineModel
from openleaf.performance.models.bada3.parsing import parse_bada3_file
from openleaf.performance.models.bada3.schemas import BADA3AerodynamicsData
from openleaf.performance.models.bada3.schemas import BADA3AircraftData
from openleaf.performance.models.bada3.schemas import BADA3Data
from openleaf.performance.models.bada3.schemas import BADA3EngineData
from openleaf.performance.models.bada3.schemas import BADA3LimitsData


__all__ = [
    "AerodynamicConfig",
    "BADA3AeroModel",
    "BADA3AerodynamicsData",
    "BADA3AircraftData",
    "BADA3Data",
    "BADA3EngineData",
    "BADA3EngineModel",
    "BADA3JetEngineModel",
    "BADA3LimitsData",
    "BADA3LimitsModel",
    "BADA3Model",
    "BADA3PistonEngineModel",
    "BADA3PropellerModel",
    "BADA3TurbineModel",
    "BADA3TurbopropEngineModel",
    "parse_bada3_file",
]
