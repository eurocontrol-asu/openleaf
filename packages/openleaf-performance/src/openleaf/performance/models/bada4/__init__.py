"""BADA4 aircraft performance model implementation.

Reference: EUROCONTROL BADA Family 4 User Manual
"""

from __future__ import annotations

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.models.bada4.limits import BADA4Limits
from openleaf.performance.models.bada4.models import BADA4AeroModel
from openleaf.performance.models.bada4.models import BADA4EngineModel
from openleaf.performance.models.bada4.models import BADA4JetEngineModel
from openleaf.performance.models.bada4.models import BADA4Model
from openleaf.performance.models.bada4.models import BADA4PistonEngineModel
from openleaf.performance.models.bada4.models import BADA4PropellerModel
from openleaf.performance.models.bada4.models import BADA4TurbineModel
from openleaf.performance.models.bada4.models import BADA4TurbopropEngineModel
from openleaf.performance.models.bada4.parsing import parse_arpm
from openleaf.performance.models.bada4.parsing import parse_bada4_file
from openleaf.performance.models.bada4.parsing import parse_limits
from openleaf.performance.models.bada4.schemas import BADA4AerodynamicsData
from openleaf.performance.models.bada4.schemas import BADA4AircraftData
from openleaf.performance.models.bada4.schemas import BADA4ARPMConfig
from openleaf.performance.models.bada4.schemas import BADA4BuffetLimitsData
from openleaf.performance.models.bada4.schemas import BADA4Data
from openleaf.performance.models.bada4.schemas import BADA4EngineData
from openleaf.performance.models.bada4.schemas import BADA4GeometricLimitsData
from openleaf.performance.models.bada4.schemas import BADA4JetEngineData
from openleaf.performance.models.bada4.schemas import BADA4JetOrTurbopropEngineData
from openleaf.performance.models.bada4.schemas import BADA4KinematicLimitsData
from openleaf.performance.models.bada4.schemas import BADA4LimitsData
from openleaf.performance.models.bada4.schemas import BADA4PistonEngineData
from openleaf.performance.models.bada4.schemas import BADA4TurbopropEngineData
from openleaf.performance.models.bada4.schemas import BADA4TurbopropOrPistonEngineData


__all__ = [
    "AerodynamicConfig",
    "BADA4ARPMConfig",
    # Aerodynamic model
    "BADA4AeroModel",
    # Data structures
    "BADA4AerodynamicsData",
    "BADA4AircraftData",
    "BADA4BuffetLimitsData",
    "BADA4Data",
    "BADA4EngineData",
    "BADA4EngineModel",
    "BADA4GeometricLimitsData",
    "BADA4JetEngineData",
    "BADA4JetEngineModel",
    "BADA4JetOrTurbopropEngineData",
    "BADA4KinematicLimitsData",
    "BADA4Limits",
    "BADA4LimitsData",
    "BADA4Model",
    "BADA4PistonEngineData",
    "BADA4PistonEngineModel",
    "BADA4PropellerModel",
    "BADA4TurbineModel",
    "BADA4TurbopropEngineData",
    "BADA4TurbopropEngineModel",
    "BADA4TurbopropOrPistonEngineData",
    # Parsing functions
    "parse_arpm",
    "parse_bada4_file",
    "parse_limits",
]
