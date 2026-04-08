"""OpenAP aircraft performance model wrapper.

This module wraps the OpenAP library to provide aircraft performance
calculations following the aetherx interface patterns.
"""

from __future__ import annotations

from openleaf.performance.models.openap.limits import OpenAPLimits
from openleaf.performance.models.openap.models import OpenAPAeroModel
from openleaf.performance.models.openap.models import OpenAPEngineModel
from openleaf.performance.models.openap.models import OpenAPModel
from openleaf.performance.models.openap.schemas import OpenAPAerodynamicsData
from openleaf.performance.models.openap.schemas import OpenAPAircraftData
from openleaf.performance.models.openap.schemas import OpenAPData
from openleaf.performance.models.openap.schemas import OpenAPEngineData
from openleaf.performance.models.openap.schemas import OpenAPLimitsData


__all__ = [
    "OpenAPAeroModel",
    "OpenAPAerodynamicsData",
    "OpenAPAircraftData",
    "OpenAPData",
    "OpenAPEngineData",
    "OpenAPEngineModel",
    "OpenAPLimits",
    "OpenAPLimitsData",
    "OpenAPModel",
]
