"""Poll-Schumann aircraft performance model implementation.

This module provides the PS (Poll-Schumann) aircraft performance model
which is designed for cruise flight analysis of civil transport aircraft.

Usage:
    from openleaf.performance.models.ps import PSModel

    # Initialize from ICAO code (requires pycontrails)
    model = PSModel("A320")

    # Initialize from pre-loaded data
    model = PSModel.from_data(ps_data)

    # List available aircraft
    aircraft = PSModel.available_aircraft()

Reference:
    Poll, D.I.A., and Schumann, U. (2021). "An estimation method for the fuel burn
    and other performance characteristics of civil transport aircraft in the cruise."
    The Aeronautical Journal, 125(1284), 550-577.
"""

from __future__ import annotations

from openleaf.performance.models.ps.limits import PSLimits
from openleaf.performance.models.ps.models import PSAeroModel
from openleaf.performance.models.ps.models import PSEngineModel
from openleaf.performance.models.ps.models import PSModel
from openleaf.performance.models.ps.ps_physics import max_mach_number_by_altitude
from openleaf.performance.models.ps.ps_physics import oswald_efficiency_factor
from openleaf.performance.models.ps.ps_physics import reynolds_number
from openleaf.performance.models.ps.ps_physics import skin_friction_coefficient
from openleaf.performance.models.ps.ps_physics import wave_drag_coefficient
from openleaf.performance.models.ps.schemas import PSAerodynamicsData
from openleaf.performance.models.ps.schemas import PSAircraftData
from openleaf.performance.models.ps.schemas import PSData
from openleaf.performance.models.ps.schemas import PSEngineData
from openleaf.performance.models.ps.schemas import PSLimitsData


__all__ = [
    "PSAeroModel",
    "PSAerodynamicsData",
    "PSAircraftData",
    "PSData",
    "PSEngineData",
    "PSEngineModel",
    "PSLimits",
    "PSLimitsData",
    "PSModel",
    # Physics functions (for advanced users)
    "max_mach_number_by_altitude",
    "oswald_efficiency_factor",
    "reynolds_number",
    "skin_friction_coefficient",
    "wave_drag_coefficient",
]
