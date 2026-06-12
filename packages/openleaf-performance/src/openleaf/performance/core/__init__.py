# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Core performance model interfaces and aircraft trajectory data structures."""

from __future__ import annotations

from openleaf.performance.core.factory import ModelFactory
from openleaf.performance.core.factory import ModelType
from openleaf.performance.core.perfos import AerodynamicsData
from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftData
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineData
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.perfos import EngineType
from openleaf.performance.core.perfos import LimitsData
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.core.trajectory import AircraftTrajectoryData
from openleaf.performance.core.trajectory import TrajectoryData
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import EnvelopeMode
from openleaf.performance.core.types import EnvelopeViolationError
from openleaf.performance.core.types import FlightPhase
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.core.validation import StateSchema
from openleaf.performance.core.validation import get_required_fields
from openleaf.performance.core.validation import requires


__all__ = [
    "AerodynamicConfig",
    "AerodynamicsData",
    "AerodynamicsModel",
    "AircraftData",
    "AircraftModel",
    "AircraftTrajectoryData",
    "EnvelopeMode",
    "EnvelopeViolationError",
    "EngineData",
    "EngineModel",
    "EngineSetting",
    "EngineType",
    "FlightPhase",
    "LandingGearPosition",
    "LimitsData",
    "LimitsModel",
    "ModelFactory",
    "ModelType",
    "StateSchema",
    "TrajectoryData",
    "get_required_fields",
    "requires",
]
