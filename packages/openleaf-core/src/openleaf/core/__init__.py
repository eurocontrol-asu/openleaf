"""openleaf-core: shared types, protocols, and exceptions.

This package holds the contract every other OpenLeaf package depends on.
Keeping it dependency-light (only numpy at runtime) means anyone can
target the contract without inheriting CasADi, PyTorch, or pandas.
"""

from __future__ import annotations


__all__ = [
    # types
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
    # protocols
    "AerodynamicsModelProtocol",
    "AircraftModelProtocol",
    "EngineModelProtocol",
    "LimitsModelProtocol",
    # exceptions
    "ConfigurationError",
    "EnvelopeError",
    "MissingFieldError",
    "ModelDataError",
    "ModelError",
    "ModelNotFoundError",
    "OpenLeafError",
    "ParsingError",
    "ValidationError",
    "ValueRangeError",
]

from openleaf.core.exceptions import ConfigurationError
from openleaf.core.exceptions import EnvelopeError
from openleaf.core.exceptions import MissingFieldError
from openleaf.core.exceptions import ModelDataError
from openleaf.core.exceptions import ModelError
from openleaf.core.exceptions import ModelNotFoundError
from openleaf.core.exceptions import OpenLeafError
from openleaf.core.exceptions import ParsingError
from openleaf.core.exceptions import ValidationError
from openleaf.core.exceptions import ValueRangeError
from openleaf.core.protocols import AerodynamicsModelProtocol
from openleaf.core.protocols import AircraftModelProtocol
from openleaf.core.protocols import EngineModelProtocol
from openleaf.core.protocols import LimitsModelProtocol
from openleaf.core.types import ArrayScalarLike
from openleaf.core.types import KeyedData
from openleaf.core.types import is_keyed_data
