"""
AetherX - Aircraft performance models for trajectory optimization.

This package provides aircraft performance models including BADA3, BADA4, OpenAP,
and Poll-Schumann models with CasADi-compatible operations for seamless trajectory
optimization.

Main modules:
    - core: Core performance model interfaces and abstract base classes
    - models: Aircraft model implementations (BADA3, BADA4, OpenAP, PS)
    - physics: Aerodynamics, ISA atmosphere, and physics utilities
    - numpy: NumPy-compatible operations supporting CasADi symbolic types
    - utils: Type definitions and utility functions

Configuration:
    BADA3 and BADA4 models require configuration of data paths:

    >>> import openleaf.performance
    >>> openleaf.performance.set_bada3_path("/path/to/bada3/data")
    >>> openleaf.performance.set_bada4_path("/path/to/bada4/data")
    >>> openleaf.performance.show_config()

Usage:
    >>> from openleaf.performance.models.bada3 import BADA3Model
    >>> from openleaf.performance.models.bada4 import BADA4Model
    >>> from openleaf.performance.models.openap import OpenAPModel
    >>> from openleaf.performance.models.ps import PSModel
    >>>
    >>> # All models use the same interface
    >>> model = BADA3Model("A320")
    >>> model = OpenAPModel("A320")
"""

from __future__ import annotations

from openleaf.performance._version import __version__
from openleaf.performance._version import __version_tuple__

# Configuration functions - available at top level for convenience
from openleaf.performance.config import config_path
from openleaf.performance.config import get_bada3_path
from openleaf.performance.config import get_bada4_path
from openleaf.performance.config import get_config
from openleaf.performance.config import set_bada3_path
from openleaf.performance.config import set_bada3_synonyms
from openleaf.performance.config import set_bada4_path
from openleaf.performance.config import set_bada4_synonyms
from openleaf.performance.config import show_config

# Exception hierarchy - available at top level for easy catching
from openleaf.performance.exceptions import OpenLeafError
from openleaf.performance.exceptions import BADA3ParseError
from openleaf.performance.exceptions import BADA4ParseError
from openleaf.performance.exceptions import ConfigurationError
from openleaf.performance.exceptions import EnvelopeError
from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.exceptions import ModelDataError
from openleaf.performance.exceptions import ModelError
from openleaf.performance.exceptions import ModelNotFoundError
from openleaf.performance.exceptions import ParsingError
from openleaf.performance.exceptions import ValidationError
from openleaf.performance.exceptions import ValueRangeError


__all__ = [
    # Version
    "__version__",
    "__version_tuple__",
    # Exceptions (grouped first as most commonly caught)
    "OpenLeafError",
    "BADA3ParseError",
    "BADA4ParseError",
    "ConfigurationError",
    "EnvelopeError",
    "MissingFieldError",
    "ModelDataError",
    "ModelError",
    "ModelNotFoundError",
    "ParsingError",
    "ValidationError",
    "ValueRangeError",
    # Configuration
    "config_path",
    "get_bada3_path",
    "get_bada4_path",
    "get_config",
    "set_bada3_path",
    "set_bada3_synonyms",
    "set_bada4_path",
    "set_bada4_synonyms",
    "show_config",
]
