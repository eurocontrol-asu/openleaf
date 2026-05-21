"""OpenLeaf exception hierarchy.

Base classes (``OpenLeafError`` and the high-level subclasses) live in
``openleaf.core.exceptions`` so every OpenLeaf package shares a single
``except OpenLeafError`` root. Provider-specific subclasses
(``BADA3ParseError``, ``BADA4ParseError``) stay here because they carry
provider-specific context attributes.

Exception hierarchy:
    OpenLeafError (openleaf.core)
    ├── ConfigurationError     - Configuration/setup issues
    ├── ModelError             - Model-related errors
    │   ├── ModelNotFoundError
    │   └── ModelDataError
    ├── ParsingError           - Data file parsing errors
    │   ├── BADA3ParseError    - (this module) BADA3-specific
    │   └── BADA4ParseError    - (this module) BADA4-specific
    ├── ValidationError        - Input/output validation errors
    │   ├── MissingFieldError
    │   └── ValueRangeError
    └── EnvelopeError          - Flight envelope violations
"""

from __future__ import annotations


__all__ = [
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
]

# Re-export the shared base hierarchy from openleaf-core.
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


# =============================================================================
# Provider-specific parsing errors (stay in openleaf-performance)
# =============================================================================


class BADA3ParseError(ParsingError):
    """Raised when BADA3 OPF/APF file parsing fails.

    Attributes:
        section: OPF section where error occurred (e.g., "AERO", "FUEL").
    """

    def __init__(
        self,
        message: str,
        filepath: str | None = None,
        section: str | None = None,
    ):
        super().__init__(message, filepath)
        self.section = section

    def __str__(self) -> str:
        base = Exception.__str__(self)
        parts = [base]
        if self.filepath:
            parts.append(f"file: {self.filepath}")
        if self.section:
            parts.append(f"section: {self.section}")
        if len(parts) > 1:
            return f"{parts[0]} ({', '.join(parts[1:])})"
        return base


class BADA4ParseError(ParsingError):
    """Raised when BADA4 XML file parsing fails.

    Attributes:
        element: XML element where error occurred.
    """

    def __init__(
        self,
        message: str,
        filepath: str | None = None,
        element: str | None = None,
    ):
        super().__init__(message, filepath)
        self.element = element

    def __str__(self) -> str:
        base = Exception.__str__(self)
        parts = [base]
        if self.filepath:
            parts.append(f"file: {self.filepath}")
        if self.element:
            parts.append(f"element: {self.element}")
        if len(parts) > 1:
            return f"{parts[0]} ({', '.join(parts[1:])})"
        return base
