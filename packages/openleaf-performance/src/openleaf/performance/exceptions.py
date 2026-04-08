"""
AetherX exception hierarchy.

Provides domain-specific exceptions for clear error handling and debugging.
All AetherX exceptions inherit from OpenLeafError for easy catching.

Exception hierarchy:
    OpenLeafError
    ├── ConfigurationError     - Configuration/setup issues
    ├── ModelError             - Model-related errors
    │   ├── ModelNotFoundError - Aircraft model not available
    │   └── ModelDataError     - Invalid model data
    ├── ParsingError           - Data file parsing errors
    │   ├── BADA3ParseError    - BADA3-specific parsing
    │   └── BADA4ParseError    - BADA4-specific parsing
    ├── ValidationError        - Input/output validation errors
    │   ├── MissingFieldError  - Required field missing from KeyedData
    │   └── ValueRangeError    - Value outside valid range
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


class OpenLeafError(Exception):
    """Base exception for all AetherX errors.

    All AetherX-specific exceptions inherit from this class,
    allowing users to catch any library error with:

        try:
            model.compute(...)
        except OpenLeafError as e:
            handle_openleaf_error(e)
    """

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(OpenLeafError):
    """Raised when configuration is missing or invalid.

    Examples:
        - BADA3/BADA4 data path not configured
        - Configuration file corrupt or unreadable
        - Invalid configuration values
    """

    pass


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(OpenLeafError):
    """Base class for model-related errors.

    Raised when there are issues with aircraft performance models
    such as initialization failures or invalid model operations.
    """

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested aircraft model is not available.

    Examples:
        - ICAO code not in synonym file
        - Model data files missing
        - Unsupported aircraft type

    Attributes:
        icao: The ICAO code that was not found.
        model_type: Type of model (e.g., "BADA3", "BADA4", "OpenAP").
    """

    def __init__(
        self,
        message: str,
        icao: str | None = None,
        model_type: str | None = None,
    ):
        super().__init__(message)
        self.icao = icao
        self.model_type = model_type


class ModelDataError(ModelError):
    """Raised when model data is invalid or inconsistent.

    Examples:
        - Missing required coefficients
        - Invalid parameter values
        - Data file format errors after parsing
    """

    pass


# =============================================================================
# Parsing Errors
# =============================================================================


class ParsingError(OpenLeafError):
    """Base class for data file parsing errors.

    Raised when there are issues parsing aircraft data files
    such as OPF, XML, or configuration files.

    Attributes:
        filepath: Path to the file that failed to parse.
        line_number: Line number where error occurred (if applicable).
    """

    def __init__(
        self,
        message: str,
        filepath: str | None = None,
        line_number: int | None = None,
    ):
        super().__init__(message)
        self.filepath = filepath
        self.line_number = line_number

    def __str__(self) -> str:
        base = super().__str__()
        if self.filepath and self.line_number:
            return f"{base} (file: {self.filepath}, line: {self.line_number})"
        elif self.filepath:
            return f"{base} (file: {self.filepath})"
        return base


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


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(OpenLeafError):
    """Base class for input/output validation errors.

    Raised when data fails validation checks such as missing fields,
    invalid types, or values outside acceptable ranges.
    """

    pass


class MissingFieldError(ValidationError):
    """Raised when a required field is missing from KeyedData.

    This error provides a helpful message indicating which field
    is missing and which method requires it.

    Attributes:
        field: Name of the missing field.
        method: Name of the method that requires the field.
    """

    def __init__(
        self,
        field: str,
        method: str | None = None,
        message: str | None = None,
    ):
        if message is None:
            if method:
                message = f"Required field '{field}' missing from data (needed by {method})"
            else:
                message = f"Required field '{field}' missing from data"
        super().__init__(message)
        self.field = field
        self.method = method


class ValueRangeError(ValidationError):
    """Raised when a value is outside its valid range.

    Attributes:
        field: Name of the field with invalid value.
        value: The invalid value.
        min_value: Minimum allowed value (if applicable).
        max_value: Maximum allowed value (if applicable).
    """

    def __init__(
        self,
        field: str,
        value: float,
        min_value: float | None = None,
        max_value: float | None = None,
        message: str | None = None,
    ):
        if message is None:
            if min_value is not None and max_value is not None:
                message = f"'{field}' value {value} outside range [{min_value}, {max_value}]"
            elif min_value is not None:
                message = f"'{field}' value {value} below minimum {min_value}"
            elif max_value is not None:
                message = f"'{field}' value {value} above maximum {max_value}"
            else:
                message = f"'{field}' value {value} is invalid"
        super().__init__(message)
        self.field = field
        self.value = value
        self.min_value = min_value
        self.max_value = max_value


# =============================================================================
# Envelope Errors
# =============================================================================


class EnvelopeError(OpenLeafError):
    """Raised when flight envelope limits are violated.

    This error indicates that computed or input values exceed
    the aircraft's operational limits (VMO, MMO, HMO, etc.).

    Attributes:
        limit_type: Type of limit violated (e.g., "VMO", "MMO", "HMO").
        value: The violating value.
        limit: The limit that was exceeded.
    """

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        value: float | None = None,
        limit: float | None = None,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.value = value
        self.limit = limit
