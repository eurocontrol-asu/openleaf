# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
Input validation utilities for OpenLeaf.

Provides helper functions for validating KeyedData inputs with clear error
messages and optional bounds checking.

Usage:
    from openleaf.performance.utils.validation import get_required, get_optional, validate_range

    # Get required field with helpful error
    cl = get_required(data, "lift_coef", method="drag_coef")

    # Get optional field with default
    config = get_optional(data, "aerodynamic_configuration", default="clean")

    # Validate value is in range
    validate_range(mach, "mach_number", min_value=0.0, max_value=1.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.exceptions import ValueRangeError


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = [
    "get_optional",
    "get_required",
    "validate_positive",
    "validate_range",
]

T = TypeVar("T")


def get_required(
    data: KeyedData,
    field: str,
    method: str | None = None,
) -> ArrayScalarLike:
    """Get a required field from KeyedData with helpful error message.

    Args:
        data: KeyedData dictionary or similar object.
        field: Name of the required field.
        method: Optional method name for error context.

    Returns:
        The field value.

    Raises:
        MissingFieldError: If the field is not present in data.

    Example:
        >>> cl = get_required(data, "lift_coef", method="drag_coef")
    """
    try:
        value = data[field]
    except KeyError:
        raise MissingFieldError(field, method=method) from None

    return value


def get_optional[T](
    data: KeyedData,
    field: str,
    default: T = None,
) -> ArrayScalarLike | T:
    """Get an optional field from KeyedData with default value.

    Args:
        data: KeyedData dictionary or similar object.
        field: Name of the optional field.
        default: Default value if field is not present.

    Returns:
        The field value or default.

    Example:
        >>> config = get_optional(data, "aerodynamic_configuration", default="clean")
    """
    return data.get(field, default)


def validate_range(
    value: ArrayScalarLike,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    raise_error: bool = True,
) -> bool:
    """Validate that a value is within the specified range.

    For array inputs, checks if any values are outside the range.

    Args:
        value: Value to validate (scalar or array).
        field: Field name for error message.
        min_value: Minimum allowed value (inclusive), or None for no minimum.
        max_value: Maximum allowed value (inclusive), or None for no maximum.
        raise_error: If True, raise ValueRangeError. If False, return bool.

    Returns:
        True if valid, False if invalid (only when raise_error=False).

    Raises:
        ValueRangeError: If value is outside range and raise_error=True.

    Example:
        >>> validate_range(mach, "mach_number", min_value=0.0, max_value=1.0)
    """
    import openleaf.performance.numpy as np

    # Convert to array for consistent handling
    arr = np.asarray(value)

    # Check minimum
    if min_value is not None and np.any(arr < min_value):
        if raise_error:
            # Get first invalid value for error message
            if np.isscalar(value):
                bad_value = float(np.asarray(value).item())
            else:
                bad_value = float(np.asarray(arr[arr < min_value][0]).item())
            raise ValueRangeError(field, bad_value, min_value=min_value)
        return False

    # Check maximum
    if max_value is not None and np.any(arr > max_value):
        if raise_error:
            if np.isscalar(value):
                bad_value = float(np.asarray(value).item())
            else:
                bad_value = float(np.asarray(arr[arr > max_value][0]).item())
            raise ValueRangeError(field, bad_value, max_value=max_value)
        return False

    return True


def validate_positive(
    value: ArrayScalarLike,
    field: str,
    *,
    allow_zero: bool = False,
    raise_error: bool = True,
) -> bool:
    """Validate that a value is positive.

    Args:
        value: Value to validate (scalar or array).
        field: Field name for error message.
        allow_zero: If True, zero is allowed. If False, must be strictly positive.
        raise_error: If True, raise ValueRangeError. If False, return bool.

    Returns:
        True if valid, False if invalid (only when raise_error=False).

    Raises:
        ValueRangeError: If value is not positive and raise_error=True.

    Example:
        >>> validate_positive(mass, "aircraft_mass")
    """
    min_value = 0.0 if allow_zero else 1e-10  # Tiny positive number
    return validate_range(value, field, min_value=min_value, raise_error=raise_error)


def require_one_of(
    data: KeyedData,
    fields: list[str],
    method: str | None = None,
) -> tuple[str, Any]:
    """Require at least one of the specified fields.

    Args:
        data: KeyedData dictionary or similar object.
        fields: List of field names, at least one must be present.
        method: Optional method name for error context.

    Returns:
        Tuple of (field_name, value) for the first field found.

    Raises:
        MissingFieldError: If none of the fields are present.

    Example:
        >>> field, value = require_one_of(data, ["altitude", "air_pressure_ratio"])
    """
    for field in fields:
        try:
            value = data[field]
            return field, value
        except KeyError:
            continue

    # None found - raise error with first field name
    fields_str = ", ".join(f"'{f}'" for f in fields)
    raise MissingFieldError(
        fields[0],
        method=method,
        message=f"At least one of {fields_str} required (needed by {method})"
        if method
        else f"At least one of {fields_str} required",
    )
