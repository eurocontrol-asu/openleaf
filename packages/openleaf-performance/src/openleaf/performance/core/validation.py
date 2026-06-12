# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
Input validation for aircraft performance model methods.

Provides the @requires decorator for declarative input validation,
allowing model methods to specify their required input fields cleanly.

Example:
    >>> from openleaf.performance.core.validation import requires
    >>>
    >>> class MyModel:
    ...     @requires("altitude", "mach_number", "air_temperature")
    ...     def drag_coef(self, data):
    ...         # data is guaranteed to have required fields
    ...         return self._compute_drag(data)
"""

from __future__ import annotations


__all__ = [
    "get_required_fields",
    "requires",
    "StateSchema",
]

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict

from openleaf.performance.exceptions import MissingFieldError


F = TypeVar("F", bound=Callable)


class StateSchema(BaseModel):
    """Schema defining required inputs for a function.

    Immutable and hashable for use as function metadata.

    Attributes:
        required: Set of required field names.
    """

    model_config = ConfigDict(frozen=True)

    required: frozenset[str]

    def validate(self, data, method: str | None = None) -> None:
        """Validate that data contains all required fields.

        Args:
            data: KeyedData-like object to validate.
            method: Method name for error messages.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        # Check each required field by trying to access it
        # This works for dicts, DataFrames, TrajectoryData (with computed properties),
        # and other KeyedData objects
        missing = []
        for key in self.required:
            try:
                # Try to access the key - this works for both stored and computed values
                _ = data[key]
            except (KeyError, TypeError):
                # KeyError: key not found
                # TypeError: data doesn't support __getitem__
                missing.append(key)

        if missing:
            # Report first missing field for clearer error message
            first_missing = sorted(missing)[0]
            raise MissingFieldError(
                field=first_missing,
                method=method,
                message=f"Missing required field(s): {sorted(missing)}",
            )


def requires(*keys: str) -> Callable[[F], F]:
    """Decorator declaring required state keys for a method.

    Validates that data contains all required keys before calling
    the decorated method. Adds introspectable _schema attribute.

    Args:
        *keys: Required field names.

    Returns:
        Decorator function.

    Example:
        >>> @requires("altitude", "mach_number")
        ... def thrust_max(self, data):
        ...     alt = data["altitude"]
        ...     mach = data["mach_number"]
        ...     return self._compute_thrust(alt, mach)
    """
    schema = StateSchema(required=frozenset(keys))

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(self, data, **kwargs):
            schema.validate(data, method=fn.__name__)
            return fn(self, data, **kwargs)

        # Attach schema for introspection
        wrapper._schema = schema  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_required_fields(fn: Callable) -> frozenset[str]:
    """Get required fields from a decorated function.

    Args:
        fn: Function decorated with @requires.

    Returns:
        Frozenset of required field names, or empty set if not decorated.
    """
    schema = getattr(fn, "_schema", None)
    if isinstance(schema, StateSchema):
        return schema.required
    return frozenset()
