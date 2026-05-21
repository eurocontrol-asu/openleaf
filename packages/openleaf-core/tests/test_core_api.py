"""Contract tests for the openleaf-core public API.

These imports define the public surface of openleaf-core. They fail before
the Phase-2 extraction (RED) and pass after (GREEN). Anything not imported
here is not part of the contract.
"""

from __future__ import annotations


def test_keyed_data_protocol_is_importable() -> None:
    """KeyedData is the structural protocol every model accepts as input."""
    from openleaf.core.types import KeyedData  # noqa: F401


def test_is_keyed_data_check_is_importable() -> None:
    """Runtime duck-type check for the KeyedData protocol."""
    from openleaf.core.types import is_keyed_data

    assert callable(is_keyed_data)
    assert is_keyed_data({"foo": 1})
    assert not is_keyed_data(42)


def test_array_scalar_like_alias_is_importable() -> None:
    """ArrayScalarLike is the cross-backend numeric annotation alias."""
    from openleaf.core.types import ArrayScalarLike  # noqa: F401


def test_base_exceptions_are_importable() -> None:
    """The base exception hierarchy lives in openleaf.core.exceptions."""
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

    # All inherit from OpenLeafError.
    for cls in (
        ConfigurationError,
        EnvelopeError,
        MissingFieldError,
        ModelDataError,
        ModelError,
        ModelNotFoundError,
        ParsingError,
        ValidationError,
        ValueRangeError,
    ):
        assert issubclass(cls, OpenLeafError), cls


def test_missing_field_error_carries_context() -> None:
    """MissingFieldError exposes ``field`` and ``method`` for diagnostics."""
    from openleaf.core.exceptions import MissingFieldError

    err = MissingFieldError(field="altitude", method="drag_coef")
    assert err.field == "altitude"
    assert err.method == "drag_coef"
    assert "altitude" in str(err)


def test_protocols_are_importable() -> None:
    """The four model Protocols live in openleaf.core.protocols.

    Sub-Protocols (Engine, Aerodynamics, Limits) are runtime-checkable.
    AircraftModelProtocol composes them via multiple inheritance and is
    deliberately NOT runtime-checkable (a Python typing limitation), but
    it must still be importable and structurally usable as a type hint.
    """
    from typing import Protocol

    from openleaf.core.protocols import AerodynamicsModelProtocol
    from openleaf.core.protocols import AircraftModelProtocol
    from openleaf.core.protocols import EngineModelProtocol
    from openleaf.core.protocols import LimitsModelProtocol

    # Every protocol has Protocol in its MRO.
    for proto in (
        AerodynamicsModelProtocol,
        AircraftModelProtocol,
        EngineModelProtocol,
        LimitsModelProtocol,
    ):
        assert Protocol in proto.__mro__, f"{proto.__name__} does not inherit from Protocol"

    # Sub-protocols are runtime-checkable.
    for proto in (
        AerodynamicsModelProtocol,
        EngineModelProtocol,
        LimitsModelProtocol,
    ):
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{proto.__name__} should be @runtime_checkable"
        )
