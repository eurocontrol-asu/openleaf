# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
Type definitions and protocols for the openleaf library.

Defines common type aliases and protocols used throughout the codebase for
type hints and runtime type checking.
"""

from __future__ import annotations


__all__ = [
    "ArrayExpression",
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
]

from typing import Any
from typing import Protocol
from typing import Self

import numpy.typing as npt
import pandas as pd


class ArrayExpression(Protocol):
    """Array value from a backend other than NumPy, used through its operators.

    Covers CasADi ``MX``/``SX``/``DM``, PyTorch tensors and JAX arrays. The
    dispatch layer in :mod:`openleaf.performance.numpy` picks the implementation
    from the runtime type, so the model code only needs the elementwise
    operators. Operands from different families (say, ``MX`` and ``SX``) do not
    combine at runtime; that is a property of the caller's inputs, which a
    single static type cannot express.
    """

    def __add__(self, other: Any, /) -> Self: ...
    def __radd__(self, other: Any, /) -> Self: ...
    def __sub__(self, other: Any, /) -> Self: ...
    def __rsub__(self, other: Any, /) -> Self: ...
    def __mul__(self, other: Any, /) -> Self: ...
    def __rmul__(self, other: Any, /) -> Self: ...
    def __truediv__(self, other: Any, /) -> Self: ...
    def __rtruediv__(self, other: Any, /) -> Self: ...
    def __pow__(self, other: Any, /) -> Self: ...
    def __rpow__(self, other: Any, /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __lt__(self, other: Any, /) -> Self: ...
    def __le__(self, other: Any, /) -> Self: ...
    def __gt__(self, other: Any, /) -> Self: ...
    def __ge__(self, other: Any, /) -> Self: ...


type ArrayScalarLike = float | int | npt.NDArray | pd.Series | ArrayExpression
"""Type alias for the scalar and array values accepted and returned by models."""


class KeyedData(Protocol):
    """Protocol for dict-like access to trajectory data.

    Any object implementing this protocol can be used where KeyedData is expected,
    allowing flexible data structure support (dicts, pandas DataFrames,
    pycontrails Flight objects, etc.).

    The ``attrs`` property is optional -- plain dicts are valid KeyedData.
    """

    def __getitem__(self, key: str) -> ArrayScalarLike:
        """Get value by key.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key.
        """
        ...

    def get[T](self, key: str, default: T = None) -> ArrayScalarLike | T:
        """Get value by key with optional default.

        Args:
            key: The key to retrieve.
            default: Default value if key is not found.

        Returns:
            The value associated with the key, or default if not found.
        """
        ...


def is_keyed_data(obj: Any) -> bool:
    """Runtime check for KeyedData protocol compliance.

    Checks for __getitem__ and get methods. The attrs property is optional
    at runtime - dicts are valid KeyedData even without attrs.

    Args:
        obj: Object to check.

    Returns:
        True if obj implements the KeyedData protocol, False otherwise.
    """
    return (
        hasattr(obj, "__getitem__")
        and callable(obj.__getitem__)
        and hasattr(obj, "get")
        and callable(obj.get)
    )
