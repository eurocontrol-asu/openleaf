"""
Type definitions and protocols for the openleaf library.

Defines common type aliases and protocols used throughout the codebase for
type hints and runtime type checking.
"""

from __future__ import annotations


__all__ = [
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
]

from typing import Any
from typing import Protocol

import casadi as ca
import numpy.typing as npt
import pandas as pd


# Extended ArrayScalarLike to include object arrays (like string arrays)
type ArrayScalarLike = float | int | npt.NDArray | pd.Series | ca.MX | ca.SX | ca.DM
"""Type alias for array or scalar-like objects supporting NumPy/CasADi operations."""


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

    def get(self, key: str, default: ArrayScalarLike = None) -> ArrayScalarLike:
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
