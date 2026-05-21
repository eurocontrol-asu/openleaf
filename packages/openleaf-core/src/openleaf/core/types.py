"""Shared type aliases and structural protocols for OpenLeaf.

Lives in ``openleaf-core`` so that downstream packages (performance,
future emissions / trajectory) can depend on a common vocabulary without
pulling each other in.

``ArrayScalarLike`` is intentionally evaluated as ``Any`` at runtime: the
full cross-backend union (NumPy, pandas, CasADi) is only visible to static
type checkers under ``TYPE_CHECKING``. This keeps openleaf-core free of
heavy optional dependencies (CasADi) while still giving callers a precise
annotation when those packages ARE installed.
"""

from __future__ import annotations


__all__ = [
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
]

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol


if TYPE_CHECKING:
    import casadi as ca
    import numpy as _np
    import numpy.typing as npt
    import pandas as pd

    # Static type checkers see the full cross-backend union.
    type ArrayScalarLike = (
        float | int | npt.NDArray[_np.float64] | pd.Series | ca.MX | ca.SX | ca.DM
    )
else:
    # At runtime, ArrayScalarLike is a structural alias — we don't import
    # CasADi (heavy SWIG dep) just to define an annotation.
    ArrayScalarLike = Any


class KeyedData(Protocol):
    """Protocol for dict-like access to trajectory or flight-state data.

    Any object implementing ``__getitem__`` and ``get`` satisfies the
    protocol: plain ``dict``, ``pandas.DataFrame``, ``pycontrails.Flight``,
    custom containers, etc. The ``attrs`` property is intentionally NOT
    part of the contract — a plain dict is valid KeyedData.
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

    Args:
        obj: Object to check.

    Returns:
        True if obj implements both ``__getitem__`` and ``get``.
    """
    return (
        hasattr(obj, "__getitem__")
        and callable(obj.__getitem__)
        and hasattr(obj, "get")
        and callable(obj.get)
    )
