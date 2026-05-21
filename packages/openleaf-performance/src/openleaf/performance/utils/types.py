"""Backward-compatible re-exports of shared types.

The canonical home of these types is ``openleaf.core.types``. This module
preserves the old import path so external consumers that imported from
``openleaf.performance.utils.types`` continue to work.
"""

from __future__ import annotations


__all__ = [
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
]

from openleaf.core.types import ArrayScalarLike
from openleaf.core.types import KeyedData
from openleaf.core.types import is_keyed_data
