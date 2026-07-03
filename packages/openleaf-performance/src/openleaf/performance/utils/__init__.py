# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Utility functions and type definitions for OpenLeaf.

This module provides:
- Type definitions for array-like and keyed data structures
- Input validation helpers for KeyedData
"""

from __future__ import annotations

from openleaf.performance.utils.types import ArrayScalarLike
from openleaf.performance.utils.types import KeyedData
from openleaf.performance.utils.types import is_keyed_data
from openleaf.performance.utils.validation import get_optional
from openleaf.performance.utils.validation import get_required
from openleaf.performance.utils.validation import require_one_of
from openleaf.performance.utils.validation import validate_positive
from openleaf.performance.utils.validation import validate_range


__all__ = [
    # Types
    "ArrayScalarLike",
    "KeyedData",
    "is_keyed_data",
    # Validation
    "get_optional",
    "get_required",
    "require_one_of",
    "validate_positive",
    "validate_range",
]
