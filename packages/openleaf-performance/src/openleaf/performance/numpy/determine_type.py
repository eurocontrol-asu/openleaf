"""
Type detection utilities for handling different array backends.

Provides functions to detect and handle different numeric types including
NumPy arrays, CasADi types, PyTorch tensors, and Python scalars.

Note:
    This module re-exports functions from backends.py for backward compatibility.
    New code should import directly from backends.py.
"""

from __future__ import annotations


__all__ = [
    "is_casadi_type",
    "is_iterable",
    "is_torch_type",
]

# Re-export from backends for backward compatibility
from openleaf.performance.numpy.backends import is_casadi_type
from openleaf.performance.numpy.backends import is_torch_type


def is_iterable(x) -> bool:
    """Check if an object is iterable.

    Args:
        x: Object to test for iterability.

    Returns:
        Boolean indicating whether the object is iterable.
    """
    try:
        iter(x)
        return True
    except TypeError:
        return False
