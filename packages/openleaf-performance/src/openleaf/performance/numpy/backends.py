"""
Backend detection and registration for multi-backend array support.

Uses the Array API standard (NEP 47) via array-api-compat for seamless
interoperability between NumPy, PyTorch, CuPy, and JAX. CasADi requires
special handling as a symbolic computation backend.

Usage:
    from openleaf.performance.numpy.backends import get_array_namespace, Backend, is_casadi_type

    # Check for CasADi first (symbolic computation needs special dispatch)
    if is_casadi_type(x):
        # CasADi-specific code
        ...
    else:
        # Use Array API for NumPy/PyTorch/CuPy/JAX
        xp = get_array_namespace(x)
        result = xp.sqrt(x)
"""

from __future__ import annotations

import importlib.util
import typing
from enum import Enum
from enum import auto
from typing import Any


if typing.TYPE_CHECKING:
    from collections.abc import Callable

import array_api_compat
import numpy as np


__all__ = [
    "Backend",
    "get_array_namespace",
    "get_backend",
    "is_casadi_type",
    "is_torch_type",
    "require_backend",
]


class Backend(Enum):
    """Supported array backends with lazy availability checking."""

    NUMPY = auto()
    CASADI = auto()
    TORCH = auto()

    @property
    def available(self) -> bool:
        """Check if this backend is installed."""
        return _check_available(self)

    @available.setter
    def available(self, value: bool) -> None:
        """Override availability (useful for testing)."""
        _BACKEND_AVAILABLE[self] = value

    @property
    def module(self) -> Any:
        """Get the backend module (imports lazily on first access)."""
        return _get_module(self)


# Lazy availability cache (checked once on first access)
_BACKEND_AVAILABLE: dict[Backend, bool] = {}
_BACKEND_MODULES: dict[Backend, Any] = {}

# Mapping from backend to import name
_IMPORT_NAMES: dict[Backend, str] = {
    Backend.NUMPY: "numpy",
    Backend.CASADI: "casadi",
    Backend.TORCH: "torch",
}


def _check_available(backend: Backend) -> bool:
    """Check if a backend is available (cached)."""
    if backend not in _BACKEND_AVAILABLE:
        import_name = _IMPORT_NAMES[backend]
        _BACKEND_AVAILABLE[backend] = importlib.util.find_spec(import_name) is not None
    return _BACKEND_AVAILABLE[backend]


def _get_module(backend: Backend) -> Any:
    """Get backend module with lazy import."""
    if backend not in _BACKEND_MODULES:
        if not _check_available(backend):
            return None
        import_name = _IMPORT_NAMES[backend]
        _BACKEND_MODULES[backend] = importlib.import_module(import_name)
    return _BACKEND_MODULES[backend]


# Initialize availability on module load (fast, doesn't import)
for _backend in Backend:
    _check_available(_backend)


def get_array_namespace(*arrays: Any) -> Any:
    """Get the Array API namespace for the given arrays.

    Uses array-api-compat to automatically detect and return the correct
    namespace for NumPy, PyTorch, CuPy, or JAX arrays.

    Args:
        *arrays: One or more arrays to get the namespace for.

    Returns:
        The array namespace module (numpy, torch, cupy, or jax.numpy).
        Returns numpy for Python scalars (int, float, complex).

    Note:
        CasADi is NOT handled by this function - check is_casadi_type() first.

    Example:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3])
        >>> xp = get_array_namespace(x)
        >>> xp.sqrt(x)
        array([1.        , 1.41421356, 1.73205081])
    """
    # Filter out Python scalars and non-array types (e.g., Enums)
    # array_namespace requires at least one proper array type
    non_scalars = []
    for a in arrays:
        # Skip Python scalars
        if isinstance(a, (int, float, complex, bool, type(None))):
            continue
        # Skip NumPy scalar types (e.g., numpy.bool_, numpy.float64).
        # These arise from operations like np.equal(enum, enum) and can
        # cause spurious namespace conflicts with JAX/PyTorch arrays.
        if isinstance(a, np.generic):
            continue
        # Skip Enum types (e.g., FlightPhase, AerodynamicConfig)
        if isinstance(a, Enum):
            continue
        # Skip strings
        if isinstance(a, str):
            continue
        non_scalars.append(a)

    if not non_scalars:
        # All inputs are scalars/enums - default to NumPy
        return Backend.NUMPY.module
    return array_api_compat.array_namespace(*non_scalars)


def get_backend(obj: Any) -> Backend:
    """Detect which backend an object belongs to.

    Args:
        obj: Array-like object to check.

    Returns:
        The Backend enum value for this object.

    Note:
        Returns NUMPY for scalars and unknown types (safest default).
    """
    obj_type = type(obj)

    # Check CasADi first (symbolic types need special handling)
    if Backend.CASADI.available:
        ca = Backend.CASADI.module
        if obj_type in (ca.MX, ca.DM, ca.SX):
            return Backend.CASADI

    # Check PyTorch
    if Backend.TORCH.available:
        torch = Backend.TORCH.module
        if isinstance(obj, torch.Tensor):
            return Backend.TORCH

    # Default to NumPy (handles ndarray, scalars, lists, etc.)
    return Backend.NUMPY


def require_backend(backend: Backend, feature: str = "This feature") -> None:
    """Raise ImportError if a backend is not available.

    Args:
        backend: The required backend.
        feature: Description of the feature requiring this backend.

    Raises:
        ImportError: If the backend is not installed.
    """
    if not backend.available:
        install_name = _IMPORT_NAMES[backend]
        extras_name = install_name  # Same for our extras
        raise ImportError(
            f"{feature} requires {install_name}. "
            f"Install with: pip install openleaf-performance[{extras_name}]"
        )


_SCALAR_TYPES: frozenset[type] = frozenset(
    {float, int, complex, bool, str, range, type(None), bytes, bytearray, memoryview}
)


def _check_container(obj: Any, checker: Callable[..., bool], *, np: Any) -> bool:
    """Check if a container holds elements satisfying *checker*.

    Handles object-dtype numpy arrays (including 0-d), list, tuple, set, and
    dict.  Returns ``False`` for non-container types so the caller can fall
    through.
    """
    t = type(obj)

    if t == np.ndarray and obj.dtype == "O":
        if obj.ndim == 0:
            return checker(obj.item(), recursive=True)
        return any(checker(element, recursive=True) for element in obj)

    if isinstance(obj, (list, tuple, set)):
        return any(checker(element, recursive=True) for element in obj)

    if isinstance(obj, dict):
        return any(checker(v, recursive=True) for v in obj.values())

    return False


def is_casadi_type(obj: Any, recursive: bool = True) -> bool:
    """Check if an object is or contains a CasADi data type.

    This is the main dispatch check - returns True if the object is a CasADi
    type (MX, DM, SX) or, if recursive=True, if any nested element is a
    CasADi type.

    Args:
        obj: The object to evaluate.
        recursive: If True, recursively check container types (list, tuple,
            dict, set) for nested CasADi types. Default is True.

    Returns:
        True if the object is (or contains) a CasADi type, False otherwise.
        Always returns False if CasADi is not installed.
    """
    if not Backend.CASADI.available:
        return False

    ca = Backend.CASADI.module
    np = Backend.NUMPY.module
    t = type(obj)

    # NumPy arrays cannot contain CasADi types (unless object dtype)
    if t == np.ndarray and obj.dtype != "O":
        return False

    # Skip Python scalar types
    if t in _SCALAR_TYPES:
        return False

    # Direct CasADi type check
    if t in (ca.MX, ca.DM, ca.SX):
        return True

    # Recursive check for containers
    if recursive:
        return _check_container(obj, is_casadi_type, np=np)

    return False


def is_torch_type(obj: Any, recursive: bool = True) -> bool:
    """Check if an object is or contains a PyTorch tensor.

    Args:
        obj: The object to evaluate.
        recursive: If True, recursively check container types for nested
            PyTorch tensors. Default is True.

    Returns:
        True if the object is (or contains) a PyTorch tensor, False otherwise.
        Always returns False if PyTorch is not installed.
    """
    if not Backend.TORCH.available:
        return False

    torch = Backend.TORCH.module
    np = Backend.NUMPY.module
    t = type(obj)

    # NumPy arrays cannot contain PyTorch tensors
    if t == np.ndarray:
        return False

    # Skip Python scalar types
    if t in _SCALAR_TYPES:
        return False

    # Direct PyTorch tensor check
    if isinstance(obj, torch.Tensor):
        return True

    # Recursive check for containers
    if recursive:
        return _check_container(obj, is_torch_type, np=np)

    return False
