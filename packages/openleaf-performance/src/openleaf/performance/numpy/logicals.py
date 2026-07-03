# SPDX-License-Identifier: EUPL-1.2 AND MIT
# Copyright (c) 2026 EUROCONTROL
#
# Portions derived from AeroSandbox (https://github.com/peterdsharpe/AeroSandbox)
# Copyright (c) 2019-2023 Peter Sharpe - MIT License (see LICENSES/MIT-aerosandbox.txt)

"""
Logical operations and boolean functions.

Provides NumPy-like logical functions (logical_and, logical_or, logical_not, all, any)
that work with NumPy arrays, CasADi symbolic types, and PyTorch tensors.
"""

from __future__ import annotations


__all__ = [
    "all",
    "any",
    "clip",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "not_equal",
]

from typing import TYPE_CHECKING

from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_array_namespace
from openleaf.performance.numpy.backends import is_casadi_type


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike


def _get_casadi():
    """Lazy import CasADi."""
    return Backend.CASADI.module


def _coerce_torch(target, other):
    """Coerce *other* to a torch tensor matching *target*'s dtype/device."""
    torch = Backend.TORCH.module
    if not isinstance(other, torch.Tensor):
        other = torch.as_tensor(other, dtype=target.dtype, device=target.device)
    return other


def _coerce_jax(target, other):
    """Coerce *other* to a JAX array matching *target*'s dtype.

    Returns *other* unchanged if JAX is not installed.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return other
    if not isinstance(other, jax.Array):
        return jnp.asarray(other, dtype=target.dtype)
    return other


def _ensure_pair(x1, x2):
    """Convert scalars/numpy values to match the backend of the other argument.

    This prevents NumPy 2.0 ``__array_wrap__`` DeprecationWarnings that arise
    when a numpy ufunc is applied to a mixed-backend pair such as
    (torch_tensor, python_scalar) or (jax_array, numpy_scalar).
    """
    from openleaf.performance.numpy.backends import get_backend

    b1, b2 = get_backend(x1), get_backend(x2)

    if b1 == Backend.TORCH:
        return x1, _coerce_torch(x1, x2)
    if b2 == Backend.TORCH:
        return _coerce_torch(x2, x1), x2

    # JAX path (no-op if JAX is not installed)
    try:
        import jax
    except ImportError:
        return x1, x2

    if isinstance(x1, jax.Array):
        return x1, _coerce_jax(x1, x2)
    if isinstance(x2, jax.Array):
        return _coerce_jax(x2, x1), x2
    return x1, x2


def equal(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise equality comparison (x1 == x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 == x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 == x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.equal(x1, x2)


def not_equal(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise inequality comparison (x1 != x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 != x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 != x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.not_equal(x1, x2)


def greater(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise greater-than comparison (x1 > x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 > x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 > x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.greater(x1, x2)


def greater_equal(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise greater-than-or-equal comparison (x1 >= x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 >= x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 >= x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.greater_equal(x1, x2)


def less(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise less-than comparison (x1 < x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 < x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 < x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.less(x1, x2)


def less_equal(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise less-than-or-equal comparison (x1 <= x2).

    Args:
        x1: First input.
        x2: Second input.

    Returns:
        Boolean array where x1 <= x2.
    """
    if is_casadi_type([x1, x2], recursive=True):
        return x1 <= x2
    x1, x2 = _ensure_pair(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.less_equal(x1, x2)


def clip(
    x: ArrayScalarLike,
    min: ArrayScalarLike | None = None,
    max: ArrayScalarLike | None = None,
) -> ArrayScalarLike:
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges. For example, if an interval of [0, 1]
    is specified, values smaller than 0 become 0, and values
    larger than 1 become 1.

    Args:
        x: Array containing elements to clip.
        min: Minimum value. If None, clipping is not performed on lower bound.
        max: Maximum value. If None, clipping is not performed on upper bound.

    Returns:
        Array with the elements of x, but where values < min are replaced
        with min, and those > max with max.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    if is_casadi_type([x, min, max], recursive=True):
        ca = _get_casadi()
        result = x
        if min is not None:
            result = ca.fmax(result, min)
        if max is not None:
            result = ca.fmin(result, max)
        return result

    # Promote mixed backends (e.g. numpy x with torch bounds) the same way
    # where() does, instead of deriving the namespace from x alone.
    from openleaf.performance.numpy.conditionals import _promote_to_common_backend

    promotable = [a for a in (x, min, max) if a is not None]
    promoted = iter(_promote_to_common_backend(*promotable))
    x = next(promoted)
    min = next(promoted) if min is not None else None
    max = next(promoted) if max is not None else None

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(*(a for a in (x, min, max) if a is not None))
    return xp.clip(x, min, max)


def logical_and(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Compute element-wise truth value of x1 AND x2.

    Args:
        x1: First input array.
        x2: Second input array.

    Returns:
        Element-wise logical AND result.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
    """
    if is_casadi_type([x1, x2], recursive=True):
        return _get_casadi().logic_and(x1, x2)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x1, x2)
    return xp.logical_and(x1, x2)


def logical_or(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """Compute element-wise truth value of x1 OR x2.

    Args:
        x1: First input array.
        x2: Second input array.

    Returns:
        Element-wise logical OR result.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html
    """
    if is_casadi_type([x1, x2], recursive=True):
        return _get_casadi().logic_or(x1, x2)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x1, x2)
    return xp.logical_or(x1, x2)


def logical_not(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute element-wise truth value of NOT x.

    Args:
        x: Input array.

    Returns:
        Element-wise logical NOT result.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
    """
    if is_casadi_type(x, recursive=False):
        return _get_casadi().logic_not(x)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.logical_not(x)


def all(a: ArrayScalarLike, axis: int | None = None) -> ArrayScalarLike:
    """Test whether all array elements along a given axis evaluate to True.

    Args:
        a: Input array or CasADi expression.
        axis: Axis or axes along which to perform the logical AND reduction.
              None means all elements.

    Returns:
        Boolean result (or array of results if axis is specified).

    Raises:
        NotImplementedError: If axis is specified for CasADi types.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.all.html
    """
    if is_casadi_type(a, recursive=False):
        ca = _get_casadi()
        # CasADi doesn't support axis parameter, only full reduction
        if axis is not None:
            msg = "axis parameter not supported for CasADi types in all()"
            raise NotImplementedError(msg)
        try:
            return ca.logic_all(a)
        except NotImplementedError:
            return False

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.all(a, axis=axis)


def any(a: ArrayScalarLike, axis: int | None = None) -> ArrayScalarLike:
    """Test whether any array element along a given axis evaluates to True.

    Args:
        a: Input array or CasADi expression.
        axis: Axis or axes along which to perform the logical OR reduction.
              None means all elements.

    Returns:
        Boolean result (or array of results if axis is specified).

    Raises:
        NotImplementedError: If axis is specified for CasADi types.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.any.html
    """
    if is_casadi_type(a, recursive=False):
        ca = _get_casadi()
        # CasADi doesn't support axis parameter, only full reduction
        if axis is not None:
            msg = "axis parameter not supported for CasADi types in any()"
            raise NotImplementedError(msg)
        try:
            return ca.logic_any(a)
        except NotImplementedError:
            return False

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.any(a, axis=axis)
