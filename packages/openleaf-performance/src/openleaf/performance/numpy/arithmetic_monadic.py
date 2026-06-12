# SPDX-License-Identifier: EUPL-1.2 AND MIT
# Copyright (c) 2026 EUROCONTROL
#
# Portions derived from AeroSandbox (https://github.com/peterdsharpe/AeroSandbox)
# Copyright (c) 2019-2023 Peter Sharpe - MIT License (see LICENSES/MIT-aerosandbox.txt)

"""
Monadic (single argument) arithmetic operations.

Provides NumPy-like arithmetic functions (sum, mean, product, abs) that work
with NumPy arrays, CasADi symbolic types, and PyTorch tensors.
"""

from __future__ import annotations


__all__ = [
    "abs",
    "cbrt",
    "mean",
    "prod",
    "sqrt",
    "sum",
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


def sum(x: ArrayScalarLike, axis: int | None = None) -> ArrayScalarLike:
    """Sum of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis along which to sum. None means all elements.

    Returns:
        Sum of elements.

    Raises:
        ValueError: If axis is invalid for CasADi types.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        # Handle negative axis indexing for CasADi types (max 2D)
        if axis == -1:
            axis = 1 if len(x.shape) > 1 else 0
        elif axis == -2:
            axis = 0

        if axis == 0:
            return ca.sum1(x).T
        elif axis == 1:
            return ca.sum2(x)
        elif axis is None:
            return sum(sum(x, axis=0), axis=0)
        else:
            msg = "CasADi types can only be up to 2D, so axis must be None, 0, 1, -1, or -2."
            raise ValueError(msg)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.sum(x, axis=axis)


def mean(x: ArrayScalarLike, axis: int | None = None) -> ArrayScalarLike:
    """Compute the arithmetic mean along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute mean. None means all elements.

    Returns:
        Mean of elements.

    Raises:
        ValueError: If axis is invalid for CasADi types.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    """
    if is_casadi_type(x):
        # Handle negative axis indexing for CasADi types (max 2D)
        if axis == -1:
            axis = 1 if len(x.shape) > 1 else 0
        elif axis == -2:
            axis = 0

        if axis == 0:
            return sum(x, axis=0) / x.shape[0]
        elif axis == 1:
            return sum(x, axis=1) / x.shape[1]
        elif axis is None:
            # mean(x, axis=0) yields a column vector, and axis=1 on a column
            # is the identity — reduce along axis=0 again (cf. sum(axis=None)).
            return mean(mean(x, axis=0), axis=0)
        else:
            msg = "CasADi types can only be up to 2D, so axis must be None, 0, 1, -1, or -2."
            raise ValueError(msg)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.mean(x, axis=axis)


def abs(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate absolute value element-wise.

    Args:
        x: Input array or scalar.

    Returns:
        Absolute values of the input.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.abs.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        return ca.fabs(x)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.abs(x)


def prod(x: ArrayScalarLike, axis: int | None = None) -> ArrayScalarLike:
    """Return the product of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute product. None means all elements.

    Returns:
        Product of elements.

    Warning:
        The CasADi path computes exp(sum(log(x))) to stay differentiable, so
        it returns NaN if any element is negative and -inf/0-related values
        for zeros, unlike numpy.prod. Restrict CasADi inputs to positive
        values.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        return ca.exp(sum(ca.log(x), axis=axis))

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.prod(x, axis=axis)


def sqrt(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the non-negative square-root element-wise.

    Args:
        x: Input array or scalar (must be non-negative).

    Returns:
        Square root of the input, element-wise.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        return ca.sqrt(x)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x)
    return xp.sqrt(x)


def cbrt(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the cube root element-wise.

    Works with both positive and negative inputs.

    Args:
        x: Input array or scalar.

    Returns:
        Cube root of the input, element-wise.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        # CasADi pow doesn't handle negative bases with fractional exponents
        # Use: cbrt(x) = sign(x) * |x|^(1/3)
        return ca.sign(x) * ca.power(ca.fabs(x), 1.0 / 3.0)

    # Array API doesn't have cbrt; use sign * |x|^(1/3) for all backends
    xp = get_array_namespace(x)
    return xp.sign(x) * xp.pow(xp.abs(x), 1.0 / 3.0)
