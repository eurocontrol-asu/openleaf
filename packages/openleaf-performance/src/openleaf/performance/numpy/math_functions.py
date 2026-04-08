"""
Mathematical functions with multi-backend support.

Provides NumPy-like mathematical functions (exp, log, power, sign, etc.)
that work with NumPy arrays, CasADi symbolic types, and PyTorch tensors.
"""

from __future__ import annotations


__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "log10",
    "log2",
    "power",
    "sign",
    "sin",
    "sinh",
    "tan",
    "tanh",
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


# --- Exponential and Logarithmic Functions ---


def exp(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the exponential of all elements in the input array.

    Args:
        x: Input array or scalar.

    Returns:
        Element-wise exponential of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.exp.html
    """
    if is_casadi_type(x):
        return _get_casadi().exp(x)
    xp = get_array_namespace(x)
    return xp.exp(x)


def log(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the natural logarithm of all elements in the input array.

    Args:
        x: Input array or scalar (must be positive).

    Returns:
        Element-wise natural logarithm of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.log.html
    """
    if is_casadi_type(x):
        return _get_casadi().log(x)
    xp = get_array_namespace(x)
    return xp.log(x)


def log10(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the base-10 logarithm of all elements in the input array.

    Args:
        x: Input array or scalar (must be positive).

    Returns:
        Element-wise base-10 logarithm of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.log10.html
    """
    if is_casadi_type(x):
        return _get_casadi().log10(x)
    xp = get_array_namespace(x)
    return xp.log10(x)


def log2(x: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate the base-2 logarithm of all elements in the input array.

    Args:
        x: Input array or scalar (must be positive).

    Returns:
        Element-wise base-2 logarithm of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.log2.html
    """
    if is_casadi_type(x):
        ca = _get_casadi()
        # CasADi doesn't have log2, use change of base formula
        return ca.log(x) / ca.log(2.0)
    xp = get_array_namespace(x)
    return xp.log2(x)


def power(x1: ArrayScalarLike, x2: ArrayScalarLike) -> ArrayScalarLike:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x1: Base array.
        x2: Exponent array.

    Returns:
        Element-wise x1 raised to the power x2.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.power.html
    """
    if is_casadi_type([x1, x2], recursive=True):
        return _get_casadi().power(x1, x2)
    xp = get_array_namespace(x1, x2)
    return xp.pow(x1, x2)


def sign(x: ArrayScalarLike) -> ArrayScalarLike:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x: Input array or scalar.

    Returns:
        -1 for negative, 0 for zero, 1 for positive values.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.sign.html
    """
    if is_casadi_type(x):
        return _get_casadi().sign(x)
    xp = get_array_namespace(x)
    return xp.sign(x)


# --- Trigonometric Functions ---


def sin(x: ArrayScalarLike) -> ArrayScalarLike:
    """Trigonometric sine, element-wise.

    Args:
        x: Angle in radians.

    Returns:
        Sine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.sin.html
    """
    if is_casadi_type(x):
        return _get_casadi().sin(x)
    xp = get_array_namespace(x)
    return xp.sin(x)


def cos(x: ArrayScalarLike) -> ArrayScalarLike:
    """Trigonometric cosine, element-wise.

    Args:
        x: Angle in radians.

    Returns:
        Cosine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.cos.html
    """
    if is_casadi_type(x):
        return _get_casadi().cos(x)
    xp = get_array_namespace(x)
    return xp.cos(x)


def tan(x: ArrayScalarLike) -> ArrayScalarLike:
    """Trigonometric tangent, element-wise.

    Args:
        x: Angle in radians.

    Returns:
        Tangent of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.tan.html
    """
    if is_casadi_type(x):
        return _get_casadi().tan(x)
    xp = get_array_namespace(x)
    return xp.tan(x)


def arcsin(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse sine, element-wise.

    Args:
        x: Value in range [-1, 1].

    Returns:
        Inverse sine of each element of x, in radians.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html
    """
    if is_casadi_type(x):
        return _get_casadi().asin(x)
    xp = get_array_namespace(x)
    return xp.asin(x)


def arccos(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse cosine, element-wise.

    Args:
        x: Value in range [-1, 1].

    Returns:
        Inverse cosine of each element of x, in radians.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
    """
    if is_casadi_type(x):
        return _get_casadi().acos(x)
    xp = get_array_namespace(x)
    return xp.acos(x)


def arctan(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse tangent, element-wise.

    Args:
        x: Input array.

    Returns:
        Inverse tangent of each element of x, in radians.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arctan.html
    """
    if is_casadi_type(x):
        return _get_casadi().atan(x)
    xp = get_array_namespace(x)
    return xp.atan(x)


def arctan2(y: ArrayScalarLike, x: ArrayScalarLike) -> ArrayScalarLike:
    """Element-wise arc tangent of y/x choosing the quadrant correctly.

    Args:
        y: Y-coordinate.
        x: X-coordinate.

    Returns:
        Angle in radians in [-pi, pi].

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    """
    if is_casadi_type([y, x], recursive=True):
        return _get_casadi().atan2(y, x)
    xp = get_array_namespace(y, x)
    return xp.atan2(y, x)


# --- Hyperbolic Functions ---


def sinh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Hyperbolic sine, element-wise.

    Args:
        x: Input array.

    Returns:
        Hyperbolic sine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.sinh.html
    """
    if is_casadi_type(x):
        return _get_casadi().sinh(x)
    xp = get_array_namespace(x)
    return xp.sinh(x)


def cosh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Hyperbolic cosine, element-wise.

    Args:
        x: Input array.

    Returns:
        Hyperbolic cosine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.cosh.html
    """
    if is_casadi_type(x):
        return _get_casadi().cosh(x)
    xp = get_array_namespace(x)
    return xp.cosh(x)


def tanh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Hyperbolic tangent, element-wise.

    Args:
        x: Input array.

    Returns:
        Hyperbolic tangent of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.tanh.html
    """
    if is_casadi_type(x):
        return _get_casadi().tanh(x)
    xp = get_array_namespace(x)
    return xp.tanh(x)


def arcsinh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse hyperbolic sine, element-wise.

    Args:
        x: Input array.

    Returns:
        Inverse hyperbolic sine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html
    """
    if is_casadi_type(x):
        return _get_casadi().asinh(x)
    xp = get_array_namespace(x)
    return xp.asinh(x)


def arccosh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x: Input array (must be >= 1).

    Returns:
        Inverse hyperbolic cosine of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html
    """
    if is_casadi_type(x):
        return _get_casadi().acosh(x)
    xp = get_array_namespace(x)
    return xp.acosh(x)


def arctanh(x: ArrayScalarLike) -> ArrayScalarLike:
    """Inverse hyperbolic tangent, element-wise.

    Args:
        x: Input array (must be in (-1, 1)).

    Returns:
        Inverse hyperbolic tangent of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html
    """
    if is_casadi_type(x):
        return _get_casadi().atanh(x)
    xp = get_array_namespace(x)
    return xp.atanh(x)


# --- Rounding Functions ---


def floor(x: ArrayScalarLike) -> ArrayScalarLike:
    """Return the floor of the input, element-wise.

    Note: Not differentiable - avoid in optimization objectives.

    Args:
        x: Input array.

    Returns:
        Floor of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.floor.html
    """
    if is_casadi_type(x):
        return _get_casadi().floor(x)
    xp = get_array_namespace(x)
    return xp.floor(x)


def ceil(x: ArrayScalarLike) -> ArrayScalarLike:
    """Return the ceiling of the input, element-wise.

    Note: Not differentiable - avoid in optimization objectives.

    Args:
        x: Input array.

    Returns:
        Ceiling of each element of x.

    See Also:
        https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
    """
    if is_casadi_type(x):
        return _get_casadi().ceil(x)
    xp = get_array_namespace(x)
    return xp.ceil(x)
