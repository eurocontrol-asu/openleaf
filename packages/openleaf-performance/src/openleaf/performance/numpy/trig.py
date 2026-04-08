"""
Trigonometric functions with degree/radian conversion.

Provides NumPy-like trigonometric functions that operate on angles in degrees
and trigonometric unit conversion functions between degrees and radians.
Works with both NumPy arrays and CasADi symbolic types.
"""

from __future__ import annotations


__all__ = [
    "arccosd",
    "arcsind",
    "arctan2d",
    "cosd",
    "degrees",
    "radians",
    "sind",
    "tand",
]

from typing import TYPE_CHECKING

from numpy import pi as _pi

from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_array_namespace
from openleaf.performance.numpy.backends import is_casadi_type


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike

_RAD_TO_DEG = 180.0 / _pi
_DEG_TO_RAD = _pi / 180.0


def _get_casadi():
    """Lazy import CasADi."""
    return Backend.CASADI.module


def degrees(x: ArrayScalarLike) -> ArrayScalarLike:
    """Convert angles from radians to degrees.

    Args:
        x: Angle in radians.

    Returns:
        Angle in degrees.
    """
    return x * _RAD_TO_DEG


def radians(x: ArrayScalarLike) -> ArrayScalarLike:
    """Convert angles from degrees to radians.

    Args:
        x: Angle in degrees.

    Returns:
        Angle in radians.
    """
    return x * _DEG_TO_RAD


def sind(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute sine of angle in degrees.

    Args:
        x: Angle in degrees.

    Returns:
        Sine of x.
    """
    rad = radians(x)
    if is_casadi_type(x):
        return _get_casadi().sin(rad)
    xp = get_array_namespace(x)
    return xp.sin(rad)


def cosd(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute cosine of angle in degrees.

    Args:
        x: Angle in degrees.

    Returns:
        Cosine of x.
    """
    rad = radians(x)
    if is_casadi_type(x):
        return _get_casadi().cos(rad)
    xp = get_array_namespace(x)
    return xp.cos(rad)


def tand(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute tangent of angle in degrees.

    Args:
        x: Angle in degrees.

    Returns:
        Tangent of x.
    """
    rad = radians(x)
    if is_casadi_type(x):
        return _get_casadi().tan(rad)
    xp = get_array_namespace(x)
    return xp.tan(rad)


def arcsind(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute arcsine, result in degrees.

    Args:
        x: Value to compute arcsine of (must be in [-1, 1]).

    Returns:
        Arcsine of x in degrees.
    """
    if is_casadi_type(x):
        return degrees(_get_casadi().asin(x))
    xp = get_array_namespace(x)
    return degrees(xp.asin(x))


def arccosd(x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute arccosine, result in degrees.

    Args:
        x: Value to compute arccosine of (must be in [-1, 1]).

    Returns:
        Arccosine of x in degrees.
    """
    if is_casadi_type(x):
        return degrees(_get_casadi().acos(x))
    xp = get_array_namespace(x)
    return degrees(xp.acos(x))


def arctan2d(y: ArrayScalarLike, x: ArrayScalarLike) -> ArrayScalarLike:
    """Compute arctan2, result in degrees.

    Args:
        y: Y-coordinate.
        x: X-coordinate.

    Returns:
        Angle in degrees of the point (x, y).
    """
    if is_casadi_type([y, x], recursive=True):
        return degrees(_get_casadi().atan2(y, x))
    xp = get_array_namespace(y, x)
    return degrees(xp.atan2(y, x))
