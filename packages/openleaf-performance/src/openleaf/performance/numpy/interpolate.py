"""
Interpolation functions for 1D and N-D data.

Provides NumPy/SciPy-like interpolation functions that work with both
NumPy arrays and CasADi symbolic types.
"""

from __future__ import annotations


__all__ = [
    "interp",
    "interpn",
    "is_data_structured",
]

from typing import TYPE_CHECKING
from typing import Literal

import numpy as _onp
from scipy import interpolate as _interpolate


if TYPE_CHECKING:
    import casadi as _cas
else:
    try:
        import casadi as _cas
    except ImportError:  # casadi is an optional backend: openleaf-performance[casadi]
        _cas = None

from openleaf.performance.numpy.array import array
from openleaf.performance.numpy.conditionals import where
from openleaf.performance.numpy.determine_type import is_casadi_type
from openleaf.performance.numpy.logicals import all
from openleaf.performance.numpy.logicals import any
from openleaf.performance.numpy.logicals import logical_or


def _require_casadi() -> None:
    """Raise a helpful error when an operation needs the optional casadi backend."""
    if _cas is None:
        raise ImportError(
            "This operation requires the optional dependency 'casadi' "
            "(needed for CasADi symbolic inputs and for interpn(method='bspline')). "
            "Install it with: pip install openleaf-performance[casadi]"
        )


def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation, analogous to numpy.interp().

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp),
    evaluated at x.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.interp.html

    Specific notes: xp is assumed to be sorted.
    """
    if not is_casadi_type([x, xp, fp], recursive=True):
        return _onp.interp(x=x, xp=xp, fp=fp, left=left, right=right, period=period)

    else:
        _require_casadi()

        ### Handle period argument
        if period is not None:
            # xp's range can only be validated when it holds numeric values; a
            # fully symbolic xp cannot be inspected at graph-construction time.
            xp_is_symbolic = is_casadi_type(xp, recursive=False) and not isinstance(xp, _cas.DM)
            if not xp_is_symbolic and any(logical_or(xp < 0, xp > period)):
                raise NotImplementedError(
                    "Haven't yet implemented handling for if xp is outside the period."
                )  # Not easy to implement because casadi doesn't have a sort feature.

            # Wrap x into [0, period) like numpy.interp. A single fmod keeps the
            # dividend's sign, which would send negative x below xp[0] instead
            # of wrapping it to the top of the period.
            x = _cas.fmod(_cas.fmod(x, period) + period, period)

        ### Make sure x isn't an int
        if isinstance(x, int):
            x = float(x)

        ### Make sure that x is an iterable
        try:
            x[0]
        except TypeError:
            x = array([x], dtype=float)

        ### Make sure xp is an iterable
        xp = array(xp, dtype=float)

        ### Do the interpolation
        if is_casadi_type([x, xp], recursive=True):
            grid = [xp.shape[0]]  # size of grid, list is used since can be multi-dimensional
            cas_interp = _cas.interpolant("cs_interp", "linear", grid, 1, {"inline": True})
            f = cas_interp(x, xp, fp)
        else:
            f = _cas.interp1d(xp, fp, x)

        ### Handle left/right
        if left is not None:
            f = where(x < xp[0], left, f)
        if right is not None:
            f = where(x > xp[-1], right, f)

        ### Return
        return f


def is_data_structured(
    x_data_coordinates: tuple[_onp.ndarray], y_data_structured: _onp.ndarray
) -> bool:
    """
    Determines if the shapes of a given dataset are consistent with "structured" (i.e. gridded) data.

    For this to evaluate True, the inputs should be:

        x_data_coordinates: A tuple or list of 1D ndarrays that represent coordinates along each axis of a N-dimensional hypercube.

        y_data_structured: The values of some scalar defined on that N-dimensional hypercube, expressed as an
        N-dimesional array. In other words, y_data_structured is evaluated at `np.meshgrid(*x_data_coordinates,
        indexing="ij")`.

    Returns: Boolean of whether the above description is true.
    """
    try:
        for coordinates in x_data_coordinates:
            if len(coordinates.shape) != 1:
                return False

        implied_y_data_shape = tuple(len(coordinates) for coordinates in x_data_coordinates)
        if y_data_structured.shape != implied_y_data_shape:
            return False
    except TypeError:  # if x_data_coordinates is not iterable, for instance
        return False
    except AttributeError:  # if y_data_structured has no shape, for instance
        return False

    return True


def _validate_interpn_inputs(
    points: tuple[_onp.ndarray],
    values: _onp.ndarray,
) -> tuple[int, ...]:
    """Validate inputs for interpn and return the implied values shape.

    Checks that points/values are not CasADi types, that each axis in
    points is 1D, and that values has the correct shape.

    Returns:
        The implied values shape derived from points.

    Raises:
        TypeError: If points or values contain CasADi types.
        ValueError: If points axes are not 1D or values shape is wrong.
    """
    if is_casadi_type([points, values], recursive=True):
        raise TypeError("The underlying dataset (points, values) must consist of NumPy arrays.")

    for points_axis in points:
        points_axis = array(points_axis)
        if not len(points_axis.shape) == 1:
            raise ValueError(
                "`points` must consist of a tuple of 1D ndarrays defining the coordinates of each axis."
            )

    implied_values_shape = tuple(len(points_axis) for points_axis in points)
    if not values.shape == implied_values_shape:
        raise ValueError(
            f"""
        The shape of `values` should be {implied_values_shape}.
        """
        )

    return implied_values_shape


def _prepare_xi_casadi(
    xi: _onp.ndarray,
    n_dimensions: int,
) -> _onp.ndarray:
    """Prepare xi for CasADi interpolation.

    Promotes scalars to arrays, reshapes 1D to 2D, and transposes
    so that xi.shape is (n_points, n_dimensions).

    Returns:
        The prepared xi array with shape (n_points, n_dimensions).

    Raises:
        ValueError: If xi cannot be reshaped to (n_points, n_dimensions).
    """
    if isinstance(xi, (int, float)):
        xi = array([xi])

    if not is_casadi_type(xi, recursive=False) and len(xi.shape) != 2:
        xi = _onp.reshape(xi, (-1, 1))

    if not len(xi.shape) == 2:
        raise ValueError("`xi` must have the shape (n_points, n_dimensions)!")

    if n_dimensions not in xi.shape:
        raise ValueError("`xi` must have the shape (n_points, n_dimensions)!")

    if xi.shape[1] != n_dimensions:
        xi = xi.T
        if xi.shape[1] != n_dimensions:
            msg = f"Expected {n_dimensions} dimensions, got {xi.shape[1]}"
            raise ValueError(msg)

    return xi


def _clamp_bounds(
    xi: _onp.ndarray,
    axis_min: list[float],
    axis_max: list[float],
    n_dimensions: int,
) -> _onp.ndarray:
    """Clamp xi to the grid domain (fill_value=None behavior).

    Operates on a copy: xi may be (a view of) the caller's own array, and the
    per-axis assignments below would otherwise write through into it.
    """
    if is_casadi_type(xi, recursive=False):
        xi = type(xi)(xi)
    else:
        xi = _onp.array(xi, dtype=float)
    for axis in range(n_dimensions):
        xi[:, axis] = where(
            xi[:, axis] > axis_max[axis],
            axis_max[axis],
            xi[:, axis],
        )
        xi[:, axis] = where(
            xi[:, axis] < axis_min[axis],
            axis_min[axis],
            xi[:, axis],
        )
    return xi


def _check_bounds(
    xi: _onp.ndarray,
    axis_min: list[float],
    axis_max: list[float],
    n_dimensions: int,
) -> None:
    """Check that xi is within grid bounds.

    Raises:
        ValueError: If xi is MX type (can't check symbolically) or
            if any point is out of bounds.
    """
    if isinstance(xi, _cas.MX):
        raise ValueError("Can't have the `bounds_error` flag as True if `xi` is of cas.MX type.")

    for axis in range(n_dimensions):
        if any(
            logical_or(
                xi[:, axis] > axis_max[axis],
                xi[:, axis] < axis_min[axis],
            )
        ):
            raise ValueError(f"One of the requested xi is out of bounds in dimension {axis}")


def _apply_fill_value(
    fi: _onp.ndarray,
    xi: _onp.ndarray,
    axis_min: list[float],
    axis_max: list[float],
    n_dimensions: int,
    fill_value: float,
) -> _onp.ndarray:
    """Replace interpolation results at out-of-bounds positions with *fill_value*."""
    for axis in range(n_dimensions):
        fi = where(xi[:, axis] > axis_max[axis], fill_value, fi)
        fi = where(xi[:, axis] < axis_min[axis], fill_value, fi)
    return fi


def _convert_casadi_result(fi: _onp.ndarray) -> _onp.ndarray | float:
    """Convert a CasADi DM result to a Python float or numpy array."""
    if isinstance(fi, _cas.DM):
        if fi.shape == (1, 1):
            return float(fi)
        return _onp.array(fi, dtype=float).reshape(-1)
    return fi


def _interpn_casadi(
    points: tuple[_onp.ndarray],
    values: _onp.ndarray,
    xi: _onp.ndarray,
    method: Literal["linear", "bspline"],
    bounds_error: bool,
    fill_value: float | None,
) -> _onp.ndarray:
    """CasADi code path for interpn (linear and bspline methods)."""
    _require_casadi()

    n_dimensions = len(points)
    xi = _prepare_xi_casadi(xi, n_dimensions)

    axis_min = [_onp.min(axis_values) for axis_values in points]
    axis_max = [_onp.max(axis_values) for axis_values in points]

    # Bounds are checked against the points as requested, BEFORE any clamping:
    # clamping first would make bounds_error=True unable to ever raise.
    if bounds_error:
        _check_bounds(xi, axis_min, axis_max, n_dimensions)

    # Workaround for CasADi bug with all-zero values in bspline.
    # See: https://github.com/casadi/casadi/issues/2837
    # The result has one value per query point, matching the interpolator output.
    if method == "bspline" and all(values == 0):
        if is_casadi_type(xi, recursive=False):
            # xi is a casadi MX/SX/DM here; all three expose .zeros()
            return _convert_casadi_result(type(xi).zeros(xi.shape[0], 1))  # type: ignore[attr-defined]
        return _onp.zeros(xi.shape[0])

    if fill_value is None:
        xi = _clamp_bounds(xi, axis_min, axis_max, n_dimensions)

    values_flattened = _onp.ravel(values, order="F")
    interpolator = _cas.interpolant("Interpolator", method, points, values_flattened)
    fi = interpolator(xi.T).T

    if fill_value is not None:
        fi = _apply_fill_value(fi, xi, axis_min, axis_max, n_dimensions, fill_value)

    return _convert_casadi_result(fi)


def interpn(
    points: tuple[_onp.ndarray],
    values: _onp.ndarray,
    xi: _onp.ndarray,
    method: Literal["linear", "bspline", "nearest"] = "linear",
    bounds_error=True,
    fill_value=_onp.nan,
) -> _onp.ndarray:
    """
    Performs multidimensional interpolation on regular grids. Analogue to scipy.interpolate.interpn().

    See syntax here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html

    Args:

        points: The points defining the regular grid in n dimensions. Tuple of coordinates of each axis. Shapes (m1,
        ), ..., (mn,)

        values: The data on the regular grid in n dimensions. Shape (m1, ..., mn)

        xi: The coordinates to sample the gridded data at. (..., ndim)

        method: The method of interpolation to perform. one of:

            * "bspline" (Note: differentiable and suitable for optimization - made of piecewise-cubics. For other
            applications, other interpolators may be faster. Not monotonicity-preserving - may overshoot.)

            * "linear" (Note: differentiable, but not suitable for use in optimization w/o subgradient treatment due
            to C1-discontinuity)

            * "nearest" (Note: NOT differentiable, don't use in optimization. Fast.)

        bounds_error: If True, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If False, then fill_value is used.

        fill_value: If provided, the value to use for points outside of the interpolation domain. If None,
        behavior depends on the code path: the NumPy/SciPy fast path ("linear"/"nearest" with NumPy inputs)
        extrapolates like scipy.interpolate.interpn, while the CasADi path ("bspline", or CasADi inputs)
        clamps out-of-domain points to the nearest grid edge (flat extrapolation).

    Returns: Interpolated values at input coordinates.

    """
    implied_values_shape = _validate_interpn_inputs(points, values)

    ### NumPy fast path — direct delegation to scipy
    if (not is_casadi_type([points, values, xi], recursive=True)) and method in (
        "linear",
        "nearest",
    ):
        xi = _onp.array(xi).reshape((-1, len(implied_values_shape)))
        return _interpolate.interpn(
            points=points,
            values=values,
            xi=xi,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    if method in ("linear", "bspline"):
        return _interpn_casadi(points, values, xi, method, bounds_error, fill_value)

    raise ValueError(
        f"{method=!r} is not a valid option. Valid options are: 'linear', 'bspline', 'nearest'."
    )
