"""
Array creation and manipulation functions.

Provides NumPy-like array operations that work with NumPy arrays,
CasADi symbolic types, and PyTorch tensors.
"""

from __future__ import annotations


__all__ = [
    "array",
    "assert_equal_shape",
    "concatenate",
    "diag",
    "dstack",
    "empty_like",
    "full_like",
    "hstack",
    "length",
    "max",
    "min",
    "ones_like",
    "ravel",
    "reshape",
    "roll",
    "stack",
    "tile",
    "vstack",
    "zeros_like",
]

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as _onp
from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_array_namespace
from openleaf.performance.numpy.backends import is_casadi_type
from openleaf.performance.numpy.backends import is_torch_type


if TYPE_CHECKING:
    from collections.abc import Sequence

# Type aliases for numpy order parameters
_OrderCF = Literal["C", "F"]
_OrderKACF = Literal["K", "A", "C", "F"]


def _get_casadi():
    """Lazy import CasADi."""
    return Backend.CASADI.module


def _get_torch():
    """Lazy import PyTorch."""
    return Backend.TORCH.module


def array(array_like, dtype=None):
    """
    Initializes a new array. Creates a NumPy array if possible; if not, creates a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.array.html
    """
    if is_casadi_type(array_like, recursive=False):
        # If you were literally given a CasADi array, just return it
        return array_like

    if is_torch_type(array_like, recursive=False):
        # If given a PyTorch tensor, return it
        return array_like

    if not is_casadi_type(array_like, recursive=True):
        # Check for PyTorch tensors in nested structure
        if is_torch_type(array_like, recursive=True):
            torch = _get_torch()
            # Convert nested structure to tensor
            if isinstance(array_like, (list, tuple)):
                return torch.stack([array(x) for x in array_like])
            return torch.tensor(array_like, dtype=dtype)

        # No CasADi or PyTorch types - use NumPy
        return _onp.array(array_like, dtype=dtype)

    else:
        # Contains CasADi types
        ca = _get_casadi()

        def make_row(contents: Sequence[Any]):
            try:
                return ca.horzcat(*contents)
            except (TypeError, RuntimeError):
                return contents

        return ca.vertcat(*[make_row(row) for row in array_like])


def concatenate(arrays: Sequence, axis: int = 0):
    """
    Join a sequence of arrays along an existing axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    """
    if is_casadi_type(arrays, recursive=True):
        ca = _get_casadi()
        if axis == 0:
            return ca.vertcat(*arrays)
        elif axis == 1:
            return ca.horzcat(*arrays)
        else:
            raise ValueError(
                "CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1."
            )

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(*arrays)
    return xp.concat(arrays, axis=axis)


def stack(arrays: Sequence, axis: int = 0):
    """
    Join a sequence of arrays along a new axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    """
    if is_casadi_type(arrays, recursive=True):
        ca = _get_casadi()
        # Validate stackability
        for arr in arrays:
            if is_casadi_type(arr, recursive=False):
                if not arr.shape[1] == 1:
                    raise ValueError("Can only stack Nx1 CasADi arrays!")
            else:
                if not len(arr.shape) == 1:
                    raise ValueError(
                        "Can only stack 1D NumPy ndarrays alongside CasADi arrays!"
                    )

        if axis == 0 or axis == -2:
            return ca.transpose(ca.horzcat(*arrays))
        elif axis == 1 or axis == -1:
            return ca.horzcat(*arrays)
        else:
            raise ValueError(
                "CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1."
            )

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(*arrays)
    return xp.stack(arrays, axis=axis)


def hstack(arrays):
    """Stack arrays in sequence horizontally (column wise).

    Args:
        arrays: Sequence of arrays to stack.

    Returns:
        Stacked array.

    Raises:
        ValueError: If arrays contain CasADi types.
    """
    if is_casadi_type(arrays, recursive=True):
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.hstack()` when dealing with CasADi arrays."
        )

    if is_torch_type(arrays, recursive=True):
        torch = _get_torch()
        return torch.hstack(list(arrays))

    return _onp.hstack(arrays)


def vstack(arrays):
    """Stack arrays in sequence vertically (row wise).

    Args:
        arrays: Sequence of arrays to stack.

    Returns:
        Stacked array.

    Raises:
        ValueError: If arrays contain CasADi types.
    """
    if is_casadi_type(arrays, recursive=True):
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.vstack()` when dealing with CasADi arrays."
        )

    if is_torch_type(arrays, recursive=True):
        torch = _get_torch()
        return torch.vstack(list(arrays))

    return _onp.vstack(arrays)


def dstack(arrays):
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        arrays: Sequence of arrays to stack.

    Returns:
        Stacked array.

    Raises:
        ValueError: If arrays contain CasADi types.
    """
    if is_casadi_type(arrays, recursive=True):
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.dstack()` when dealing with CasADi arrays."
        )

    if is_torch_type(arrays, recursive=True):
        torch = _get_torch()
        return torch.dstack(list(arrays))

    return _onp.dstack(arrays)


def length(array) -> int:
    """Return the length of an array-like object.

    An extension of len() that handles CasADi types and returns 1 for scalars.

    Args:
        array: Array-like object to get length of.

    Returns:
        Length of the array (1 for scalars).
    """
    if is_casadi_type(array, recursive=False):
        if array.shape[0] != 1:
            return array.shape[0]
        else:
            return array.shape[1]

    # For non-CasADi arrays, use standard Python len() with scalar handling
    try:
        if hasattr(array, "shape"):
            # Handle 0-dim arrays
            if len(array.shape) == 0:
                return 1
            return array.shape[0]
        return len(array)
    except TypeError:
        return 1


def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    """
    if is_casadi_type(v, recursive=False):
        ca = _get_casadi()
        if 1 in v.shape:  # If v is a 1D array, construct a diagonal matrix
            if v.shape[0] == 1:
                v = v.T

            if k == 0:
                return ca.diag(v)
            else:
                n = v.shape[0]
                res = type(v).zeros(n + abs(k), n + abs(k))
                for i in range(n):
                    if k >= 0:
                        res[i, i + k] = v[i]
                    else:
                        res[i - k, i] = v[i]
                return res

        elif v.shape[0] == v.shape[1]:  # Square matrix - extract diagonal
            n = v.shape[0]
            if k >= 0:
                return array([v[i, i + k] for i in range(n - k)])
            else:
                return array([v[i - k, i] for i in range(n + k)])
        else:
            raise NotImplementedError(
                "Haven't yet added logic for non-square matrices."
            )

    if is_torch_type(v, recursive=False):
        torch = _get_torch()
        return torch.diag(v, diagonal=k)

    return _onp.diag(v, k=k)


def roll(a, shift, axis: int | None = None):
    """
    Roll array elements along a given axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
    """
    if is_casadi_type(a, recursive=False):
        if axis is None:
            a_flat = reshape(a, -1)
            result = roll(a_flat, shift, axis=0)
            return reshape(result, a.shape)
        elif isinstance(axis, int):
            shift = shift % a.shape[axis]
            if shift != 0:
                slice1 = [slice(None)] * 2
                slice1[axis] = slice(-shift, None)
                slice2 = [slice(None)] * 2
                slice2[axis] = slice(-shift)
                result = concatenate([a[tuple(slice1)], a[tuple(slice2)]], axis=axis)
            else:
                result = a
            return result
        elif isinstance(axis, tuple):
            result = a
            if not isinstance(shift, tuple):
                shift = (shift,) * len(axis)
            for ax, sh in zip(axis, shift, strict=True):
                result = roll(result, sh, ax)
            return result
        else:
            raise ValueError("'axis' must be None, an integer or a tuple of integers")

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.roll(a, shift, axis=axis)


def max(a, axis=None):
    """
    Return the maximum of an array or maximum along an axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.max.html
    """
    if is_casadi_type(a, recursive=False):
        ca = _get_casadi()
        if axis is None:
            return ca.mmax(a)
        if axis == 0:
            if a.shape[1] == 1:
                return ca.mmax(a)
            else:
                return array([ca.mmax(a[:, i]) for i in range(a.shape[1])])
        elif axis == 1:
            if a.shape[0] == 1:
                return ca.mmax(a)
            else:
                return array([ca.mmax(a[i, :]) for i in range(a.shape[0])])
        else:
            raise ValueError(f"Invalid axis {axis} for CasADi array.")

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.max(a, axis=axis)


def min(a, axis=None):
    """
    Return the minimum of an array or minimum along an axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.min.html
    """
    if is_casadi_type(a, recursive=False):
        ca = _get_casadi()
        if axis is None:
            return ca.mmin(a)
        if axis == 0:
            if a.shape[1] == 1:
                return ca.mmin(a)
            else:
                return array([ca.mmin(a[:, i]) for i in range(a.shape[1])])
        elif axis == 1:
            if a.shape[0] == 1:
                return ca.mmin(a)
            else:
                return array([ca.mmin(a[i, :]) for i in range(a.shape[0])])
        else:
            raise ValueError(f"Invalid axis {axis} for CasADi array.")

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.min(a, axis=axis)


def reshape(a, newshape, order: _OrderCF = "C"):
    """
    Gives a new shape to an array without changing its data.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    """
    if is_casadi_type(a, recursive=False):
        ca = _get_casadi()
        if isinstance(newshape, int):
            newshape = (newshape, 1)
        elif len(newshape) == 1:
            newshape = (newshape[0], 1)
        elif len(newshape) == 2:
            newshape = tuple(newshape)
        elif len(newshape) > 2:
            raise ValueError(
                "CasADi data types are limited to no more than 2 dimensions."
            )

        if order == "C":
            return ca.reshape(a.T, newshape[::-1]).T
        elif order == "F":
            return ca.reshape(a, newshape)
        else:
            raise NotImplementedError("Only C and F orders are supported.")

    # Array API reshape doesn't support order parameter
    if order != "C":
        # Fall back to NumPy for non-C order
        return _onp.reshape(a, newshape, order=order)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    if isinstance(newshape, int):
        newshape = (newshape,)
    return xp.reshape(a, newshape)


def ravel(a, order: _OrderCF = "C"):
    """
    Returns a contiguous flattened array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    """
    if is_casadi_type(a, recursive=False):
        return reshape(a, -1, order=order)

    # Use reshape(-1) for Array API compatibility
    if order != "C":
        # Fall back to NumPy for non-C order
        return _onp.ravel(a, order=order)

    xp = get_array_namespace(a)
    return xp.reshape(a, (-1,))


def tile(A, reps):
    """
    Construct an array by repeating A the number of times given by reps.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
    """
    if is_casadi_type(A, recursive=False):
        ca = _get_casadi()
        if len(reps) == 1:
            return ca.repmat(A, reps[0], 1)
        elif len(reps) == 2:
            return ca.repmat(A, reps[0], reps[1])
        else:
            raise ValueError(
                "Cannot have >2D arrays when using CasADi numeric backend!"
            )

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(A)
    return xp.tile(A, reps)


def zeros_like(a, dtype=None, order: _OrderKACF = "K", subok: bool = True, shape=None):
    """Return an array of zeros with the same shape and type as a given array."""
    if is_casadi_type(a, recursive=False):
        return _onp.zeros(shape=length(a))

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.zeros_like(a, dtype=dtype)


def ones_like(a, dtype=None, order: _OrderKACF = "K", subok: bool = True, shape=None):
    """Return an array of ones with the same shape and type as a given array."""
    if is_casadi_type(a, recursive=False):
        return _onp.ones(shape=length(a))

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.ones_like(a, dtype=dtype)


def empty_like(
    prototype, dtype=None, order: _OrderKACF = "K", subok: bool = True, shape=None
):
    """Return a new array with the same shape and type as a given array."""
    if is_casadi_type(prototype, recursive=False):
        return zeros_like(prototype)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(prototype)
    return xp.empty_like(prototype, dtype=dtype)


def full_like(
    a, fill_value, dtype=None, order: _OrderKACF = "K", subok: bool = True, shape=None
):
    """Return a full array with the same shape and type as a given array."""
    if is_casadi_type(a, recursive=False):
        return fill_value * ones_like(a)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(a)
    return xp.full_like(a, fill_value, dtype=dtype)


def assert_equal_shape(
    arrays: list[_onp.ndarray] | dict[str, _onp.ndarray],
) -> None:
    """
    Assert that all of the given arrays are the same shape.

    Args:
        arrays: The arrays to be evaluated. Can be provided as a list or
            a dictionary of name:array pairs.

    Raises:
        ValueError: If arrays have different shapes.
    """
    names: list[str] | None = None
    array_list: list[_onp.ndarray]

    if isinstance(arrays, dict):
        names = list(arrays.keys())
        array_list = list(arrays.values())
    else:
        array_list = arrays

    def get_shape(arr) -> tuple:
        try:
            return arr.shape
        except AttributeError:
            return ()

    shape = get_shape(array_list[0])

    for arr in array_list[1:]:
        if get_shape(arr) != shape:
            if names is None:
                raise ValueError("The given arrays do not have the same shape!")
            else:
                namelist = ", ".join(names)
                raise ValueError(
                    f"The given arrays {namelist} do not have the same shape!"
                )
