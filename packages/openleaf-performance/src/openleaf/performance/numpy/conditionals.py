"""
Conditional selection and comparison operations.

Provides NumPy-like conditional functions (where, maximum, minimum) that work
with NumPy arrays, CasADi symbolic types, and PyTorch tensors.
"""

from __future__ import annotations


__all__ = [
    "maximum",
    "minimum",
    "where",
]

from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_array_namespace
from openleaf.performance.numpy.backends import get_backend
from openleaf.performance.numpy.backends import is_casadi_type


def _get_casadi():
    """Lazy import CasADi."""
    return Backend.CASADI.module


def _promote_to_common_backend(*args):
    """Promote all arguments to a common non-numpy backend when one is present.

    If any argument is a PyTorch tensor or JAX array, convert all remaining
    numpy arrays/scalars to that backend so that array_api_compat sees a single
    namespace.  This handles the case where e.g. ``np.where(scalar_cond, a, b)``
    returns a 0-d numpy array which is then passed as a value to a subsequent
    ``np.where`` whose condition is a JAX or torch array.
    """
    from openleaf.performance.numpy.backends import is_torch_type

    has_torch = any(is_torch_type(a, recursive=False) for a in args)
    if has_torch:
        torch = Backend.TORCH.module

        # Find reference device and the widest floating-point dtype
        ref_device = None
        ref_dtype = None
        for a in args:
            if isinstance(a, torch.Tensor):
                if ref_device is None:
                    ref_device = a.device
                if a.dtype.is_floating_point and (
                    ref_dtype is None or a.dtype == torch.float64
                ):
                    ref_dtype = a.dtype

        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            t = torch.as_tensor(x)
            if t.dtype == torch.bool:
                return t.to(device=ref_device) if ref_device is not None else t
            return torch.as_tensor(x, dtype=ref_dtype, device=ref_device)

        return tuple(_to_tensor(a) for a in args)

    # Check for JAX arrays — import lazily to avoid hard dependency at import time
    try:
        import jax
        import jax.numpy as jnp

        has_jax = any(isinstance(a, jax.Array) for a in args)
    except ImportError:
        has_jax = False

    if has_jax:

        def _to_jax(x):
            if isinstance(x, jax.Array):
                return x
            # Let JAX infer the dtype from the value itself — do NOT force the
            # condition's bool dtype onto value arrays.
            return jnp.asarray(x)

        return tuple(_to_jax(a) for a in args)

    return args


def _ensure_tensor_pair(x1, x2):
    """Ensure both arguments are tensors when one is a PyTorch tensor.

    PyTorch's maximum/minimum require both args to be Tensor, unlike NumPy
    which accepts scalar + array.
    """
    if get_backend(x1) == Backend.TORCH:
        torch = Backend.TORCH.module
        if not isinstance(x2, torch.Tensor):
            x2 = torch.as_tensor(x2, dtype=x1.dtype)
    elif get_backend(x2) == Backend.TORCH:
        torch = Backend.TORCH.module
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1, dtype=x2.dtype)
    return x1, x2


def where(
    condition,
    value_if_true,
    value_if_false,
):
    """
    Return elements chosen from x or y depending on condition.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    """
    if is_casadi_type([condition, value_if_true, value_if_false], recursive=True):
        ca = _get_casadi()
        return ca.if_else(condition, value_if_true, value_if_false)

    # When mixing torch tensors with numpy arrays/scalars, promote to torch
    condition, value_if_true, value_if_false = _promote_to_common_backend(
        condition, value_if_true, value_if_false
    )

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(condition, value_if_true, value_if_false)
    return xp.where(condition, value_if_true, value_if_false)


def maximum(
    x1,
    x2,
):
    """
    Element-wise maximum of two arrays.

    Note: not differentiable at the crossover point, will cause issues if you try to optimize across a crossover.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """
    if is_casadi_type([x1, x2], recursive=True):
        return where(x1 >= x2, x1, x2)

    # Ensure both args are tensors for PyTorch compatibility
    x1, x2 = _ensure_tensor_pair(x1, x2)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x1, x2)
    return xp.maximum(x1, x2)


def minimum(
    x1,
    x2,
):
    """
    Element-wise minimum of two arrays.

    Note: not differentiable at the crossover point, will cause issues if you try to optimize across a crossover.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """
    if is_casadi_type([x1, x2], recursive=True):
        return where(x1 <= x2, x1, x2)

    # Ensure both args are tensors for PyTorch compatibility
    x1, x2 = _ensure_tensor_pair(x1, x2)

    # Use Array API for NumPy/PyTorch/CuPy/JAX
    xp = get_array_namespace(x1, x2)
    return xp.minimum(x1, x2)
