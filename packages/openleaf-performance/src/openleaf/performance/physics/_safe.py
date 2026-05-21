"""Safe arithmetic helpers for physics functions.

The aerodynamic helpers (``aerodynamic_coefficient``, ``aerodynamic_force``)
divide by quantities like ``q * S`` that approach zero near the lower edge
of the flight envelope. A plain ``a / b`` produces ``NaN`` there, which
silently poisons every downstream computation. ``safe_divide`` swaps the
quotient for a configurable default when the denominator falls within an
epsilon band of zero.

The helper is backend-agnostic and works element-wise for both Python
scalars and NumPy arrays. It avoids ``np.where(b != 0, a/b, default)``,
which still evaluates ``a/b`` and so still emits a runtime warning even
when the result is masked out.
"""

from __future__ import annotations


__all__ = ["safe_divide"]

from openleaf.core.types import ArrayScalarLike
from openleaf.performance import numpy as np


# A conservative absolute tolerance well below any physical denominator
# we expect (dynamic pressure at 30 kt CAS at sea level is ~120 Pa;
# 1e-10 is safely below the smallest meaningful value).
_DEFAULT_EPS: float = 1e-10


def safe_divide(
    numerator: ArrayScalarLike,
    denominator: ArrayScalarLike,
    *,
    default: ArrayScalarLike = 0.0,
    eps: float = _DEFAULT_EPS,
) -> ArrayScalarLike:
    """Element-wise division that swaps to ``default`` near zero denominators.

    Args:
        numerator: Dividend.
        denominator: Divisor. May contain zeros or near-zeros.
        default: Value returned where ``|denominator| < eps``. Defaults to 0.0.
        eps: Absolute tolerance band around zero. Defaults to 1e-10.

    Returns:
        ``numerator / denominator`` where it's safe, else ``default``.
    """
    safe_den = np.where(np.abs(denominator) < eps, 1.0, denominator)
    return np.where(np.abs(denominator) < eps, default, numerator / safe_den)
