# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""OpenAP math backend that dispatches through openleaf.performance.numpy.

OpenAP (>= 2.5) evaluates its thrust, drag and fuel-flow equations through a
pluggable math backend (``openap.backends.MathBackend``) passed to each model's
constructor. Its own backends are bound to one array library each, whereas an
openleaf model is evaluated with whatever values the caller passes: NumPy
arrays, CasADi symbols, PyTorch tensors or JAX arrays.

:class:`OpenleafBackend` forwards every operation to
:mod:`openleaf.performance.numpy`, which picks the implementation from the
type of the operands. No NumPy function is ever applied to a CasADi value,
which CasADi 3.8 deprecates and older CasADi releases reject under NumPy 2.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike


__all__ = ["OpenleafBackend"]


class OpenleafBackend:
    """OpenAP math backend backed by openleaf's type-dispatching numpy layer.

    Implements the operations that OpenAP's ``Thrust``, ``Drag`` and
    ``FuelFlow`` models call.
    """

    smooth_guards = False
    """Keep OpenAP's exact equations (``maximum``/``where``) on every backend.

    OpenAP's CasADi backend enables smooth approximations of its guards; leaving
    them off means a symbolic evaluation computes the same model as a numeric one.
    """

    pi = np.pi

    @staticmethod
    def sqrt(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.sqrt(x)

    @staticmethod
    def exp(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.exp(x)

    @staticmethod
    def log(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.log(x)

    @staticmethod
    def power(x: ArrayScalarLike, y: ArrayScalarLike) -> ArrayScalarLike:
        return np.power(x, y)

    @staticmethod
    def sin(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.sin(x)

    @staticmethod
    def cos(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.cos(x)

    @staticmethod
    def tan(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.tan(x)

    @staticmethod
    def arcsin(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.arcsin(x)

    @staticmethod
    def arccos(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.arccos(x)

    @staticmethod
    def arctan(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.arctan(x)

    @staticmethod
    def arctan2(y: ArrayScalarLike, x: ArrayScalarLike) -> ArrayScalarLike:
        return np.arctan2(y, x)

    @staticmethod
    def abs(x: ArrayScalarLike) -> ArrayScalarLike:
        return np.abs(x)

    @staticmethod
    def where(
        condition: ArrayScalarLike, x: ArrayScalarLike, y: ArrayScalarLike
    ) -> ArrayScalarLike:
        return np.where(condition, x, y)

    @staticmethod
    def maximum(x: ArrayScalarLike, y: ArrayScalarLike) -> ArrayScalarLike:
        return np.maximum(x, y)

    @staticmethod
    def minimum(x: ArrayScalarLike, y: ArrayScalarLike) -> ArrayScalarLike:
        return np.minimum(x, y)

    @staticmethod
    def clip(
        x: ArrayScalarLike, min_val: ArrayScalarLike, max_val: ArrayScalarLike
    ) -> ArrayScalarLike:
        return np.clip(x, min_val, max_val)

    @staticmethod
    def interp(x: ArrayScalarLike, xp: ArrayScalarLike, fp: ArrayScalarLike) -> ArrayScalarLike:
        return np.interp(x, xp, fp)
