# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
NumPy-compatible array operations supporting CasADi symbolic types.

This module provides a drop-in replacement for NumPy that works seamlessly
with both NumPy arrays and CasADi symbolic types. Functions are automatically
dispatched to either NumPy or CasADi based on input types.

Usage:
    import openleaf.performance.numpy as np

    # Works with regular NumPy arrays
    x = np.array([1, 2, 3])
    y = np.sqrt(x)

    # Also works with CasADi symbolic types
    import casadi as ca
    x_sym = ca.MX.sym("x", 3)
    y_sym = np.sqrt(x_sym)

CasADi Compatibility Notes:
---------------------------
The following functions work with both NumPy arrays and CasADi symbolic types:
- Arithmetic: sqrt, cbrt, abs, sum, mean, prod, exp, log, log2, log10, power
- Trigonometric: sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh
- Degree-based: sind, cosd, tand, arcsind, arccosd, arctan2d
- Conditional: where, maximum, minimum, clip
- Logical: logical_and, logical_or, logical_not, all, any
- Array: array, concatenate, stack, reshape, roll, diag

The following functions are NumPy-only (will NOT work with CasADi):
- Sorting: sort, argsort, argmax, argmin
- Statistics: std, var (use mean instead)
- Array creation: linspace, arange, meshgrid, zeros, ones, full
  (These create NumPy arrays; CasADi arrays must be created via casadi)
"""

from __future__ import annotations

# =============================================================================
# NumPy re-exports (functions that don't need CasADi overrides)
# These either work with CasADi naturally or are array-creation only
# =============================================================================
from numpy import allclose
from numpy import arange
from numpy import argmax  # NumPy-only
from numpy import argmin  # NumPy-only
from numpy import argsort  # NumPy-only
from numpy import asarray
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import bool_
from numpy import broadcast_to
from numpy import cross
from numpy import cumprod  # NumPy-only
from numpy import cumsum  # NumPy-only
from numpy import diff  # NumPy-only
from numpy import dot
from numpy import e
from numpy import empty
from numpy import expand_dims
from numpy import eye
from numpy import flip
from numpy import fliplr
from numpy import flipud
from numpy import float32
from numpy import float64
from numpy import full
from numpy import gradient  # NumPy-only
from numpy import identity
from numpy import inf
from numpy import int32
from numpy import int64
from numpy import isclose
from numpy import isfinite
from numpy import isinf
from numpy import isnan
from numpy import isscalar
from numpy import linspace
from numpy import logspace
from numpy import meshgrid
from numpy import nan
from numpy import ndarray
from numpy import newaxis
from numpy import ones
from numpy import outer
from numpy import pi
from numpy import pow
from numpy import rot90
from numpy import round
from numpy import searchsorted  # NumPy-only
from numpy import sort  # NumPy-only
from numpy import squeeze
from numpy import std  # NumPy-only
from numpy import swapaxes
from numpy import testing
from numpy import transpose
from numpy import unique  # NumPy-only
from numpy import var  # NumPy-only
from numpy import zeros

# =============================================================================
# CasADi-compatible function overrides
# These work with both NumPy arrays and CasADi symbolic types
# =============================================================================
# Monadic arithmetic (single argument operations)
from openleaf.performance.numpy.arithmetic_monadic import abs
from openleaf.performance.numpy.arithmetic_monadic import cbrt
from openleaf.performance.numpy.arithmetic_monadic import mean
from openleaf.performance.numpy.arithmetic_monadic import prod
from openleaf.performance.numpy.arithmetic_monadic import sqrt
from openleaf.performance.numpy.arithmetic_monadic import sum

# Array creation and manipulation
from openleaf.performance.numpy.array import array
from openleaf.performance.numpy.array import assert_equal_shape
from openleaf.performance.numpy.array import concatenate
from openleaf.performance.numpy.array import diag
from openleaf.performance.numpy.array import dstack
from openleaf.performance.numpy.array import empty_like
from openleaf.performance.numpy.array import full_like
from openleaf.performance.numpy.array import hstack
from openleaf.performance.numpy.array import length
from openleaf.performance.numpy.array import max
from openleaf.performance.numpy.array import min
from openleaf.performance.numpy.array import ones_like
from openleaf.performance.numpy.array import ravel
from openleaf.performance.numpy.array import reshape
from openleaf.performance.numpy.array import roll
from openleaf.performance.numpy.array import stack
from openleaf.performance.numpy.array import tile
from openleaf.performance.numpy.array import vstack
from openleaf.performance.numpy.array import zeros_like

# Backend detection and Array API
from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_array_namespace
from openleaf.performance.numpy.backends import get_backend
from openleaf.performance.numpy.backends import require_backend

# Conditional operations
from openleaf.performance.numpy.conditionals import maximum
from openleaf.performance.numpy.conditionals import minimum
from openleaf.performance.numpy.conditionals import where

# Type detection utilities
from openleaf.performance.numpy.determine_type import is_casadi_type
from openleaf.performance.numpy.determine_type import is_torch_type

# Interpolation
from openleaf.performance.numpy.interpolate import interp
from openleaf.performance.numpy.interpolate import interpn
from openleaf.performance.numpy.interpolate import is_data_structured

# Logical operations
from openleaf.performance.numpy.logicals import all
from openleaf.performance.numpy.logicals import any
from openleaf.performance.numpy.logicals import clip
from openleaf.performance.numpy.logicals import equal
from openleaf.performance.numpy.logicals import greater
from openleaf.performance.numpy.logicals import greater_equal
from openleaf.performance.numpy.logicals import less
from openleaf.performance.numpy.logicals import less_equal
from openleaf.performance.numpy.logicals import logical_and
from openleaf.performance.numpy.logicals import logical_not
from openleaf.performance.numpy.logicals import logical_or
from openleaf.performance.numpy.logicals import not_equal

# CasADi-compatible math functions (override numpy versions)
from openleaf.performance.numpy.math_functions import arccos
from openleaf.performance.numpy.math_functions import arccosh
from openleaf.performance.numpy.math_functions import arcsin
from openleaf.performance.numpy.math_functions import arcsinh
from openleaf.performance.numpy.math_functions import arctan
from openleaf.performance.numpy.math_functions import arctan2
from openleaf.performance.numpy.math_functions import arctanh
from openleaf.performance.numpy.math_functions import ceil
from openleaf.performance.numpy.math_functions import cos
from openleaf.performance.numpy.math_functions import cosh
from openleaf.performance.numpy.math_functions import exp
from openleaf.performance.numpy.math_functions import floor
from openleaf.performance.numpy.math_functions import log
from openleaf.performance.numpy.math_functions import log2
from openleaf.performance.numpy.math_functions import log10
from openleaf.performance.numpy.math_functions import power
from openleaf.performance.numpy.math_functions import sign
from openleaf.performance.numpy.math_functions import sin
from openleaf.performance.numpy.math_functions import sinh
from openleaf.performance.numpy.math_functions import tan
from openleaf.performance.numpy.math_functions import tanh

# Trigonometric helpers (degree-based)
from openleaf.performance.numpy.trig import arccosd
from openleaf.performance.numpy.trig import arcsind
from openleaf.performance.numpy.trig import arctan2d
from openleaf.performance.numpy.trig import cosd
from openleaf.performance.numpy.trig import degrees
from openleaf.performance.numpy.trig import radians
from openleaf.performance.numpy.trig import sind
from openleaf.performance.numpy.trig import tand


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Monadic arithmetic
    "abs",
    "cbrt",
    "mean",
    "prod",
    "sqrt",
    "sum",
    # Array operations
    "all",
    "allclose",
    "any",
    "arange",
    "arccos",
    "arccosd",
    "arccosh",
    "arcsin",
    "arcsind",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctan2d",
    "arctanh",
    "argmax",
    "argmin",
    "argsort",
    "array",
    "asarray",
    "assert_equal_shape",
    "atleast_1d",
    "atleast_2d",
    "bool_",
    "broadcast_to",
    "ceil",
    "clip",
    "concatenate",
    "cos",
    "cosd",
    "cosh",
    "cross",
    "cumprod",
    "cumsum",
    "degrees",
    "diag",
    "diff",
    "dot",
    "dstack",
    "e",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "eye",
    "flip",
    "fliplr",
    "flipud",
    "float32",
    "float64",
    "floor",
    "full",
    "full_like",
    "gradient",
    "greater",
    "greater_equal",
    "hstack",
    "identity",
    "inf",
    "int32",
    "int64",
    "interp",
    "interpn",
    "is_casadi_type",
    "is_torch_type",
    "is_data_structured",
    "Backend",
    "get_array_namespace",
    "get_backend",
    "require_backend",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isscalar",
    "length",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log2",
    "log10",
    "logical_and",
    "logical_not",
    "logical_or",
    "logspace",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "nan",
    "ndarray",
    "newaxis",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pi",
    "pow",
    "power",
    "radians",
    "ravel",
    "reshape",
    "roll",
    "rot90",
    "round",
    "searchsorted",
    "sign",
    "sin",
    "sind",
    "sinh",
    "sort",
    "sqrt",
    "squeeze",
    "stack",
    "std",
    "swapaxes",
    "tan",
    "tand",
    "tanh",
    "testing",
    "tile",
    "transpose",
    "unique",
    "var",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]
