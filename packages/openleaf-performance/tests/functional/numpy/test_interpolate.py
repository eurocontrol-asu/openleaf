from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.interpolate import _interpn_casadi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_grid():
    """Simple 2D linear interpolation grid."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    # values = x + 2*y on the grid
    values = np.array(
        [
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]
    )
    return (x, y), values


@pytest.fixture
def bspline_grid():
    """1D grid for bspline interpolation."""
    x = np.linspace(0, 2 * np.pi, 20)
    values = np.sin(x)
    return (x,), values


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_interpn_casadi_linear_unchanged(linear_grid):
    """Linear interpolation produces same results as before refactor."""
    points, values = linear_grid
    xi = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 0.5]])

    result = _interpn_casadi(
        points, values, xi, method="linear", bounds_error=False, fill_value=None
    )

    expected = np.array([1.5, 3.0, 2.5])  # x + 2*y
    np.testing.assert_allclose(np.array(result, dtype=float).reshape(-1), expected, atol=1e-10)


def test_interpn_casadi_bspline_unchanged(bspline_grid):
    """Bspline interpolation produces same results as before refactor."""
    points, values = bspline_grid
    xi = np.array([0.5, 1.0, 2.0, 3.0])

    result = _interpn_casadi(
        points, values, xi, method="bspline", bounds_error=False, fill_value=None
    )

    result_arr = np.array(result, dtype=float).reshape(-1)
    expected = np.sin(xi)
    np.testing.assert_allclose(result_arr, expected, atol=0.05)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_interpn_casadi_bspline_all_zeros():
    """All-zero bspline returns zeros_like(xi) (early return)."""
    x = np.linspace(0, 1, 10)
    values = np.zeros_like(x)
    xi = np.array([0.2, 0.5, 0.8])

    result = _interpn_casadi(
        (x,), values, xi, method="bspline", bounds_error=False, fill_value=None
    )

    result_arr = np.array(result, dtype=float).reshape(-1)
    np.testing.assert_allclose(result_arr, 0.0)


def test_interpn_casadi_no_fill_value(linear_grid):
    """fill_value=None clamps bounds, no fill masking applied."""
    points, values = linear_grid
    # Query outside grid — should be clamped, not masked
    xi = np.array([[3.0, 1.0], [-1.0, 1.0]])

    result = _interpn_casadi(
        points, values, xi, method="linear", bounds_error=False, fill_value=None
    )

    result_arr = np.array(result, dtype=float).reshape(-1)
    # Clamped to grid edges: (2,1) → 4.0; (0,1) → 2.0
    expected = np.array([4.0, 2.0])
    np.testing.assert_allclose(result_arr, expected, atol=1e-10)


def test_interpn_casadi_scalar_result():
    """Single-point interpolation returns a Python float."""
    x = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 1.0, 4.0])
    xi = np.array([1.0])

    result = _interpn_casadi(
        (x,), values, xi, method="linear", bounds_error=False, fill_value=None
    )

    assert isinstance(result, float)
    assert result == pytest.approx(1.0)
