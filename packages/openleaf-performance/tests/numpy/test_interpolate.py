"""Tests for interpn in openleaf.performance.numpy.interpolate.

Covers unit tests, functional paths, and edge cases for the interpn
function across NumPy and CasADi backends.
"""

from __future__ import annotations

import casadi as cas
import numpy as np
import pytest
from scipy import interpolate as scipy_interpolate

from openleaf.performance.numpy.interpolate import interpn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_2d():
    """2D regular grid with known values for interpolation tests.

    Uses enough points per axis (10) for CasADi bspline (cubic needs >= 8).
    """
    x = np.linspace(0.0, 2.0, 10)
    y = np.linspace(0.0, 3.0, 10)
    # f(x, y) = x + y
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = xx + yy
    return (x, y), values


@pytest.fixture()
def grid_2d_zeros():
    """2D regular grid with all-zero values (CasADi bspline edge case)."""
    x = np.linspace(0.0, 2.0, 10)
    y = np.linspace(0.0, 2.0, 10)
    values = np.zeros((10, 10))
    return (x, y), values


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestInterpnNumpyLinear:
    """Test interpn with NumPy arrays and linear method."""

    def test_interpn_numpy_linear(self, grid_2d):
        """Result matches scipy.interpolate.interpn for linear method."""
        points, values = grid_2d
        xi = np.array([[0.5, 1.5], [1.0, 2.0]])

        result = interpn(points, values, xi, method="linear")
        expected = scipy_interpolate.interpn(points, values, xi, method="linear")

        np.testing.assert_allclose(result, expected)


class TestInterpnNumpyNearest:
    """Test interpn with NumPy arrays and nearest method."""

    def test_interpn_numpy_nearest(self, grid_2d):
        """Result matches scipy.interpolate.interpn for nearest method."""
        points, values = grid_2d
        xi = np.array([[0.3, 1.7], [1.6, 0.4]])

        result = interpn(points, values, xi, method="nearest")
        expected = scipy_interpolate.interpn(points, values, xi, method="nearest")

        np.testing.assert_allclose(result, expected)


class TestInterpnCasadiLinear:
    """Test interpn with CasADi MX symbolic inputs."""

    def test_interpn_casadi_linear(self, grid_2d):
        """Returns CasADi symbolic output when xi is MX."""
        points, values = grid_2d
        xi = cas.MX.sym("xi", 2, 2)  # 2 points, 2 dimensions

        result = interpn(points, values, xi, method="linear", bounds_error=False)

        assert isinstance(result, cas.MX)


class TestInterpnCasadiBspline:
    """Test interpn with CasADi DM inputs and bspline method."""

    def test_interpn_casadi_bspline(self, grid_2d):
        """Returns numeric interpolated values with DM xi."""
        points, values = grid_2d
        xi = cas.DM([[0.5, 1.5], [1.0, 2.0]])

        result = interpn(points, values, xi, method="bspline", bounds_error=False)

        # Should return numeric values (ndarray or float)
        assert not isinstance(result, cas.MX)
        # Verify interpolation produces reasonable values
        result_arr = np.atleast_1d(np.array(result, dtype=float))
        assert result_arr.shape[0] == 2
        # Values should be within the range of the grid values
        assert np.all(result_arr >= -1.0)  # allow small overshoot from bspline
        assert np.all(result_arr <= 6.0)

    def test_interpn_casadi_bspline_zeros(self, grid_2d_zeros):
        """All-zero values grid returns zeros_like(xi)."""
        points, values = grid_2d_zeros
        xi = cas.DM([[0.5, 0.5], [1.0, 1.0]])

        result = interpn(points, values, xi, method="bspline", bounds_error=False)

        result_arr = np.atleast_1d(np.array(result, dtype=float).reshape(-1))
        np.testing.assert_allclose(result_arr, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestInterpnEdgeCases:
    """Edge cases for interpn."""

    def test_casadi_points_raises_type_error(self, grid_2d):
        """CasADi arrays in points raises TypeError."""
        _, values = grid_2d
        casadi_points = (cas.DM([0.0, 1.0, 2.0]), cas.DM([0.0, 1.0, 2.0, 3.0]))
        xi = np.array([[0.5, 1.5]])

        with pytest.raises(TypeError, match="NumPy arrays"):
            interpn(casadi_points, values, xi)

    def test_casadi_values_raises_type_error(self, grid_2d):
        """CasADi arrays in values raises TypeError."""
        points, values = grid_2d
        casadi_values = cas.DM(values)
        xi = np.array([[0.5, 1.5]])

        with pytest.raises(TypeError, match="NumPy arrays"):
            interpn(points, casadi_values, xi)

    def test_wrong_values_shape(self, grid_2d):
        """values.shape != implied_values_shape raises ValueError."""
        points, _ = grid_2d
        wrong_values = np.zeros((2, 4))  # Should be (3, 4)
        xi = np.array([[0.5, 1.5]])

        with pytest.raises(ValueError, match="shape"):
            interpn(points, wrong_values, xi)

    def test_out_of_bounds_with_bounds_error(self, grid_2d):
        """xi outside grid with bounds_error=True raises ValueError."""
        points, values = grid_2d
        xi = np.array([[5.0, 1.0]])  # x=5.0 is outside [0, 2]

        with pytest.raises(ValueError, match="out of bounds"):
            interpn(points, values, xi, method="bspline", bounds_error=True)

    def test_out_of_bounds_fill_value_none(self, grid_2d):
        """xi outside grid with fill_value=None clamps to nearest grid point."""
        points, values = grid_2d
        xi_oob = np.array([[5.0, 1.5]])  # x=5.0 is outside [0, 2]
        xi_clamped = np.array([[2.0, 1.5]])  # clamped to x=2.0

        result_oob = interpn(
            points,
            values,
            xi_oob,
            method="bspline",
            bounds_error=False,
            fill_value=None,
        )
        result_clamped = interpn(
            points,
            values,
            xi_clamped,
            method="bspline",
            bounds_error=False,
            fill_value=None,
        )

        np.testing.assert_allclose(
            np.atleast_1d(np.array(result_oob, dtype=float)).reshape(-1),
            np.atleast_1d(np.array(result_clamped, dtype=float)).reshape(-1),
            atol=1e-10,
        )

    def test_out_of_bounds_fill_value_zero(self, grid_2d):
        """xi outside grid with fill_value=0 returns 0 for OOB points."""
        points, values = grid_2d
        xi = np.array([[5.0, 1.5]])  # x=5.0 is outside [0, 2]

        result = interpn(
            points,
            values,
            xi,
            method="bspline",
            bounds_error=False,
            fill_value=0,
        )

        result_arr = np.atleast_1d(np.array(result, dtype=float)).reshape(-1)
        np.testing.assert_allclose(result_arr, 0.0, atol=1e-12)

    def test_scalar_xi(self):
        """Scalar xi (int/float) is promoted to array, returns scalar."""
        points = (np.linspace(0.0, 2.0, 10),)
        values = np.linspace(10.0, 30.0, 10)
        xi = 1.5

        result = interpn(
            points,
            values,
            xi,
            method="bspline",
            bounds_error=False,
            fill_value=None,
        )

        # Should return a scalar (float)
        result_float = float(result)
        assert 15.0 - 1e-9 <= result_float <= 25.0 + 1e-9

    def test_1d_xi(self):
        """1D xi array is reshaped to 2D."""
        points = (np.linspace(0.0, 2.0, 10),)
        values = np.linspace(10.0, 30.0, 10)
        xi = np.array([0.5, 1.5])

        result = interpn(
            points,
            values,
            xi,
            method="bspline",
            bounds_error=False,
            fill_value=None,
        )

        result_arr = np.atleast_1d(np.array(result, dtype=float)).reshape(-1)
        assert result_arr.shape[0] == 2

    def test_mx_with_bounds_error_raises(self, grid_2d):
        """CasADi MX xi with bounds_error=True raises ValueError."""
        points, values = grid_2d
        xi = cas.MX.sym("xi", 2, 2)  # 2 points, 2 dimensions

        with pytest.raises(ValueError, match="MX type"):
            interpn(points, values, xi, method="linear", bounds_error=True)

    def test_dm_scalar_output(self):
        """1-point DM interpolation returning DM(1,1) converts to float."""
        points = (np.linspace(0.0, 2.0, 10),)
        values = np.linspace(10.0, 30.0, 10)
        xi = cas.DM([[1.0]])

        result = interpn(
            points,
            values,
            xi,
            method="bspline",
            bounds_error=False,
            fill_value=None,
        )

        assert isinstance(result, float)
        np.testing.assert_allclose(result, 20.0, atol=1e-10)
