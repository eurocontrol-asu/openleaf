"""Tests for aetherx.numpy module.

Tests both NumPy and CasADi backends for mathematical operations.
"""

from __future__ import annotations

import casadi as cas
import numpy as np
import pytest

import openleaf.performance.numpy as anp


class TestLogicals:
    """Test logical operations."""

    # NumPy backend tests
    def test_logical_and_numpy(self):
        """Test logical_and with NumPy arrays."""
        x = np.array([True, True, False, False])
        y = np.array([True, False, True, False])
        result = anp.logical_and(x, y)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_logical_or_numpy(self):
        """Test logical_or with NumPy arrays."""
        x = np.array([True, True, False, False])
        y = np.array([True, False, True, False])
        result = anp.logical_or(x, y)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_logical_not_numpy(self):
        """Test logical_not with NumPy arrays."""
        x = np.array([True, False, True, False])
        result = anp.logical_not(x)
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_all_numpy(self):
        """Test all with NumPy arrays."""
        assert anp.all(np.array([True, True, True])) == True  # noqa: E712
        assert anp.all(np.array([True, False, True])) == False  # noqa: E712

    def test_any_numpy(self):
        """Test any with NumPy arrays."""
        assert anp.any(np.array([False, False, True])) == True  # noqa: E712
        assert anp.any(np.array([False, False, False])) == False  # noqa: E712

    def test_clip_numpy(self):
        """Test clip with NumPy arrays."""
        x = np.array([-1.0, 0.5, 1.5])
        result = anp.clip(x, 0.0, 1.0)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_clip_min_only_numpy(self):
        """Test clip with min only."""
        x = np.array([-1.0, 0.5, 1.5])
        result = anp.clip(x, min=0.0)
        expected = np.array([0.0, 0.5, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_clip_max_only_numpy(self):
        """Test clip with max only."""
        x = np.array([-1.0, 0.5, 1.5])
        result = anp.clip(x, max=1.0)
        expected = np.array([-1.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    # CasADi backend tests
    def test_logical_and_casadi(self):
        """Test logical_and with CasADi symbols."""
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        result = anp.logical_and(x > 0, y > 0)
        assert isinstance(result, cas.SX)

    def test_logical_or_casadi(self):
        """Test logical_or with CasADi symbols."""
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        result = anp.logical_or(x > 0, y > 0)
        assert isinstance(result, cas.SX)

    def test_logical_not_casadi(self):
        """Test logical_not with CasADi symbols."""
        x = cas.SX.sym("x")
        result = anp.logical_not(x > 0)
        assert isinstance(result, cas.SX)

    def test_clip_casadi(self):
        """Test clip with CasADi symbols."""
        x = cas.SX.sym("x")
        result = anp.clip(x, 0.0, 1.0)
        assert isinstance(result, cas.SX)

    def test_all_casadi_raises_on_axis(self):
        """Test all with CasADi raises on axis parameter."""
        x = cas.SX.sym("x", 3)
        with pytest.raises(NotImplementedError):
            anp.all(x, axis=0)

    def test_any_casadi_raises_on_axis(self):
        """Test any with CasADi raises on axis parameter."""
        x = cas.SX.sym("x", 3)
        with pytest.raises(NotImplementedError):
            anp.any(x, axis=0)


class TestTrigonometric:
    """Test trigonometric functions."""

    # Degree/radian conversion
    def test_degrees(self):
        """Test degrees conversion."""
        result = anp.degrees(np.pi)
        np.testing.assert_almost_equal(result, 180.0)

    def test_radians(self):
        """Test radians conversion."""
        result = anp.radians(180.0)
        np.testing.assert_almost_equal(result, np.pi)

    # Sine functions
    def test_sind_numpy(self):
        """Test sind (sine in degrees) with NumPy."""
        result = anp.sind(90.0)
        np.testing.assert_almost_equal(result, 1.0)
        result = anp.sind(0.0)
        np.testing.assert_almost_equal(result, 0.0)

    def test_sind_casadi(self):
        """Test sind with CasADi."""
        x = cas.SX.sym("x")
        result = anp.sind(x)
        assert isinstance(result, cas.SX)

    # Cosine functions
    def test_cosd_numpy(self):
        """Test cosd (cosine in degrees) with NumPy."""
        result = anp.cosd(0.0)
        np.testing.assert_almost_equal(result, 1.0)
        result = anp.cosd(90.0)
        np.testing.assert_almost_equal(result, 0.0, decimal=10)

    def test_cosd_casadi(self):
        """Test cosd with CasADi."""
        x = cas.SX.sym("x")
        result = anp.cosd(x)
        assert isinstance(result, cas.SX)

    # Tangent functions
    def test_tand_numpy(self):
        """Test tand (tangent in degrees) with NumPy."""
        result = anp.tand(45.0)
        np.testing.assert_almost_equal(result, 1.0)
        result = anp.tand(0.0)
        np.testing.assert_almost_equal(result, 0.0)

    def test_tand_casadi(self):
        """Test tand with CasADi."""
        x = cas.SX.sym("x")
        result = anp.tand(x)
        assert isinstance(result, cas.SX)

    # Inverse trig functions
    def test_arcsind_numpy(self):
        """Test arcsind (arcsine returning degrees) with NumPy."""
        result = anp.arcsind(1.0)
        np.testing.assert_almost_equal(result, 90.0)
        result = anp.arcsind(0.0)
        np.testing.assert_almost_equal(result, 0.0)

    def test_arcsind_casadi(self):
        """Test arcsind with CasADi."""
        x = cas.SX.sym("x")
        result = anp.arcsind(x)
        assert isinstance(result, cas.SX)

    def test_arccosd_numpy(self):
        """Test arccosd (arccosine returning degrees) with NumPy."""
        result = anp.arccosd(1.0)
        np.testing.assert_almost_equal(result, 0.0)
        result = anp.arccosd(0.0)
        np.testing.assert_almost_equal(result, 90.0)

    def test_arccosd_casadi(self):
        """Test arccosd with CasADi."""
        x = cas.SX.sym("x")
        result = anp.arccosd(x)
        assert isinstance(result, cas.SX)

    def test_arctan2d_numpy(self):
        """Test arctan2d (arctan2 returning degrees) with NumPy."""
        result = anp.arctan2d(1.0, 1.0)
        np.testing.assert_almost_equal(result, 45.0)
        result = anp.arctan2d(1.0, 0.0)
        np.testing.assert_almost_equal(result, 90.0)

    def test_arctan2d_casadi(self):
        """Test arctan2d with CasADi."""
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        result = anp.arctan2d(y, x)
        assert isinstance(result, cas.SX)


class TestMathFunctions:
    """Test mathematical functions."""

    # Power and root functions
    def test_sqrt_numpy(self):
        """Test sqrt with NumPy."""
        result = anp.sqrt(4.0)
        np.testing.assert_almost_equal(result, 2.0)

    def test_sqrt_array(self):
        """Test sqrt with array."""
        x = np.array([1.0, 4.0, 9.0, 16.0])
        result = anp.sqrt(x)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_numpy(self):
        """Test power with NumPy."""
        result = anp.power(2.0, 3.0)
        np.testing.assert_almost_equal(result, 8.0)

    # Exponential and logarithm
    def test_exp_numpy(self):
        """Test exp with NumPy."""
        result = anp.exp(0.0)
        np.testing.assert_almost_equal(result, 1.0)
        result = anp.exp(1.0)
        np.testing.assert_almost_equal(result, np.e)

    def test_log_numpy(self):
        """Test log (natural logarithm) with NumPy."""
        result = anp.log(np.e)
        np.testing.assert_almost_equal(result, 1.0)

    def test_log10_numpy(self):
        """Test log10 with NumPy."""
        result = anp.log10(100.0)
        np.testing.assert_almost_equal(result, 2.0)

    # Standard trig functions (radians)
    def test_sin_numpy(self):
        """Test sin with NumPy."""
        result = anp.sin(np.pi / 2)
        np.testing.assert_almost_equal(result, 1.0)

    def test_cos_numpy(self):
        """Test cos with NumPy."""
        result = anp.cos(0.0)
        np.testing.assert_almost_equal(result, 1.0)

    def test_tan_numpy(self):
        """Test tan with NumPy."""
        result = anp.tan(np.pi / 4)
        np.testing.assert_almost_equal(result, 1.0)

    def test_arcsin_numpy(self):
        """Test arcsin with NumPy."""
        result = anp.arcsin(1.0)
        np.testing.assert_almost_equal(result, np.pi / 2)

    def test_arccos_numpy(self):
        """Test arccos with NumPy."""
        result = anp.arccos(1.0)
        np.testing.assert_almost_equal(result, 0.0)

    def test_arctan_numpy(self):
        """Test arctan with NumPy."""
        result = anp.arctan(1.0)
        np.testing.assert_almost_equal(result, np.pi / 4)

    def test_arctan2_numpy(self):
        """Test arctan2 with NumPy."""
        result = anp.arctan2(1.0, 1.0)
        np.testing.assert_almost_equal(result, np.pi / 4)


class TestArithmetic:
    """Test arithmetic operations."""

    def test_abs_numpy(self):
        """Test abs with NumPy."""
        result = anp.abs(-5.0)
        assert result == 5.0
        result = anp.abs(np.array([-1.0, 2.0, -3.0]))
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_abs_casadi(self):
        """Test abs with CasADi."""
        x = cas.SX.sym("x")
        result = anp.abs(x)
        assert isinstance(result, cas.SX)

    def test_sign_numpy(self):
        """Test sign with NumPy."""
        x = np.array([-5.0, 0.0, 3.0])
        result = anp.sign(x)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_floor_numpy(self):
        """Test floor with NumPy."""
        x = np.array([1.5, 2.9, -1.5])
        result = anp.floor(x)
        expected = np.array([1.0, 2.0, -2.0])
        np.testing.assert_array_equal(result, expected)

    def test_ceil_numpy(self):
        """Test ceil with NumPy."""
        x = np.array([1.5, 2.9, -1.5])
        result = anp.ceil(x)
        expected = np.array([2.0, 3.0, -1.0])
        np.testing.assert_array_equal(result, expected)


class TestComparisons:
    """Test comparison operations."""

    def test_equal_numpy(self):
        """Test equal with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([1, 0, 3])
        result = anp.equal(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_equal_casadi(self):
        """Test equal with CasADi."""
        x = cas.SX.sym("x")
        result = anp.equal(x, 1.0)
        assert isinstance(result, cas.SX)

    def test_not_equal_numpy(self):
        """Test not_equal with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([1, 0, 3])
        result = anp.not_equal(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_greater_numpy(self):
        """Test greater with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([0, 2, 4])
        result = anp.greater(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_greater_equal_numpy(self):
        """Test greater_equal with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([0, 2, 4])
        result = anp.greater_equal(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_less_numpy(self):
        """Test less with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([0, 2, 4])
        result = anp.less(x, y)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_less_equal_numpy(self):
        """Test less_equal with NumPy."""
        x = np.array([1, 2, 3])
        y = np.array([0, 2, 4])
        result = anp.less_equal(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result, expected)


class TestReductions:
    """Test reduction operations."""

    def test_sum_numpy(self):
        """Test sum with NumPy."""
        x = np.array([1.0, 2.0, 3.0])
        result = anp.sum(x)
        assert result == 6.0

    def test_minimum_numpy(self):
        """Test minimum with NumPy."""
        result = anp.minimum(3.0, 5.0)
        assert result == 3.0
        x = np.array([1.0, 5.0, 3.0])
        y = np.array([2.0, 4.0, 3.0])
        result = anp.minimum(x, y)
        expected = np.array([1.0, 4.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_maximum_numpy(self):
        """Test maximum with NumPy."""
        result = anp.maximum(3.0, 5.0)
        assert result == 5.0
        x = np.array([1.0, 5.0, 3.0])
        y = np.array([2.0, 4.0, 3.0])
        result = anp.maximum(x, y)
        expected = np.array([2.0, 5.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestConditionals:
    """Test conditional operations."""

    def test_where_numpy(self):
        """Test where with NumPy."""
        condition = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = anp.where(condition, x, y)
        expected = np.array([1.0, 5.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_where_scalar(self):
        """Test where with scalars."""
        result = anp.where(True, 1.0, 2.0)
        assert result == 1.0
        result = anp.where(False, 1.0, 2.0)
        assert result == 2.0

    def test_where_casadi(self):
        """Test where with CasADi."""
        x = cas.SX.sym("x")
        result = anp.where(x > 0, x, -x)
        assert isinstance(result, cas.SX)


class TestArrayCreation:
    """Test array creation functions."""

    def test_asarray(self):
        """Test asarray with list."""
        result = anp.asarray([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_zeros_like(self):
        """Test zeros_like."""
        x = np.array([1.0, 2.0, 3.0])
        result = anp.zeros_like(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_ones_like(self):
        """Test ones_like."""
        x = np.array([1.0, 2.0, 3.0])
        result = anp.ones_like(x)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)


class TestCasADiSpecific:
    """Test CasADi-specific behavior."""

    def test_sqrt_casadi(self):
        """Test sqrt with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.sqrt(x)
        assert isinstance(result, cas.SX)
        # Evaluate
        f = cas.Function("f", [x], [result])
        assert float(f(4.0)) == pytest.approx(2.0)

    def test_power_casadi(self):
        """Test power with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.power(x, 2.0)
        assert isinstance(result, cas.SX)
        f = cas.Function("f", [x], [result])
        assert float(f(3.0)) == pytest.approx(9.0)

    def test_exp_casadi(self):
        """Test exp with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.exp(x)
        assert isinstance(result, cas.SX)

    def test_log_casadi(self):
        """Test log with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.log(x)
        assert isinstance(result, cas.SX)

    def test_sin_casadi(self):
        """Test sin with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.sin(x)
        assert isinstance(result, cas.SX)

    def test_cos_casadi(self):
        """Test cos with CasADi symbol."""
        x = cas.SX.sym("x")
        result = anp.cos(x)
        assert isinstance(result, cas.SX)

    def test_minimum_casadi(self):
        """Test minimum with CasADi symbols."""
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        result = anp.minimum(x, y)
        assert isinstance(result, cas.SX)

    def test_maximum_casadi(self):
        """Test maximum with CasADi symbols."""
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        result = anp.maximum(x, y)
        assert isinstance(result, cas.SX)
