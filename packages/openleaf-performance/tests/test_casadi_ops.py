"""Tests for CasADi-specific branches in aetherx.numpy.

These tests cover the CasADi code paths in array.py and arithmetic_monadic.py
that aren't reached by the default NumPy tests.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

import openleaf.performance.numpy as anp


# =============================================================================
# Arithmetic monadic: CasADi paths
# =============================================================================


class TestSumCasADi:
    """Tests for sum() with CasADi types."""

    def test_sum_axis_0(self):
        """Sum along rows (axis=0)."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.sum(x, axis=0)
        np.testing.assert_allclose(np.array(result).flatten(), [4, 6])

    def test_sum_axis_1(self):
        """Sum along columns (axis=1)."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.sum(x, axis=1)
        np.testing.assert_allclose(np.array(result).flatten(), [3, 7])

    def test_sum_axis_neg1(self):
        """Negative axis -1 maps to last axis."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.sum(x, axis=-1)
        np.testing.assert_allclose(np.array(result).flatten(), [3, 7])

    def test_sum_axis_neg2(self):
        """Negative axis -2 maps to axis 0."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.sum(x, axis=-2)
        np.testing.assert_allclose(np.array(result).flatten(), [4, 6])

    def test_sum_invalid_axis_raises(self):
        """Invalid axis raises ValueError."""
        x = ca.DM([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="CasADi types can only be up to 2D"):
            anp.sum(x, axis=2)


class TestMeanCasADi:
    """Tests for mean() with CasADi types."""

    def test_mean_axis_0(self):
        """Mean along axis 0."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.mean(x, axis=0)
        np.testing.assert_allclose(np.array(result).flatten(), [2, 3])

    def test_mean_axis_1(self):
        """Mean along axis 1."""
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.mean(x, axis=1)
        np.testing.assert_allclose(np.array(result).flatten(), [1.5, 3.5])

    def test_mean_axis_neg1(self):
        """Negative axis -1."""
        x = ca.DM([[2, 4], [6, 8]])
        result = anp.mean(x, axis=-1)
        np.testing.assert_allclose(np.array(result).flatten(), [3, 7])

    def test_mean_axis_neg2(self):
        """Negative axis -2."""
        x = ca.DM([[2, 4], [6, 8]])
        result = anp.mean(x, axis=-2)
        np.testing.assert_allclose(np.array(result).flatten(), [4, 6])

    def test_mean_invalid_axis_raises(self):
        """Invalid axis raises ValueError."""
        x = ca.DM([[1, 2]])
        with pytest.raises(ValueError, match="CasADi types can only be up to 2D"):
            anp.mean(x, axis=2)


class TestProdCasADi:
    """Tests for prod() with CasADi types."""

    def test_prod_column(self):
        """Product of column vector elements via log/exp trick."""
        x = ca.DM([2, 3, 4])
        result = float(anp.prod(x))
        assert result == pytest.approx(24.0, rel=1e-6)


class TestCbrtCasADi:
    """Tests for cbrt() with CasADi types."""

    def test_cbrt_positive(self):
        result = float(anp.cbrt(ca.DM(27)))
        assert result == pytest.approx(3.0, rel=1e-6)

    def test_cbrt_negative(self):
        result = float(anp.cbrt(ca.DM(-8)))
        assert result == pytest.approx(-2.0, rel=1e-6)


# =============================================================================
# Array operations: CasADi paths
# =============================================================================


class TestLengthCasADi:
    """Tests for length() with CasADi types."""

    def test_length_column_vector(self):
        """Column vector length."""
        x = ca.DM([1, 2, 3])
        assert anp.length(x) == 3

    def test_length_row_vector(self):
        """Row vector length uses second dimension."""
        x = ca.DM([1, 2, 3]).T
        assert anp.length(x) == 3

    def test_length_scalar(self):
        """Scalar returns 1."""
        assert anp.length(42) == 1

    def test_length_0d_array(self):
        """0-dim numpy array returns 1."""
        assert anp.length(np.float64(3.14)) == 1


class TestDiagCasADi:
    """Tests for diag() with CasADi types."""

    def test_diag_construct(self):
        """Construct diagonal matrix from vector."""
        v = ca.DM([1, 2, 3])
        result = anp.diag(v)
        assert result.shape == (3, 3)
        assert float(result[0, 0]) == 1
        assert float(result[1, 1]) == 2
        assert float(result[2, 2]) == 3
        assert float(result[0, 1]) == 0

    def test_diag_off_diagonal_construct(self):
        """Off-diagonal construction."""
        v = ca.DM([1, 2])
        result = anp.diag(v, k=1)
        assert result.shape == (3, 3)
        assert float(result[0, 1]) == 1
        assert float(result[1, 2]) == 2

    def test_diag_negative_offset_construct(self):
        """Negative offset diagonal construction."""
        v = ca.DM([5, 6])
        result = anp.diag(v, k=-1)
        assert result.shape == (3, 3)
        assert float(result[1, 0]) == 5
        assert float(result[2, 1]) == 6

    def test_diag_row_vector(self):
        """Row vector is transposed first."""
        v = ca.DM([1, 2, 3]).T
        result = anp.diag(v)
        assert result.shape == (3, 3)


class TestMaxMinCasADi:
    """Tests for max()/min() with CasADi types."""

    def test_max_all(self):
        x = ca.DM([[1, 5], [3, 2]])
        assert float(anp.max(x)) == 5

    def test_max_axis_0_column(self):
        """Single-column matrix axis=0."""
        x = ca.DM([3, 1, 5])
        assert float(anp.max(x, axis=0)) == 5

    def test_max_axis_1_row(self):
        """Single-row matrix axis=1."""
        x = ca.DM([3, 1, 5]).T
        assert float(anp.max(x, axis=1)) == 5

    def test_max_invalid_axis(self):
        x = ca.DM([[1, 2]])
        with pytest.raises(ValueError, match="Invalid axis"):
            anp.max(x, axis=2)

    def test_min_all(self):
        x = ca.DM([[1, 5], [3, 2]])
        assert float(anp.min(x)) == 1

    def test_min_axis_0_column(self):
        """Single-column matrix axis=0."""
        x = ca.DM([3, 1, 5])
        assert float(anp.min(x, axis=0)) == 1

    def test_min_axis_1_row(self):
        """Single-row matrix axis=1."""
        x = ca.DM([3, 1, 5]).T
        assert float(anp.min(x, axis=1)) == 1

    def test_min_invalid_axis(self):
        x = ca.DM([[1, 2]])
        with pytest.raises(ValueError, match="Invalid axis"):
            anp.min(x, axis=2)


class TestReshapeCasADi:
    """Tests for reshape() with CasADi types."""

    def test_reshape_to_column(self):
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.reshape(x, -1)
        assert result.shape[0] == 4

    def test_reshape_2d(self):
        x = ca.DM([1, 2, 3, 4])
        result = anp.reshape(x, (2, 2))
        assert result.shape == (2, 2)

    def test_reshape_1d_tuple(self):
        x = ca.DM([1, 2, 3])
        result = anp.reshape(x, (3,))
        assert result.shape[0] == 3

    def test_reshape_fortran_order(self):
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.reshape(x, (4, 1), order="F")
        assert result.shape[0] == 4

    def test_reshape_3d_raises(self):
        x = ca.DM([1, 2, 3, 4])
        with pytest.raises(ValueError, match="limited to no more than 2 dimensions"):
            anp.reshape(x, (2, 2, 1))

    def test_reshape_invalid_order_raises(self):
        x = ca.DM([1, 2])
        with pytest.raises(NotImplementedError, match="Only C and F orders"):
            anp.reshape(x, (2, 1), order="A")


class TestRavelCasADi:
    """Tests for ravel() with CasADi types."""

    def test_ravel(self):
        x = ca.DM([[1, 2], [3, 4]])
        result = anp.ravel(x)
        assert result.shape[0] == 4


class TestTileCasADi:
    """Tests for tile() with CasADi types."""

    def test_tile_1d(self):
        x = ca.DM([1, 2])
        result = anp.tile(x, (3,))
        assert result.shape[0] == 6

    def test_tile_2d(self):
        x = ca.DM([[1], [2]])
        result = anp.tile(x, (1, 3))
        assert result.shape == (2, 3)

    def test_tile_3d_raises(self):
        x = ca.DM([1])
        with pytest.raises(ValueError, match="Cannot have >2D"):
            anp.tile(x, (1, 2, 3))


class TestLikesCasADi:
    """Tests for zeros_like, ones_like, empty_like, full_like with CasADi."""

    def test_zeros_like(self):
        x = ca.DM([1, 2, 3])
        result = anp.zeros_like(x)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_ones_like(self):
        x = ca.DM([1, 2, 3])
        result = anp.ones_like(x)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_empty_like(self):
        x = ca.DM([1, 2])
        result = anp.empty_like(x)
        assert len(result) == 2

    def test_full_like(self):
        x = ca.DM([1, 2, 3])
        result = anp.full_like(x, 7.0)
        np.testing.assert_array_equal(result, [7, 7, 7])


class TestAssertEqualShape:
    """Tests for assert_equal_shape()."""

    def test_same_shape_list(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        anp.assert_equal_shape([a, b])

    def test_different_shape_list_raises(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        with pytest.raises(ValueError, match="do not have the same shape"):
            anp.assert_equal_shape([a, b])

    def test_same_shape_dict(self):
        a = np.array([1, 2])
        b = np.array([3, 4])
        anp.assert_equal_shape({"a": a, "b": b})

    def test_different_shape_dict_raises(self):
        a = np.array([1, 2])
        b = np.array([3, 4, 5])
        with pytest.raises(ValueError, match="a, b"):
            anp.assert_equal_shape({"a": a, "b": b})

    def test_scalars_same_shape(self):
        anp.assert_equal_shape([1.0, 2.0])


class TestReshapeNonCOrder:
    """Tests for reshape/ravel with non-C order (NumPy fallback)."""

    def test_reshape_fortran_order_numpy(self):
        x = np.array([[1, 2], [3, 4]])
        result = anp.reshape(x, (4,), order="F")
        np.testing.assert_array_equal(result, [1, 3, 2, 4])

    def test_ravel_fortran_order_numpy(self):
        x = np.array([[1, 2], [3, 4]])
        result = anp.ravel(x, order="F")
        np.testing.assert_array_equal(result, [1, 3, 2, 4])


# =============================================================================
# Math functions: CasADi paths
# =============================================================================


class TestMathFunctionsCasADi:
    """Tests for CasADi branches in math_functions.py."""

    def test_log10(self):
        result = float(anp.log10(ca.DM(100)))
        assert result == pytest.approx(2.0, rel=1e-10)

    def test_log2(self):
        result = float(anp.log2(ca.DM(8)))
        assert result == pytest.approx(3.0, rel=1e-10)

    def test_sign_positive(self):
        result = float(anp.sign(ca.DM(5)))
        assert result == 1.0

    def test_sign_negative(self):
        result = float(anp.sign(ca.DM(-3)))
        assert result == -1.0

    def test_tan(self):
        result = float(anp.tan(ca.DM(0)))
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_arcsin(self):
        result = float(anp.arcsin(ca.DM(0.5)))
        assert result == pytest.approx(np.arcsin(0.5), rel=1e-10)

    def test_arccos(self):
        result = float(anp.arccos(ca.DM(0.5)))
        assert result == pytest.approx(np.arccos(0.5), rel=1e-10)

    def test_arctan(self):
        result = float(anp.arctan(ca.DM(1.0)))
        assert result == pytest.approx(np.arctan(1.0), rel=1e-10)

    def test_sinh(self):
        result = float(anp.sinh(ca.DM(1.0)))
        assert result == pytest.approx(np.sinh(1.0), rel=1e-10)

    def test_cosh(self):
        result = float(anp.cosh(ca.DM(1.0)))
        assert result == pytest.approx(np.cosh(1.0), rel=1e-10)

    def test_tanh(self):
        result = float(anp.tanh(ca.DM(0.5)))
        assert result == pytest.approx(np.tanh(0.5), rel=1e-10)

    def test_arcsinh(self):
        result = float(anp.arcsinh(ca.DM(1.0)))
        assert result == pytest.approx(np.arcsinh(1.0), rel=1e-10)

    def test_arccosh(self):
        result = float(anp.arccosh(ca.DM(2.0)))
        assert result == pytest.approx(np.arccosh(2.0), rel=1e-10)

    def test_arctanh(self):
        result = float(anp.arctanh(ca.DM(0.5)))
        assert result == pytest.approx(np.arctanh(0.5), rel=1e-10)

    def test_floor(self):
        result = float(anp.floor(ca.DM(2.7)))
        assert result == 2.0

    def test_ceil(self):
        result = float(anp.ceil(ca.DM(2.3)))
        assert result == 3.0


# =============================================================================
# Logical operations: CasADi paths
# =============================================================================


class TestLogicalsCasADi:
    """Tests for CasADi branches in logicals.py."""

    def test_clip_min_only(self):
        result = float(anp.clip(ca.DM(-1), min=0))
        assert result == 0.0

    def test_clip_max_only(self):
        result = float(anp.clip(ca.DM(5), max=3))
        assert result == 3.0

    def test_clip_both(self):
        result = float(anp.clip(ca.DM(5), min=0, max=3))
        assert result == 3.0

    def test_all_casadi_axis_raises(self):
        x = ca.DM([1, 1, 1])
        with pytest.raises(NotImplementedError, match="axis parameter not supported"):
            anp.all(x, axis=0)

    def test_any_casadi_axis_raises(self):
        x = ca.DM([0, 1, 0])
        with pytest.raises(NotImplementedError, match="axis parameter not supported"):
            anp.any(x, axis=0)
