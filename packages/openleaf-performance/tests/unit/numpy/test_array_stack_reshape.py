from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.array import reshape
from openleaf.performance.numpy.array import stack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ca():
    """Import and return the casadi module, skip if unavailable."""
    casadi = pytest.importorskip("casadi")
    return casadi


@pytest.fixture()
def casadi_vectors(ca):
    """Three CasADi Nx1 column vectors."""
    return [ca.DM([[1], [2], [3]]), ca.DM([[4], [5], [6]]), ca.DM([[7], [8], [9]])]


@pytest.fixture()
def casadi_matrix(ca):
    """A 2x3 CasADi matrix."""
    return ca.DM([[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# Unit tests — stack
# ---------------------------------------------------------------------------


class TestStackCasadiAxis0:
    """test_stack_casadi_axis0: List of CasADi Nx1 vectors, axis=0 → rows."""

    def test_stacked_rows(self, ca, casadi_vectors):
        result = stack(casadi_vectors, axis=0)
        expected = ca.DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_axis_minus2_is_same_as_axis0(self, ca, casadi_vectors):
        r0 = stack(casadi_vectors, axis=0)
        rm2 = stack(casadi_vectors, axis=-2)
        np.testing.assert_array_equal(np.array(r0), np.array(rm2))


class TestStackCasadiAxis1:
    """test_stack_casadi_axis1: List of CasADi Nx1 vectors, axis=1 → cols."""

    def test_stacked_cols(self, ca, casadi_vectors):
        result = stack(casadi_vectors, axis=1)
        expected = ca.DM([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_axis_minus1_is_same_as_axis1(self, ca, casadi_vectors):
        r1 = stack(casadi_vectors, axis=1)
        rm1 = stack(casadi_vectors, axis=-1)
        np.testing.assert_array_equal(np.array(r1), np.array(rm1))


# ---------------------------------------------------------------------------
# Unit tests — reshape
# ---------------------------------------------------------------------------


class TestReshapeCasadiCOrder:
    """test_reshape_casadi_c_order: CasADi matrix, order='C'."""

    def test_c_order_reshape(self, ca, casadi_matrix):
        result = reshape(casadi_matrix, (3, 2), order="C")
        # C-order: row-major flattening → [[1,2],[3,4],[5,6]]
        expected = ca.DM([[1, 2], [3, 4], [5, 6]])
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(np.array(result), np.array(expected))


class TestReshapeCasadiFOrder:
    """test_reshape_casadi_f_order: CasADi matrix, order='F'."""

    def test_f_order_reshape(self, ca, casadi_matrix):
        result = reshape(casadi_matrix, (3, 2), order="F")
        # F-order: column-major flattening → [[1,5],[4,3],[2,6]]
        expected = ca.DM([[1, 5], [4, 3], [2, 6]])
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(np.array(result), np.array(expected))


# ---------------------------------------------------------------------------
# Edge cases — stack
# ---------------------------------------------------------------------------


class TestStackEdgeCases:
    def test_non_nx1_casadi_raises(self, ca):
        """CasADi array with shape (N,M) where M>1 → ValueError."""
        wide = ca.DM([[1, 2], [3, 4]])
        col = ca.DM([[5], [6]])
        with pytest.raises(ValueError, match="Can only stack Nx1 CasADi"):
            stack([wide, col], axis=0)

    def test_mixed_numpy_1d_and_casadi(self, ca):
        """NumPy 1D + CasADi Nx1 stacks correctly."""
        np_arr = np.array([1.0, 2.0, 3.0])
        ca_arr = ca.DM([[4], [5], [6]])
        result = stack([np_arr, ca_arr], axis=0)
        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_invalid_axis_raises(self, ca, casadi_vectors):
        """Axis outside {0, 1, -1, -2} → ValueError."""
        with pytest.raises(ValueError, match="axis"):
            stack(casadi_vectors, axis=2)


# ---------------------------------------------------------------------------
# Edge cases — reshape
# ---------------------------------------------------------------------------


class TestReshapeEdgeCases:
    def test_int_newshape_treated_as_column(self, ca):
        """reshape(casadi, 6) → treated as (6, 1)."""
        arr = ca.DM([[1, 2, 3], [4, 5, 6]])
        result = reshape(arr, 6, order="F")
        assert result.shape == (6, 1)

    def test_greater_than_2d_raises(self, ca):
        """newshape with >2 dims → ValueError."""
        arr = ca.DM([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="no more than 2 dimensions"):
            reshape(arr, (2, 3, 4))

    def test_unsupported_order_raises(self, ca):
        """Order other than C/F → NotImplementedError."""
        arr = ca.DM([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError, match="Only C and F"):
            reshape(arr, (4, 1), order="A")

    def test_1d_newshape_becomes_column(self, ca):
        """reshape(casadi, (6,)) → (6, 1)."""
        arr = ca.DM([[1, 2, 3], [4, 5, 6]])
        result = reshape(arr, (6,), order="F")
        assert result.shape == (6, 1)
