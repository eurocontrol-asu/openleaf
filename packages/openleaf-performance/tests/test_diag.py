from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.array import diag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ca():
    """Return the casadi module."""
    casadi = pytest.importorskip("casadi")
    return casadi


@pytest.fixture
def casadi_col_vector(ca):
    """A CasADi column vector [1, 2, 3]."""
    return ca.DM([1, 2, 3])


@pytest.fixture
def casadi_square_matrix(ca):
    """A 3x3 CasADi matrix with known values."""
    return ca.DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# ---------------------------------------------------------------------------
# Unit tests — construct diagonal matrix from vector
# ---------------------------------------------------------------------------


def test_diag_construct_casadi_zero_offset(ca, casadi_col_vector):
    """k=0: values on main diagonal."""
    result = diag(casadi_col_vector, k=0)
    expected = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    np.testing.assert_array_equal(np.array(result), np.array(expected))


def test_diag_construct_casadi_positive_offset(ca, casadi_col_vector):
    """k=1: values on the superdiagonal."""
    result = diag(casadi_col_vector, k=1)
    expected = np.zeros((4, 4))
    expected[0, 1] = 1
    expected[1, 2] = 2
    expected[2, 3] = 3
    np.testing.assert_array_equal(np.array(result), expected)


def test_diag_construct_casadi_negative_offset(ca, casadi_col_vector):
    """k=-1: values on the subdiagonal."""
    result = diag(casadi_col_vector, k=-1)
    expected = np.zeros((4, 4))
    expected[1, 0] = 1
    expected[2, 1] = 2
    expected[3, 2] = 3
    np.testing.assert_array_equal(np.array(result), expected)


# ---------------------------------------------------------------------------
# Unit tests — extract diagonal from square matrix
# ---------------------------------------------------------------------------


def test_diag_extract_casadi(ca, casadi_square_matrix):
    """k=0: extract main diagonal."""
    result = diag(casadi_square_matrix, k=0)
    expected = np.array([1, 5, 9], dtype=float)
    np.testing.assert_array_equal(np.array(result.full()).flatten(), expected)


def test_diag_extract_casadi_offset(ca, casadi_square_matrix):
    """k=1: extract superdiagonal."""
    result = diag(casadi_square_matrix, k=1)
    expected = np.array([2, 6], dtype=float)
    np.testing.assert_array_equal(np.array(result.full()).flatten(), expected)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_diag_row_vector_input(ca):
    """Row vector (1, N) should be transposed to (N, 1) then construct."""
    row = ca.DM([[1, 2, 3]])  # shape (1, 3)
    result = diag(row, k=0)
    expected = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    np.testing.assert_array_equal(np.array(result), np.array(expected))


def test_diag_non_square_matrix(ca):
    """Non-square matrix should raise NotImplementedError."""
    mat = ca.DM([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # (3, 4)
    with pytest.raises(NotImplementedError):
        diag(mat)
