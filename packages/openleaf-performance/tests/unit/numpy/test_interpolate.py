from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from openleaf.performance.numpy.interpolate import _apply_fill_value
from openleaf.performance.numpy.interpolate import _convert_casadi_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oob_data():
    """Grid [0,1] x [0,1] with xi containing out-of-bounds points."""
    fi = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0])  # 5x1 column (matches interpolator output shape)
    xi = np.array(
        [
            [0.5, 0.5],  # in-bounds
            [1.5, 0.5],  # OOB axis 0 (above max)
            [0.5, -0.5],  # OOB axis 1 (below min)
            [-0.1, 0.5],  # OOB axis 0 (below min)
            [0.5, 1.5],  # OOB axis 1 (above max)
        ]
    )
    axis_min = [0.0, 0.0]
    axis_max = [1.0, 1.0]
    return fi, xi, axis_min, axis_max


# ---------------------------------------------------------------------------
# Unit tests — _apply_fill_value
# ---------------------------------------------------------------------------


def test_apply_fill_value_masks_oob(oob_data):
    """OOB positions are replaced with fill_value."""
    fi, xi, axis_min, axis_max = oob_data
    n_dimensions = 2
    fill_value = -999.0

    result = _apply_fill_value(fi, xi, axis_min, axis_max, n_dimensions, fill_value)

    result_arr = np.array(result, dtype=float).reshape(-1)
    # Index 0 is in-bounds → original value preserved
    assert result_arr[0] == pytest.approx(1.0)
    # Indices 1-4 are OOB → replaced with fill_value
    assert result_arr[1] == pytest.approx(fill_value)
    assert result_arr[2] == pytest.approx(fill_value)
    assert result_arr[3] == pytest.approx(fill_value)
    assert result_arr[4] == pytest.approx(fill_value)


# ---------------------------------------------------------------------------
# Unit tests — _convert_casadi_result
# ---------------------------------------------------------------------------


def test_convert_casadi_result_scalar():
    """1x1 DM returns a Python float."""
    dm = ca.DM(3.14)
    result = _convert_casadi_result(dm)

    assert isinstance(result, float)
    assert result == pytest.approx(3.14)


def test_convert_casadi_result_array():
    """Nx1 DM returns a 1D numpy array."""
    dm = ca.DM([1.0, 2.0, 3.0])
    result = _convert_casadi_result(dm)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_convert_casadi_result_passthrough():
    """MX symbolic is returned unchanged."""
    mx = ca.MX.sym("x", 3)
    result = _convert_casadi_result(mx)

    assert result is mx
