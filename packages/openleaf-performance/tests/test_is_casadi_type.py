from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import is_casadi_type


@pytest.fixture()
def casadi_module():
    """Return the casadi module if available, skip otherwise."""
    if not Backend.CASADI.available:
        pytest.skip("CasADi not installed")
    return Backend.CASADI.module


@pytest.fixture()
def casadi_dm(casadi_module):
    """Return a CasADi DM scalar."""
    return casadi_module.DM(1.0)


# --- Unit tests ---


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        1,
        True,
        "hello",
        None,
        b"bytes",
        bytearray(b"ba"),
        memoryview(b"mv"),
        1 + 2j,
        range(10),
    ],
    ids=[
        "float",
        "int",
        "bool",
        "str",
        "None",
        "bytes",
        "bytearray",
        "memoryview",
        "complex",
        "range",
    ],
)
def test_is_casadi_type_scalar_types(casadi_module, value):
    """Scalar Python types must always return False."""
    assert is_casadi_type(value) is False
    assert is_casadi_type(value, recursive=False) is False


def test_is_casadi_type_numpy_array(casadi_module):
    """Standard float64 numpy array must return False."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert is_casadi_type(arr) is False


def test_is_casadi_type_object_array_nested(casadi_module):
    """Numpy object array containing CasADi MX returns True with recursive=True."""
    ca = casadi_module
    mx = ca.MX.sym("x")
    arr = np.array([mx], dtype=object)
    assert is_casadi_type(arr, recursive=True) is True


def test_is_casadi_type_container_recursive_list(casadi_dm):
    """List containing CasADi DM: True with recursive, False without."""
    assert is_casadi_type([casadi_dm], recursive=True) is True
    assert is_casadi_type([casadi_dm], recursive=False) is False


def test_is_casadi_type_container_recursive_tuple(casadi_dm):
    """Tuple containing CasADi DM: True with recursive, False without."""
    assert is_casadi_type((casadi_dm,), recursive=True) is True
    assert is_casadi_type((casadi_dm,), recursive=False) is False


def test_is_casadi_type_container_recursive_set(casadi_dm):
    """Set containing CasADi DM: True with recursive, False without."""
    assert is_casadi_type({casadi_dm}, recursive=True) is True
    assert is_casadi_type({casadi_dm}, recursive=False) is False


def test_is_casadi_type_container_recursive_dict(casadi_dm):
    """Dict containing CasADi DM value: True with recursive, False without."""
    assert is_casadi_type({"key": casadi_dm}, recursive=True) is True
    assert is_casadi_type({"key": casadi_dm}, recursive=False) is False


# --- Edge cases ---


def test_is_casadi_type_zero_d_object_array(casadi_module):
    """0-d numpy object array wrapping CasADi MX calls .item() and returns True."""
    ca = casadi_module
    mx = ca.MX.sym("x")
    arr = np.array(mx, dtype=object)
    assert arr.ndim == 0
    assert is_casadi_type(arr, recursive=True) is True


def test_is_casadi_type_empty_containers(casadi_module):
    """Empty list, dict, and set must return False."""
    assert is_casadi_type([]) is False
    assert is_casadi_type({}) is False
    assert is_casadi_type(set()) is False


def test_is_casadi_type_not_installed(monkeypatch):
    """When CasADi is not available, return False immediately."""
    monkeypatch.setattr(Backend.CASADI, "available", False)
    assert is_casadi_type(42) is False
    assert is_casadi_type([1, 2, 3], recursive=True) is False
