from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.conditionals import _promote_to_common_backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def torch():
    """Import torch or skip."""
    return pytest.importorskip("torch")


@pytest.fixture()
def jax():
    """Import jax or skip."""
    return pytest.importorskip("jax")


@pytest.fixture()
def jnp(jax):
    """Import jax.numpy or skip."""
    import jax.numpy as jnp

    return jnp


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_promote_torch_mixed(torch):
    """Numpy array + torch tensor -> both returned as torch tensors on same device."""
    np_arr = np.array([1.0, 2.0])
    t_arr = torch.tensor([3.0, 4.0])

    a, b = _promote_to_common_backend(np_arr, t_arr)

    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert a.device == b.device


def test_promote_torch_bool_condition(torch):
    """Torch bool tensor + numpy float -> bool keeps dtype, float converted to torch."""
    cond = torch.tensor(True)
    val = np.float64(3.14)

    a, b = _promote_to_common_backend(cond, val)

    assert isinstance(a, torch.Tensor)
    assert a.dtype == torch.bool
    assert isinstance(b, torch.Tensor)


def test_promote_jax_mixed(jax, jnp):
    """Numpy array + jax array -> both returned as jax arrays."""
    np_arr = np.array([1.0, 2.0])
    jax_arr = jnp.array([3.0, 4.0])

    a, b = _promote_to_common_backend(np_arr, jax_arr)

    assert isinstance(a, jax.Array)
    assert isinstance(b, jax.Array)


def test_promote_no_backend():
    """All numpy arrays -> returns args unchanged."""
    a = np.array([1.0])
    b = np.array([2.0])

    result = _promote_to_common_backend(a, b)

    assert result[0] is a
    assert result[1] is b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_promote_torch_float64_promotion(torch):
    """torch.float32 + torch.float64 + numpy -> all promoted to float64."""
    t32 = torch.tensor([1.0], dtype=torch.float32)
    t64 = torch.tensor([2.0], dtype=torch.float64)
    np_arr = np.array([3.0])

    a, b, c = _promote_to_common_backend(t32, t64, np_arr)

    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(c, torch.Tensor)
    # float64 is the widest -> numpy should be promoted to float64
    assert c.dtype == torch.float64


def test_promote_torch_bool_plus_float(torch):
    """Bool tensor stays bool, float tensor unchanged."""
    cond = torch.tensor(True)
    val = torch.tensor(1.5)

    a, b = _promote_to_common_backend(cond, val)

    assert a.dtype == torch.bool
    assert b.dtype == val.dtype


def test_promote_jax_not_installed(monkeypatch):
    """Only numpy args, JAX import fails -> returns args as-is, no error."""
    import builtins

    real_import = builtins.__import__

    def _block_jax(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_jax)

    a = np.array([1.0])
    b = np.array([2.0])

    result = _promote_to_common_backend(a, b)

    assert result[0] is a
    assert result[1] is b


def test_promote_empty_args():
    """Empty call -> returns empty tuple."""
    result = _promote_to_common_backend()

    assert result == ()
