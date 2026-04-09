from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.logicals import _ensure_pair


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def torch():
    """Import torch or skip."""
    return pytest.importorskip("torch")


@pytest.fixture
def jax():
    """Import jax or skip."""
    return pytest.importorskip("jax")


@pytest.fixture
def jnp(jax):
    """Import jax.numpy or skip."""
    import jax.numpy as jnp

    return jnp


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_ensure_pair_torch_scalar(torch):
    """Torch tensor + Python float → both Torch tensors with matching dtype/device."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    scalar = 3.0

    r1, r2 = _ensure_pair(t, scalar)

    assert isinstance(r1, torch.Tensor)
    assert isinstance(r2, torch.Tensor)
    assert r2.dtype == t.dtype
    assert r2.device == t.device


def test_ensure_pair_jax_numpy(jax, jnp):
    """JAX array + NumPy scalar → both JAX arrays with matching dtype."""
    j = jnp.array([1.0, 2.0], dtype=jnp.float32)
    n = np.float64(3.0)

    r1, r2 = _ensure_pair(j, n)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    assert r2.dtype == j.dtype


def test_ensure_pair_numpy_numpy():
    """Two NumPy arrays → returned unchanged."""
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])

    r1, r2 = _ensure_pair(a, b)

    assert r1 is a
    assert r2 is b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ensure_pair_jax_not_installed(monkeypatch):
    """When JAX is not installed and torch not involved, returns inputs unchanged."""
    a = np.array([1.0])
    b = np.array([2.0])

    import builtins

    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("No module named 'jax'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    r1, r2 = _ensure_pair(a, b)

    assert r1 is a
    assert r2 is b


def test_ensure_pair_both_torch(torch):
    """Two Torch tensors → returned unchanged (no conversion)."""
    t1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    t2 = torch.tensor([3.0, 4.0], dtype=torch.float32)

    r1, r2 = _ensure_pair(t1, t2)

    assert isinstance(r1, torch.Tensor)
    assert isinstance(r2, torch.Tensor)
    # Should be the original tensors, not converted copies
    assert torch.equal(r1, t1)
    assert torch.equal(r2, t2)
