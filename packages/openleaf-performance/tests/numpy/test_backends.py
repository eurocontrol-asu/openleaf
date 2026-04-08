"""Tests for the multi-backend dispatch system."""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.numpy.backends import Backend
from openleaf.performance.numpy.backends import get_backend
from openleaf.performance.numpy.backends import is_casadi_type
from openleaf.performance.numpy.backends import is_torch_type
from openleaf.performance.numpy.backends import require_backend


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_numpy_always_available(self):
        """NumPy should always be available."""
        assert Backend.NUMPY.available is True
        assert Backend.NUMPY.module is not None

    def test_casadi_availability(self):
        """CasADi availability should be correctly detected."""
        # This test just verifies the property works without error
        available = Backend.CASADI.available
        assert isinstance(available, bool)

        if available:
            assert Backend.CASADI.module is not None
        else:
            assert Backend.CASADI.module is None

    def test_torch_availability(self):
        """PyTorch availability should be correctly detected."""
        available = Backend.TORCH.available
        assert isinstance(available, bool)

        if available:
            assert Backend.TORCH.module is not None
        else:
            assert Backend.TORCH.module is None


class TestGetBackend:
    """Tests for get_backend function."""

    def test_numpy_array(self):
        """NumPy arrays should return NUMPY backend."""
        x = np.array([1, 2, 3])
        assert get_backend(x) == Backend.NUMPY

    def test_python_scalars(self):
        """Python scalars should return NUMPY backend."""
        assert get_backend(1) == Backend.NUMPY
        assert get_backend(1.0) == Backend.NUMPY
        assert get_backend([1, 2, 3]) == Backend.NUMPY

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_casadi_types(self):
        """CasADi types should return CASADI backend."""
        ca = Backend.CASADI.module
        assert get_backend(ca.DM([1, 2, 3])) == Backend.CASADI
        assert get_backend(ca.MX.sym("x")) == Backend.CASADI
        assert get_backend(ca.SX.sym("x")) == Backend.CASADI

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_torch_tensors(self):
        """PyTorch tensors should return TORCH backend."""
        torch = Backend.TORCH.module
        x = torch.tensor([1.0, 2.0, 3.0])
        assert get_backend(x) == Backend.TORCH


class TestIsCasadiType:
    """Tests for is_casadi_type function."""

    def test_numpy_array_returns_false(self):
        """NumPy arrays should not be CasADi types."""
        x = np.array([1, 2, 3])
        assert is_casadi_type(x) is False

    def test_python_scalars_return_false(self):
        """Python scalars should not be CasADi types."""
        assert is_casadi_type(1) is False
        assert is_casadi_type(1.0) is False
        assert is_casadi_type("string") is False

    def test_list_of_numpy_returns_false(self):
        """Lists of NumPy arrays should not contain CasADi types."""
        x = [np.array([1, 2]), np.array([3, 4])]
        assert is_casadi_type(x, recursive=True) is False

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_casadi_dm(self):
        """CasADi DM should be detected."""
        ca = Backend.CASADI.module
        x = ca.DM([1, 2, 3])
        assert is_casadi_type(x) is True

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_casadi_mx(self):
        """CasADi MX should be detected."""
        ca = Backend.CASADI.module
        x = ca.MX.sym("x", 3)
        assert is_casadi_type(x) is True

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_casadi_sx(self):
        """CasADi SX should be detected."""
        ca = Backend.CASADI.module
        x = ca.SX.sym("x", 3)
        assert is_casadi_type(x) is True

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_list_with_casadi_recursive(self):
        """Lists containing CasADi types should be detected recursively."""
        ca = Backend.CASADI.module
        x = [np.array([1, 2]), ca.DM([3, 4])]
        assert is_casadi_type(x, recursive=True) is True
        assert is_casadi_type(x, recursive=False) is False


class TestIsTorchType:
    """Tests for is_torch_type function."""

    def test_numpy_array_returns_false(self):
        """NumPy arrays should not be torch types."""
        x = np.array([1, 2, 3])
        assert is_torch_type(x) is False

    def test_python_scalars_return_false(self):
        """Python scalars should not be torch types."""
        assert is_torch_type(1) is False
        assert is_torch_type(1.0) is False

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_torch_tensor(self):
        """PyTorch tensors should be detected."""
        torch = Backend.TORCH.module
        x = torch.tensor([1.0, 2.0, 3.0])
        assert is_torch_type(x) is True

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_list_with_torch_recursive(self):
        """Lists containing PyTorch tensors should be detected recursively."""
        torch = Backend.TORCH.module
        x = [np.array([1, 2]), torch.tensor([3.0, 4.0])]
        assert is_torch_type(x, recursive=True) is True
        assert is_torch_type(x, recursive=False) is False


class TestRequireBackend:
    """Tests for require_backend function."""

    def test_numpy_never_raises(self):
        """Requiring NumPy should never raise."""
        require_backend(Backend.NUMPY, "Test feature")  # Should not raise

    def test_unavailable_backend_raises(self):
        """Requiring unavailable backend should raise ImportError."""
        # Create a mock unavailable backend scenario by checking a known unavailable one
        # or by testing the error message format
        if not Backend.TORCH.available:
            with pytest.raises(ImportError, match="requires torch"):
                require_backend(Backend.TORCH, "Test feature")

    @pytest.mark.skipif(not Backend.CASADI.available, reason="CasADi not installed")
    def test_casadi_available_no_raise(self):
        """Requiring CasADi when available should not raise."""
        require_backend(Backend.CASADI, "Test feature")  # Should not raise

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_torch_available_no_raise(self):
        """Requiring PyTorch when available should not raise."""
        require_backend(Backend.TORCH, "Test feature")  # Should not raise


class TestArrayAPIDispatch:
    """Tests for Array API dispatch across backends."""

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_sqrt_numpy_and_torch(self):
        """sqrt should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import sqrt

        torch = Backend.TORCH.module
        x_np = np.array([1.0, 4.0, 9.0, 16.0])
        x_torch = torch.tensor([1.0, 4.0, 9.0, 16.0])

        result_np = sqrt(x_np)
        result_torch = sqrt(x_torch)

        np.testing.assert_allclose(result_np, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result_torch.numpy(), [1.0, 2.0, 3.0, 4.0])

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_exp_log_numpy_and_torch(self):
        """exp and log should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import exp
        from openleaf.performance.numpy import log

        torch = Backend.TORCH.module
        x_np = np.array([1.0, 2.0, 3.0])
        x_torch = torch.tensor([1.0, 2.0, 3.0])

        # Test exp
        exp_np = exp(x_np)
        exp_torch = exp(x_torch)
        np.testing.assert_allclose(exp_torch.numpy(), exp_np, rtol=1e-6)

        # Test log
        log_np = log(x_np)
        log_torch = log(x_torch)
        np.testing.assert_allclose(log_torch.numpy(), log_np, rtol=1e-6)

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_trig_functions_numpy_and_torch(self):
        """Trigonometric functions should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import cos
        from openleaf.performance.numpy import sin
        from openleaf.performance.numpy import tan

        torch = Backend.TORCH.module
        x_np = np.array([0.0, np.pi / 4, np.pi / 2])
        x_torch = torch.tensor([0.0, np.pi / 4, np.pi / 2])

        # Use atol for values near zero (float32 vs float64 precision)
        np.testing.assert_allclose(
            sin(x_torch).numpy(), sin(x_np), rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            cos(x_torch).numpy(), cos(x_np), rtol=1e-6, atol=1e-7
        )
        # tan at pi/2 is undefined, so test only valid values
        x_np = np.array([0.0, np.pi / 4, np.pi / 3])
        x_torch = torch.tensor([0.0, np.pi / 4, np.pi / 3])
        np.testing.assert_allclose(tan(x_torch).numpy(), tan(x_np), rtol=1e-6)

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_sum_mean_numpy_and_torch(self):
        """sum and mean should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import mean
        from openleaf.performance.numpy import sum

        torch = Backend.TORCH.module
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Full reduction
        np.testing.assert_allclose(float(sum(x_torch)), float(sum(x_np)))
        np.testing.assert_allclose(float(mean(x_torch)), float(mean(x_np)))

        # Axis reduction
        np.testing.assert_allclose(sum(x_torch, axis=0).numpy(), sum(x_np, axis=0))
        np.testing.assert_allclose(sum(x_torch, axis=1).numpy(), sum(x_np, axis=1))
        np.testing.assert_allclose(mean(x_torch, axis=0).numpy(), mean(x_np, axis=0))

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_abs_numpy_and_torch(self):
        """abs should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import abs

        torch = Backend.TORCH.module
        x_np = np.array([-1.0, 2.0, -3.0, 4.0])
        x_torch = torch.tensor([-1.0, 2.0, -3.0, 4.0])

        np.testing.assert_allclose(abs(x_torch).numpy(), abs(x_np))

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_where_numpy_and_torch(self):
        """where should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import where

        torch = Backend.TORCH.module
        cond_np = np.array([True, False, True, False])
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        y_np = np.array([10.0, 20.0, 30.0, 40.0])

        cond_torch = torch.tensor([True, False, True, False])
        x_torch = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_torch = torch.tensor([10.0, 20.0, 30.0, 40.0])

        result_np = where(cond_np, x_np, y_np)
        result_torch = where(cond_torch, x_torch, y_torch)

        np.testing.assert_allclose(result_torch.numpy(), result_np)

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_maximum_minimum_numpy_and_torch(self):
        """maximum and minimum should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import maximum
        from openleaf.performance.numpy import minimum

        torch = Backend.TORCH.module
        x_np = np.array([1.0, 5.0, 3.0])
        y_np = np.array([2.0, 2.0, 4.0])

        x_torch = torch.tensor([1.0, 5.0, 3.0])
        y_torch = torch.tensor([2.0, 2.0, 4.0])

        np.testing.assert_allclose(
            maximum(x_torch, y_torch).numpy(), maximum(x_np, y_np)
        )
        np.testing.assert_allclose(
            minimum(x_torch, y_torch).numpy(), minimum(x_np, y_np)
        )

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_clip_numpy_and_torch(self):
        """clip should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import clip

        torch = Backend.TORCH.module
        x_np = np.array([1.0, 5.0, 10.0, 15.0])
        x_torch = torch.tensor([1.0, 5.0, 10.0, 15.0])

        np.testing.assert_allclose(
            clip(x_torch, 3.0, 12.0).numpy(), clip(x_np, 3.0, 12.0)
        )

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_reshape_numpy_and_torch(self):
        """reshape should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import reshape

        torch = Backend.TORCH.module
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        x_torch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        np.testing.assert_allclose(
            reshape(x_torch, (2, 3)).numpy(), reshape(x_np, (2, 3))
        )

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_concatenate_numpy_and_torch(self):
        """concatenate should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import concatenate

        torch = Backend.TORCH.module
        a_np = np.array([1.0, 2.0])
        b_np = np.array([3.0, 4.0])

        a_torch = torch.tensor([1.0, 2.0])
        b_torch = torch.tensor([3.0, 4.0])

        np.testing.assert_allclose(
            concatenate([a_torch, b_torch]).numpy(), concatenate([a_np, b_np])
        )

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_stack_numpy_and_torch(self):
        """stack should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import stack

        torch = Backend.TORCH.module
        a_np = np.array([1.0, 2.0])
        b_np = np.array([3.0, 4.0])

        a_torch = torch.tensor([1.0, 2.0])
        b_torch = torch.tensor([3.0, 4.0])

        np.testing.assert_allclose(
            stack([a_torch, b_torch]).numpy(), stack([a_np, b_np])
        )

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_zeros_like_ones_like_numpy_and_torch(self):
        """zeros_like and ones_like should work with PyTorch."""
        from openleaf.performance.numpy import ones_like
        from openleaf.performance.numpy import zeros_like

        torch = Backend.TORCH.module
        x_torch = torch.tensor([1.0, 2.0, 3.0])

        zeros = zeros_like(x_torch)
        ones = ones_like(x_torch)

        assert isinstance(zeros, torch.Tensor)
        assert isinstance(ones, torch.Tensor)
        np.testing.assert_allclose(zeros.numpy(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ones.numpy(), [1.0, 1.0, 1.0])

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_logical_ops_numpy_and_torch(self):
        """Logical operations should work identically for NumPy and PyTorch."""
        from openleaf.performance.numpy import logical_and
        from openleaf.performance.numpy import logical_not
        from openleaf.performance.numpy import logical_or

        torch = Backend.TORCH.module
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])

        a_torch = torch.tensor([True, True, False, False])
        b_torch = torch.tensor([True, False, True, False])

        np.testing.assert_array_equal(
            logical_and(a_torch, b_torch).numpy(), logical_and(a_np, b_np)
        )
        np.testing.assert_array_equal(
            logical_or(a_torch, b_torch).numpy(), logical_or(a_np, b_np)
        )
        np.testing.assert_array_equal(logical_not(a_torch).numpy(), logical_not(a_np))

    @pytest.mark.skipif(not Backend.TORCH.available, reason="PyTorch not installed")
    def test_get_array_namespace(self):
        """get_array_namespace should return correct namespace."""
        from openleaf.performance.numpy.backends import get_array_namespace

        torch = Backend.TORCH.module

        # NumPy array
        x_np = np.array([1.0, 2.0, 3.0])
        xp_np = get_array_namespace(x_np)
        assert xp_np is not None

        # PyTorch tensor
        x_torch = torch.tensor([1.0, 2.0, 3.0])
        xp_torch = get_array_namespace(x_torch)
        assert xp_torch is not None

        # Scalars should fall back to NumPy
        xp_scalar = get_array_namespace(1.0, 2.0)
        assert xp_scalar is not None
