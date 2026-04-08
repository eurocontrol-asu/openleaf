"""GPU/CUDA validation tests for PyTorch backend.

These tests verify that AetherX functions produce correct results on GPU
when CUDA is available. All tests are automatically skipped if no GPU is
present, so they are safe to run in any environment.
"""

from __future__ import annotations

import pytest

import numpy as np


torch = pytest.importorskip("torch", reason="PyTorch not installed")

# Skip entire module if no CUDA GPU is available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU not available",
)


import openleaf.performance.numpy as anp
from openleaf.performance.physics import aero
from openleaf.performance.physics import isa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gpu(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a CUDA tensor (float64)."""
    return torch.tensor(arr, dtype=torch.float64, device="cuda")


def _assert_close(
    gpu_result: torch.Tensor, expected: np.ndarray, rtol: float = 1e-10
) -> None:
    """Assert GPU result matches expected values."""
    actual = gpu_result.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=rtol)


# ---------------------------------------------------------------------------
# ISA functions on GPU
# ---------------------------------------------------------------------------


class TestISAOnGPU:
    """Verify ISA atmosphere functions produce correct results on GPU."""

    altitudes_m = np.array([0.0, 5000.0, 10668.0, 15000.0])

    def test_theta_gpu_matches_cpu(self):
        cpu = isa.theta(self.altitudes_m)
        gpu = isa.theta(_to_gpu(self.altitudes_m))
        _assert_close(gpu, cpu)

    def test_delta_gpu_matches_cpu(self):
        cpu = isa.delta(self.altitudes_m)
        gpu = isa.delta(_to_gpu(self.altitudes_m))
        _assert_close(gpu, cpu)

    def test_air_temperature_gpu_matches_cpu(self):
        cpu = isa.air_temperature(self.altitudes_m)
        gpu = isa.air_temperature(_to_gpu(self.altitudes_m))
        _assert_close(gpu, cpu)

    def test_air_pressure_gpu_matches_cpu(self):
        cpu = isa.air_pressure(self.altitudes_m)
        gpu = isa.air_pressure(_to_gpu(self.altitudes_m))
        _assert_close(gpu, cpu)

    def test_speed_of_sound_gpu_matches_cpu(self):
        cpu_state = isa.AtmosphericState(self.altitudes_m)
        gpu_state = isa.AtmosphericState(_to_gpu(self.altitudes_m))
        _assert_close(gpu_state.speed_of_sound, cpu_state.speed_of_sound)

    def test_result_stays_on_gpu(self):
        """Ensure output tensor remains on the same CUDA device."""
        gpu_alt = _to_gpu(self.altitudes_m)
        result = isa.theta(gpu_alt)
        assert result.device.type == "cuda"


# ---------------------------------------------------------------------------
# Aerodynamic functions on GPU
# ---------------------------------------------------------------------------


class TestAeroOnGPU:
    """Verify aerodynamic helper functions on GPU."""

    def test_dynamic_pressure_gpu_matches_cpu(self):
        mach = np.array([0.3, 0.5, 0.78])
        pressure = np.array([101325.0, 54048.0, 22632.0])
        cpu = aero.dynamic_pressure(mach, pressure)
        gpu = aero.dynamic_pressure(_to_gpu(mach), _to_gpu(pressure))
        _assert_close(gpu, cpu)


# ---------------------------------------------------------------------------
# aetherx.numpy operations on GPU
# ---------------------------------------------------------------------------


class TestNumpyOpsOnGPU:
    """Verify aetherx.numpy dispatches correctly on GPU tensors."""

    def test_sin_cos_gpu(self):
        x_np = np.linspace(0, 2 * np.pi, 100)
        x_gpu = _to_gpu(x_np)

        sin_cpu = np.sin(x_np)
        sin_gpu = anp.sin(x_gpu)
        _assert_close(sin_gpu, sin_cpu, rtol=1e-9)

        cos_cpu = np.cos(x_np)
        cos_gpu = anp.cos(x_gpu)
        _assert_close(cos_gpu, cos_cpu, rtol=1e-9)

    def test_exp_log_gpu(self):
        x_np = np.array([0.1, 1.0, 2.0, 5.0])
        x_gpu = _to_gpu(x_np)

        exp_cpu = np.exp(x_np)
        exp_gpu = anp.exp(x_gpu)
        _assert_close(exp_gpu, exp_cpu, rtol=1e-9)

        log_cpu = np.log(x_np)
        log_gpu = anp.log(x_gpu)
        _assert_close(log_gpu, log_cpu, rtol=1e-9)

    def test_sqrt_gpu(self):
        x_np = np.array([1.0, 4.0, 9.0, 16.0])
        x_gpu = _to_gpu(x_np)
        cpu = np.sqrt(x_np)
        gpu = anp.sqrt(x_gpu)
        _assert_close(gpu, cpu)

    def test_where_gpu(self):
        cond_np = np.array([True, False, True, False])
        a_np = np.array([1.0, 2.0, 3.0, 4.0])
        b_np = np.array([10.0, 20.0, 30.0, 40.0])
        cpu = np.where(cond_np, a_np, b_np)

        gpu = anp.where(
            torch.tensor(cond_np, dtype=torch.bool, device="cuda"),
            _to_gpu(a_np),
            _to_gpu(b_np),
        )
        _assert_close(gpu, cpu)

    def test_clip_gpu(self):
        x_np = np.array([-1.0, 0.5, 1.5, 3.0])
        cpu = np.clip(x_np, 0.0, 2.0)
        gpu = anp.clip(_to_gpu(x_np), 0.0, 2.0)
        _assert_close(gpu, cpu)

    def test_sum_gpu(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        cpu_all = np.sum(x_np)
        gpu_all = anp.sum(_to_gpu(x_np))
        assert float(gpu_all.cpu()) == pytest.approx(float(cpu_all))


# ---------------------------------------------------------------------------
# Gradient flow on GPU
# ---------------------------------------------------------------------------


class TestGradientFlowOnGPU:
    """Verify that autograd gradients flow through AetherX on GPU."""

    def test_isa_gradient_on_gpu(self):
        h = torch.tensor(
            [10000.0], dtype=torch.float64, device="cuda", requires_grad=True
        )
        T = isa.air_temperature(h)
        T.backward()
        assert h.grad is not None
        assert h.grad.device.type == "cuda"
        assert torch.isfinite(h.grad).all()

    def test_multi_variable_gradient_on_gpu(self):
        h = torch.tensor(
            [10668.0], dtype=torch.float64, device="cuda", requires_grad=True
        )
        m = torch.tensor([0.78], dtype=torch.float64, device="cuda", requires_grad=True)
        T = isa.air_temperature(h)
        p = isa.air_pressure(h)
        q = aero.dynamic_pressure(m, p)
        loss = (T * q).sum()
        loss.backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()
        assert m.grad is not None and torch.isfinite(m.grad).all()
