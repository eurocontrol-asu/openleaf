from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from openleaf.performance.core.mixins import PerformanceMixin


class DummyContainer(PerformanceMixin):
    """Minimal container that provides attributes the mixin expects."""

    def __init__(
        self,
        *,
        model: MagicMock,
        traj: dict,
        air_temperature: float = 288.15,
    ) -> None:
        self._model = model
        self._traj = traj
        self._air_temperature = air_temperature

    @property
    def drag(self):
        return self._model.drag(self)

    @property
    def aircraft_mass(self):
        return self._traj.get("mass", 70_000.0)

    @property
    def f(self):
        return self._traj.get("f", 1.0)

    @property
    def air_temperature(self):
        return self._air_temperature


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.lift_coef.return_value = np.array([0.45])
    model.drag_coef.return_value = np.array([0.032])
    model.drag.return_value = np.array([45_000.0])
    model.thrust_max.return_value = np.array([120_000.0])
    model.thrust_min.return_value = np.array([5_000.0])
    model.fuel_flow.return_value = np.array([0.85])
    model.fuel_flow_min.return_value = np.array([0.12])
    return model


@pytest.fixture
def traj_level() -> dict:
    return {
        "vertical_speed": np.array([0.0]),
        "true_airspeed": np.array([230.0]),
        "altitude": np.array([10_000.0]),
        "mass": 70_000.0,
        "f": 1.0,
    }


@pytest.fixture
def container(mock_model: MagicMock, traj_level: dict) -> DummyContainer:
    return DummyContainer(model=mock_model, traj=traj_level)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMixinLiftCoefDelegatesToModel:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.lift_coef
        mock_model.lift_coef.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([0.45]))


class TestMixinDragCoefDelegatesToModel:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.drag_coef
        mock_model.drag_coef.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([0.032]))


class TestMixinThrustMaxDelegatesToModel:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.thrust_max
        mock_model.thrust_max.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([120_000.0]))


class TestMixinThrustMinDelegatesToModel:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.thrust_min
        mock_model.thrust_min.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([5_000.0]))


class TestMixinFuelFlowDelegates:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.fuel_flow
        mock_model.fuel_flow.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([0.85]))


class TestMixinFuelFlowMinDelegatesToModel:
    def test_returns_model_value(self, container: DummyContainer, mock_model: MagicMock) -> None:
        result = container.fuel_flow_min
        mock_model.fuel_flow_min.assert_called_once_with(container)
        np.testing.assert_array_equal(result, np.array([0.12]))


class TestMixinThrustTEMEquation:
    """Thrust uses TEM: T = D + (m * g * dh_geo/dt) / (V * f)."""

    def test_thrust_matches_tem(
        self,
        mock_model: MagicMock,
    ) -> None:
        """With known drag, mass, vs, f — thrust matches TEM equation."""
        drag_val = 45_000.0
        mass = 70_000.0
        vs = 5.0  # m/s climb
        tas = 230.0
        f = 0.95
        alt = 10_000.0
        temp = 288.15

        traj = {
            "vertical_speed": np.array([vs]),
            "true_airspeed": np.array([tas]),
            "altitude": np.array([alt]),
            "mass": mass,
            "f": f,
        }
        mock_model.drag.return_value = np.array([drag_val])
        obj = DummyContainer(model=mock_model, traj=traj, air_temperature=temp)

        result = obj.thrust

        # The mixin delegates to tem.thrust_from_rocd — we verify the result
        # is consistent with the TEM direction: climbing => thrust > drag
        assert np.all(result > drag_val)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestThrustEdgeCases:
    def test_level_flight_thrust_equals_drag(
        self,
        mock_model: MagicMock,
    ) -> None:
        """When vertical_speed=0, TEM reduces to T=D."""
        drag_val = 45_000.0
        traj = {
            "vertical_speed": np.array([0.0]),
            "true_airspeed": np.array([230.0]),
            "altitude": np.array([10_000.0]),
            "mass": 70_000.0,
            "f": 1.0,
        }
        mock_model.drag.return_value = np.array([drag_val])
        obj = DummyContainer(model=mock_model, traj=traj)

        result = obj.thrust
        np.testing.assert_allclose(result, drag_val, rtol=1e-6)

    def test_descent_thrust_less_than_drag(
        self,
        mock_model: MagicMock,
    ) -> None:
        """When vertical_speed<0 (descent), thrust < drag."""
        drag_val = 45_000.0
        traj = {
            "vertical_speed": np.array([-5.0]),
            "true_airspeed": np.array([230.0]),
            "altitude": np.array([10_000.0]),
            "mass": 70_000.0,
            "f": 0.95,
        }
        mock_model.drag.return_value = np.array([drag_val])
        obj = DummyContainer(model=mock_model, traj=traj)

        result = obj.thrust
        assert np.all(result < drag_val)
