"""Sanity tests for the BaseEngineModel skeleton.

The skeleton itself is intentionally thin — it carries no provider math.
These tests just verify that the migration contract works:

- the generic over (EngineDataT, CoeffSetT) compiles and accepts both;
- the coefficient set survives round-trip from constructor to property;
- subclasses must still implement the abstract engine methods inherited
  from EngineModel (so the @abstractmethod gate is not bypassed by the
  extra constructor parameter).

Phase 4 will replace this file with the real per-provider migration tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from openleaf.performance.core.engine_base import BaseEngineModel
from openleaf.performance.core.validation import requires


@dataclass(frozen=True)
class _StubEngineData:
    """Minimal engine data placeholder."""

    name: str
    n_engines: int


@dataclass(frozen=True)
class _StubCoeffs:
    """Minimal coefficient set placeholder."""

    thrust_scale: float
    fuel_scale: float


class _StubEngine(BaseEngineModel[_StubEngineData, _StubCoeffs]):
    """A trivial concrete subclass used purely to exercise the skeleton.

    All four engine methods are implemented as constant returns scaled by
    the coefficient set so the test can confirm the wiring without
    depending on any provider library.
    """

    @requires("altitude")
    def thrust_max(self, data):
        return float(data["altitude"]) * self.coefficients.thrust_scale

    @requires("altitude")
    def thrust_min(self, data):
        return float(data["altitude"]) * self.coefficients.thrust_scale * 0.1

    @requires("thrust")
    def fuel_flow(self, data):
        return float(data["thrust"]) * self.coefficients.fuel_scale

    @requires("altitude")
    def fuel_flow_min(self, data):
        return float(data["altitude"]) * self.coefficients.fuel_scale * 0.01


class TestBaseEngineModelSkeleton:
    def test_constructor_stores_data_and_coefficients(self) -> None:
        engine_data = _StubEngineData(name="stub", n_engines=2)
        coeffs = _StubCoeffs(thrust_scale=0.5, fuel_scale=0.0001)
        eng = _StubEngine(engine_data, coeffs)

        assert eng.data is engine_data
        assert eng.coefficients is coeffs

    def test_subclass_must_still_implement_abstract_engine_methods(self) -> None:
        """The extra ``coefficients`` argument doesn't bypass @abstractmethod."""

        class _IncompleteEngine(BaseEngineModel[_StubEngineData, _StubCoeffs]):
            """No engine methods implemented."""

        with pytest.raises(TypeError):
            _IncompleteEngine(  # type: ignore[abstract]
                _StubEngineData(name="x", n_engines=1),
                _StubCoeffs(thrust_scale=1.0, fuel_scale=1.0),
            )

    def test_dispatch_uses_coefficient_set(self) -> None:
        coeffs = _StubCoeffs(thrust_scale=2.0, fuel_scale=0.5)
        eng = _StubEngine(_StubEngineData(name="x", n_engines=1), coeffs)

        assert eng.thrust_max({"altitude": 10.0}) == 20.0
        assert eng.fuel_flow({"thrust": 4.0}) == 2.0

    def test_concrete_methods_have_required_decorator(self) -> None:
        """The skeleton does not weaken the @requires enforcement gate."""
        for name in ("thrust_max", "thrust_min", "fuel_flow", "fuel_flow_min"):
            fn = getattr(_StubEngine, name)
            assert hasattr(fn, "_schema"), f"{name} is missing @requires"
