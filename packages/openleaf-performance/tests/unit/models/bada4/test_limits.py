"""Unit tests for the BADA4 geometric limits (Eq. 6.1-1), on stand-in data."""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.models.bada4.limits import BADA4Limits
from openleaf.performance.models.bada4.models import BADA4Model
from openleaf.performance.models.bada4.schemas import BADA4BuffetLimitsData
from openleaf.performance.models.bada4.schemas import BADA4GeometricLimitsData
from openleaf.performance.models.bada4.schemas import BADA4KinematicLimitsData
from openleaf.performance.models.bada4.schemas import BADA4LimitsData


HMO = 41_000.0  # [ft]
MFA = 20_000.0  # [ft]


@pytest.fixture
def limits() -> BADA4Limits:
    """BADA4 limits with distinct HMO and MFA."""
    data = BADA4LimitsData(
        geometric=BADA4GeometricLimitsData(hmo=HMO, mfa=MFA),
        kinematic=BADA4KinematicLimitsData(vmo=350.0, mmo=0.82),
        buffet=BADA4BuffetLimitsData(),
    )
    return BADA4Limits(data)


@pytest.fixture
def model(limits: BADA4Limits) -> BADA4Model:
    """BADA4 model carrying only the stand-in limits (no BADA4 files)."""
    model = BADA4Model.__new__(BADA4Model)
    model.limits = limits
    return model


class TestMaxAltitude:
    """BADA4Limits.max_altitude: HMO when clean, MFA with high lift deployed."""

    def test_defaults_to_clean(self, limits: BADA4Limits) -> None:
        assert limits.max_altitude({}) == HMO

    @pytest.mark.parametrize(("hl_deployed", "expected"), [(False, HMO), (True, MFA)])
    def test_hl_deployed(self, limits: BADA4Limits, hl_deployed: bool, expected: float) -> None:
        assert limits.max_altitude({"hl_deployed": hl_deployed}) == expected

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (AerodynamicConfig.CR, HMO),
            (AerodynamicConfig.IC, MFA),
            (AerodynamicConfig.TO, MFA),
            (AerodynamicConfig.AP, MFA),
            (AerodynamicConfig.LD, MFA),
        ],
    )
    def test_aerodynamic_configuration(
        self, limits: BADA4Limits, config: AerodynamicConfig, expected: float
    ) -> None:
        assert limits.max_altitude({"aerodynamic_configuration": config}) == expected

    def test_aerodynamic_configuration_takes_precedence(self, limits: BADA4Limits) -> None:
        data = {"aerodynamic_configuration": AerodynamicConfig.CR, "hl_deployed": True}
        assert limits.max_altitude(data) == HMO

    def test_hl_deployed_vectorized(self, limits: BADA4Limits) -> None:
        result = limits.max_altitude({"hl_deployed": np.array([False, True])})
        np.testing.assert_array_equal(result, [HMO, MFA])


class TestMaxAltitudeFor:
    """BADA4Model.max_altitude_for agrees with BADA4Limits.max_altitude."""

    @pytest.mark.parametrize(("hl_deployed", "expected"), [(False, HMO), (True, MFA)])
    def test_hl_deployed(self, model: BADA4Model, hl_deployed: bool, expected: float) -> None:
        assert model.max_altitude_for(hl_deployed) == expected

    def test_default_is_clean(self, model: BADA4Model) -> None:
        assert model.max_altitude_for() == HMO
