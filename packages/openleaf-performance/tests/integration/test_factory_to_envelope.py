"""End-to-end integration: factory -> trajectory -> performance -> envelope.

These tests exercise the full public surface in one shot. They are intentionally
coarser than the unit tests in ``tests/unit/`` — the goal is to catch any
breakage that only surfaces when the layers are wired together.

Two variants:

- ``test_openap_end_to_end``: uses OpenAP, which ships with its own data, so it
  runs for every contributor locally.
- ``test_bada4_end_to_end``: uses BADA4 and skips cleanly when
  ``OPENLEAF_BADA4_PATH`` is unset (no proprietary data available locally).

CI runs both because it checks out the private bada-data repo and sets the
env var.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from openleaf.performance.core.factory import ModelFactory
from openleaf.performance.core.trajectory import TrajectoryData
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import EnvelopeMode


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared trajectory fixture
# ---------------------------------------------------------------------------


def _cruise_trajectory() -> TrajectoryData:
    """Return a two-point cruise trajectory at FL330 -> FL345.

    Both points sit inside any reasonable transport-category envelope, so the
    test stays meaningful regardless of which model is plugged in.
    """
    return TrajectoryData(
        {
            "altitude": np.array([10_000.0, 10_500.0]),  # m
            "true_airspeed": np.array([230.0, 232.0]),  # m/s
            "vertical_speed": np.array([5.0, 5.0]),  # m/s
            "aircraft_mass": np.array([65_000.0, 64_900.0]),  # kg
            "acceleration": np.array([0.1, 0.1]),  # m/s²
            "aerodynamic_configuration": np.array([AerodynamicConfig.CR, AerodynamicConfig.CR]),
        }
    )


def _assert_realistic_cruise_outputs(perf) -> None:
    """Assert that a two-point A320-class cruise produces physically plausible numbers."""
    # Per-row outputs are length 2.
    assert len(perf) == 2

    # Aerodynamic coefficients in a sane range for a transport jet at cruise.
    cd = np.asarray(perf.drag_coef)
    cl = np.asarray(perf.lift_coef)
    assert np.all((cd > 0.01) & (cd < 0.1)), f"drag_coef out of range: {cd}"
    assert np.all((cl > 0.1) & (cl < 1.0)), f"lift_coef out of range: {cl}"

    # Forces and fuel flow strictly positive for a climbing aircraft.
    assert np.all(np.asarray(perf.drag) > 0)
    assert np.all(np.asarray(perf.thrust) > 0)
    assert np.all(np.asarray(perf.fuel_flow) > 0)

    # Envelope limits readable and ordered consistently.
    assert float(perf.max_altitude) > 30_000.0  # ft
    assert float(perf.max_cas) > float(perf.min_cas)
    assert float(perf.max_mach) > 0.5


# ---------------------------------------------------------------------------
# OpenAP — always runs
# ---------------------------------------------------------------------------


def test_openap_end_to_end() -> None:
    """ModelFactory -> TrajectoryData -> AircraftTrajectoryData with OpenAP.

    OpenAP ships with its own data so this runs locally for any contributor.
    Exercises factory dispatch, model construction, trajectory wrapping,
    derived properties (mach, density), and the performance chain
    (lift_coef -> drag_coef -> drag -> thrust -> fuel_flow).
    """
    model = ModelFactory.load("A320", "openap")
    assert model.model == "A320"
    assert model.mtom > 0.0

    traj = _cruise_trajectory()
    perf = traj.with_aircraft(model, envelope_mode=EnvelopeMode.IGNORE)

    # The wrapping layer should re-expose the trajectory's derived properties.
    mach = np.asarray(perf.mach_number)
    assert np.all((mach > 0.5) & (mach < 1.0))

    _assert_realistic_cruise_outputs(perf)


# ---------------------------------------------------------------------------
# BADA4 — only runs when proprietary data is available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENLEAF_BADA4_PATH"),
    reason="OPENLEAF_BADA4_PATH not set; skipping BADA4 integration test "
    "(BADA data is proprietary and not redistributed with this repo).",
)
def test_bada4_end_to_end() -> None:
    """Same end-to-end flow as the OpenAP test, but using BADA4.

    Skipped automatically when ``OPENLEAF_BADA4_PATH`` is unset so contributors
    without BADA access see a green build. CI sets the env var and runs the
    test against the private bada-data checkout.
    """
    # The factory must auto-detect that BADA4 is available before the user
    # explicitly asks for it. Passing a non-None model_type narrows the
    # return type to list[str].
    available = ModelFactory.available_aircraft("bada4")
    assert isinstance(available, list)
    assert "A320" in available or any(a.startswith("A320") for a in available)

    model = ModelFactory.load("A320", "bada4")
    assert model.mtom > 0.0

    traj = _cruise_trajectory()
    perf = traj.with_aircraft(model, envelope_mode=EnvelopeMode.IGNORE)
    _assert_realistic_cruise_outputs(perf)
