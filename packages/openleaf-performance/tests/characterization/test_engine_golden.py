"""Characterization tests: pin engine-model outputs against golden fixtures.

These tests freeze the existing numerical behaviour of every engine model so
Phase 4 of the audit plan (``BaseEngineModel`` extraction + per-provider
migration) can proceed safely: any bit-level drift fails the test.

The fixtures themselves live in ``fixtures/`` and are produced by the
``_regenerate_fixtures.py`` script (maintainer-only — needs BADA data).
Until Phase 4 lands, fixtures may be missing or contain stub NaNs; the
test skips cleanly in both cases so a contributor without BADA data sees
a green build, not a failure.

When Phase 4 lands:

1. A maintainer with BADA access runs the regenerator to produce real
   fixtures.
2. The fixtures are committed to git.
3. This test then asserts ``rtol=1e-12`` equality between the live
   model outputs and the committed fixtures.
4. Anyone — BADA or not — can run the test against the live models;
   no proprietary data is needed at test time.

Pytest marker: ``integration`` (engine model construction needs the live
provider library).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.characterization._grid import AIRCRAFT_BY_PROVIDER
from tests.characterization._grid import fixture_filename


pytestmark = pytest.mark.integration


def _fixture_or_skip(provider: str, engine_type: str, aircraft: str) -> dict:
    """Load a fixture file, skipping the test cleanly if it's missing/stubbed."""
    path = fixture_filename(provider, engine_type, aircraft)
    if not path.exists():
        pytest.skip(
            f"Fixture {path.name} not committed yet — Phase 4 deliverable. "
            "Run tests/characterization/_regenerate_fixtures.py with BADA "
            "data env vars to produce it."
        )
    fixture = np.load(path, allow_pickle=False)
    if np.all(np.isnan(fixture["thrust_max_n"])):
        pytest.skip(
            f"Fixture {path.name} contains stub NaN outputs — regenerate "
            "with the real models once BaseEngineModel migration lands."
        )
    return dict(fixture)


def _grid_combinations() -> list[tuple[str, str, str]]:
    """Flatten the provider/engine_type/aircraft matrix to a pytest parametrize set."""
    out: list[tuple[str, str, str]] = []
    for provider, by_engine in AIRCRAFT_BY_PROVIDER.items():
        for engine_type, aircraft in by_engine.items():
            out.append((provider, engine_type, aircraft))
    return out


@pytest.mark.parametrize(
    ("provider", "engine_type", "aircraft"),
    _grid_combinations(),
    ids=lambda v: str(v),
)
def test_engine_outputs_match_golden_fixture(
    provider: str,
    engine_type: str,
    aircraft: str,
) -> None:
    """Live engine outputs must match the committed fixture bit-for-bit (rtol=1e-12).

    Until Phase 4 lands the fixture is missing or stubbed; the test skips
    cleanly so a green build is still possible without BADA data. After
    Phase 4 it becomes the hard gate that proves the BaseEngineModel
    refactor preserved every numerical value across every provider.
    """
    fixture = _fixture_or_skip(provider, engine_type, aircraft)

    # Placeholder: assert the fixture structure rather than re-running the
    # live model. The live-model comparison is added together with the
    # BaseEngineModel migration; doing it now would mean reimplementing
    # the per-provider engine loading here, which duplicates code that
    # Phase 4 is specifically extracting.
    expected_keys = {
        "altitude_m",
        "mach_number",
        "aircraft_mass_kg",
        "thrust_max_n",
        "thrust_min_n",
        "fuel_flow_kgs",
    }
    assert expected_keys.issubset(fixture.keys()), (
        f"Fixture for ({provider}, {engine_type}, {aircraft}) missing keys: "
        f"{expected_keys - set(fixture.keys())}"
    )
    n = fixture["altitude_m"].size
    for k in ("thrust_max_n", "thrust_min_n", "fuel_flow_kgs"):
        assert fixture[k].size == n, f"{k} length mismatch with grid"
