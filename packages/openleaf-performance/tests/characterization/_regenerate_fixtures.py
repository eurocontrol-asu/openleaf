"""Regenerate engine-model characterization fixtures.

MAINTAINER-ONLY: this script needs the proprietary BADA data and the OpenAP
distribution to compute reference values. Contributors without BADA data
should NOT run this — they should rely on the fixtures already committed
to ``fixtures/``.

The leading underscore in the filename prevents pytest from collecting it.

Run as::

    OPENLEAF_BADA3_PATH=/path/to/bada/3.16 \
    OPENLEAF_BADA4_PATH=/path/to/bada/4.2.1 \
    uv run python -m tests.characterization._regenerate_fixtures

It writes one ``.npz`` per (provider, engine_type, aircraft) triple in
``tests/characterization/fixtures/``. Each file holds the input grid and
the reference outputs (thrust_max, thrust_min, fuel_flow) so
``test_engine_golden.py`` can assert behaviour preservation without
needing the proprietary data at runtime.

NOTE: this scaffold writes the grid metadata but stubs the actual
computation — Phase 4 of the audit plan replaces the stub with calls to
each provider's engine model. Doing so requires (1) BADA data, (2) the
``BaseEngineModel`` extraction, and (3) a maintainer to confirm the
values are correct before committing. This file is included now so the
infrastructure is in place when Phase 4 lands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from tests.characterization._grid import AIRCRAFT_BY_PROVIDER
from tests.characterization._grid import DEFAULT_GRID
from tests.characterization._grid import fixture_filename


def _require_bada_data() -> None:
    """Abort with a clear message if BADA data env vars are not set."""
    missing = [
        var for var in ("OPENLEAF_BADA3_PATH", "OPENLEAF_BADA4_PATH") if not os.environ.get(var)
    ]
    if missing:
        print(
            f"Refusing to regenerate without {', '.join(missing)} set. "
            "BADA data is proprietary; only maintainers with access should "
            "regenerate fixtures.",
            file=sys.stderr,
        )
        sys.exit(2)


def regenerate_one(
    provider: str,
    engine_type: str,
    aircraft: str,
    fixture_path: Path,
) -> None:
    """Write the fixture file for a single (provider, engine_type, aircraft).

    Phase 4 will replace the body with calls to the real models. Until
    BaseEngineModel exists and the provider migrations are done, the
    function only writes the input grid plus NaN-filled output arrays so
    the file structure is correct and downstream tests can skip cleanly.
    """
    alt, mach, mass = DEFAULT_GRID.as_arrays()
    n = alt.size
    nan = np.full(n, np.nan)
    np.savez_compressed(
        fixture_path,
        # Inputs
        altitude_m=alt,
        mach_number=mach,
        aircraft_mass_kg=mass,
        # Outputs (filled in by Phase 4)
        thrust_max_n=nan,
        thrust_min_n=nan,
        fuel_flow_kgs=nan,
        # Metadata
        provider=np.array(provider),
        engine_type=np.array(engine_type),
        aircraft=np.array(aircraft),
    )
    print(f"wrote {fixture_path.relative_to(Path.cwd())} ({n} points)")


def main() -> int:
    """Walk the (provider, engine_type, aircraft) matrix and write fixtures."""
    _require_bada_data()
    for provider, by_engine in AIRCRAFT_BY_PROVIDER.items():
        for engine_type, aircraft in by_engine.items():
            path = fixture_filename(provider, engine_type, aircraft)
            path.parent.mkdir(parents=True, exist_ok=True)
            regenerate_one(provider, engine_type, aircraft, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
