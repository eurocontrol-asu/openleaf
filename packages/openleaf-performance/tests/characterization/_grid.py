"""Shared grid definition for engine-model characterization fixtures.

Both the regenerator and the test consumer import from here so the grid
cannot drift between fixture file and assertion. Changing the grid means
regenerating every fixture — that's deliberate.
"""

from __future__ import annotations


__all__ = [
    "AIRCRAFT_BY_PROVIDER",
    "DEFAULT_GRID",
    "FIXTURE_DIR",
    "FixtureGrid",
    "fixture_filename",
]

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FIXTURE_DIR: Path = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class FixtureGrid:
    """Cartesian grid of flight states used as inputs to every engine call."""

    altitudes_m: tuple[float, ...]
    mach_numbers: tuple[float, ...]
    masses_kg: tuple[float, ...]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened, length-aligned NumPy arrays of every grid point."""
        alt, mach, mass = np.meshgrid(
            np.asarray(self.altitudes_m, dtype=float),
            np.asarray(self.mach_numbers, dtype=float),
            np.asarray(self.masses_kg, dtype=float),
            indexing="ij",
        )
        return alt.ravel(), mach.ravel(), mass.ravel()


DEFAULT_GRID = FixtureGrid(
    altitudes_m=(3_000.0, 7_000.0, 10_000.0, 11_500.0),
    mach_numbers=(0.4, 0.6, 0.78),
    masses_kg=(55_000.0, 65_000.0, 75_000.0),
)


# One representative aircraft per (provider, engine_type) cell.
# Chosen for breadth, not depth: A320 covers civil jet, AT75 covers turboprop,
# C172 covers piston. Each provider gets to map its own ICAO codes here so we
# can extend or override per provider in one place.
AIRCRAFT_BY_PROVIDER: dict[str, dict[str, str]] = {
    "bada3": {"jet": "A320", "turboprop": "AT75", "piston": "C172"},
    "bada4": {"jet": "A320", "turboprop": "AT75"},
    "openap": {"jet": "A320", "turboprop": "AT75"},
    "ps": {"jet": "A320"},
}


def fixture_filename(provider: str, engine_type: str, aircraft: str) -> Path:
    """Canonical fixture path for a (provider, engine_type, aircraft) triple."""
    return FIXTURE_DIR / f"{provider}_{engine_type}_{aircraft}.npz"
